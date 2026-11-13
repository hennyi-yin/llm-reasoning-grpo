"""Independent 50-step MATH GRPO pilot: screen, train, and evaluate."""

import argparse
from functools import lru_cache
import re

from .common import load_config, sha256_file, write_json
from .evaluate import judge
from . import followup_answers
from .followup_answers import final_candidate
from .math_pilot_data import CONFIG, OUT, WORK, items, prepare
from .prompts import build_messages
from .stats import mcnemar_exact
from .training import memory_decision


@lru_cache(maxsize=4096)
def parse_math(candidate):
    from math_verify import LatexExtractionConfig, parse
    if not candidate or len(candidate) > 300 or "\\boxed{" in candidate:
        return ()
    # Pass only the identified final answer, never the complete derivation.
    results = parse(r"\boxed{" + candidate + "}",
                    extraction_config=[LatexExtractionConfig(boxed_match_priority=0)],
                    extraction_mode="first_match", fallback_mode="no_fallback",
                    parsing_timeout=2)
    return tuple(results)


def normalize_word(text):
    wrapper = re.fullmatch(r"\\(?:text|mathrm|mbox)\{([A-Za-z][A-Za-z\s-]+)\}", text)
    normalized = " ".join((wrapper[1] if wrapper else text).casefold().split())
    labels = {"yes", "no", "true", "false", "none", "undefined", "no solution",
              "does not exist", "all real numbers", "infinitely many solutions", "impossible"}
    return normalized if wrapper or normalized in labels else ""


def score(gold, completion, *, truncated=False):
    from math_verify import verify
    answer, source = final_candidate(completion)
    if answer is None or (truncated and source not in {"boxed", "terminal_display"}):
        return None, "unresolved", False
    if source == "explicit_final" and re.search(r"\b(?:or|maybe|perhaps|either)\b", answer, re.I):
        return answer, "unresolved", False
    target_word, prediction_word = normalize_word(gold), normalize_word(answer)
    if target_word and prediction_word:
        return answer, source, target_word == prediction_word
    gold_parsed = parse_math(gold)
    prediction = parse_math(answer)
    if not gold_parsed or not prediction:
        return answer, "unresolved", False
    return answer, source, bool(verify(list(gold_parsed), list(prediction), timeout_seconds=2))


def reward(completions, gold, **kwargs):
    from .rewards import completion_text
    if len(completions) != len(gold):
        raise ValueError("Unpaired training completions")
    return [float(score(g, completion_text(c))[2]) for g, c in zip(gold, completions)]


def local_base():
    from huggingface_hub import snapshot_download
    decision = memory_decision()
    path = snapshot_download(decision["model_id"], revision=decision["revision"],
                             local_files_only=True,
                             allow_patterns=["*.json", "*.safetensors", "*.txt", "tokenizer.*"])
    return path, decision


def infer(source, *, n, temperature, max_tokens, seed, adapter=None, enable_lora=False):
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    base, _ = local_base()
    llm = LLM(model=base, dtype="bfloat16", seed=seed, gpu_memory_utilization=0.58,
              max_model_len=3072, max_num_seqs=64, enforce_eager=True,
              enable_flashinfer_autotune=False if enable_lora else None,
              enable_lora=enable_lora, max_lora_rank=16)
    params = SamplingParams(n=n, temperature=temperature, top_p=1.0, top_k=-1,
                            max_tokens=max_tokens, seed=seed)
    lora = LoRARequest("math-pilot", 1, str(adapter)) if adapter else None
    outputs = llm.chat([build_messages(row["question"]) for row in source], params,
                       lora_request=lora)
    if len(outputs) != len(source) or any(len(x.outputs) != n for x in outputs):
        raise RuntimeError("Incomplete MATH generations")
    return [[{"text": c.text, "n_tokens": len(c.token_ids), "truncated": c.finish_reason == "length"}
             for c in o.outputs] for o in outputs]


def screen(*, reuse_generations=False):
    cfg = load_config(CONFIG)["sampling"]
    source = items("screen")
    invalid = [row["id"] for row in source if not parse_math(row["gold"]) and not normalize_word(row["gold"])]
    invalid_set = set(invalid)
    eligible = [row for row in source if row["id"] not in invalid_set]
    if not eligible:
        raise RuntimeError("No verifiable MATH references")
    raw_path = WORK / "math_screen_generations.json"
    if reuse_generations:
        saved = {x["id"]: x["outputs"] for x in load_config(raw_path)["rows"]}
        if not all(x["id"] in saved and len(saved[x["id"]]) == cfg["n"] for x in eligible):
            raise RuntimeError("Saved screening generations do not match the fixed split")
        generations = [saved[x["id"]] for x in eligible]
    else:
        generations = infer(eligible, n=cfg["n"], temperature=cfg["temperature"],
                            max_tokens=cfg["max_tokens"], seed=cfg["seed"])
    selected = []
    group_counts = {"all_correct": 0, "all_wrong": 0, "mixed": 0}
    details = []
    for row, generated in zip(eligible, generations):
        for out in generated:
            out["candidate"], out["source"], out["correct"] = score(
                row["gold"], out["text"], truncated=out["truncated"])
        correct = sum(out["correct"] for out in generated)
        group = "all_correct" if correct == cfg["n"] else "all_wrong" if not correct else "mixed"
        group_counts[group] += 1
        # Require at least one resolved wrong answer as well as a correct one.
        resolved = [out["correct"] for out in generated
                    if out["source"] != "unresolved" and not out["truncated"]]
        if group == "mixed" and any(resolved) and not all(resolved):
            selected.append(row)
        details.append({"id": row["id"], "group": group, "outputs": generated})
    total = cfg["n"] * len(eligible)
    truncated = sum(o["truncated"] for row in details for o in row["outputs"])
    resolved = sum(o["source"] != "unresolved" for row in details for o in row["outputs"])
    train_path = WORK / "math_train_selected.json"
    write_json(train_path, {"items": selected})
    write_json(raw_path, {"rows": details})
    # Coverage is enforced for reference answers. Missing output answers are
    # counted separately and never counted as resolved evidence of a wrong answer.
    coverage = len(eligible) / len(source)
    result = {"status": "ready" if len(selected) >= cfg["min_mixed_prompts"]
              and truncated / total <= cfg["max_truncation_fraction"]
              and coverage >= cfg["min_judge_coverage"] else "gate_failed",
              "screened": len(source), "invalid_reference_ids": invalid, "groups": group_counts,
              "selected_ids": [row["id"] for row in selected], "train_sha256": sha256_file(train_path),
              "rollouts": total, "truncation_fraction": truncated / total,
              "resolved_output_fraction": resolved / total, "reference_coverage": coverage,
              "sampling": cfg, "extractor_sha256": sha256_file(followup_answers.__file__),
              "scorer_sha256": sha256_file(__file__)}
    write_json(OUT / "math_screen.json", result)
    print(result, flush=True)


def train(resume=None):
    from datasets import Dataset
    from peft import LoraConfig
    import torch
    from transformers import set_seed
    from trl import GRPOConfig, GRPOTrainer
    from .training import logging_callbacks, prepare_trainer_kwargs, package_versions, ensure_new_or_resumed
    from .train_grpo import resume_safe_trainer
    if not torch.cuda.is_available():
        raise RuntimeError("MATH GRPO pilot requires a CUDA GPU")
    cfg = load_config(CONFIG)["training"]
    screened = load_config(OUT / "math_screen.json")
    if not resume and (screened.get("extractor_sha256") != sha256_file(followup_answers.__file__)
                       or screened.get("scorer_sha256") != sha256_file(__file__)):
        screen(reuse_generations=True)
        screened = load_config(OUT / "math_screen.json")
    train_path = WORK / "math_train_selected.json"
    if screened["status"] != "ready" or screened["train_sha256"] != sha256_file(train_path):
        raise RuntimeError("Screening gate failed or selected training data was changed")
    source = load_config(train_path)["items"]
    base, decision = local_base()
    template = load_config("configs/grpo_no_sft.yaml")
    path = WORK / "math_adapter"
    config = {"output_dir": str(path), "trainer": dict(template["trainer"])}
    ensure_new_or_resumed(config, resume)
    config["trainer"].update(decision["grpo_overrides"])
    config["trainer"].update(
        {k: cfg[k] for k in ("max_steps", "generation_batch_size", "gradient_accumulation_steps",
                              "save_steps", "max_completion_length")})
    config["trainer"].update(
        reward_weights=[1.0], log_completions=False, logging_steps=1,
        save_total_limit=2, mask_truncated_completions=True,
        transformers_continuous_batching_config={"max_memory_percent": 0.32})
    set_seed(42)
    args = GRPOConfig(**prepare_trainer_kwargs(config, decision["revision"]))
    data = Dataset.from_list([{"prompt": build_messages(x["question"]), "gold": x["gold"]} for x in source])
    log_dir = WORK / "train"
    callback = logging_callbacks(str(log_dir))
    manifest_path = OUT / "math_train_manifest.json"
    manifest = {"model": decision["model_id"], "revision": decision["revision"],
                "package_versions": package_versions(), "train_count": len(data),
                "train_sha256": screened["train_sha256"], "lora": template["lora"],
                "trainer": args.to_dict(), "reward": "binary_final_answer_no_format_bonus",
                "extractor_sha256": sha256_file(followup_answers.__file__),
                "scorer_sha256": sha256_file(__file__),
                "monitor": {"max_gpu_used_fraction": cfg["max_gpu_used_fraction"],
                            "memory_grace_seconds": cfg["memory_grace_seconds"]}}
    # Round-trip the manifest comparison because YAML/Trainer may provide tuples.
    import json
    if resume and load_config(manifest_path) != json.loads(json.dumps(manifest)):
        raise ValueError("Cannot resume with a different pilot training protocol")
    if not resume:
        write_json(manifest_path, manifest)
    trainer = resume_safe_trainer(GRPOTrainer)(model=base, args=args, reward_funcs=[reward],
                train_dataset=data, peft_config=LoraConfig(**template["lora"]), callbacks=[callback])
    trainer.train(resume_from_checkpoint=resume)
    final = path / "final_adapter"
    trainer.save_model(str(final))
    trainer.processing_class.save_pretrained(final)
    measurements = callback.measurements()
    if trainer.state.global_step != cfg["max_steps"] or not torch.cuda.is_available():
        raise RuntimeError("Training did not reach the planned GPU steps")
    write_json(OUT / "math_train_summary.json", {**measurements, "final_global_step": trainer.state.global_step})


def evaluate(split, label):
    if split == "confirmation" and load_config(OUT / "math_dev_comparison.json")["decision"] != "confirm":
        raise RuntimeError("Confirmation set is reserved for a positive development gate")
    cfg = load_config(CONFIG)["evaluation"]
    source = items(split)
    adapter = WORK / "math_adapter/final_adapter" if label == "pilot" else None
    if adapter and not (adapter / "adapter_config.json").is_file():
        raise RuntimeError("Missing completed pilot adapter")
    outputs = infer(source, n=1, temperature=0.0, max_tokens=cfg["max_tokens"], seed=cfg["seed"],
                    adapter=adapter, enable_lora=True)
    rows = []
    for problem, (output,) in zip(source, outputs):
        answer, via, correct = score(problem["gold"], output["text"], truncated=output["truncated"])
        strict = judge("math500", problem["gold"], output["text"])
        rows.append({"id": problem["id"], "question_sha256": problem["question_sha256"], **output,
                     "candidate": answer, "source": via, "correct": correct, "strict": strict})
    write_json(WORK / f"{split}_{label}.json", {"rows": rows, "protocol": cfg,
               "split_sha256": sha256_file(WORK / f"math_{split}.json")})


def compare(split):
    import numpy as np
    left, right = (load_config(WORK / f"{split}_{label}.json") for label in ("base", "pilot"))
    if (left["protocol"], left["split_sha256"]) != (right["protocol"], right["split_sha256"]):
        raise ValueError("Evaluation protocol or split differs")
    a, b = left["rows"], right["rows"]
    identity = lambda rows: [(x["id"], x["question_sha256"]) for x in rows]
    if identity(a) != identity(b) or identity(a) != identity(items(split)):
        raise ValueError("Evaluation pair identity differs")
    n = len(a)
    measures = {}
    for key in ("correct", "strict"):
        aa, bb = [x[key] for x in a], [x[key] for x in b]
        gains = sum(not x and y for x, y in zip(aa, bb))
        losses = sum(x and not y for x, y in zip(aa, bb))
        draws = np.random.default_rng(0).multinomial(n, [gains/n, losses/n, (n-gains-losses)/n], 10_000)
        measures[key] = {"base": sum(aa), "pilot": sum(bb), "delta_pp": 100 * (gains - losses) / n,
                         "gains": gains, "losses": losses,
                         "paired_bootstrap_95_ci_pp": np.percentile(
                             100 * (draws[:, 0] - draws[:, 1]) / n, [2.5, 97.5]).tolist(),
                         "paired_p_descriptive": mcnemar_exact(aa, bb)}
    unknown = [sum(x["source"] == "unresolved" for x in rows) for rows in (a, b)]
    truncation = [sum(x["truncated"] for x in rows) for rows in (a, b)]
    cfg = left["protocol"]
    go = (measures["correct"]["delta_pp"] >= cfg["min_dev_gain_pp"]
          and max(unknown) / n <= cfg["max_unresolved_fraction"]
          and 100 * abs(unknown[1] - unknown[0]) / n <= cfg["max_unknown_delta_pp"])
    confirmed = go and measures["correct"]["paired_bootstrap_95_ci_pp"][0] > 0
    result = {"n": n, "split": split, "metrics": measures, "unresolved_base_pilot": unknown,
              "truncated_base_pilot": truncation, "decision": ("confirm" if go else "stop")
              if split == "dev" else ("supported" if confirmed else "inconclusive")}
    write_json(OUT / f"math_{split}_comparison.json", result)
    write_json(OUT / f"math_{split}_paired.json",
               {"split_sha256": left["split_sha256"], "rows": [
                   {"id": x["id"], "question_sha256": x["question_sha256"],
                    "base": {k: x[k] for k in ("source", "correct", "strict", "truncated", "n_tokens")},
                    "pilot": {k: y[k] for k in ("source", "correct", "strict", "truncated", "n_tokens")}}
                   for x, y in zip(a, b)]})
    print(result, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "screen", "train", "eval", "compare"))
    parser.add_argument("--split", choices=("dev", "confirmation"), default="dev")
    parser.add_argument("--label", choices=("base", "pilot"), default="base")
    parser.add_argument("--resume")
    args = parser.parse_args()
    if args.stage == "eval":
        evaluate(args.split, args.label)
    elif args.stage == "compare":
        compare(args.split)
    elif args.stage == "train":
        train(args.resume)
    else:
        {"prepare": prepare, "screen": screen}[args.stage]()
