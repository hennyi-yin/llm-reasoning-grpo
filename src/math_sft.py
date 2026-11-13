"""Matched worked-reference versus answer-only LoRA SFT on MATH."""

import argparse
from collections import Counter, defaultdict
import math
import os
from pathlib import Path
import random
import re

from .common import ROOT, load_config, sha256_file, write_json
from .evaluate import judge
from .extract import last_boxed
from .followup_answers import final_candidate
from .math_pilot import infer, local_base, score
from .math_pilot_data import question_key
from .prompts import build_messages
from .stats import mcnemar_exact


PILOT = ROOT / "results/followup"
CONFIG = ROOT / "configs/math_sft.yaml"
OUT = Path(os.environ.get("LLM_MATH_SFT_OUT", PILOT / "work/math_sft")).resolve()
ARMS = ("base", "short", "worked")
PROSE = re.compile(
    r"(?i)^(?:so|thus|therefore),?\s+[^\n\d]{1,100}?"
    r"\b(?:is|are|has|have|equals?)\s+([+−-]?\d+(?:\.\d+)?)\s+"
    r"[a-z]+(?:\s+[a-z]+){0,4}\.?$"
)


def score_v2(gold, text, *, truncated=False):
    answer, via, ok = score(gold, text, truncated=truncated)
    if via != "unresolved" or truncated:
        return answer, via, ok
    # Accept literal matching math only from an identified final segment.
    candidate, source = final_candidate(text)
    if candidate and source in {"boxed", "terminal_display", "explicit_final", "terminal_expression"}:
        if re.search(r"[0-9\\]", candidate) and not re.search(r"\b(?:or|maybe|perhaps|either)\b", candidate, re.I):
            normalize = lambda s: re.sub(r"\s+", "", s)
            if normalize(candidate) == normalize(gold):
                return candidate, "literal_final_math", True
    final_line = next((x.strip() for x in reversed(text.splitlines()) if x.strip()), "")
    match = PROSE.fullmatch(final_line)
    if match:
        answer, via, ok = score(gold, f"Final answer: {match[1]}")
        if via != "unresolved":
            return answer, "terminal_prose_number", ok
    return None, "unresolved", False


def fit_ids(selected, forbidden, expected):
    ids = [row["id"] for row in selected]
    keys = [row["question_sha256"] for row in selected]
    if (len(ids) != expected or len(set(ids)) != expected or len(set(keys)) != expected
            or any(k in forbidden for k in keys)):
        raise ValueError("SFT training identities overlap held-out questions or contain duplicates")


def chat_lengths(tokenizer, question, reply):
    prompt = build_messages(question)
    prefix = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(
        prompt + [{"role": "assistant", "content": reply}], tokenize=True)
    if isinstance(prefix, dict) or hasattr(prefix, "keys"):
        prefix, full = prefix["input_ids"], full["input_ids"]
    if list(full[:len(prefix)]) != list(prefix):
        raise ValueError("Completion did not preserve the prompt token prefix")
    return len(full), len(full) - len(prefix)


def fixed_split(name):
    entry = load_config(PILOT / "math_split.json")["splits"][name]
    path = PILOT / "work" / f"math_{name}.json"
    if not path.is_file():
        from .math_pilot_data import prepare
        prepare()
    if sha256_file(path) != entry["sha256"]:
        raise ValueError(f"Frozen follow-up split checksum differs: {name}")
    return load_config(path)["items"]


def prepare():
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    cfg = load_config(CONFIG)
    split = load_config(PILOT / "math_split.json")
    blocked_ids = {k for entry in split["splits"].values() for k in entry["ids"]}
    blocked = {x["question_sha256"] for name in split["splits"] for x in fixed_split(name)}
    blocked.update(question_key(load_config(p)["problem"]["question"]) for p in
                   (ROOT / "results/final_eval/base--math500/raw").glob("*.json"))
    base, decision = local_base()
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    rng = random.Random(cfg["seed"])
    selected, rejected = [], Counter()
    seen = set(blocked)
    for filename, digest in sorted(split["file_sha256"].items()):
        subject = filename.split("/")[0]
        path = hf_hub_download(split["repo"], filename, revision=split["revision"], repo_type="dataset",
                               local_files_only=True)
        if sha256_file(path) != digest:
            raise ValueError(f"MATH source file checksum differs: {filename}")
        entries = list(enumerate(pq.read_table(path).to_pylist()))
        rng.shuffle(entries)
        accepted = 0
        for i, item in entries:
            question, solution = item["problem"].strip(), item["solution"].strip()
            key, identity = question_key(question), f"{subject}-{i}"
            if identity in blocked_ids or key in seen:
                rejected["held_out_or_duplicate"] += 1
                continue
            gold = last_boxed(solution)
            if not gold or "[asy]" in question or "[asy]" in solution or solution.count(r"\boxed{") != 1:
                rejected["diagram_or_no_unique_box"] += 1
                continue
            # Require substantive source explanation, a parseable final answer,
            # and an intact training sequence. The source solution is not a
            # model-generated trajectory or an independently certified proof.
            total, completion = chat_lengths(tokenizer, question, solution)
            if total > cfg["max_sequence_tokens"] or completion < cfg["min_solution_tokens"]:
                rejected["length"] += 1
                continue
            if not judge("math500", gold, solution):
                rejected["unverifiable_reference"] += 1
                continue
            short = rf"The final answer is \boxed{{{gold.strip()}}}."
            _, short_tokens = chat_lengths(tokenizer, question, short)
            seen.add(key)
            selected.append(dict(id=identity, question_sha256=key, subject=subject, level=item["level"],
                                 question=question, worked=solution, short=short, gold=gold.strip(),
                                 worked_tokens=completion, short_tokens=short_tokens))
            accepted += 1
            if accepted == cfg["per_subject"]:
                break
        if accepted != cfg["per_subject"]:
            raise RuntimeError(f"Insufficient worked references for {subject}: {accepted}")
    rng.shuffle(selected)
    fit_ids(selected, blocked, cfg["per_subject"] * len(split["file_sha256"]))
    dataset = OUT / "training.json"
    write_json(dataset, {"rows": selected})
    # Audit the same number from each subject before training.
    grouped = defaultdict(list)
    for row in selected:
        grouped[row["subject"]].append(row)
    audit_rng = random.Random(41)
    sample = [row for group in grouped.values() for row in audit_rng.sample(group, 3)]
    write_json(OUT / "review_queue.json", {"rows": sample})
    write_json(OUT / "data_manifest.json", dict(
        model=decision["model_id"], revision=decision["revision"], dataset_sha256=sha256_file(dataset),
        source_revision=split["revision"], source_file_sha256=split["file_sha256"],
        selected_ids=[x["id"] for x in selected], subject_counts=dict(Counter(x["subject"] for x in selected)),
        rejected=dict(rejected), review_ids=sorted(x["id"] for x in sample),
        target_tokens={label: sum(x[f"{label}_tokens"] for x in selected) for label in ("short", "worked")}))


def reviewed_data():
    manifest = load_config(OUT / "data_manifest.json")
    reviews = load_config(ROOT / "results/sft_followup/reference_review.json")["reviews"]
    if sha256_file(OUT / "training.json") != manifest["dataset_sha256"]:
        raise ValueError("Training data changed")
    if set(reviews) != set(manifest["review_ids"]):
        raise ValueError("Fixed reference review is incomplete")
    if sum(v["accept"] for v in reviews.values()) < math.ceil(0.90 * len(reviews)):
        raise RuntimeError("Worked reference review did not meet its 90% acceptance gate")
    rows = load_config(OUT / "training.json")["rows"]
    return [x for x in rows if x["id"] not in reviews or reviews[x["id"]]["accept"]]


def train(label, resume=None):
    from datasets import Dataset
    from peft import LoraConfig
    import torch
    from transformers import set_seed
    from trl import SFTConfig, SFTTrainer
    from .training import logging_callbacks, package_versions

    if label not in ("short", "worked") or not torch.cuda.is_available():
        raise ValueError("SFT requires a treatment arm and a CUDA GPU")
    cfg = load_config(CONFIG)
    base, decision = local_base()
    rows = reviewed_data()
    arm = OUT / label
    if arm.exists() and any(arm.glob("checkpoint-*")) and not resume:
        raise FileExistsError(f"Existing checkpoints in {arm}; explicitly resume")
    template = load_config("configs/sft.yaml")
    values = dict(template["trainer"])
    values.update(output_dir=str(arm), num_train_epochs=1, learning_rate=cfg["learning_rate"],
                  gradient_accumulation_steps=cfg["batch_accumulation"],
                  max_length=cfg["max_sequence_tokens"], logging_steps=2,
                  save_strategy="steps", save_steps=cfg["checkpoint_steps"], save_total_limit=2)
    values["model_init_kwargs"] = {**values["model_init_kwargs"], "local_files_only": True}
    set_seed(cfg["seed"])
    args = SFTConfig(**values)
    data = Dataset.from_list([{"prompt": build_messages(x["question"]),
                              "completion": [{"role": "assistant", "content": x[label]}]} for x in rows])
    callback = logging_callbacks(str(OUT / f"{label}_logs"))
    trainer = SFTTrainer(model=base, args=args, train_dataset=data,
                         peft_config=LoraConfig(**template["lora"]), callbacks=[callback])
    batch = next(iter(trainer.get_train_dataloader()))
    if not (batch["labels"] == -100).any() or not (batch["labels"] != -100).any():
        raise RuntimeError("SFT prompt masking or completion labels are missing")
    manifest_path = OUT / f"{label}_manifest.json"
    manifest = dict(model=decision["model_id"], revision=decision["revision"], arm=label,
                    versions=package_versions(), ids=[x["id"] for x in rows],
                    target_tokens=sum(x[f"{label}_tokens"] for x in rows), trainer=args.to_dict(),
                    dataset_sha256=sha256_file(OUT / "training.json"),
                    review_sha256=sha256_file(ROOT / "results/sft_followup/reference_review.json"))
    if resume and load_config(manifest_path) != manifest:
        raise ValueError("Cannot resume a changed SFT arm")
    if not resume:
        write_json(manifest_path, manifest)
    trainer.train(resume_from_checkpoint=resume)
    if trainer.state.global_step != math.ceil(len(rows) / cfg["batch_accumulation"]):
        raise RuntimeError("SFT did not complete its planned optimizer steps")
    trainer.save_model(str(arm / "final_adapter"))
    trainer.processing_class.save_pretrained(arm / "final_adapter")
    write_json(OUT / f"{label}_train.json", dict(
        **callback.measurements(), final_global_step=trainer.state.global_step, n=len(rows),
        adapter_sha256=sha256_file(arm / "final_adapter/adapter_model.safetensors")))


def evaluate(split, label, *, adapter_override=None):
    if split == "confirmation" and load_config(OUT / "dev_comparison.json")["decision"] != "confirm":
        raise RuntimeError("Confirmation requires a positive dev gate")
    source = fixed_split(split)
    cfg = load_config("configs/math_pilot.yaml")["evaluation"]
    path = PILOT / "work/dev_base.json"
    if split == "dev" and label == "base" and path.is_file():
        if sha256_file(path) != load_config(PILOT / "math_pilot_audit.json")["local_raw_sha256"]["base"]:
            raise ValueError("Existing base generations differ from the audited run")
        old = load_config(path)
        if [(r["id"], r["question_sha256"]) for r in old["rows"]] != [
                (r["id"], r["question_sha256"]) for r in source]:
            raise ValueError("Base generations do not match the evaluation rows")
        generated = old["rows"]
    else:
        adapter = (adapter_override or OUT / label / "final_adapter") if label != "base" else None
        if adapter and not (adapter / "adapter_config.json").is_file():
            raise FileNotFoundError(f"Missing adapter: {adapter}")
        generated = [one[0] for one in infer(source, n=1, temperature=0.0, max_tokens=cfg["max_tokens"],
                                               seed=cfg["seed"], adapter=adapter, enable_lora=True)]
    rows = []
    for item, out in zip(source, generated, strict=True):
        answer, via, correct = score_v2(item["gold"], out["text"], truncated=out["truncated"])
        _, _, previous = score(item["gold"], out["text"], truncated=out["truncated"])
        rows.append(dict(id=item["id"], question_sha256=item["question_sha256"], text=out["text"],
                         candidate=answer, source=via, correct=correct, previous=previous,
                         strict=judge("math500", item["gold"], out["text"]),
                         truncated=out["truncated"], n_tokens=out["n_tokens"]))
    write_json(OUT / f"{split}_{label}.json", {"rows": rows})


def probe_scale():
    """Test a fixed quarter-strength adapter on dev only after a failed gate."""
    import shutil

    if load_config(OUT / "dev_comparison.json")["decision"] != "stop":
        raise RuntimeError("Scale probe requires a completed stopped SFT experiment")
    original = OUT / "worked/final_adapter"
    probe = OUT / "worked_quarter"
    training = load_config(OUT / "worked_train.json")
    weights = original / "adapter_model.safetensors"
    if sha256_file(weights) != training["adapter_sha256"]:
        raise ValueError("Worked adapter weight checksum changed")
    config = load_config(original / "adapter_config.json")
    if (config["r"] != 16 or config["lora_alpha"] != 32 or config.get("alpha_pattern")
            or config.get("rank_pattern") or config.get("use_rslora") or config.get("use_dora")):
        raise ValueError("Unexpected adapter scaling configuration")
    probe.mkdir(exist_ok=True)
    shutil.copyfile(weights, probe / weights.name)
    write_json(probe / "adapter_config.json", {**config, "lora_alpha": 8})
    evaluate("dev", probe.name, adapter_override=probe)
    base = load_config(OUT / "dev_base.json")["rows"]
    quarter = load_config(OUT / f"dev_{probe.name}.json")["rows"]
    if [(r["id"], r["question_sha256"]) for r in base] != [(r["id"], r["question_sha256"]) for r in quarter]:
        raise ValueError("Unpaired dev probe outcomes")
    keys = ("correct", "previous", "strict", "source", "truncated", "n_tokens")
    evidence = {
        "split": "dev", "scale_relative_to_trained_adapter": 0.25,
        "adapter_weights_sha256": training["adapter_sha256"],
        "local_raw_sha256": sha256_file(OUT / f"dev_{probe.name}.json"),
        "comparisons": {k: paired([x[k] for x in base], [x[k] for x in quarter])
                        for k in ("correct", "previous", "strict")},
        "paired": [dict(id=b["id"], question_sha256=b["question_sha256"],
                        base={k: b[k] for k in keys}, quarter={k: q[k] for k in keys})
                   for b, q in zip(base, quarter)],
    }
    write_json(OUT / "dev_scale_probe.json", evidence)


def paired(left, right):
    import numpy as np
    if len(left) != len(right) or not left:
        raise ValueError("Unpaired SFT outcomes")
    n = len(left)
    wins = sum(not a and b for a, b in zip(left, right))
    losses = sum(a and not b for a, b in zip(left, right))
    draws = np.random.default_rng(0).multinomial(n, [wins/n, losses/n, (n-wins-losses)/n], 10_000)
    return dict(left=sum(left), right=sum(right), delta_pp=100*(wins-losses)/n, gains=wins, losses=losses,
                ci_pp=np.percentile(100*(draws[:, 0]-draws[:, 1])/n, [2.5, 97.5]).tolist(),
                mcnemar_p=mcnemar_exact(left, right))


def summarize(split, groups, gate):
    n = len(groups["base"])
    if not n or any(len(rows) != n for rows in groups.values()):
        raise ValueError("Unpaired SFT outcomes")
    counts = {a: {k: sum(r[k] for r in rows) for k in ("correct", "strict", "previous", "truncated")} |
                 {"unresolved": sum(r["source"] == "unresolved" for r in rows),
                  "mean_tokens": sum(r["n_tokens"] for r in rows)/n} for a, rows in groups.items()}
    fields = ("correct", "previous", "strict")
    results = {key: {a: paired([r[key] for r in groups[a]], [r[key] for r in groups["worked"]])
                     for a in ("base", "short")} for key in fields}
    passes = (all(results["correct"][a]["delta_pp"] >= gate[f"min_gain_vs_{a}_pp"]
                  and results["previous"][a]["delta_pp"] > 0
                  and 100*(counts["worked"]["unresolved"]-counts[a]["unresolved"])/n <= gate["max_extra_unresolved_pp"]
                  for a in ("base", "short"))
              and max(x["unresolved"] for x in counts.values())/n <= gate["max_unresolved_fraction"])
    if split == "dev":
        decision = "confirm" if passes else "stop"
    else:
        decision = "supported" if passes and all(x["ci_pp"][0] > 0 for x in results["correct"].values()) else "inconclusive"
    return dict(split=split, n=n, counts=counts, worked_vs=results, decision=decision)


def compare(split):
    groups = {label: load_config(OUT / f"{split}_{label}.json")["rows"] for label in ARMS}
    keys = [(x["id"], x["question_sha256"]) for x in fixed_split(split)]
    if any([(x["id"], x["question_sha256"]) for x in rows] != keys for rows in groups.values()):
        raise ValueError("Matched SFT groups differ")
    fields = ("correct", "previous", "strict")
    result = summarize(split, groups, load_config(CONFIG)["gate"])
    write_json(OUT / f"{split}_comparison.json", result)
    write_json(OUT / f"{split}_paired.json", {"rows": [
        dict(id=key[0], question_sha256=key[1], **{
            arm: {k: groups[arm][i][k] for k in (*fields, "source", "truncated", "n_tokens")}
            for arm in ARMS}) for i, key in enumerate(keys)]})
    print(result, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "train", "eval", "compare", "probe-scale"))
    parser.add_argument("--arm", choices=ARMS, default="base")
    parser.add_argument("--split", choices=("dev", "confirmation"), default="dev")
    parser.add_argument("--resume")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    elif args.action == "train":
        train(args.arm, args.resume)
    elif args.action == "eval":
        evaluate(args.split, args.arm)
    elif args.action == "probe-scale":
        probe_scale()
    else:
        compare(args.split)
