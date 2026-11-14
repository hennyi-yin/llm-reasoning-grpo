"""Small matched pilot of independently reviewed self-generated MATH solutions."""

import argparse
from collections import Counter, defaultdict
import math
import random

from .common import ROOT, load_config, sha256_file, write_json
from .evaluate import judge
from .extract import last_boxed
from .math_independent import REPORT as INDEPENDENT
from .math_pilot import infer, local_base, normalize_word, parse_math, score
from .math_pilot_data import question_key
from .math_sft import PILOT, chat_lengths, fixed_split, paired, score_v2
from .prompts import build_messages

CFG = ROOT / "configs/math_self_sft.yaml"
WORK = PILOT / "work/math_self_sft"
REPORT = ROOT / "results/math_self_sft"
ORIGINAL = PILOT / "work/math_sft"
ARMS = ("base", "reference", "self")
FIELDS = ("correct", "previous", "strict", "source", "truncated", "n_tokens")


def cfg():
    return load_config(CFG)


def checked_source():
    path = ORIGINAL / "training.json"
    source = load_config(ROOT / "results/sft_followup/training_manifest.json")
    if sha256_file(path) != source["dataset_sha256"]:
        raise ValueError("Previously frozen reference data changed")
    ids = set(source["training_ids"])
    rows = [r for r in load_config(path)["rows"] if r["id"] in ids]
    if [r["id"] for r in rows] != source["training_ids"]:
        raise ValueError("Previously frozen training IDs differ")
    return rows


def prepare():
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    config = cfg()
    policy = load_config(INDEPENDENT / "comparison.json")
    if not isinstance(policy["serving_gate"], bool):
        raise ValueError("Independent test gate is missing")
    if policy["serving_gate"] and not (INDEPENDENT / "serving.json").is_file():
        raise RuntimeError("Run the gated serving comparison before step 3")
    prior = load_config(PILOT / "math_split.json")
    excluded = {r["question_sha256"] for r in load_config(INDEPENDENT / "protocol.json")["items"]}
    excluded.update(r["question_sha256"] for r in checked_source())
    excluded.update(r["question_sha256"] for name in prior["splits"] for r in fixed_split(name))
    excluded.update(question_key(load_config(p)["problem"]["question"]) for p in
                    (ROOT / "results/final_eval/base--math500/raw").glob("*.json"))
    base, _ = local_base()
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    rng = random.Random(config["seed"])
    dev = []
    for filename, digest in sorted(prior["file_sha256"].items()):
        subject = filename.split("/")[0]
        path = hf_hub_download(prior["repo"], filename, revision=prior["revision"], repo_type="dataset",
                               local_files_only=True)
        if sha256_file(path) != digest:
            raise ValueError("Source checksum differs")
        entries = list(enumerate(pq.read_table(path).to_pylist()))
        rng.shuffle(entries)
        group = []
        for i, entry in entries:
            q, solution = entry["problem"].strip(), entry["solution"]
            key, gold = question_key(q), last_boxed(solution)
            if (key in excluded or "[asy]" in q or "[asy]" in solution
                    or solution.count(r"\boxed{") != 1 or not gold
                    or not (parse_math(gold.strip()) or normalize_word(gold.strip()))):
                continue
            n, _ = chat_lengths(tokenizer, q, rf"\boxed{{{gold}}}")
            if n > 1024:
                continue
            group.append(dict(id=f"{subject}-{i}", question=q, question_sha256=key,
                              gold=gold.strip(), subject=subject, level=entry["level"]))
            excluded.add(key)
            if len(group) == config["dev_per_subject"]:
                break
        if len(group) != config["dev_per_subject"]:
            raise ValueError(f"Cannot fill fresh dev: {subject}")
        dev.extend(group)
    rng.shuffle(dev)
    pools = defaultdict(list)
    for item in checked_source():
        pools[item["subject"]].append(item)
    n_per, extra = divmod(config["review_pool"], len(pools))
    pool = [item for i, (subject, items) in enumerate(sorted(pools.items()))
            for item in rng.sample(items, n_per + (i < extra))]
    rng.shuffle(pool)
    write_json(WORK / "dev_items.json", {"rows": dev})
    write_json(WORK / "pool.json", {"rows": pool})
    write_json(REPORT / "protocol.json", {
        "config": config, "dev_sha256": sha256_file(WORK / "dev_items.json"),
        "pool_sha256": sha256_file(WORK / "pool.json"),
        "dev_ids": [{"id": r["id"], "question_sha256": r["question_sha256"], "subject": r["subject"]}
                    for r in dev],
        "pool_ids": [r["id"] for r in pool], "dataset_revision": prior["revision"],
        "teacher": "same pinned unadapted Qwen2.5-1.5B-Instruct, prompt only",
        "scorer_sha256": {name: sha256_file(ROOT / "src" / name)
                          for name in ("math_sft.py", "math_pilot.py", "followup_answers.py", "prompts.py")},
    })


def generate():
    from transformers import AutoTokenizer

    c = cfg()
    frozen = load_config(REPORT / "protocol.json")
    if frozen["config"] != c or frozen["pool_sha256"] != sha256_file(WORK / "pool.json"):
        raise ValueError("Frozen self-generation inputs changed")
    pool = load_config(WORK / "pool.json")["rows"]
    base, _ = local_base()
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    output = infer(pool, n=c["generation_samples"], temperature=c["generation_temperature"],
                   max_tokens=c["generation_max_tokens"], seed=c["seed"], enable_lora=True)
    candidates = []
    for item, generations in zip(pool, output, strict=True):
        for i, response in enumerate(generations):
            text = response["text"].strip()
            if (response["truncated"] or text.count(r"\boxed{") != 1
                    or not score_v2(item["gold"], text)[2] or not judge("math500", item["gold"], text)):
                continue
            length, completion = chat_lengths(tokenizer, item["question"], text)
            if length > c["max_sequence_tokens"] or completion < c["min_completion_tokens"]:
                continue
            candidates.append(dict(id=item["id"], sample=i, question=item["question"], gold=item["gold"],
                                   self=text, reference=item["worked"], self_tokens=completion,
                                   reference_tokens=item["worked_tokens"], subject=item["subject"]))
            break
    write_json(WORK / "generated.json", {"rows": output})
    write_json(WORK / "review_queue.json", {"rows": candidates})
    print(f"{len(candidates)} of {len(pool)} had a complete answer-checked candidate", flush=True)


def approved():
    queue = load_config(WORK / "review_queue.json")["rows"]
    audit = load_config(REPORT / "review.json")
    if audit["queue_sha256"] != sha256_file(WORK / "review_queue.json"):
        raise ValueError("Manual audit does not bind to this review queue")
    index = {x["id"]: x for x in queue}
    decisions = audit["reviews"]
    if len(index) != len(queue) or set(decisions) != set(index) or any(
            not isinstance(v["accept"], bool) or not v["note"] for v in decisions.values()):
        raise ValueError("Review decisions are incomplete or stale")
    c = cfg()
    rows = [index[x["id"]] for x in queue if decisions.get(x["id"], {}).get("accept")][:c["max_approved_pairs"]]
    if len(rows) < c["min_approved_pairs"]:
        raise RuntimeError(f"Review gate failed: only {len(rows)} approved matched pairs")
    return rows


def train(arm):
    from datasets import Dataset
    from peft import LoraConfig
    import torch
    from transformers import set_seed
    from trl import SFTConfig, SFTTrainer
    from .training import logging_callbacks, package_versions

    if arm not in ARMS[1:] or not torch.cuda.is_available():
        raise ValueError("Expected a treatment arm and CUDA")
    c, rows = cfg(), approved()
    if c != load_config(REPORT / "protocol.json")["config"]:
        raise ValueError("Training protocol changed after selecting examples")
    output = WORK / arm
    template = load_config("configs/sft.yaml")
    v = dict(template["trainer"])
    v.update(output_dir=str(output), num_train_epochs=c["epochs"], learning_rate=c["learning_rate"],
             gradient_accumulation_steps=c["gradient_accumulation"], max_length=c["max_sequence_tokens"],
             save_strategy="epoch", save_total_limit=2, logging_steps=1)
    v["model_init_kwargs"] = {**v["model_init_kwargs"], "local_files_only": True}
    base, model = local_base()
    set_seed(c["seed"])
    v.update(seed=c["seed"], data_seed=c["seed"])
    data = Dataset.from_list([{"prompt": build_messages(x["question"]),
                              "completion": [{"role": "assistant", "content": x[arm]}]} for x in rows])
    callback = logging_callbacks(str(WORK / f"{arm}_logs"))
    trainer = SFTTrainer(model=base, args=SFTConfig(**v), train_dataset=data,
                         peft_config=LoraConfig(**template["lora"]), callbacks=[callback])
    batch = next(iter(trainer.get_train_dataloader()))
    if not (batch["labels"] == -100).any() or not (batch["labels"] != -100).any():
        raise ValueError("Prompt/completion masking missing")
    checkpoints = sorted(output.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    resume = next((p for p in reversed(checkpoints) if all((p / f).is_file() for f in
                   ("trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth",
                    "adapter_model.safetensors"))), None)
    receipt = {"arm": arm, "ids": [r["id"] for r in rows], "dataset_review": sha256_file(REPORT / "review.json"),
               "config": c, "model": model["model_id"], "revision": model["revision"],
               "versions": package_versions(),
               "target_tokens_per_epoch": sum(r[f"{arm}_tokens"] for r in rows)}
    path = WORK / f"{arm}_train_plan.json"
    if path.exists() and load_config(path) != receipt:
        raise ValueError("Matched training inputs changed")
    if not path.exists():
        write_json(path, receipt)
    trainer.train(resume_from_checkpoint=str(resume) if resume else None)
    expected = math.ceil(len(rows) / c["gradient_accumulation"]) * c["epochs"]
    if trainer.state.global_step != expected:
        raise ValueError("Unexpected optimizer step count")
    trainer.save_model(str(output / "final_adapter"))
    write_json(WORK / f"{arm}_train.json", {**receipt, "steps": expected, **callback.measurements(),
               "adapter_sha256": sha256_file(output / "final_adapter/adapter_model.safetensors")})


def evaluate(arm):
    frozen = load_config(REPORT / "protocol.json")
    if (frozen["dev_sha256"] != sha256_file(WORK / "dev_items.json")
            or any(sha256_file(ROOT / "src" / name) != digest for name, digest in frozen["scorer_sha256"].items())):
        raise ValueError("Fresh development split changed")
    items = load_config(WORK / "dev_items.json")["rows"]
    adapter = WORK / arm / "final_adapter" if arm != "base" else None
    outputs = infer(items, n=1, temperature=0, max_tokens=2048, seed=0,
                    adapter=adapter, enable_lora=True)
    rows = []
    for item, (out,) in zip(items, outputs, strict=True):
        g, text = item["gold"], out["text"]
        _, source, correct = score_v2(g, text, truncated=out["truncated"])
        _, _, previous = score(g, text, truncated=out["truncated"])
        rows.append(dict(id=item["id"], question_sha256=item["question_sha256"], text=text,
                         correct=correct, previous=previous, strict=judge("math500", g, text),
                         source=source, truncated=out["truncated"], n_tokens=out["n_tokens"]))
    write_json(WORK / f"dev_{arm}.json", {"rows": rows})


def summarize(rows, gate):
    n = len(rows)
    comparisons = {a: {k: paired([x[a][k] for x in rows], [x["self"][k] for x in rows])
                       for k in ("correct", "previous", "strict")} for a in ("base", "reference")}
    counts = {a: {"correct": sum(x[a]["correct"] for x in rows),
                  "unresolved": sum(x[a]["source"] == "unresolved" for x in rows),
                  "truncated": sum(x[a]["truncated"] for x in rows),
                  "mean_tokens": sum(x[a]["n_tokens"] for x in rows)/n} for a in ARMS}
    signal = (all(comparisons[a]["correct"]["delta_pp"] >= gate["min_gain_pp"]
                  and comparisons[a]["correct"]["ci_pp"][0] > 0
                  and all(comparisons[a][k]["delta_pp"] > 0 for k in ("previous", "strict"))
                  and 100 * (counts["self"]["unresolved"] - counts[a]["unresolved"]) / n
                  <= gate["max_extra_unresolved_pp"] for a in ("base", "reference"))
              and max(x["unresolved"] for x in counts.values())/n <= gate["max_unresolved_fraction"])
    return dict(n=n, counts=counts, self_vs=comparisons,
                decision="new_confirmatory_study" if signal else "stop",
                grpo_started=False)


def compare():
    plan = load_config(REPORT / "protocol.json")
    arms = [load_config(WORK / f"dev_{label}.json")["rows"] for label in ARMS]
    rows = []
    for key, *group in zip(plan["dev_ids"], *arms, strict=True):
        if any((x["id"], x["question_sha256"]) != (key["id"], key["question_sha256"]) for x in group):
            raise ValueError("Matched pilot evaluation rows differ")
        rows.append({**key, **{a: {k: x[k] for k in FIELDS} for a, x in zip(ARMS, group)}})
    write_json(REPORT / "paired.json", {"rows": rows})
    train = {a: load_config(WORK / f"{a}_train.json") for a in ARMS[1:]}
    if any(train["reference"][k] != train["self"][k] for k in ("ids", "steps", "versions", "model", "revision")):
        raise ValueError("Training pair or optimizer step count differs")
    write_json(REPORT / "training.json", {
        **train, "raw_sha256": {a: sha256_file(WORK / f"dev_{a}.json") for a in ARMS},
        "memory": {a: load_config(WORK / f"train_{a}_memory.json") for a in ARMS[1:]},
        "execution": {
            "generated_examples_sha256": sha256_file(WORK / "generated.json"),
            "review_queue_sha256": sha256_file(WORK / "review_queue.json"),
        },
    })
    result = summarize(rows, plan["config"]["gate"])
    write_json(REPORT / "comparison.json", result)
    print(result, flush=True)


def verify():
    plan = load_config(REPORT / "protocol.json")
    rows = load_config(REPORT / "paired.json")["rows"]
    if [{k: x[k] for k in ("id", "question_sha256", "subject")} for x in rows] != plan["dev_ids"]:
        raise ValueError("Public pilot ledger differs from frozen development split")
    independent = {x["question_sha256"] for x in load_config(INDEPENDENT / "protocol.json")["items"]}
    if independent & {x["question_sha256"] for x in rows}:
        raise ValueError("Independent test leaked into pilot dev")
    arms = load_config(REPORT / "training.json")
    audit = load_config(REPORT / "review.json")
    review = audit["reviews"]
    a, b = (arms[x] for x in ARMS[1:])
    if (any(a[k] != b[k] for k in ("ids", "steps", "versions", "model", "revision"))
            or set(a["ids"]) & {x["id"] for x in rows}
            or not plan["config"]["min_approved_pairs"] <= len(set(a["ids"])) <= plan["config"]["max_approved_pairs"]
            or any(not review[x]["accept"] for x in a["ids"]) or not set(a["ids"]) <= set(plan["pool_ids"])
            or audit["queue_sha256"] != arms["execution"]["review_queue_sha256"]
            or any(t["dataset_review"] != sha256_file(REPORT / "review.json") or t["config"] != plan["config"]
                   for t in (a, b))):
        raise ValueError("Training match, split boundary, or review approval differs")
    result = summarize(rows, plan["config"]["gate"])
    if result != load_config(REPORT / "comparison.json"):
        raise ValueError("Public pilot report differs")
    print(result)


def stages(mode):
    module = "src.math_self_sft"
    jobs = [("prepare", module, ["prepare"], REPORT / "protocol.json", False),
            ("generate", module, ["generate"], WORK / "review_queue.json", True)]
    if mode == "self-compare":
        approved()
        for a in ARMS:
            if a != "base":
                jobs.append((f"train_{a}", module, ["train", "--arm", a], WORK / f"{a}_train.json", True))
            jobs.append((f"dev_{a}", module, ["eval", "--arm", a], WORK / f"dev_{a}.json", True))
        jobs.append(("compare", module, ["compare"], REPORT / "comparison.json", False))
    return WORK, jobs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "generate", "train", "eval", "compare", "verify"))
    parser.add_argument("--arm", choices=ARMS, default="base")
    args = parser.parse_args()
    if args.stage == "train":
        train(args.arm)
    elif args.stage == "eval":
        evaluate(args.arm)
    else:
        {"prepare": prepare, "generate": generate, "compare": compare, "verify": verify}[args.stage]()
