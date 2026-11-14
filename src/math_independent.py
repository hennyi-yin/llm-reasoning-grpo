"""Frozen paired MATH test evaluation of base and the existing low-rate adapter."""

import argparse
from collections import Counter
from pathlib import Path
import random
import shutil
import time

from .common import ROOT, load_config, sha256_file, write_json
from .evaluate import judge
from .extract import last_boxed
from .math_pilot import local_base, normalize_word, parse_math, score
from .math_pilot_data import question_key
from .math_sft import paired, score_v2
from .prompts import build_messages

CFG = ROOT / "configs/math_independent.yaml"
PILOT = ROOT / "results/followup"
WORK = PILOT / "work/math_independent"
REPORT = ROOT / "results/math_independent"
ADAPTER = PILOT / "work/math_sft_conservative/worked/final_adapter"
PRIOR = ROOT / "results/sft_conservative/run.json"
LABELS = ("base", "candidate")
FIELDS = ("correct", "previous", "strict", "source", "truncated", "n_tokens")
PROTOCOL_FILES = (
    "src/math_independent.py", "src/math_sft.py", "src/followup_answers.py",
    "src/math_pilot.py", "src/extract.py", "src/evaluate.py", "src/prompts.py",
)


def protocol_hashes():
    return {name: sha256_file(ROOT / name) for name in PROTOCOL_FILES}


def adapter_hashes():
    prior = load_config(PRIOR)
    weights = sha256_file(ADAPTER / "adapter_model.safetensors")
    if weights != prior["training"]["adapter_sha256"]:
        raise ValueError("The low-rate candidate differs from the previously recorded adapter")
    config = load_config(ADAPTER / "adapter_config.json")
    if (config["r"], config["lora_alpha"], config.get("use_rslora"), config.get("use_dora")) != (16, 32, False, False):
        raise ValueError("Expected the original full-strength rank-16 adapter")
    return {"weights": weights, "config": sha256_file(ADAPTER / "adapter_config.json")}


def select_subject(rows, blocked, n, rng):
    by_level = {}
    for row in rows:
        key = row["question_sha256"]
        if key not in blocked:
            blocked.add(key)
            by_level.setdefault(row["level"], []).append(row)
    if sum(map(len, by_level.values())) < n:
        raise ValueError("Insufficient eligible test questions")
    # Largest remainder preserves eligible subject and difficulty proportions.
    total = sum(map(len, by_level.values()))
    counts = {level: n * len(group) // total for level, group in by_level.items()}
    order = sorted(by_level, key=lambda l: (-(n * len(by_level[l]) % total), l))
    for level in order[:n - sum(counts.values())]:
        counts[level] += 1
    return [row for level in sorted(by_level) for row in rng.sample(by_level[level], counts[level])]


def prepare():
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    if (WORK / "items.json").exists() or (REPORT / "protocol.json").exists():
        raise FileExistsError("Test selection already exists; do not redraw")
    cfg = load_config(CFG)
    previous = load_config(PILOT / "math_split.json")
    files, blocked, eligible, counts = {}, set(), [], Counter()
    rng = random.Random(cfg["seed"])
    base, decision = local_base()
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    generation = cfg["generation"]
    budget = generation["max_model_len"] - generation["max_tokens"]
    for name, checksum in previous["file_sha256"].items():
        path = hf_hub_download(previous["repo"], name, repo_type="dataset",
                               revision=previous["revision"], local_files_only=True)
        if sha256_file(path) != checksum:
            raise ValueError("Source training checksum differs")
        blocked.update(question_key(q) for q in pq.read_table(path, columns=["problem"])["problem"].to_pylist())
    frozen = set()
    for path in (ROOT / "results/final_eval/base--math500/raw").glob("*.json"):
        frozen.add(question_key(load_config(path)["problem"]["question"]))
    if len(frozen) != 500:
        raise ValueError("Expected all 500 previously evaluated fingerprints")
    blocked.update(frozen)
    excluded_before_test = set(blocked)
    counts["unique_train_and_math500_fingerprints"] = len(blocked)
    for train_name in sorted(previous["file_sha256"]):
        name = train_name.replace("/train-", "/test-")
        subject = name.split("/")[0]
        path = hf_hub_download(previous["repo"], name, repo_type="dataset", revision=previous["revision"])
        files[name] = sha256_file(path)
        for i, row in enumerate(pq.read_table(path).to_pylist()):
            counts["official_test_rows"] += 1
            question, solution = row["problem"].strip(), row["solution"]
            key = question_key(question)
            if key in excluded_before_test:
                counts["excluded_train_or_math500"] += 1
                continue
            gold = last_boxed(solution)
            if ("[asy]" in question or "[asy]" in solution or not gold or solution.count(r"\boxed{") != 1
                    or not (parse_math(gold.strip()) or normalize_word(gold.strip()))):
                counts["excluded_diagram_or_unverifiable"] += 1
                continue
            tokens = tokenizer.apply_chat_template(build_messages(question), tokenize=True, add_generation_prompt=True)
            length = len(tokens["input_ids"] if hasattr(tokens, "keys") else tokens)
            if length > budget:
                counts["excluded_context"] += 1
                continue
            eligible.append(dict(id=f"{subject}-test-{i}", subject=subject, level=row["level"],
                                 question=question, question_sha256=key, gold=gold.strip(), prompt_tokens=length))
    if counts["official_test_rows"] != cfg["expected_test_rows"]:
        raise ValueError(f"Unexpected MATH official test count: {counts['official_test_rows']}")
    # Sample proportional to the filtered pool rather than over-representing
    # small subjects. This also avoids an impossible equal-subject quota.
    eligible = [{**row, "level": (row["subject"], row["level"])} for row in eligible]
    selected = select_subject(eligible, blocked, cfg["questions"], rng)
    for row in selected:
        row["level"] = row["level"][1]
    counts["eligible_unique"] = len(blocked) - len(excluded_before_test)
    rng.shuffle(selected)
    write_json(WORK / "items.json", {"rows": selected})
    write_json(REPORT / "protocol.json", {
        "policy": cfg, "adapter": adapter_hashes(), "code_sha256": protocol_hashes(),
        "model": decision["model_id"], "revision": decision["revision"],
        "dataset": previous["repo"], "dataset_revision": previous["revision"], "test_file_sha256": files,
        "exclusion_counts": dict(counts), "items_sha256": sha256_file(WORK / "items.json"),
        "items": [{k: x[k] for k in ("id", "subject", "level", "question_sha256")} for x in selected],
        "prompt": "src.prompts.build_messages", "primary": "src.math_sft.score_v2",
        "temperature": 0.0, "sampling": "greedy", "project_holdout_only": True,
    })
    print(f"Frozen {len(selected)} test items; {dict(counts)}", flush=True)


def ready():
    plan = load_config(REPORT / "protocol.json")
    if (plan["policy"] != load_config(CFG) or plan["adapter"] != adapter_hashes()
            or plan["code_sha256"] != protocol_hashes()
            or plan["items_sha256"] != sha256_file(WORK / "items.json")):
        raise ValueError("Frozen evaluation inputs changed")
    items = load_config(WORK / "items.json")["rows"]
    return plan, items


def generate(label):
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    plan, items = ready()
    g = plan["policy"]["generation"]
    size = g["shard_size"]
    chunks = [items[i:i+size] for i in range(0, len(items), size)]
    base, decision = local_base()
    if decision["revision"] != plan["revision"] or decision["model_id"] != plan["model"]:
        raise ValueError("Base revision differs")
    llm = LLM(model=base, dtype="bfloat16", seed=g["seed"], gpu_memory_utilization=g["gpu_memory_utilization"],
              max_model_len=g["max_model_len"], max_num_seqs=g["max_num_seqs"], enforce_eager=True,
              enable_flashinfer_autotune=False, enable_lora=True, max_lora_rank=16)
    params = SamplingParams(n=1, temperature=0, top_p=1, top_k=-1, max_tokens=g["max_tokens"], seed=g["seed"])
    lora = LoRARequest("low-rate", 1, str(ADAPTER)) if label == "candidate" else None
    for i, chunk in enumerate(chunks):
        path = WORK / f"raw_{label}_{i:02}.json"
        if path.exists():
            old = load_config(path)
            if old["protocol"] != sha256_file(REPORT / "protocol.json") or [x["id"] for x in old["rows"]] != [
                    x["id"] for x in chunk]:
                raise ValueError("Cannot reuse a mismatched shard")
            continue
        start = time.perf_counter()
        outputs = llm.chat([build_messages(q["question"]) for q in chunk], params, lora_request=lora)
        elapsed = time.perf_counter() - start
        rows = []
        for q, o in zip(chunk, outputs, strict=True):
            (c,) = o.outputs
            rows.append(dict(id=q["id"], question_sha256=q["question_sha256"], text=c.text,
                             truncated=c.finish_reason == "length", n_tokens=len(c.token_ids)))
        write_json(path, {"protocol": sha256_file(REPORT / "protocol.json"), "rows": rows, "elapsed": elapsed})
        print(f"{label}: {min((i+1)*size, len(items))}/{len(items)}; shard {elapsed:.1f} s", flush=True)
    write_json(WORK / f"{label}_complete.json", {"n": len(items), "shards": len(chunks)})


def judge_arm(label):
    plan, items = ready()
    raw = [r for path in sorted(WORK.glob(f"raw_{label}_*.json")) for r in load_config(path)["rows"]]
    ledger = []
    for i, (q, r) in enumerate(zip(items, raw, strict=True), 1):
        if (r["id"], r["question_sha256"]) != (q["id"], q["question_sha256"]):
            raise ValueError("Raw identities differ")
        _, source, correct = score_v2(q["gold"], r["text"], truncated=r["truncated"])
        _, _, previous = score(q["gold"], r["text"], truncated=r["truncated"])
        ledger.append(dict(id=q["id"], question_sha256=q["question_sha256"], correct=correct,
                           source=source, previous=previous, strict=judge("math500", q["gold"], r["text"]),
                           truncated=r["truncated"], n_tokens=r["n_tokens"]))
        if i % 200 == 0:
            print(f"judge {label}: {i}/{len(items)}", flush=True)
    write_json(WORK / f"judged_{label}.json", {"rows": ledger})


def summarize(rows, cfg):
    n = len(rows)
    comparisons = {key: paired([r["base"][key] for r in rows], [r["candidate"][key] for r in rows])
                   for key in ("correct", "previous", "strict")}
    counts = {a: {"unresolved": sum(r[a]["source"] == "unresolved" for r in rows),
                  "truncated": sum(r[a]["truncated"] for r in rows),
                  "mean_tokens": sum(r[a]["n_tokens"] for r in rows)/n} for a in LABELS}
    coverage = (max(x["unresolved"] for x in counts.values()) / n <= cfg["max_unresolved_fraction"]
                and 100 * (counts["candidate"]["unresolved"] - counts["base"]["unresolved"]) / n
                <= cfg["max_extra_unresolved_pp"])
    lo, hi = comparisons["correct"]["ci_pp"]
    preserved = coverage and lo > -cfg["noninferiority_margin_pp"]
    superiority = (preserved and lo > 0 and comparisons["correct"]["delta_pp"] >= cfg["worthwhile_gain_pp"]
                   and all(comparisons[k]["delta_pp"] > 0 for k in ("previous", "strict")))
    reduction = 1 - counts["candidate"]["mean_tokens"] / counts["base"]["mean_tokens"]
    return dict(n=n, comparisons=comparisons, counts=counts, noninferior=preserved, improved=superiority,
                inferiority_supported=hi < 0, token_reduction_fraction=reduction,
                serving_gate=preserved and reduction >= cfg["min_token_reduction_fraction"],
                default="candidate" if superiority else "base")


def compare():
    plan, items = ready()
    arms = [load_config(WORK / f"judged_{label}.json")["rows"] for label in LABELS]
    rows = []
    for q, a, b in zip(items, *arms, strict=True):
        if any(x["id"] != q["id"] or x["question_sha256"] != q["question_sha256"] for x in (a, b)):
            raise ValueError("Paired identities differ")
        rows.append(dict(id=q["id"], question_sha256=q["question_sha256"],
                         base={k: a[k] for k in FIELDS}, candidate={k: b[k] for k in FIELDS}))
    write_json(REPORT / "paired.json", {"rows": rows})
    result = summarize(rows, plan["policy"]["decision"])
    write_json(REPORT / "comparison.json", result)
    print(result, flush=True)


def verify():
    plan = load_config(REPORT / "protocol.json")
    rows = load_config(REPORT / "paired.json")["rows"]
    if [(x["id"], x["question_sha256"]) for x in rows] != [(x["id"], x["question_sha256"]) for x in plan["items"]]:
        raise ValueError("Ledger and frozen test plan differ")
    if len({x["question_sha256"] for x in rows}) != len(rows):
        raise ValueError("Duplicate evaluation question")
    if len(rows) != plan["policy"]["questions"] or {x["subject"] for x in plan["items"]} != {
            x.split("/")[0] for x in plan["test_file_sha256"]}:
        raise ValueError("Subject allocation differs")
    report = summarize(rows, plan["policy"]["decision"])
    if report != load_config(REPORT / "comparison.json"):
        raise ValueError("Independent test report differs")
    print(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "generate", "judge", "compare", "verify"))
    parser.add_argument("--arm", choices=LABELS, default="base")
    args = parser.parse_args()
    if args.stage == "generate":
        generate(args.arm)
    elif args.stage == "judge":
        judge_arm(args.arm)
    else:
        {"prepare": prepare, "compare": compare, "verify": verify}[args.stage]()
