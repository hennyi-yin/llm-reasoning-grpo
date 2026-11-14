"""Paired answer coverage and reference-blind selection on fresh MATH questions."""

import argparse
from collections import Counter
from fractions import Fraction
from math import comb
import random
import re
import time

from . import math_independent
from .common import ROOT, load_config, sha256_file, write_json
from .evaluate import judge
from .extract import last_boxed
from .followup_answers import final_candidate
from .math_pilot import local_base, normalize_word, parse_math
from .math_pilot_data import question_key
from .math_sft import PROSE, score_v2
from .prompts import build_messages

CFG = ROOT / "configs/math_multisample.yaml"
REPORT = ROOT / "results/math_multisample"
WORK = ROOT / "results/followup/work/math_multisample"
PILOT = ROOT / "results/followup"
ADAPTER = PILOT / "work/math_adapter/final_adapter"
ARMS = ("base", "grpo")
SOURCES = ("src/math_multisample.py", "src/math_independent.py", "src/math_sft.py",
           "src/math_pilot.py", "src/followup_answers.py", "src/extract.py",
           "src/evaluate.py", "src/prompts.py", "src/math_pilot_data.py", "src/stats.py")


def adapter_hashes():
    expected = load_config(PILOT / "math_pilot_audit.json")["local_final_adapter_sha256"]
    weights = sha256_file(ADAPTER / "adapter_model.safetensors")
    if weights != expected:
        raise ValueError("Adapter does not match the audited 50-step MATH GRPO weights")
    config = load_config(ADAPTER / "adapter_config.json")
    if (config["r"], config["lora_alpha"]) != (16, 32) or any(config.get(k) for k in
            ("use_rslora", "use_dora", "rank_pattern", "alpha_pattern")):
        raise ValueError("Unexpected LoRA configuration")
    return {"weights": weights, "config": sha256_file(ADAPTER / "adapter_config.json")}


def code_hashes():
    return {name: sha256_file(ROOT / name) for name in SOURCES}


def prepare():
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    protocol = REPORT / "protocol.json"
    if protocol.exists():
        raise FileExistsError("Audit set is already frozen; do not redraw")
    config = load_config(CFG)
    previous = load_config(math_independent.REPORT / "protocol.json")
    original = load_config(PILOT / "math_split.json")
    base, model = local_base()
    locked_adapter = adapter_hashes()
    if (model["model_id"], model["revision"]) != (previous["model"], previous["revision"]):
        raise ValueError("Baseline revision differs")
    tok = AutoTokenizer.from_pretrained(base, local_files_only=True)
    blocked = {r["question_sha256"] for r in previous["items"]}
    counts = Counter(excluded_train_or_previously_used=0, excluded_reference_or_diagram=0,
                     excluded_context=0, official_test_rows=0)
    for name, digest in original["file_sha256"].items():
        path = hf_hub_download(original["repo"], name, repo_type="dataset",
                               revision=original["revision"], local_files_only=True)
        if sha256_file(path) != digest:
            raise ValueError("Train file hash differs")
        blocked.update(question_key(q) for q in pq.read_table(path, columns=["problem"])["problem"].to_pylist())
    old_test = list((ROOT / "results/final_eval/base--math500/raw").glob("*.json"))
    if len(old_test) != 500:
        raise ValueError("Missing frozen MATH-500 outputs")
    blocked.update(question_key(load_config(p)["problem"]["question"]) for p in old_test)
    only_blocked = set(blocked)
    eligible = []
    g = config["sampling"]
    for name, digest in sorted(previous["test_file_sha256"].items()):
        subject = name.split("/")[0]
        path = hf_hub_download(previous["dataset"], name, repo_type="dataset",
                               revision=previous["dataset_revision"], local_files_only=True)
        if sha256_file(path) != digest:
            raise ValueError("Official test file hash differs")
        for i, row in enumerate(pq.read_table(path).to_pylist()):
            counts["official_test_rows"] += 1
            question, solution = row["problem"].strip(), row["solution"]
            key = question_key(question)
            if key in only_blocked:
                counts["excluded_train_or_previously_used"] += 1
                continue
            gold = last_boxed(solution)
            if ("[asy]" in question or "[asy]" in solution or not gold or solution.count(r"\boxed{") != 1
                    or not (parse_math(gold.strip()) or normalize_word(gold.strip()))):
                counts["excluded_reference_or_diagram"] += 1
                continue
            ids = tok.apply_chat_template(build_messages(question), tokenize=True, add_generation_prompt=True)
            if len(ids["input_ids"] if hasattr(ids, "keys") else ids) + g["max_tokens"] > g["max_model_len"]:
                counts["excluded_context"] += 1
                continue
            eligible.append(dict(id=f"{subject}-test-{i}", subject=subject, level=(subject, row["level"]),
                                 question=question, question_sha256=key, gold=gold.strip()))
    if counts["official_test_rows"] != config["expected_test_rows"]:
        raise ValueError("Official test row count differs")
    rng = random.Random(config["selection_seed"])
    rows = math_independent.select_subject(eligible, blocked, config["questions"], rng)
    counts["unique_eligible"] = len(blocked) - len(only_blocked)
    for row in rows:
        row["level"] = row["level"][1]
    rng.shuffle(rows)
    write_json(WORK / "items.json", {"rows": rows})
    write_json(protocol, {"config": config, "adapter": locked_adapter, "source_hashes": code_hashes(),
               "model": model["model_id"], "revision": model["revision"], "counts": dict(counts),
               "items_sha256": sha256_file(WORK / "items.json"),
               "dataset": previous["dataset"], "dataset_revision": previous["dataset_revision"],
               "test_file_sha256": previous["test_file_sha256"],
               "prior_test_protocol_sha256": sha256_file(math_independent.REPORT / "protocol.json"),
               "primary": "paired answer pass@8 under src.math_sft.score_v2",
               "items": [{k: r[k] for k in ("id", "subject", "level", "question_sha256")} for r in rows]})
    print(f"Froze {len(rows)} questions; {dict(counts)}", flush=True)


def ready():
    plan = load_config(REPORT / "protocol.json")
    if (plan["config"] != load_config(CFG) or plan["adapter"] != adapter_hashes()
            or plan["source_hashes"] != code_hashes()
            or plan["items_sha256"] != sha256_file(WORK / "items.json")):
        raise ValueError("Frozen sampling inputs changed")
    return plan, load_config(WORK / "items.json")["rows"]


def read_shard(path, digest, group, n):
    payload = load_config(path)
    if payload["protocol"] != digest or [
            (r["id"], r["question_sha256"]) for r in payload["rows"]] != [
            (r["id"], r["question_sha256"]) for r in group] or any(len(r["samples"]) != n for r in payload["rows"]):
        raise ValueError(f"Invalid sampling shard: {path.name}")
    return payload


def generate(arm):
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    plan, items = ready()
    g = plan["config"]["sampling"]
    digest = sha256_file(REPORT / "protocol.json")
    base, model = local_base()
    if (model["model_id"], model["revision"]) != (plan["model"], plan["revision"]):
        raise ValueError("Pinned base changed")
    engine = LLM(model=base, dtype="bfloat16", seed=g["seed"],
                 gpu_memory_utilization=g["gpu_memory_utilization"], max_model_len=g["max_model_len"],
                 max_num_seqs=g["max_num_seqs"], enforce_eager=True, enable_flashinfer_autotune=False,
                 enable_lora=True, max_lora_rank=16)
    lora = LoRARequest("math-grpo50", 1, str(ADAPTER)) if arm == "grpo" else None
    size = g["shard_size"]
    for j, start in enumerate(range(0, len(items), size)):
        group = items[start:start + size]
        path = WORK / f"raw_{arm}_{j:02}.json"
        if path.exists():
            read_shard(path, digest, group, g["n"])
            continue
        params = [SamplingParams(n=g["n"], temperature=g["temperature"], top_p=g["top_p"],
                  max_tokens=g["max_tokens"], seed=g["seed"] + start+i) for i in range(len(group))]
        before = time.perf_counter()
        outputs = engine.chat([build_messages(r["question"]) for r in group], params, lora_request=lora)
        elapsed = time.perf_counter() - before
        rows = []
        for item, result in zip(group, outputs, strict=True):
            ordered = sorted(result.outputs, key=lambda out: out.index)
            if [c.index for c in ordered] != list(range(g["n"])):
                raise RuntimeError("Unexpected sampling multiplicity")
            rows.append(dict(id=item["id"], question_sha256=item["question_sha256"], samples=[
                dict(text=c.text, n_tokens=len(c.token_ids), truncated=c.finish_reason == "length")
                for c in ordered]))
        write_json(path, {"protocol": digest, "rows": rows, "elapsed": elapsed})
        print(f"{arm}: {start+len(group)}/{len(items)} questions, shard {elapsed:.1f}s", flush=True)
    write_json(WORK / f"{arm}_complete.json", {"questions": len(items), "samples_per_question": g["n"]})


UNSURE = re.compile(r"\b(?:or|maybe|perhaps|either)\b", re.I)
RATIONAL = re.compile(r"([+-]?\d+)\s*/\s*([+-]?\d+)")
LATEX_FRACTION = re.compile(r"([+-]?)\\(?:d|t)?frac\{([+-]?\d+)\}\{([+-]?\d+)\}")
DECIMAL = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)")


def vote_key(text, truncated):
    """Normalize extracted finals only; this function never receives gold or correctness."""
    x, source = final_candidate(text)
    if truncated and source not in {"boxed", "terminal_display"}:
        return None
    if not x:
        tail = next((line.strip() for line in reversed(text.splitlines()) if line.strip()), "")
        prose = PROSE.fullmatch(tail)
        x = prose[1] if prose else None
    if not x or len(x) > 300 or UNSURE.search(x):
        return None
    word = normalize_word(x)
    if word:
        return "word:" + word
    x = re.sub(r"\s+", "", x).replace("−", "-").replace(r"\left", "").replace(r"\right", "")
    if re.fullmatch(r"[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?", x):
        x = x.replace(",", "")
    try:
        if DECIMAL.fullmatch(x):
            return "number:" + str(Fraction(x))
        if m := LATEX_FRACTION.fullmatch(x):
            return "number:" + str(Fraction(int(m[2]) * (-1 if m[1] == "-" else 1), int(m[3])))
        if m := RATIONAL.fullmatch(x):
            return "number:" + str(Fraction(int(m[1]), int(m[2])))
    except ZeroDivisionError:
        return None
    if not (parse_math(x) or (source == "boxed" and re.search(r"[0-9\\]", x))):
        return None
    return "expression:" + x


def vote_index(keys):
    counts = Counter(k for k in keys if k is not None)
    if not counts:
        return None
    most = max(counts.values())
    return next(i for i, k in enumerate(keys) if counts[k] == most)


def pass_k(correct, k):
    n, c = len(correct), sum(correct)
    if not 1 <= k <= n:
        raise ValueError("k outside sampling budget")
    return 1 - comb(n-c, k) / comb(n, k)


def judge_arm(arm):
    plan, items = ready()
    digest = sha256_file(REPORT / "protocol.json")
    cfg = plan["config"]["sampling"]
    size = cfg["shard_size"]
    out = []
    for j, start in enumerate(range(0, len(items), size)):
        group = items[start:start + size]
        shard = read_shard(WORK / f"raw_{arm}_{j:02}.json", digest, group, cfg["n"])
        for q, raw in zip(group, shard["rows"], strict=True):
            samples = []
            for c in raw["samples"]:
                key = vote_key(c["text"], c["truncated"])
                _, source, correct = score_v2(q["gold"], c["text"], truncated=c["truncated"])
                samples.append(dict(vote_key=key, correct=correct, source=source,
                                    strict=judge("math500", q["gold"], c["text"]),
                                    truncated=c["truncated"], n_tokens=c["n_tokens"]))
            out.append(dict(id=q["id"], question_sha256=q["question_sha256"], samples=samples))
        print(f"judge {arm}: {start+len(group)}/{len(items)}", flush=True)
    write_json(WORK / f"judged_{arm}.json", {"rows": out})


def summarize(rows, config):
    import numpy as np
    from .stats import mcnemar_exact

    n = len(rows)
    ks = config["analysis"]["k"]
    if n != config["questions"] or any(len(r[a]) != config["sampling"]["n"] for r in rows for a in ARMS):
        raise ValueError("Incomplete paired sampling")
    rng = np.random.default_rng(config["analysis"]["bootstrap_seed"])
    draws = rng.integers(n, size=(config["analysis"]["bootstrap_replicates"], n))

    def compare(a, b, binary=False):
        delta = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
        interval = np.percentile(delta[draws].mean(axis=1)*100, [2.5, 97.5]).tolist()
        result = dict(base_pct=100*float(np.mean(a)), grpo_pct=100*float(np.mean(b)),
                      delta_pp=100*float(delta.mean()), paired_95_ci_pp=interval)
        if binary:
            result.update(gains=sum(not x and y for x, y in zip(a, b)), losses=sum(x and not y for x, y in zip(a, b)),
                          mcnemar_p_descriptive=mcnemar_exact(a, b))
        return result

    def coverage(arm, k, field):
        return [pass_k([x[field] for x in r[arm]], k) for r in rows]

    def vote(arm):
        choices = []
        for row in rows:
            samples = row[arm]
            i = vote_index([c["vote_key"] for c in samples])
            choices.append(False if i is None else samples[i]["correct"])
        return choices

    votes = {a: vote(a) for a in ARMS}
    curves = {f"pass@{k}": compare(*(coverage(a, k, "correct") for a in ARMS), binary=k == max(ks))
              for k in ks}
    selection = compare(*(votes[a] for a in ARMS), binary=True)
    arms = {}
    for a in ARMS:
        samples = [c for r in rows for c in r[a]]
        coverage_n = sum(any(c["correct"] for c in r[a]) for r in rows)
        arms[a] = dict(correct_samples=sum(x["correct"] for x in samples), samples=len(samples),
                       unresolved=sum(x["source"] == "unresolved" for x in samples),
                       truncated=sum(x["truncated"] for x in samples),
                       mean_tokens_per_sample=sum(x["n_tokens"] for x in samples)/len(samples),
                       vote_abstentions=sum(all(c["vote_key"] is None for c in r[a]) for r in rows),
                       coverage_n=coverage_n, vote_correct=sum(votes[a]),
                       covered_but_not_selected=coverage_n - sum(votes[a]))
    return {"n": n, "coverage": curves, "vote_at_8": selection, "arms": arms,
            "strict": {f"pass@{k}": compare(*(coverage(a, k, "strict") for a in ARMS),
                                            binary=k == max(ks)) for k in (1, max(ks))}}


def compare():
    from importlib.metadata import version

    plan, items = ready()
    judges = [load_config(WORK / f"judged_{a}.json")["rows"] for a in ARMS]
    rows = []
    for item, *arms in zip(items, *judges, strict=True):
        if any((r["id"], r["question_sha256"]) != (item["id"], item["question_sha256"]) for r in arms):
            raise ValueError("Paired sampling identities differ")
        rows.append(dict(id=item["id"], question_sha256=item["question_sha256"],
                         **{a: r["samples"] for a, r in zip(ARMS, arms)}))
    write_json(REPORT / "paired.json", {"rows": rows})
    report = summarize(rows, plan["config"])
    digest = sha256_file(REPORT / "protocol.json")
    g = plan["config"]["sampling"]
    receipt = {}
    for a in ARMS:
        shards = []
        for j, start in enumerate(range(0, len(items), g["shard_size"])):
            path = WORK / f"raw_{a}_{j:02}.json"
            data = read_shard(path, digest, items[start:start+g["shard_size"]], g["n"])
            shards.append(dict(path=path.name, sha256=sha256_file(path), offline_seconds=data["elapsed"]))
        receipt[a] = {"raw_shards": shards, **load_config(WORK / f"generate_{a}_memory.json")}
    write_json(REPORT / "run.json", {"protocol_sha256": digest, "arms": receipt, "versions": {
        p: version(p) for p in ("torch", "vllm", "transformers", "math-verify", "numpy", "scipy")}})
    write_json(REPORT / "comparison.json", report)
    print(report, flush=True)


def verify():
    plan = load_config(REPORT / "protocol.json")
    previous = load_config(math_independent.REPORT / "protocol.json")
    ledger = load_config(REPORT / "paired.json")["rows"]
    keys = [(x["id"], x["question_sha256"]) for x in ledger]
    fingerprints = {k[1] for k in keys}
    run = load_config(REPORT / "run.json")
    if (keys != [(x["id"], x["question_sha256"]) for x in plan["items"]] or len(fingerprints) != len(keys)
            or fingerprints & {r["question_sha256"] for r in previous["items"]}
            or plan["config"] != load_config(CFG) or run["protocol_sha256"] != sha256_file(REPORT / "protocol.json")):
        raise ValueError("Public ledger, metadata, or test boundary differs")
    result = summarize(ledger, plan["config"])
    if result != load_config(REPORT / "comparison.json"):
        raise ValueError("Saved sampling comparison differs")
    print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "generate", "judge", "compare", "verify"))
    parser.add_argument("--arm", choices=ARMS, default="base")
    args = parser.parse_args()
    if args.stage == "generate":
        generate(args.arm)
    elif args.stage == "judge":
        judge_arm(args.arm)
    else:
        {"prepare": prepare, "compare": compare, "verify": verify}[args.stage]()
