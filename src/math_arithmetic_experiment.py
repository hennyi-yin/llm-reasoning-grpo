"""Freeze an arithmetic reranker and evaluate it on fresh MATH samples."""

import argparse
from importlib.metadata import version
import time

from . import math_multisample as pilot, math_selector_experiment as selector
from .common import ROOT, load_config, sha256_file, write_json
from .math_arithmetic import checks, choose, develop, diagnostics
from .math_pilot import local_base
from .math_selector import summarize
from .math_sft import score_v2
from .prompts import build_messages


CFG = ROOT / "configs/math_arithmetic.yaml"
REPORT = ROOT / "results/math_arithmetic"
WORK = pilot.WORK.parent / "math_arithmetic"
DEV = REPORT / "development.json"
PRIOR_PLAN = selector.REPORT / "protocol.json"
PRIOR_LEDGER = selector.REPORT / "paired.json"
PRIOR_RUN = selector.REPORT / "run.json"
SOURCES = (*selector.SOURCES, "src/math_arithmetic.py", "src/math_arithmetic_experiment.py")


def hashes():
    paths = [DEV, PRIOR_PLAN, PRIOR_LEDGER, PRIOR_RUN, selector.DEV_PROTOCOL, selector.PRIOR_TEST,
             selector.TRAIN_SPLIT, *(ROOT / s for s in SOURCES)]
    return {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in paths}


def development_rows():
    saved = load_config(PRIOR_LEDGER)["rows"]
    records = load_config(DEV)["rows"]
    output = []
    for old, item in zip(saved, records, strict=True):
        if selector.identity(old) != selector.identity(item):
            raise ValueError("Development identities differ")
        output.append({"samples": [{**s, **stats}
                       for s, stats in zip(old["samples"], item["checks"], strict=True)]})
    return output


def build_dev(prior):
    run = load_config(PRIOR_RUN)
    if run["protocol_sha256"] != sha256_file(PRIOR_PLAN):
        raise ValueError("Development generation receipt differs")
    records = []
    old_sampling = prior["sampling"]
    size = old_sampling["shard_size"]
    old_items = prior["items"]
    for offset, entry in enumerate(run["shards"]):
        path = selector.WORK / entry["file"]
        if sha256_file(path) != entry["sha256"]:
            raise ValueError(f"Development shard changed: {path.name}")
        group = old_items[offset * size:(offset + 1) * size]
        shard = pilot.read_shard(path, run["protocol_sha256"], group, old_sampling["n"])
        for row in shard["rows"]:
            features = []
            for s in row["samples"]:
                key = pilot.vote_key(s["text"], s["truncated"])
                features.append(checks(s["text"], key))
            records.append({"id": row["id"], "question_sha256": row["question_sha256"], "checks": features})
    write_json(DEV, {"rows": records})


def freeze():
    if (REPORT / "protocol.json").exists():
        raise FileExistsError("Test protocol is already frozen")
    config = load_config(CFG)
    prior = load_config(PRIOR_PLAN)
    dataset = load_config(selector.DEV_PROTOCOL)
    dataset["items"] += prior["items"]
    build_dev(prior)
    dev = develop(development_rows(), config)
    base, model = local_base()
    adapter = pilot.adapter_hashes()
    if (model["model_id"], model["revision"], adapter) != (prior["model"], prior["revision"], prior["adapter"]):
        raise ValueError("Model or adapter differs")
    sampling = {**prior["sampling"], "seed": config["sampling_seed"]}
    items, counts = selector.select_questions(config, dataset, sampling, base)
    items_path = WORK / "items.json"
    write_json(items_path, {"rows": items})
    write_json(REPORT / "protocol.json", {
        "config": config, "sampling": sampling, "input_hashes": hashes(), "development": dev,
        "model": prior["model"], "revision": prior["revision"], "adapter": adapter, "counts": counts,
        "primary": "paired final-answer accuracy of exact-arithmetic reranking minus plurality",
        "items_sha256": sha256_file(items_path),
        "items": [{k: r[k] for k in ("id", "question_sha256", "subject", "level")} for r in items],
    })
    print(f"Frozen arithmetic development: {dev}; test filtering: {counts}", flush=True)


def ready():
    path = REPORT / "protocol.json"
    plan = load_config(path)
    items = WORK / "items.json"
    if (plan["input_hashes"] != hashes() or plan["config"] != load_config(CFG)
            or plan["adapter"] != pilot.adapter_hashes() or plan["items_sha256"] != sha256_file(items)):
        raise ValueError("Frozen inputs changed")
    return plan, load_config(items)["rows"], sha256_file(path)


def shards(items, sampling):
    size = sampling["shard_size"]
    for start in range(0, len(items), size):
        yield start, items[start:start+size], WORK / f"raw_grpo_{start // size:02}.json"


def generate():
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    plan, items, digest = ready()
    g = plan["sampling"]
    base, model = local_base()
    if (model["model_id"], model["revision"]) != (plan["model"], plan["revision"]):
        raise ValueError("Pinned base changed")
    engine = LLM(model=base, dtype="bfloat16", seed=g["seed"], max_model_len=g["max_model_len"],
                 max_num_seqs=g["max_num_seqs"], gpu_memory_utilization=g["gpu_memory_utilization"],
                 enforce_eager=True, enable_flashinfer_autotune=False, enable_lora=True, max_lora_rank=16)
    lora = LoRARequest("math-grpo50", 1, str(pilot.ADAPTER))
    for start, group, path in shards(items, g):
        if path.exists():
            pilot.read_shard(path, digest, group, g["n"])
            continue
        params = [SamplingParams(n=g["n"], temperature=g["temperature"], top_p=g["top_p"],
                                 max_tokens=g["max_tokens"], seed=selector.question_seed(g, start + i))
                  for i in range(len(group))]
        before = time.perf_counter()
        outputs = engine.chat([build_messages(r["question"]) for r in group], params, lora_request=lora)
        elapsed = time.perf_counter() - before
        rows = []
        for item, output in zip(group, outputs, strict=True):
            completions = sorted(output.outputs, key=lambda sample: sample.index)
            if [s.index for s in completions] != list(range(g["n"])):
                raise ValueError("Unexpected completion multiplicity")
            rows.append({"id": item["id"], "question_sha256": item["question_sha256"],
                         "samples": [{"text": s.text, "truncated": s.finish_reason == "length",
                                      "n_tokens": len(s.token_ids)} for s in completions]})
        write_json(path, {"protocol": digest, "rows": rows, "elapsed": elapsed})
        print(f"grpo: {start + len(group)}/{len(items)} questions, shard {elapsed:.1f}s", flush=True)
    write_json(WORK / "generation_complete.json", {"questions": len(items), "samples_per_question": g["n"]})


def blind_features(completion):
    key = pilot.vote_key(completion["text"], completion["truncated"])
    return {"vote_key": key, "n_tokens": completion["n_tokens"], "truncated": completion["truncated"],
            **checks(completion["text"], key)}


def summary(rows, config):
    return {**summarize(rows, config), "arithmetic": diagnostics(rows)}


def test():
    plan, items, digest = ready()
    rows, receipts = [], []
    g = plan["sampling"]
    weight = plan["development"]["selected_weight"]
    for start, group, path in shards(items, g):
        shard = pilot.read_shard(path, digest, group, g["n"])
        receipts.append({"file": path.name, "sha256": sha256_file(path), "generation_seconds": shard["elapsed"]})
        for item, raw in zip(group, shard["rows"], strict=True):
            samples = [blind_features(s) for s in raw["samples"]]
            vote = pilot.vote_index([s["vote_key"] for s in samples])
            picked = choose(samples, weight)
            for sample, completion in zip(samples, raw["samples"], strict=True):
                _, _, sample["correct"] = score_v2(item["gold"], completion["text"],
                                                     truncated=sample["truncated"])
            rows.append({"id": item["id"], "question_sha256": item["question_sha256"],
                         "vote_index": vote, "rank_index": picked, "samples": samples})
        print(f"test: {start + len(group)}/{len(items)} questions", flush=True)
    write_json(REPORT / "paired.json", {"rows": rows})
    write_json(REPORT / "run.json", {
        "protocol_sha256": digest, "shards": receipts, **load_config(WORK / "generate_grpo_memory.json"),
        "versions": {p: version(p) for p in ("torch", "vllm", "transformers", "math-verify", "numpy", "scipy")},
    })
    write_json(REPORT / "comparison.json", summary(rows, plan["config"]))


def verify():
    path = REPORT / "protocol.json"
    plan = load_config(path)
    run = load_config(REPORT / "run.json")
    config = load_config(CFG)
    if (plan["input_hashes"] != hashes() or plan["config"] != config
            or run["protocol_sha256"] != sha256_file(path)):
        raise ValueError("Public protocol provenance differs")
    rows = load_config(REPORT / "paired.json")["rows"]
    used = {q["question_sha256"] for p in (PRIOR_PLAN, selector.DEV_PROTOCOL, selector.PRIOR_TEST)
            for q in load_config(p)["items"]}
    fingerprints = {r["question_sha256"] for r in rows}
    if (list(map(selector.identity, rows)) != list(map(selector.identity, plan["items"]))
            or used & fingerprints or len(rows) != len(fingerprints) or len(rows) != config["test_questions"]):
        raise ValueError("Public question identities or test boundary differs")
    if develop(development_rows(), config) != plan["development"]:
        raise ValueError("Frozen development result differs")
    weight = plan["development"]["selected_weight"]
    for row in rows:
        samples = row["samples"]
        if (len(samples) != plan["sampling"]["n"]
                or row["vote_index"] != pilot.vote_index([s["vote_key"] for s in samples])
                or row["rank_index"] != choose(samples, weight)):
            raise ValueError(f"Selection differs for {row['id']}")
    result = summary(rows, config)
    if result != load_config(REPORT / "comparison.json"):
        raise ValueError("Saved comparison differs")
    print(result)


if __name__ == "__main__":
    commands = {"freeze": freeze, "generate": generate, "test": test, "verify": verify}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=commands)
    commands[parser.parse_args().stage]()
