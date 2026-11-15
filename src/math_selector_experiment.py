"""Develop on saved GRPO samples and test the frozen selector on fresh MATH rows."""

import argparse
from collections import Counter
from importlib.metadata import version
import random
import time

from . import math_independent, math_multisample as pilot
from .common import ROOT, load_config, sha256_file, write_json
from .extract import last_boxed
from .math_pilot import local_base, normalize_word, parse_math
from .math_pilot_data import question_key
from .math_selector import develop, evidence, rank_index, summarize
from .math_sft import score_v2
from .prompts import build_messages


CFG = ROOT / "configs/math_selector.yaml"
REPORT = ROOT / "results/math_selector"
WORK = pilot.WORK.parent / "math_selector"
DEV_PROTOCOL = pilot.REPORT / "protocol.json"
DEV_LEDGER = pilot.REPORT / "paired.json"
PRIOR_TEST = math_independent.REPORT / "protocol.json"
TRAIN_SPLIT = pilot.PILOT / "math_split.json"
SOURCES = (*pilot.SOURCES, "src/common.py", "src/math_selector.py", "src/math_selector_experiment.py")


def input_hashes():
    paths = [DEV_PROTOCOL, DEV_LEDGER, PRIOR_TEST, TRAIN_SPLIT, *(ROOT / s for s in SOURCES)]
    return {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in paths}


def select_questions(config, dataset, sampling, base):
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    def read(repo, revision, filename, digest, columns=None):
        path = hf_hub_download(repo, filename, repo_type="dataset", revision=revision, local_files_only=True)
        if sha256_file(path) != digest:
            raise ValueError(f"Dataset hash differs for {filename}")
        return pq.read_table(path, columns=columns).to_pylist()

    previous = load_config(PRIOR_TEST)
    blocked = {q["question_sha256"] for data in (previous, dataset) for q in data["items"]}
    train = load_config(TRAIN_SPLIT)
    for filename, digest in train["file_sha256"].items():
        for row in read(train["repo"], train["revision"], filename, digest, ["problem"]):
            blocked.add(question_key(row["problem"]))
    math500 = list((ROOT / "results/final_eval/base--math500/raw").glob("*.json"))
    if len(math500) != 500:
        raise ValueError("Missing original MATH-500 rows")
    blocked.update(question_key(load_config(p)["problem"]["question"]) for p in math500)

    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    budget = sampling["max_model_len"] - sampling["max_tokens"]
    eligible = []
    counts = Counter(official_rows=0, previously_used_or_train=0, invalid_reference_or_diagram=0,
                     context=0, eligible_unique=0)
    for filename, digest in sorted(dataset["test_file_sha256"].items()):
        subject = filename.split("/")[0]
        for i, row in enumerate(read(dataset["dataset"], dataset["dataset_revision"], filename, digest)):
            counts["official_rows"] += 1
            question, solution = row["problem"].strip(), row["solution"]
            key = question_key(question)
            if key in blocked:
                counts["previously_used_or_train"] += 1
                continue
            gold = (last_boxed(solution) or "").strip()
            if ("[asy]" in question or "[asy]" in solution or solution.count(r"\boxed{") != 1
                    or not (parse_math(gold) or normalize_word(gold))):
                counts["invalid_reference_or_diagram"] += 1
                continue
            encoded = tokenizer.apply_chat_template(build_messages(question), tokenize=True,
                                                    add_generation_prompt=True)
            if len(encoded["input_ids"] if hasattr(encoded, "keys") else encoded) > budget:
                counts["context"] += 1
                continue
            eligible.append({"id": f"{subject}-test-{i}", "subject": subject,
                             "level": (subject, row["level"]), "question_sha256": key,
                             "question": question, "gold": gold})
    if counts["official_rows"] != dataset["config"]["expected_test_rows"]:
        raise ValueError("Official test row count differs")
    rng = random.Random(config["selection_seed"])
    excluded = len(blocked)
    items = math_independent.select_subject(eligible, blocked, config["test_questions"], rng)
    counts["eligible_unique"] = len(blocked) - excluded
    for item in items:
        item["level"] = item["level"][1]
    rng.shuffle(items)
    return items, dict(counts)


def identity(row):
    return row["id"], row["question_sha256"]


def freeze():
    if (REPORT / "protocol.json").exists():
        raise FileExistsError("The test protocol is already frozen")
    config = load_config(CFG)
    prior = load_config(DEV_PROTOCOL)
    dev = load_config(DEV_LEDGER)["rows"]
    if list(map(identity, dev)) != list(map(identity, prior["items"])):
        raise ValueError("Development ledger does not match the frozen source")
    selector = develop(dev, config)
    base, model = local_base()
    adapter = pilot.adapter_hashes()
    if (model["model_id"], model["revision"], adapter) != (prior["model"], prior["revision"], prior["adapter"]):
        raise ValueError("Model or adapter differs from the development run")
    sampling = {**prior["config"]["sampling"], "seed": config["sampling_seed"]}
    test, counts = select_questions(config, prior, sampling, base)
    write_json(WORK / "items.json", {"rows": test})
    write_json(REPORT / "protocol.json", {
        "config": config, "sampling": sampling, "input_hashes": input_hashes(),
        "model": prior["model"], "revision": prior["revision"], "adapter": adapter,
        "development": selector, "counts": counts, "seed_stride": sampling["n"],
        "primary": "paired exact-final-answer accuracy of frozen reranker minus plurality",
        "items_sha256": sha256_file(WORK / "items.json"),
        "items": [{k: r[k] for k in ("id", "question_sha256", "subject", "level")} for r in test],
    })
    print(f"Frozen development result: {selector}; test filtering: {counts}", flush=True)


def ready():
    plan = load_config(REPORT / "protocol.json")
    path = WORK / "items.json"
    if (plan["input_hashes"] != input_hashes() or plan["config"] != load_config(CFG)
            or plan["adapter"] != pilot.adapter_hashes() or plan["items_sha256"] != sha256_file(path)):
        raise ValueError("Frozen selector or test inputs changed")
    return plan, load_config(path)["rows"], sha256_file(REPORT / "protocol.json")


def question_seed(sampling, position):
    return sampling["seed"] + sampling["n"] * position


def shard_groups(items, sampling):
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
    for start, group, path in shard_groups(items, g):
        if path.exists():
            pilot.read_shard(path, digest, group, g["n"])
            continue
        params = [SamplingParams(n=g["n"], temperature=g["temperature"], top_p=g["top_p"],
                                 max_tokens=g["max_tokens"], seed=question_seed(g, start + i))
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


def test():
    plan, items, digest = ready()
    rows, shards = [], []
    g = plan["sampling"]
    for start, group, path in shard_groups(items, g):
        shard = pilot.read_shard(path, digest, group, g["n"])
        shards.append({"file": path.name, "sha256": sha256_file(path), "generation_seconds": shard["elapsed"]})
        for item, raw in zip(group, shard["rows"], strict=True):
            samples = [dict(vote_key=pilot.vote_key(s["text"], s["truncated"]),
                            n_tokens=s["n_tokens"], truncated=s["truncated"]) for s in raw["samples"]]
            vote = pilot.vote_index([s["vote_key"] for s in samples])
            rank = rank_index(samples, plan["development"]["weights"])
            for sample, completion in zip(samples, raw["samples"], strict=True):
                _, _, sample["correct"] = score_v2(item["gold"], completion["text"],
                                                     truncated=sample["truncated"])
            rows.append({"id": item["id"], "question_sha256": item["question_sha256"],
                         "vote_index": vote, "rank_index": rank, "samples": samples})
        print(f"test: {start + len(group)}/{len(items)} questions", flush=True)
    write_json(REPORT / "paired.json", {"rows": rows})
    write_json(REPORT / "run.json", {
        "protocol_sha256": digest, "shards": shards,
        **load_config(WORK / "generate_grpo_memory.json"),
        "versions": {p: version(p) for p in ("torch", "vllm", "transformers", "math-verify", "numpy", "scipy")},
    })
    write_json(REPORT / "comparison.json", summarize(rows, plan["config"]))


def verify():
    from math import isclose

    path = REPORT / "protocol.json"
    plan = load_config(path)
    run = load_config(REPORT / "run.json")
    if (plan["input_hashes"] != input_hashes() or plan["config"] != load_config(CFG)
            or run["protocol_sha256"] != sha256_file(path)):
        raise ValueError("Public protocol provenance differs")
    rows = load_config(REPORT / "paired.json")["rows"]
    used = {q["question_sha256"] for p in (DEV_PROTOCOL, PRIOR_TEST) for q in load_config(p)["items"]}
    fingerprints = {r["question_sha256"] for r in rows}
    if (list(map(identity, rows)) != list(map(identity, plan["items"])) or used & fingerprints
            or len(rows) != len(fingerprints) or len(rows) != plan["config"]["test_questions"]):
        raise ValueError("Public question identities or test boundary differs")
    fresh_fit = develop(load_config(DEV_LEDGER)["rows"], plan["config"])
    dev = plan["development"]
    if (any(v != dev[k] for k, v in fresh_fit.items() if k != "weights")
            or not all(isclose(a, b, abs_tol=1e-6) for a, b in zip(fresh_fit["weights"], dev["weights"], strict=True))):
        raise ValueError("Frozen development result could not be reproduced")
    for row in rows:
        blind = evidence(row["samples"])
        if (len(blind) != plan["sampling"]["n"]
                or row["vote_index"] != pilot.vote_index([s["vote_key"] for s in blind])
                or row["rank_index"] != rank_index(blind, dev["weights"])):
            raise ValueError(f"Selection differs for {row['id']}")
    result = summarize(rows, plan["config"])
    if result != load_config(REPORT / "comparison.json"):
        raise ValueError("Saved comparison differs")
    print(result)


if __name__ == "__main__":
    commands = {"freeze": freeze, "generate": generate, "test": test, "verify": verify}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=commands)
    commands[parser.parse_args().stage]()
