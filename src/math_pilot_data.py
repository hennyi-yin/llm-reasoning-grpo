"""Pinned MATH training-only data and a stratified, disjoint follow-up split."""

from collections import defaultdict
import hashlib
import os
from pathlib import Path
import random
import re
import unicodedata

from .common import ROOT, load_config, write_json, sha256_file
from .extract import last_boxed

OUT = Path(os.environ.get("LLM_MATH_PILOT_OUT", ROOT / "results/followup")).resolve()
WORK = OUT / "work"
CONFIG = ROOT / "configs/math_pilot.yaml"


def question_key(question):
    normal = unicodedata.normalize("NFKC", question).casefold()
    normal = re.sub(r"\s+", "", normal)
    return hashlib.sha256(normal.encode()).hexdigest()


def split_groups(rows, subjects, config):
    rng = random.Random(config["seed"])
    groups = defaultdict(list)
    for row in rows:
        groups[row["subject"]].append(row)
    keys = ("dev", "confirmation", "screen")
    counts = (config["dev_count"], config["confirmation_count"], config["screening_count"])
    result = {key: [] for key in keys}
    for i, subject in enumerate(subjects):
        group = sorted(groups[subject], key=lambda x: x["id"])
        rng.shuffle(group)
        offset = 0
        for key, total in zip(keys, counts):
            count = total // len(subjects) + int(i < total % len(subjects))
            result[key].extend(group[offset:offset + count])
            offset += count
        if len(group) < offset:
            raise ValueError(f"Insufficient eligible questions for {subject}")
    if any(len(result[k]) != v for k, v in zip(keys, counts)):
        raise ValueError("Split count mismatch")
    return result


def prepare():
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    cfg = load_config(CONFIG)["data"]
    if not re.fullmatch(r"[0-9a-f]{40}", cfg["revision"]):
        raise ValueError("MATH dataset revision is not pinned")
    # The original frozen test ledgers provide only question fingerprints for
    # exclusion. They are never used for fitting, scoring dev, or selection.
    frozen = set()
    for raw in sorted((ROOT / "results/final_eval/base--math500/raw").glob("*.json")):
        frozen.add(question_key(load_config(raw)["problem"]["question"]))
    if len(frozen) < 490:
        raise RuntimeError("Missing frozen MATH-500 fingerprint coverage")
    files, rows, seen = {}, [], set()
    counts = defaultdict(int, frozen_test_exact_overlap=0, normalized_duplicate=0,
                         not_text_with_boxed_reference=0, raw_train=0)
    for subject in cfg["subjects"]:
        name = f"{subject}/train-00000-of-00001.parquet"
        path = hf_hub_download(cfg["repo"], name, repo_type="dataset", revision=cfg["revision"])
        files[name] = sha256_file(path)
        for i, row in enumerate(pq.read_table(path).to_pylist()):
            counts["raw_train"] += 1
            question = row["problem"].strip()
            digest = question_key(question)
            if digest in frozen:
                counts["frozen_test_exact_overlap"] += 1
                continue
            if digest in seen:
                counts["normalized_duplicate"] += 1
                continue
            gold = last_boxed(row["solution"])
            if not gold or not gold.strip() or "[asy]" in question:
                counts["not_text_with_boxed_reference"] += 1
                continue
            seen.add(digest)
            rows.append({"id": f"{subject}-{i}", "question": question, "gold": gold.strip(),
                         "subject": subject, "level": row["level"], "question_sha256": digest})
    if counts["raw_train"] != cfg["train_count"]:
        raise ValueError(f"Unexpected MATH training size: {counts['raw_train']}")
    split = split_groups(rows, cfg["subjects"], cfg)
    for key, items in split.items():
        write_json(WORK / f"math_{key}.json", {"items": items})
    manifest = {"repo": cfg["repo"], "revision": cfg["revision"], "file_sha256": files,
                "counts": dict(counts), "frozen_test_fingerprints": len(frozen),
                "eligible_train_source": len(rows), "seed": cfg["seed"],
                "splits": {k: {"ids": [x["id"] for x in v],
                               "sha256": sha256_file(WORK / f"math_{k}.json")} for k, v in split.items()}}
    write_json(OUT / "math_split.json", manifest)
    return manifest


def items(split):
    manifest = load_config(OUT / "math_split.json")
    path = WORK / f"math_{split}.json"
    if sha256_file(path) != manifest["splits"][split]["sha256"]:
        raise ValueError(f"Split hash mismatch: {split}")
    return load_config(path)["items"]
