"""Pin Hub metadata, persist split IDs, and cache only training data."""
from __future__ import annotations

import argparse
import random
from urllib.request import urlopen
import json
from .common import ROOT, write_json


def hub_revision(kind: str, repo: str) -> str:
    with urlopen(f"https://huggingface.co/api/{kind}/{repo}", timeout=60) as response:
        sha = json.load(response)["sha"]
    if len(sha) != 40:
        raise ValueError(f"Invalid Hub revision: {repo}")
    return sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    data_path = ROOT / "configs/data.yaml"
    if not data_path.exists():
        import yaml
        datasets = {
            "gsm8k": {"repo": "openai/gsm8k", "subset": "main", "counts": {"train": 7473, "test": 1319}},
            "math500": {"repo": "HuggingFaceH4/MATH-500", "counts": {"test": 500}},
        }
        for config in datasets.values():
            config["revision"] = hub_revision("datasets", config["repo"])
            from huggingface_hub import HfApi
            files = HfApi().list_repo_files(config["repo"], repo_type="dataset", revision=config["revision"])
            config["files"] = {}
            for split in config["counts"]:
                matches = [path for path in files if path.endswith((".parquet", ".jsonl")) and
                           (f"/{split}-" in path or f"/{split}.parquet" in path or path == f"{split}.jsonl") and
                           (not config.get("subset") or path.startswith(config["subset"] + "/"))]
                if len(matches) != 1:
                    raise ValueError(f"Expected one pinned data file for {config['repo']}/{split}: {matches}")
                config["files"][split] = matches[0]
        data_path.write_text(yaml.safe_dump(datasets, sort_keys=False), encoding="utf-8")
    model_path = ROOT / "configs/models.yaml"
    if not model_path.exists():
        import yaml
        models = {size: {"model_id": f"Qwen/Qwen2.5-{size}-Instruct",
                         "revision": hub_revision("models", f"Qwen/Qwen2.5-{size}-Instruct")}
                  for size in ("1.5B", "0.5B")}
        model_path.write_text(yaml.safe_dump(models, sort_keys=False), encoding="utf-8")
    split_path = ROOT / "splits/gsm8k_split.json"
    ids = list(range(7473))
    random.Random(42).shuffle(ids)
    expected = {"dev": sorted(ids[:500]), "train": sorted(ids[500:])}
    if split_path.exists():
        if json.loads(split_path.read_text()) != expected:
            raise ValueError("Existing split differs from frozen seed-42 split")
    else:
        write_json(split_path, expected)
    if not args.metadata_only:
        from .data import load_gsm8k_train
        load_gsm8k_train()
    print("Pinned revisions and 6973/500 split ready; test data has not been loaded.")


if __name__ == "__main__":
    main()
