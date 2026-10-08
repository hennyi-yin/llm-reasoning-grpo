"""Pinned datasets with stable identities and an explicit train/dev boundary."""
from __future__ import annotations

import json
from .common import ROOT, load_config


def load_split() -> dict:
    result = json.loads((ROOT / "splits/gsm8k_split.json").read_text(encoding="utf-8"))
    train, dev = result["train"], result["dev"]
    if len(train) != 6973 or len(dev) != 500 or set(train) & set(dev):
        raise ValueError("Invalid train/dev split")
    if set(train) | set(dev) != set(range(7473)):
        raise ValueError("Split does not partition the complete GSM8K training set")
    return result


def _dataset(name: str, split: str):
    from datasets import load_dataset
    config = load_config("configs/data.yaml")[name]
    if len(config["revision"]) != 40:
        raise ValueError("Dataset revision must be a pinned commit")
    if split == "test":
        from .protocol import require_final_eligibility
        require_final_eligibility()
    # load_dataset(repo, split='train') may prepare every split. Read only the
    # pinned split file so preparing train never downloads/materializes test.
    data_file = config["files"][split]
    url = f"hf://datasets/{config['repo']}@{config['revision']}/{data_file}"
    loader = "parquet" if data_file.endswith(".parquet") else "json"
    ds = load_dataset(loader, data_files={split: url}, split=split)
    expected = config["counts"][split]
    if len(ds) != expected:
        raise ValueError(f"{name}/{split}: expected {expected}, got {len(ds)}")
    return ds


def load_gsm8k_train():
    return _dataset("gsm8k", "train")


def load_items(task: str, split: str) -> list[dict]:
    if task == "gsm8k":
        if split in {"dev", "train"}:
            ds = load_gsm8k_train()
            ids = load_split()[split]
            prefix = "train"
        elif split == "test":
            ds = _dataset("gsm8k", "test")
            ids = range(len(ds))
            prefix = "test"
        else:
            raise ValueError(f"Unknown split: {split}")
        result = []
        for idx in ids:
            row = ds[int(idx)]
            if "####" not in row["answer"]:
                raise ValueError(f"GSM8K missing answer delimiter at {idx}")
            result.append({"id": f"{prefix}-{idx}", "question": row["question"],
                           "gold": row["answer"].rsplit("####", 1)[1].strip(), "answer": row["answer"]})
        return result
    if task == "math500" and split == "test":
        ds = _dataset("math500", "test")
        return [{"id": f"test-{i}", "question": row["problem"], "gold": row["answer"],
                 "answer": row.get("solution", row["answer"]),
                 "subject": row.get("subject"), "level": row.get("level")}
                for i, row in enumerate(ds)]
    raise ValueError(f"Unsupported task/split: {task}/{split}")
