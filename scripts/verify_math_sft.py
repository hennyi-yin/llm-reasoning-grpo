"""Verify the published SFT comparisons from paired CPU-side evidence."""

import argparse
import json
from pathlib import Path

from src.common import ROOT, load_config, sha256_file
from src.math_sft import ARMS, CONFIG, paired as compare_binary, summarize


def verify(directory, pilot):
    cfg = load_config(CONFIG)
    splits = load_config(pilot / "math_split.json")["splits"]
    training = load_config(directory / "training_manifest.json")
    if (training["config_sha256"] != sha256_file(CONFIG)
            or training["review_sha256"] != sha256_file(directory / "reference_review.json")):
        raise ValueError("Training configuration or review record differs")
    ids = set(training["training_ids"])
    if len(ids) != len(training["training_ids"]) or any(ids.intersection(x["ids"]) for x in splits.values()):
        raise ValueError("Training IDs duplicate or overlap a previous split")
    if training["arms"]["short"]["final_global_step"] != training["arms"]["worked"]["final_global_step"]:
        raise ValueError("Training arm update count mismatch")
    dev = load_config(directory / "dev_comparison.json")
    expected = ("dev", "confirmation") if dev["decision"] == "confirm" else ("dev",)
    if dev["decision"] == "stop" and (directory / "confirmation_paired.json").exists():
        raise ValueError("Confirmation evidence exists after a failed dev gate")
    results = {}
    for split in expected:
        paired = load_config(directory / f"{split}_paired.json")["rows"]
        if [x["id"] for x in paired] != splits[split]["ids"]:
            raise ValueError("Held-out IDs differ from the original split")
        groups = {label: [r[label] for r in paired] for label in ARMS}
        result = summarize(split, groups, cfg["gate"])
        if result != load_config(directory / f"{split}_comparison.json"):
            raise ValueError(f"Saved {split} metrics differ")
        if split == "dev":
            review = load_config(directory / "dev_unresolved_review.json")["reviews"]
            unresolved = {x["id"] for x in paired if x["worked"]["source"] == "unresolved"
                          and not x["worked"]["truncated"]}
            if set(review) != unresolved or any(not isinstance(v["matches"], bool) for v in review.values()):
                raise ValueError("Manual audit does not cover the full unclipped unresolved subset")
        results[split] = result
    if training["status"] != results[expected[-1]]["decision"]:
        raise ValueError("Recorded stop or confirmation status differs")
    probe_path = directory / "dev_scale_probe.json"
    if probe_path.is_file():
        probe = load_config(probe_path)
        original = load_config(directory / "dev_paired.json")["rows"]
        rows = probe["paired"]
        if (probe["split"] != "dev" or probe["scale_relative_to_trained_adapter"] != 0.25
                or probe["adapter_weights_sha256"] != training["arms"]["worked"]["adapter_sha256"]
                or len(rows) != len(original)
                or any((a["id"], a["question_sha256"], a["base"]) != (b["id"], b["question_sha256"], b["base"])
                       for a, b in zip(original, rows))):
            raise ValueError("Scale probe is not paired with the original base")
        for metric in ("correct", "previous", "strict"):
            calculated = compare_binary([r["base"][metric] for r in rows], [r["quarter"][metric] for r in rows])
            if calculated != probe["comparisons"][metric]:
                raise ValueError(f"Scale probe comparison differs: {metric}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT / "results/sft_followup")
    parser.add_argument("--pilot", type=Path, default=ROOT / "results/followup")
    args = parser.parse_args()
    print(json.dumps(verify(args.directory, args.pilot), indent=2))
