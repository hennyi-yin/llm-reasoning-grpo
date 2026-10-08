"""Read pipeline progress from durable logs without loading any model or dataset."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from .common import ROOT


def read_records(path):
    """A live callback may be halfway through its final append when read."""
    content = path.read_text(encoding="utf-8")
    lines = content.splitlines()
    records = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines) - 1 or content.endswith("\n"):
                raise
    return records


def progress() -> dict:
    path = ROOT / "results/pipeline_state.json"
    state = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"stages": {}}
    output = {"stages": state["stages"], "training": {}, "smoke": []}
    for stage in ("sft", "grpo", "grpo_no_sft"):
        path = ROOT / "logs" / stage / "step_times.jsonl"
        if path.exists():
            records = read_records(path)
            if records:
                current = {"step": records[-1]["step"], "latest_step_seconds": records[-1]["seconds"]}
                manifest_path = ROOT / "logs" / stage / "run_manifest.json"
                if manifest_path.is_file():
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    config = manifest["config"]["trainer"]
                    total = config["max_steps"]
                    if total < 0:
                        batch = config["per_device_train_batch_size"] * config["gradient_accumulation_steps"]
                        total = math.ceil(math.ceil(manifest["train_count"] / batch) * config["num_train_epochs"])
                    median = statistics.median(row["seconds"] for row in records[-30:])
                    current.update(total_steps=total, progress_percent=100 * current["step"] / total,
                                   recent_median_step_seconds=median)
                    pipeline_stage = "ablation" if stage == "grpo_no_sft" else stage
                    if state["stages"].get(f"{pipeline_stage}_train", {}).get("status") == "running":
                        current["estimated_remaining_seconds"] = max(0, total - current["step"]) * median
                output["training"][stage] = current
    output["dev_evaluation"] = []
    for path in sorted((ROOT / "results/dev").glob("*/*_raw/manifest.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        output["dev_evaluation"].append({"stage": path.parent.parent.name, "checkpoint": path.parent.name,
                                         "status": manifest["status"],
                                         "committed_predictions": len(manifest.get("raw_sha256", {})),
                                         "total_problems": len(manifest.get("items", []))})
    for path in sorted((ROOT / "logs/smoke").glob("*/*/step_times.jsonl")):
        records = read_records(path)
        if records:
            output["smoke"].append({"attempt": path.parent.parent.name, "plan": path.parent.name,
                                     "step": records[-1]["step"],
                                     "latest_step_seconds": records[-1]["seconds"]})
    path = ROOT / "logs/windows_runner.json"
    if path.exists():
        output["windows_runner"] = json.loads(path.read_text(encoding="utf-8-sig"))
    path = ROOT / "logs/release_runner.json"
    if path.exists():
        output["release_runner"] = json.loads(path.read_text(encoding="utf-8-sig"))
    return output


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps(progress(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
