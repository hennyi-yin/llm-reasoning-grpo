"""Read pipeline progress from durable logs without loading any model or dataset."""
from __future__ import annotations

import argparse
import json
from .common import ROOT


def progress() -> dict:
    path = ROOT / "results/pipeline_state.json"
    state = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"stages": {}}
    output = {"stages": state["stages"], "training": {}, "smoke": []}
    for stage in ("sft", "grpo", "grpo_no_sft"):
        path = ROOT / "logs" / stage / "step_times.jsonl"
        if path.exists():
            records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            if records:
                output["training"][stage] = {"step": records[-1]["step"],
                                              "latest_step_seconds": records[-1]["seconds"]}
    for path in sorted((ROOT / "logs/smoke").glob("*/*/step_times.jsonl")):
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if records:
            output["smoke"].append({"attempt": path.parent.parent.name, "plan": path.parent.name,
                                     "step": records[-1]["step"],
                                     "latest_step_seconds": records[-1]["seconds"]})
    path = ROOT / "logs/windows_runner.json"
    if path.exists():
        output["windows_runner"] = json.loads(path.read_text(encoding="utf-8-sig"))
    return output


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps(progress(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
