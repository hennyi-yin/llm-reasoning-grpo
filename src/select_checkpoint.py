"""Select saved adapters using GSM8K dev accuracy, breaking ties by earliest step."""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

from .common import ROOT, load_config, sha256_file, write_json
from .training import assert_frozen, memory_decision, project_path, resolve_model


def checkpoint_step(path: Path) -> int:
    match = re.fullmatch(r"checkpoint-(\d+)", path.name)
    if not match:
        raise ValueError(f"Not a numbered Trainer checkpoint: {path}")
    return int(match.group(1))


def artifact_fingerprint(path: Path) -> dict:
    weights = sorted(path.glob("adapter_model*.safetensors"))
    if not weights or not (path / "adapter_config.json").is_file():
        raise FileNotFoundError(f"Missing LoRA adapter files: {path}")
    return {item.name: sha256_file(item) for item in [path / "adapter_config.json", *weights]}


def relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def worker(config_path: str, checkpoint: str, output: str) -> None:
    assert_frozen()
    decision = memory_decision()
    config = load_config(project_path(config_path))
    base, revision = resolve_model(config, decision)
    adapter = project_path(checkpoint)
    raw = project_path(output).with_suffix(".csv")
    from .evaluate import run_dev

    rows = run_dev(base, revision=revision, adapter_path=str(adapter),
                   config_path=config["selection"]["eval_config"], output_path=str(raw))
    if len(rows) != 500 or len({row["id"] for row in rows}) != 500:
        raise RuntimeError("Checkpoint selection must evaluate all 500 unique dev problems.")
    correct = sum(bool(row["correct"]) for row in rows)
    write_json(project_path(output), {
        "checkpoint": relative_path(adapter), "step": checkpoint_step(adapter), "split": "dev",
        "task": "gsm8k", "correct": correct, "total": 500, "accuracy": correct / 500,
        "raw_predictions": relative_path(raw), "artifact_sha256": artifact_fingerprint(adapter),
        "raw_predictions_sha256": sha256_file(raw),
        "eval_config_sha256": sha256_file(config["selection"]["eval_config"]),
        "frozen_manifest_sha256": sha256_file("docs/frozen_manifest.json"),
        "base_model": base, "revision": revision,
    })


def select(config_path: str, *, reuse: bool = False) -> dict:
    assert_frozen()
    decision = memory_decision()
    config = load_config(project_path(config_path))
    base, revision = resolve_model(config, decision)
    selection = config["selection"]
    if selection["tie_break"] != "earliest_step":
        raise ValueError("Frozen selection rule is maximum correct count, then earliest step.")
    if not (project_path(config["log_dir"]) / "training_summary.json").is_file():
        raise RuntimeError("Training has not completed; no complete-run summary is available.")
    candidates = sorted(project_path(config["output_dir"]).glob("checkpoint-*"), key=checkpoint_step)
    if not candidates:
        raise FileNotFoundError("No saved checkpoints to select; train the configured stage first.")
    history = []
    directory = project_path(selection["result_dir"])
    directory.mkdir(parents=True, exist_ok=True)
    for checkpoint in candidates:
        output = directory / f"{checkpoint.name}.json"
        if output.exists():
            if not reuse:
                raise FileExistsError(f"Dev result exists: {output}. Use --reuse to verify and use recorded predictions.")
            row = json.loads(output.read_text(encoding="utf-8"))
            if row["artifact_sha256"] != artifact_fingerprint(checkpoint):
                raise RuntimeError(f"Adapter changed after dev evaluation: {checkpoint}")
            if row["eval_config_sha256"] != sha256_file(selection["eval_config"]):
                raise RuntimeError("Evaluation config changed after dev selection.")
            if row.get("base_model") != base or row.get("revision") != revision:
                raise RuntimeError("Cached dev result used a different starting model.")
            if row.get("frozen_manifest_sha256") != sha256_file("docs/frozen_manifest.json"):
                raise RuntimeError("Cached dev result used a different frozen protocol.")
            raw = project_path(row["raw_predictions"])
            if row.get("raw_predictions_sha256") != sha256_file(raw):
                raise RuntimeError("Raw dev predictions changed after evaluation.")
            with raw.open(newline="", encoding="utf-8") as stream:
                raw_rows = list(csv.DictReader(stream))
            raw_correct = sum(item["correct"].lower() in {"true", "1"} for item in raw_rows)
            if (row.get("split") != "dev" or row.get("total") != 500 or len(raw_rows) != 500
                    or len({item["id"] for item in raw_rows}) != 500 or row.get("correct") != raw_correct):
                raise RuntimeError("Cached selection does not match the 500 raw dev predictions.")
        else:
            subprocess.run([sys.executable, "-m", "src.select_checkpoint", "--config", config_path,
                            "--worker", str(checkpoint), "--worker-output", str(output)],
                           cwd=ROOT, check=True)
            row = json.loads(output.read_text(encoding="utf-8"))
        history.append(row)
        write_json(directory / "selection_history.json", history)
    best = max(history, key=lambda row: (row["correct"], -row["step"]))
    result = {**best, "selection_rule": "maximum_dev_correct_then_earliest_step",
              "candidates": history, "test_used": False}
    write_json(selection["output_path"], result)
    print(json.dumps({key: result[key] for key in ("checkpoint", "step", "accuracy", "split")}, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=["sft", "grpo", "ablation"])
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        if not args.worker_output:
            parser.error("--worker requires --worker-output")
        worker(args.config, args.worker, args.worker_output)
    else:
        config = load_config(project_path(args.config))
        expected = "sft" if config["stage"] == "sft" else "ablation" if config["model"] == "base" else "grpo"
        if args.stage and args.stage != expected:
            parser.error("--stage differs from the selected config")
        select(args.config, reuse=args.reuse)


if __name__ == "__main__":
    main()
