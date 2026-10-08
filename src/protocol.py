"""Fail-closed protocol, model selection, and test-access preconditions."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from .common import ROOT, resolve, sha256_file, write_json

FROZEN_PATHS = ["src/prompts.py", "src/extract.py", "src/evaluate.py", "configs/eval.yaml",
                "src/data.py", "configs/data.yaml", "configs/models.yaml", "splits/gsm8k_split.json",
                "src/rewards.py", "src/stats.py", "src/protocol.py",
                "tests/test_extract.py", "tests/test_rewards.py", "tests/test_stats.py",
                "configs/smoke.yaml"]


def verify_frozen() -> dict:
    path = ROOT / "docs/frozen_manifest.json"
    if not path.exists():
        raise RuntimeError("Freeze and commit the evaluation protocol before training/evaluation")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        if sha256_file(name) != expected:
            raise RuntimeError(f"Frozen protocol changed: {name}")
    tracked = subprocess.check_output(["git", "ls-files", "docs/frozen_manifest.json"], cwd=ROOT, text=True).strip()
    if not tracked:
        raise RuntimeError("Frozen manifest has not been committed")
    commits = subprocess.check_output(["git", "log", "--reverse", "--format=%H", "--", "docs/frozen_manifest.json"],
                                      cwd=ROOT, text=True).splitlines()
    if not commits:
        raise RuntimeError("Frozen protocol requires an actual commit")
    original = subprocess.check_output(["git", "show", f"{commits[0]}:docs/frozen_manifest.json"], cwd=ROOT)
    if original != path.read_bytes():
        raise RuntimeError("Frozen manifest differs from its first committed version")
    return manifest


def require_memory_decision() -> dict:
    verify_frozen()
    path = ROOT / "configs/memory_decision.json"
    if not path.exists():
        raise RuntimeError("R12 memory decision is required before training or dev evaluation")
    decision = json.loads(path.read_text(encoding="utf-8"))
    if decision.get("status") != "selected" or decision.get("plan") not in {"A", "B", "C"}:
        raise RuntimeError("R12 has no successful memory plan")
    if not decision.get("decided_before_evaluation") or not re.fullmatch(r"[0-9a-f]{40}", decision.get("revision", "")):
        raise RuntimeError("Invalid model revision or R12 provenance")
    if subprocess.run(["git", "cat-file", "-e", "HEAD:configs/memory_decision.json"], cwd=ROOT, capture_output=True).returncode:
        raise RuntimeError("Commit the R12 decision before evaluating or training")
    commits = subprocess.check_output(["git", "log", "--reverse", "--format=%H", "--", "configs/memory_decision.json"],
                                      cwd=ROOT, text=True).splitlines()
    original = subprocess.check_output(["git", "show", f"{commits[0]}:configs/memory_decision.json"], cwd=ROOT)
    if original != path.read_bytes():
        raise RuntimeError("R12 decision differs from its first committed version")
    return decision


require_training_eligibility = require_memory_decision


def require_final_eligibility() -> dict:
    decision = require_memory_decision()
    registry_path = ROOT / "configs/final_models.json"
    if not registry_path.exists():
        raise RuntimeError("Final model registry is not ready")
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    labels = {"base", "sft", "grpo", "ablation", "quantized"}
    if set(registry.get("models", {})) != labels:
        raise RuntimeError("Final evaluation requires all five models")
    selection = json.loads(resolve(registry.get("selection_manifest", "results/selection.json")).read_text())
    for stage in ("sft", "grpo", "ablation"):
        record = selection.get(stage)
        if not record or record.get("split") != "dev" or not record.get("checkpoint"):
            raise RuntimeError(f"Missing dev-only checkpoint selection for {stage}")
    for label, record in registry["models"].items():
        if label == "base":
            if record.get("path") != decision["model_id"] or record.get("revision") != decision["revision"]:
                raise RuntimeError("Base model differs from the pre-evaluation R12 decision")
            continue
        path = resolve(record["path"])
        if not (path / "config.json").is_file() or not list(path.glob("*.safetensors")):
            raise RuntimeError(f"Model artifact is missing: {label}: {path}")
        if not (path / "tokenizer_config.json").is_file():
            raise RuntimeError(f"Tokenizer artifact is missing: {label}: {path}")
    quantization = json.loads((ROOT / "results/quantization.json").read_text())
    from .data import load_split
    training_ids = {f"train-{i}" for i in load_split()["train"]}
    calibration_ids = quantization.get("calibration_ids", [])
    if len(calibration_ids) != 256 or len(set(calibration_ids)) != 256 or not set(calibration_ids) <= training_ids:
        raise RuntimeError("Quantization calibration is not 256 unique training questions")
    return registry


def freeze() -> None:
    path = ROOT / "docs/frozen_manifest.json"
    if path.exists():
        raise RuntimeError("Protocol already frozen; do not replace its manifest")
    write_json(path, {"version": 1, "files": {name: sha256_file(name) for name in FROZEN_PATHS}})
    print("Protocol hashes written. Commit the manifest and frozen files before training.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["freeze", "verify"])
    args = parser.parse_args()
    freeze() if args.action == "freeze" else verify_frozen()


if __name__ == "__main__":
    main()
