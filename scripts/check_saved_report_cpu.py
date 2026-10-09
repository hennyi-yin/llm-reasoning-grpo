"""Reproduce completed tables from saved raw outputs without model packages or generation."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
from importlib import metadata
import json
from pathlib import Path
import subprocess
import sys

from src.common import ROOT, sha256_file, write_json
from src.report import assemble, render_markdown
from src.stats import EXPECTED_COUNTS, MODEL_LABELS


def installed_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-minimal-env", action="store_true",
                        help="Require an environment without GPU/model/training packages")
    parser.add_argument("--record", type=Path, help="Write verification evidence after every check passes")
    args = parser.parse_args()

    model_packages = {name: version for name in ("torch", "vllm", "transformers", "trl", "peft", "datasets")
                      if (version := installed_version(name)) is not None}
    if args.require_minimal_env and model_packages:
        raise RuntimeError(f"Use the fresh CPU workflow environment; installed model packages: {model_packages}")

    summary_path = ROOT / "results/summary.json"
    expected = json.loads(summary_path.read_text(encoding="utf-8"))
    if expected.get("status") != "measured":
        raise RuntimeError("Complete every final experiment before verifying the final tables")

    actual = assemble()  # Rejudge and hash-check every saved output; recompute statistics and serving values.
    actual["figures"] = expected["figures"]  # Preserve the saved figure list; do not rerender or modify files.
    if actual != expected:
        raise RuntimeError("Reproduced summary differs from the recorded final summary")
    markdown = render_markdown(actual)
    if markdown != (ROOT / "RESULTS.md").read_text(encoding="utf-8"):
        raise RuntimeError("Reproduced result table differs from RESULTS.md")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    begin, end = "<!-- BEGIN GENERATED RESULTS -->", "<!-- END GENERATED RESULTS -->"
    if readme.count(begin) != 1 or readme.count(end) != 1:
        raise RuntimeError("README must contain exactly one generated-results marker pair")
    generated = readme.split(begin, 1)[1].split(end, 1)[0]
    if generated != "\n\n" + markdown + "\n":
        raise RuntimeError("README result block differs from the reproduced final tables")

    with (ROOT / "results/per_problem.csv").open(newline="", encoding="utf-8") as handle:
        counts = Counter((row["model"], row["task"]) for row in csv.DictReader(handle))
    expected_counts = {(model, task): count for model in MODEL_LABELS for task, count in EXPECTED_COUNTS.items()}
    if dict(counts) != expected_counts:
        raise RuntimeError(f"Incomplete final per-problem records: {counts}")
    imported_model_packages = [name for name in ("torch", "vllm", "transformers", "trl", "peft", "datasets")
                               if name in sys.modules]
    if imported_model_packages:
        raise RuntimeError(f"CPU report verification unexpectedly imported model packages: {imported_model_packages}")

    evidence = {
        "status": "verified",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "python": sys.version,
        "python_executable": sys.executable,
        "primary_dependency_versions": {name: installed_version(name)
                                        for name in ("pytest", "numpy", "scipy", "PyYAML", "math-verify")},
        "installed_model_packages": model_packages,
        "imported_model_packages": imported_model_packages,
        "gpu_used": False,
        "new_generation": False,
        "hub_dataset_access": False,
        "raw_final_outputs_rejudged_and_hash_checked": True,
        "summary_matches": True,
        "results_markdown_matches": True,
        "readme_generated_results_match": True,
        "per_problem_rows": sum(counts.values()),
        "task_counts": [{"model": model, "task": task, "n": count}
                        for (model, task), count in sorted(counts.items())],
        "source_hashes": {str(path.relative_to(ROOT)): sha256_file(path)
                          for path in (summary_path, ROOT / "RESULTS.md", ROOT / "README.md",
                                       ROOT / "results/per_problem.csv")},
    }
    if args.record:
        record_path = args.record if args.record.is_absolute() else ROOT / args.record
        write_json(record_path, evidence)
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
