"""Create an independent reproduction without altering this run or its ledgers."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


EXCLUDED_DIRS = {"results", "logs", "figures", "checkpoints", "models", "release_assets"}
EXCLUDED_FILES = {"configs/memory_decision.json", "configs/final_models.json",
                  "RESULTS.md", "docs/execution.md"}


def create(destination: Path) -> Path:
    root = Path(__file__).resolve().parents[1]
    destination = destination.resolve()
    if destination.is_relative_to(root):
        raise ValueError("Use a destination outside the original experiment")
    if destination.exists():
        raise FileExistsError(f"Choose a new directory: {destination}")
    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    # Use committed content, so a running pipeline's mutable logs cannot enter
    # the new run, and the recorded source commit actually identifies the code.
    paths = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", "-z", source_commit], cwd=root)
    destination.mkdir(parents=True)
    for name in paths.decode("utf-8").split("\0"):
        if not name or Path(name).parts[0] in EXCLUDED_DIRS or name in EXCLUDED_FILES:
            continue
        target = destination / name
        if not target.resolve().is_relative_to(destination):
            raise ValueError(f"Invalid tracked path: {name}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(subprocess.check_output(["git", "show", f"{source_commit}:{name}"], cwd=root))
    protocol = destination / "docs/protocol.md"
    text = protocol.read_text(encoding="utf-8")
    text = text.split("## R12 decision")[0] + (
        "## R12 decision\n\nPending: no model-size decision has been made and no dev/test evaluation is permitted.\n")
    protocol.write_text(text, encoding="utf-8")
    readme = destination / "README.md"
    text = readme.read_text(encoding="utf-8")
    if "The actual RTX5070 passed" in text:
        before, remaining = text.split("The actual RTX5070 passed", 1)
        text = before + "<!-- BEGIN GENERATED RESULTS -->" + remaining.split("<!-- BEGIN GENERATED RESULTS -->", 1)[1]
    readme.write_text(text, encoding="utf-8")
    provenance = {"independent_experiment": True, "source_commit": source_commit,
                  "copied_previous_run_evidence": False, "copied_previous_model_selection": False,
                  "evaluation_code": "unchanged source protocol hashes",
                  "test_policy": "once per model/task in this independent reproduction"}
    (destination / "docs/reproduction_source.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    (destination / "docs/execution.md").write_text(
        f"# Independent reproduction\n\nSource commit: `{source_commit}`.\n\n"
        "Original results, logs, model selections and evaluation ledgers were excluded. "
        "The evaluation code and split hashes are unchanged. Run preparation and the full "
        "pipeline in this directory; it will measure and commit a new memory decision first.\n",
        encoding="utf-8")
    subprocess.run([sys.executable, "-m", "src.report", "--update-readme", "--no-plots"],
                   cwd=destination, check=True, capture_output=True, text=True)
    for command in (["git", "init", "-q", "-b", "main"],
                    ["git", "config", "user.name", "Codex"],
                    ["git", "config", "user.email", "codex@local"],
                    ["git", "add", "."],
                    ["git", "commit", "-q", "-m", "Freeze unchanged source protocol for independent reproduction"]):
        subprocess.run(command, cwd=destination, check=True)
    subprocess.run([sys.executable, "-m", "src.protocol", "verify"], cwd=destination, check=True)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(create(args.destination))


if __name__ == "__main__":
    main()
