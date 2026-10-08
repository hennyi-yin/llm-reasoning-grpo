"""Package selected adapters and W4A16 locally; this never uploads or publishes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import os
import subprocess
import sys
import tarfile


def hash_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package(root: Path, output: Path) -> dict:
    summary = json.loads((root / "results/summary.json").read_text())
    if summary["status"] != "measured":
        raise RuntimeError("Finish all measured experiments before packaging final Release artifacts")
    decision = json.loads((root / "configs/memory_decision.json").read_text())
    license_path = root / "licenses" / (decision["model_id"].replace("/", "--") + "-LICENSE")
    if not license_path.is_file():
        raise FileNotFoundError(f"Missing pinned upstream model license: {license_path}")
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        expected = {f"{stage}-adapter.tar.gz" for stage in ("sft", "grpo", "ablation")}
        expected |= {f"{label}.tar.gz" for label in ("sft-bf16", "final-bf16", "ablation-bf16", "final-w4a16")}
        if (manifest.get("summary_sha256") != hash_file(root / "results/summary.json")
                or {row["name"] for row in manifest.get("artifacts", [])} != expected):
            raise RuntimeError("Existing Release manifest differs from the completed experiment")
        for row in manifest["artifacts"]:
            if hash_file(output / row["name"]) != row["sha256"]:
                raise RuntimeError("Existing Release archive changed; preserve it for inspection")
            for record in row["source_files"]:
                source = (root / record["path"]).resolve()
                if not source.is_relative_to(root.resolve()) or hash_file(source) != record["sha256"]:
                    raise RuntimeError("Release source artifact changed after packaging")
        return manifest
    sources = []
    for stage in ("sft", "grpo", "ablation"):
        selection = json.loads((root / f"results/selected_{stage}.json").read_text())
        directory = root / selection["checkpoint"]
        files = [directory / "adapter_config.json", *directory.glob("adapter_model*.safetensors")]
        actual = {path.name: hash_file(path) for path in files}
        if actual != selection["artifact_sha256"]:
            raise RuntimeError(f"Selected {stage} adapter changed before Release packaging")
        files += [p for p in directory.glob("*token*") if p.is_file()]
        files += [p for p in directory.glob("*.txt") if p.is_file()]
        sources.append((f"{stage}-adapter", directory, sorted(set(files))))
    for label, relative in (("sft-bf16", "checkpoints/sft_merged"), ("final-bf16", "models/final-bf16"),
                            ("ablation-bf16", "models/ablation-bf16"), ("final-w4a16", "models/final-w4a16")):
        directory = root / relative
        if not (directory / "config.json").is_file() or not list(directory.glob("*.safetensors")):
            raise FileNotFoundError(f"Missing complete model: {directory}")
        sources.append((label, directory, sorted(p for p in directory.rglob("*") if p.is_file())))
    output.mkdir(parents=True, exist_ok=True)
    archives = []
    for label, directory, files in sources:
        archive = output / f"{label}.tar.gz"
        if archive.exists():
            raise FileExistsError(f"Preserve the existing release archive: {archive}")
        records = []
        temporary = archive.with_suffix(archive.suffix + ".tmp")
        if temporary.exists():
            raise FileExistsError(f"Inspect interrupted Release archive: {temporary}")
        with tarfile.open(temporary, "w:gz", compresslevel=1) as tar:
            for path in files:
                if path.is_symlink() or not path.resolve().is_relative_to(directory.resolve()):
                    raise ValueError(f"Unexpected model symlink or out-of-directory file: {path}")
                records.append({"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size,
                                "sha256": hash_file(path)})
                tar.add(path, arcname=path.relative_to(directory))
            records.append({"path": license_path.relative_to(root).as_posix(), "bytes": license_path.stat().st_size,
                            "sha256": hash_file(license_path), "archive_path": "UPSTREAM_MODEL_LICENSE"})
            tar.add(license_path, arcname="UPSTREAM_MODEL_LICENSE")
        os.replace(temporary, archive)
        archives.append({"name": archive.name, "bytes": archive.stat().st_size,
                         "sha256": hash_file(archive), "source_files": records})
    manifest = {"uploaded": False, "artifacts": archives, "resume_values": summary["resume_values"],
                "summary_sha256": hash_file(root / "results/summary.json"),
                "upstream_model": {"model_id": decision["model_id"], "revision": decision["revision"],
                                   "license_sha256": hash_file(license_path)}}
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, manifest_path)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("release_assets"))
    options = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    # Validate completion, immutable ledgers and selected/quantized artifacts
    # before packing. These checks only read already recorded final outputs.
    sys.path.insert(0, str(root))
    from src.pipeline import Pipeline, final_preflight
    pipeline = Pipeline(str(Path.home() / ".venvs/llm-reasoning-grpo-quant/bin/python"))
    pipeline.validate_stage("report")
    pipeline.validate_registry()
    for label in ("sft", "ablation"):
        final_preflight(label, "gsm8k", require_complete=True)
    output = options.output if options.output.is_absolute() else root / options.output
    manifest = package(root, output)
    lightweight = {key: value for key, value in manifest.items() if key != "artifacts"}
    lightweight["artifacts"] = [{key: value for key, value in row.items() if key != "source_files"}
                                  for row in manifest["artifacts"]]
    (root / "results/release_package.json").write_text(json.dumps(lightweight, indent=2) + "\n", encoding="utf-8")
    subprocess.run(["git", "add", "--", "results/release_package.json"], cwd=root, check=True)
    subprocess.run(["git", "commit", "--only", "-m", "Record verified local Release archives", "--", "results/release_package.json"],
                   cwd=root, check=False)
    print(json.dumps(lightweight, indent=2))


if __name__ == "__main__":
    main()
