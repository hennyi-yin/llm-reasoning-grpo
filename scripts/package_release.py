"""Package selected adapters and W4A16 locally; this never uploads or publishes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def hash_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("release_assets"))
    options = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = options.output if options.output.is_absolute() else root / options.output
    output.mkdir(parents=True, exist_ok=True)
    summary = json.loads((root / "results/summary.json").read_text())
    if summary["status"] != "measured":
        raise RuntimeError("Finish all measured experiments before packaging final Release artifacts")
    archives = []
    for stage in ("sft", "grpo", "ablation"):
        selection = json.loads((root / f"results/selected_{stage}.json").read_text())
        directory = root / selection["checkpoint"]
        files = [directory / "adapter_config.json", *directory.glob("adapter_model*.safetensors")]
        files += [p for p in directory.glob("*token*") if p.is_file()]
        files += [p for p in directory.glob("*.txt") if p.is_file()]
        archive = output / f"{stage}-adapter.tar.gz"
        if archive.exists():
            raise FileExistsError(f"Preserve the existing release archive: {archive}")
        with tarfile.open(archive, "w:gz") as tar:
            for path in sorted(set(files)):
                if not path.is_file():
                    raise FileNotFoundError(path)
                tar.add(path, arcname=path.name)
        archives.append({"name": archive.name, "bytes": archive.stat().st_size,
                         "sha256": hash_file(archive), "selected_checkpoint": selection["checkpoint"]})
    directory = root / "models/final-w4a16"
    archive = output / "final-w4a16.tar.gz"
    if archive.exists():
        raise FileExistsError(f"Preserve the existing release archive: {archive}")
    with tarfile.open(archive, "w:gz") as tar:
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                tar.add(path, arcname=path.relative_to(directory))
    archives.append({"name": archive.name, "bytes": archive.stat().st_size, "sha256": hash_file(archive)})
    (output / "manifest.json").write_text(json.dumps({"uploaded": False, "artifacts": archives,
        "resume_values": summary["resume_values"]}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(archives, indent=2))


if __name__ == "__main__":
    main()
