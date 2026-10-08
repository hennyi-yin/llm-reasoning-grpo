"""Merge a dev-selected LoRA adapter into its exact bf16 starting model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import load_config, write_json
from .select_checkpoint import artifact_fingerprint, relative_path
from .training import assert_frozen, memory_decision, project_path, resolve_model


def merge(base: str, adapter: str, output: str, revision: str | None = None,
          selection: dict | None = None) -> dict:
    assert_frozen()
    decision = memory_decision()
    adapter_path = project_path(adapter)
    fingerprint = artifact_fingerprint(adapter_path)
    if not selection:
        raise ValueError("Merge requires the recorded dev-selected checkpoint JSON.")
    if selection.get("split") != "dev" or selection.get("checkpoint") != relative_path(adapter_path):
        raise ValueError("Merge selection must refer to this dev-selected adapter.")
    if selection.get("artifact_sha256") != fingerprint:
        raise ValueError("Adapter weights have changed since dev selection.")
    target = project_path(output)
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"Merged output already contains files: {target}")
    local_base = project_path(base)
    if local_base.is_dir():
        base = str(local_base)
        lineage = local_base / "merge_manifest.json"
        if not lineage.is_file():
            raise RuntimeError("A local starting model requires merge lineage.")
        previous = json.loads(lineage.read_text(encoding="utf-8"))
        if previous.get("selected_model_id") != decision["model_id"] or previous.get("selected_revision") != decision["revision"]:
            raise ValueError("Starting model lineage differs from the R12 decision.")
    elif base != decision["model_id"] or revision != decision["revision"]:
        raise ValueError("Hub base must be the R12 model at its exact frozen revision.")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base, revision=revision, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(base, revision=revision, dtype=torch.bfloat16,
                                               device_map="cpu", trust_remote_code=False)
    model = PeftModel.from_pretrained(model, str(adapter_path))
    merged = model.merge_and_unload(safe_merge=True)
    merged.save_pretrained(target, safe_serialization=True)
    tokenizer.save_pretrained(target)
    manifest = {
        "base": base, "base_revision": revision, "adapter": relative_path(adapter_path),
        "adapter_sha256": fingerprint, "selection": selection,
        "selected_model_id": decision["model_id"], "selected_revision": decision["revision"],
        "dtype": "bfloat16", "output": relative_path(target),
    }
    write_json(target / "merge_manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Resolve base and dev-selected adapter from a training config")
    parser.add_argument("--selection", help="Selected checkpoint JSON (default: config selection.output_path)")
    parser.add_argument("--base")
    parser.add_argument("--revision")
    parser.add_argument("--adapter")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    selection = None
    if args.config:
        config = load_config(project_path(args.config))
        base, revision = resolve_model(config, memory_decision())
        selection_path = args.selection or config["selection"]["output_path"]
        selection = load_config(project_path(selection_path))
        adapter = selection["checkpoint"]
    else:
        if not args.base or not args.adapter or not args.selection:
            parser.error("Provide --config, or --base, --adapter and --selection")
        base, revision, adapter = args.base, args.revision, args.adapter
        if args.selection:
            selection = load_config(project_path(args.selection))
    result = merge(base, adapter, args.output, revision, selection)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
