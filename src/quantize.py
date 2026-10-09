"""GPTQ W4A16 with precisely 256 training questions and checkpoint size accounting."""
from __future__ import annotations

import argparse
from importlib.metadata import version
import json

from .benchmark_common import checkpoint_manifest, digest, evaluation_model_sha256, rooted
from .common import ROOT, load_config, sha256_file, write_json


def validate_config(config: dict) -> None:
    for key, expected in {"num_calibration_samples": 256, "targets": "Linear",
                          "scheme": "W4A16", "ignore": ["lm_head"]}.items():
        if config[key] != expected:
            raise ValueError(f"Quantization protocol requires {key}={expected}")


def calibration_records(tokenizer, rows: list[dict], max_seq_length: int) -> list[dict]:
    from .prompts import build_messages
    records = []
    for row in rows:
        text = tokenizer.apply_chat_template(build_messages(row["question"]), tokenize=False,
                                             add_generation_prompt=True)
        # BatchEncoding is a mapping, but the JSON provenance encoder requires
        # plain dictionaries. Keep exactly the same token lists for calibration.
        records.append(dict(tokenizer(text, add_special_tokens=False, padding=False,
                                      truncation=True, max_length=max_seq_length)))
    return records


def run(config: dict) -> dict:
    validate_config(config)
    from .protocol import require_memory_decision, verify_frozen
    verify_frozen()
    require_memory_decision()
    from .data import load_items
    import torch
    import llmcompressor
    if version("llmcompressor") != "0.14.0":
        raise RuntimeError("The verified quantization API requires llmcompressor==0.14.0")
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from llmcompressor import oneshot
    from llmcompressor.modifiers.gptq import GPTQModifier
    source, target = rooted(config["source"]), rooted(config["output"])
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"Refusing to overwrite quantized checkpoint: {target}")
    source_manifest = checkpoint_manifest(source)
    model = AutoModelForCausalLM.from_pretrained(source, dtype=torch.bfloat16,
                                                local_files_only=True)
    tok = AutoTokenizer.from_pretrained(source, local_files_only=True)
    rows = load_items(task="gsm8k", split="train")[:256]
    if len(rows) != 256 or len({r["id"] for r in rows}) != 256:
        raise ValueError("Calibration requires exactly 256 unique training questions")
    records = calibration_records(tok, rows, config["max_seq_length"])
    # Validate provenance serialization before starting the expensive GPU work.
    calibration_tokens_sha256 = digest(records)
    dataset = Dataset.from_list(records)
    torch.manual_seed(config["seed"])
    recipe = GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])
    oneshot(model=model, dataset=dataset, recipe=recipe,
        max_seq_length=config["max_seq_length"], num_calibration_samples=256)
    target.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(target, save_compressed=True)
    tok.save_pretrained(target)
    compressed_manifest = checkpoint_manifest(target)
    calibration_ids = [r["id"] for r in rows]
    metadata = {"status": "measured", "config": config,
        "protocol_sha256": sha256_file(ROOT / "docs/frozen_manifest.json"),
        "llmcompressor_version": getattr(llmcompressor, "__version__", "unknown"),
        "source": source_manifest, "compressed": compressed_manifest,
        "calibration_ids": calibration_ids, "calibration_ids_sha256": digest(calibration_ids),
        "calibration_tokens_sha256": calibration_tokens_sha256, "calibration_split": "train",
        "weight_size_ratio": source_manifest["weight_bytes"] / compressed_manifest["weight_bytes"],
        "size_definition": "referenced HF weight files only; excludes config, tokenizer and optimizer",
        "vllm_weight_memory_gib": {"bf16": None, "w4a16": None}}
    write_json(target / "quantization_provenance.json", metadata)
    # Keep complete evaluator identities outside the model to avoid self-hashing
    # provenance: the evaluator hashes every JSON file inside the checkpoint.
    metadata["source_model_sha256"] = evaluation_model_sha256(source)
    metadata["compressed_model_sha256"] = evaluation_model_sha256(target)
    write_json("results/quantization.json", metadata)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/quantize.yaml")
    args = parser.parse_args()
    print(json.dumps(run(load_config(args.config)), indent=2))


if __name__ == "__main__":
    main()
