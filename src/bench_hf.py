"""Strongest fitting HF generate baseline; invoke with python -m src.bench_hf."""
from __future__ import annotations

import argparse
import gc
import json
import time

from .benchmark_common import (benchmark_inputs, checkpoint_manifest, evaluation_model_sha256, rooted,
                               measured_trial, summarize_trials, validate_config)
from .common import load_config, write_json


def run(config: dict) -> dict:
    validate_config(config)
    from .protocol import require_memory_decision, verify_frozen
    verify_frozen()
    require_memory_decision()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .common import sha256_file, ROOT
    if not torch.cuda.is_available():
        raise RuntimeError("HF serving benchmark requires the CUDA GPU")
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    folder = rooted(config["model"])
    manifest = checkpoint_manifest(folder)
    tok = AutoTokenizer.from_pretrained(folder, padding_side="left", local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tokens, identity = benchmark_inputs(tok, config["num_prompts"])
    model = AutoModelForCausalLM.from_pretrained(folder, dtype=torch.bfloat16,
             attn_implementation=config.get("hf_attention", "sdpa"), local_files_only=True)
    model.to("cuda").eval()
    result = {"status": "running", "engine": "hf", "precision": "bf16",
        "config": config, "model_manifest": manifest,
        "model_weights_sha256": manifest["weights_sha256"],
        "protocol_sha256": sha256_file(ROOT / "docs/frozen_manifest.json"),
        "model_sha256": evaluation_model_sha256(folder),
        "gpu": torch.cuda.get_device_name(), "torch_version": torch.__version__,
        "num_prompts": 512, "output_tokens": 256, "warmup_runs": 1,
        "repeats": 3, "timing_scope": "generation and host-to-device inputs; excludes tokenization",
        **identity, "batch_results": []}
    for batch_size in config["hf_batch_sizes"]:
        # Prepare left-padded CPU inputs before timing, equally excluding tokenization.
        batches = [tok.pad({"input_ids": tokens[i:i + batch_size]},
                          padding=True, return_tensors="pt")
                   for i in range(0, len(tokens), batch_size)]

        def trial() -> dict:
            counts = []
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.inference_mode():
                for batch in batches:
                    inputs = {k: v.to("cuda") for k, v in batch.items()}
                    generated = model.generate(**inputs, do_sample=False,
                        min_new_tokens=256, max_new_tokens=256, num_beams=1,
                        pad_token_id=tok.pad_token_id, use_cache=True)
                    counts.extend([generated.shape[1] - inputs["input_ids"].shape[1]]
                                  * generated.shape[0])
                    del generated, inputs
            torch.cuda.synchronize()
            return measured_trial(time.perf_counter() - started, counts)

        torch.cuda.reset_peak_memory_stats()
        try:
            warmup = trial()
            trials = [trial() for _ in range(3)]
            row = {"batch_size": batch_size, "status": "measured", "warmup": warmup,
                   "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                   **summarize_trials(trials)}
        except torch.OutOfMemoryError as error:
            row = {"batch_size": batch_size, "status": "oom", "error": str(error)}
        if row["status"] == "oom":
            # Leave the except block before collection: its traceback can keep
            # the failed trial's CUDA tensors alive until the exception clears.
            gc.collect()
            torch.cuda.empty_cache()
        result["batch_results"].append(row)
        write_json("results/offline_hf.json", result)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    good = [r for r in result["batch_results"] if r["status"] == "measured"]
    if not good:
        result["status"] = "failed"
        write_json("results/offline_hf.json", result)
        raise RuntimeError("All required HF batch sizes ran out of memory")
    best = max(good, key=lambda r: r["output_tokens_per_second"])
    result.update(status="measured", best_batch_size=best["batch_size"],
                  output_tokens_per_second=best["output_tokens_per_second"])
    write_json("results/offline_hf.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/bench.yaml")
    args = parser.parse_args()
    print(json.dumps(run(load_config(args.config)), indent=2))


if __name__ == "__main__":
    main()
