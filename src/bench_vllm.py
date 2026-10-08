"""Fixed-work vLLM offline benchmark; invoke with python -m src.bench_vllm."""
from __future__ import annotations

import argparse
import json
import time

from .benchmark_common import (benchmark_inputs, checkpoint_manifest, evaluation_model_sha256, rooted,
                               measured_trial, summarize_trials, validate_config)
from .common import ROOT, load_config, sha256_file, write_json


def run(config: dict) -> dict:
    validate_config(config)
    from .protocol import require_memory_decision, verify_frozen
    verify_frozen()
    require_memory_decision()
    import torch
    import vllm
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    if not torch.cuda.is_available():
        raise RuntimeError("vLLM serving benchmark requires the CUDA GPU")
    folder = rooted(config["model"])
    manifest = checkpoint_manifest(folder)
    tok = AutoTokenizer.from_pretrained(folder, local_files_only=True)
    tokens, identity = benchmark_inputs(tok, 512)
    # Disable prefix caching: repeated prompts must still perform prefill on every run.
    llm = LLM(model=str(folder), dtype="bfloat16", seed=config["seed"],
        gpu_memory_utilization=config["vllm_gpu_memory_utilization"],
        max_model_len=config["vllm_max_model_len"], enable_prefix_caching=False)
    sampling = SamplingParams(temperature=0.0, max_tokens=256, min_tokens=256,
                              ignore_eos=True, seed=config["seed"])
    requests = [{"prompt_token_ids": ids} for ids in tokens]

    def trial() -> dict:
        torch.cuda.synchronize()
        started = time.perf_counter()
        outputs = llm.generate(requests, sampling, use_tqdm=False)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        return measured_trial(elapsed, [len(o.outputs[0].token_ids) for o in outputs])

    warmup = trial()
    trials = [trial() for _ in range(3)]
    result = {"status": "measured", "engine": "vllm", "precision": "bf16",
        "config": config, "model_manifest": manifest,
        "model_weights_sha256": manifest["weights_sha256"],
        "protocol_sha256": sha256_file(ROOT / "docs/frozen_manifest.json"),
        "model_sha256": evaluation_model_sha256(folder),
        "gpu": torch.cuda.get_device_name(), "vllm_version": vllm.__version__,
        "num_prompts": 512, "output_tokens": 256, "warmup_runs": 1,
        "repeats": 3, "prefix_caching": False,
        "timing_scope": "generation and request submission; excludes tokenization",
        **identity, "warmup": warmup, **summarize_trials(trials)}
    write_json("results/offline_vllm.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/bench.yaml")
    args = parser.parse_args()
    print(json.dumps(run(load_config(args.config)), indent=2))


if __name__ == "__main__":
    main()
