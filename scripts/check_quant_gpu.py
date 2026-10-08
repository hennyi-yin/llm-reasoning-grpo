"""Exercise W4A16 GPU kernels on the synthetic CPU compatibility checkpoint.

Run serially after R12 and before real training. This is an engine integration
check, not a reasoning evaluation, throughput benchmark, or R10 measurement.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone


def prepare_synthetic_checkpoint(quant_python: str) -> Path:
    """Build and read the fixture using both locks instead of depending on /tmp."""
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts/check_quant_compat.py"
    produced = subprocess.run([quant_python, str(script), "produce"], cwd=root,
                              capture_output=True, text=True)
    log_path = root / "logs/quantization-compatibility-cpu.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(produced.stdout + produced.stderr, encoding="utf-8")
    if produced.returncode:
        raise RuntimeError(f"Synthetic quantization producer failed; inspect {log_path}")
    paths = [line.removeprefix("CPU_COMPAT_CHECKPOINT=") for line in produced.stdout.splitlines()
             if line.startswith("CPU_COMPAT_CHECKPOINT=")]
    if len(paths) != 1 or not (Path(paths[0]) / "model.safetensors").is_file():
        raise RuntimeError("Synthetic producer did not report one complete checkpoint")
    source = Path(paths[0])
    consumed = subprocess.run([sys.executable, str(script), "consume", "--checkpoint", str(source)],
                              cwd=root, capture_output=True, text=True)
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write(consumed.stdout + consumed.stderr)
    if consumed.returncode:
        raise RuntimeError(f"Synthetic quantization consumer failed; inspect {log_path}")
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--quant-python", default=str(Path.home() / ".venvs/llm-reasoning-grpo-quant/bin/python"))
    options = parser.parse_args()
    if options.checkpoint:
        source = options.checkpoint
    else:
        source = prepare_synthetic_checkpoint(options.quant_python)
    target = Path(tempfile.mkdtemp(prefix="llm-w4a16-kernel-gpu-"))
    # The CPU reference weights must not be seen as another HF weight shard.
    for path in source.iterdir():
        if path.is_file() and path.name != "reference_ct019.safetensors":
            shutil.copy2(path, target / path.name)
    import torch
    from vllm import LLM, SamplingParams
    if not torch.cuda.is_available():
        raise RuntimeError("The GPU compatibility check requires CUDA")
    model = LLM(model=str(target), dtype="bfloat16", max_model_len=128,
                max_num_seqs=1, max_num_batched_tokens=128, seed=13,
                gpu_memory_utilization=0.2, enforce_eager=True,
                enable_prefix_caching=False)
    output = model.generate([{"prompt_token_ids": [4, 5, 6]}],
                            SamplingParams(temperature=0, max_tokens=4, ignore_eos=True))
    if len(output) != 1 or len(output[0].outputs[0].token_ids) != 4:
        raise RuntimeError("W4A16 GPU kernel check failed to generate four tokens")
    evidence = {
        "status": "synthetic_gpu_kernel_check_passed",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "one-layer random Qwen2; synthetic token IDs; no project score or speedup",
        "gpu": torch.cuda.get_device_name(0), "compute_capability": torch.cuda.get_device_capability(0),
        "versions": {name: importlib.metadata.version(name) for name in ("vllm", "compressed-tensors", "torch")},
        "scheme": "W4A16", "output_tokens": len(output[0].outputs[0].token_ids),
        "real_checkpoint_tested": False, "test_dataset_accessed": False,
    }
    root = Path(__file__).resolve().parents[1]
    path = root / "results/compatibility/quantization_gpu.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
