"""Compile and execute FlashInfer's sampler without loading any problem data."""
import os
import shutil
import subprocess
from datetime import datetime, timezone

import torch
import flashinfer

from src.common import write_json


def main():
    nvcc = shutil.which("nvcc")
    if not nvcc:
        raise RuntimeError("Source scripts/runtime_env.sh after activating the environment")
    version = subprocess.check_output([nvcc, "--version"], text=True).strip()
    logits = torch.tensor([[0.0, 1.0, 2.0, 3.0]], device="cuda")
    sampled = flashinfer.sampling.top_k_top_p_sampling_from_logits(
        logits, top_k=1, top_p=1.0)
    torch.cuda.synchronize()
    if sampled.tolist() != [3]:
        raise RuntimeError("FlashInfer sampler returned an unexpected token")
    result = {"status": "passed", "timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "cuda_home": os.environ.get("CUDA_HOME"), "nvcc": nvcc,
              "nvcc_version": version, "torch_cuda": torch.version.cuda,
              "max_jobs": os.environ.get("MAX_JOBS"), "sampled_token": sampled.item(),
              "dataset_accessed": False}
    write_json("results/compatibility/cuda_sampler_gpu.json", result)
    print("FlashInfer CUDA sampler compiled and executed successfully", flush=True)


if __name__ == "__main__":
    main()
