"""Record installed packages and GPU facts without collecting credentials."""
from __future__ import annotations

import importlib.metadata
import platform
import subprocess
import sys
from datetime import datetime, timezone
from .common import ROOT, write_json


def main():
    import torch
    packages = {item.metadata["Name"]: item.version for item in importlib.metadata.distributions()}
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used",
                          "--format=csv,noheader"], capture_output=True, text=True)
    result = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
              "platform": platform.platform(), "packages": packages, "gpu": gpu.stdout.strip(),
              "torch_cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        result["compute_capability"] = torch.cuda.get_device_capability(0)
        torch.ones(8, device="cuda").sum().item()
    write_json("results/environment.json", result)
    for name in ("vllm", "trl", "torch", "transformers", "peft", "datasets"):
        print(f"{name}=={packages.get(name, 'MISSING')}")
    print(f"GPU: {gpu.stdout.strip()}; torch CUDA {torch.version.cuda}; available={torch.cuda.is_available()}")


if __name__ == "__main__":
    main()
