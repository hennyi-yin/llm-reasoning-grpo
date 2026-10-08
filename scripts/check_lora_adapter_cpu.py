"""Read an actual PEFT adapter through vLLM without generation or CUDA access."""
from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["VLLM_TARGET_DEVICE"] = "cpu"

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", default="results/compatibility/sft_adapter_cpu.json")
    args = parser.parse_args()
    import torch
    from vllm.config import LoRAConfig
    from vllm.lora.peft_helper import PEFTHelper
    from vllm.lora.lora_model import LoRAModel
    from src.common import resolve, write_json
    from src.select_checkpoint import artifact_fingerprint, relative_path
    checkpoint = resolve(args.checkpoint)
    fingerprint = artifact_fingerprint(checkpoint)
    assert not torch.cuda.is_available()
    helper = PEFTHelper.from_local_dir(str(checkpoint), max_position_embeddings=4096)
    helper.validate_legal(LoRAConfig(max_lora_rank=16))
    modules = {"q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "gate_proj", "down_proj"}
    adapter = LoRAModel.from_local_checkpoint(str(checkpoint), expected_lora_modules=modules,
        peft_helper=helper, device="cpu", dtype=torch.bfloat16)
    devices = {str(tensor.device) for layer in adapter.loras.values()
               for tensor in (layer.lora_a, layer.lora_b)}
    if adapter.rank != 16 or not adapter.loras or devices != {"cpu"}:
        raise RuntimeError("Actual LoRA adapter failed the rank/device compatibility check")
    if artifact_fingerprint(checkpoint) != fingerprint:
        raise RuntimeError("Adapter files changed during the read-only compatibility check")
    write_json(args.output, {"status": "cpu_adapter_loading_verified",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": relative_path(checkpoint), "artifact_sha256": fingerprint,
        "rank": adapter.rank, "loaded_layers": len(adapter.loras), "devices": sorted(devices),
        "versions": {name: version(name) for name in ("vllm", "peft", "torch")},
        "generated_answers": False, "test_dataset_accessed": False, "gpu_kernels_tested": False})


if __name__ == "__main__":
    main()
