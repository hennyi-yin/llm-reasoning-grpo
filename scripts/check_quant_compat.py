"""CPU-only synthetic W4A16 serializer/reader check across the two locked envs.

Producer runs in the quantization env; consumer runs in the vLLM env. Checkpoints
stay in /tmp. This compatibility evidence is never used as project R10 results.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["VLLM_TARGET_DEVICE"] = "cpu"
os.environ["LLMCOMPRESSOR_DISABLE_GPTQ_TRITON"] = "1"

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import tempfile


def produce():
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import Qwen2Config, Qwen2ForCausalLM, PreTrainedTokenizerFast
    from datasets import Dataset
    from llmcompressor import oneshot
    from llmcompressor.modifiers.gptq import GPTQModifier
    from compressed_tensors.quantization import QuantizationScheme
    from compressed_tensors.compressors.pack_quantized import PackedQuantizationCompressor
    from safetensors.torch import load_file, save_file
    assert version("llmcompressor") == "0.14.0"
    assert version("compressed-tensors") == "0.19.0"
    assert not torch.cuda.is_available()
    torch.set_num_threads(2)
    torch.manual_seed(13)
    config = Qwen2Config(vocab_size=64, hidden_size=128, intermediate_size=256,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        max_position_embeddings=128, bos_token_id=1, eos_token_id=2, pad_token_id=0,
        tie_word_embeddings=False)
    model = Qwen2ForCausalLM(config).to(dtype=torch.bfloat16, device="cpu").eval()
    backend = Tokenizer(WordLevel({f"token_{i}": i for i in range(64)}, unk_token="token_3"))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="token_0",
        bos_token="token_1", eos_token="token_2", unk_token="token_3")
    dataset = Dataset.from_list([{"input_ids": torch.randint(4, 64, (16,)).tolist(),
                                 "attention_mask": [1] * 16} for _ in range(4)])
    base = Path(tempfile.mkdtemp(prefix="llm-bf16-schema-cpu-"))
    model.save_pretrained(base)
    tokenizer.save_pretrained(base)
    model = Qwen2ForCausalLM.from_pretrained(base, dtype=torch.bfloat16,
        local_files_only=True, attn_implementation="eager")
    oneshot(model=model, processor=tokenizer, dataset=dataset,
        recipe=GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"]),
        num_calibration_samples=4, max_seq_length=16, pipeline="basic")
    assert all(p.device.type == "cpu" for p in model.parameters())
    path = Path(tempfile.mkdtemp(prefix="llm-w4a16-schema-cpu-"))
    model.save_pretrained(path, save_compressed=True)
    tokenizer.save_pretrained(path)
    raw = json.loads((path / "config.json").read_text())["quantization_config"]
    scheme = QuantizationScheme.model_validate(raw["config_groups"]["group_0"])
    state = load_file(str(path / "model.safetensors"), device="cpu")
    reference = {}
    for name in state:
        if name.endswith(".weight_packed"):
            prefix = name[:-len(".weight_packed")]
            local = {k[len(prefix) + 1:]: v for k, v in state.items() if k.startswith(prefix + ".")}
            reference[prefix + ".weight"] = PackedQuantizationCompressor.decompress(local, scheme)["weight"].contiguous()
    assert len(reference) == 7
    save_file(reference, str(path / "reference_ct019.safetensors"))
    (path / "producer_versions.json").write_text(json.dumps({k: version(k) for k in
        ("llmcompressor", "compressed-tensors", "torch", "transformers")}, indent=2))
    print(f"CPU_COMPAT_CHECKPOINT={path}", flush=True)


def consume(path, output):
    import torch
    from safetensors.torch import load_file
    from compressed_tensors.quantization import QuantizationConfig, QuantizationScheme
    from compressed_tensors.compressors.pack_quantized import PackedQuantizationCompressor
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import CompressedTensorsConfig
    assert version("compressed-tensors") == "0.17.0"
    assert version("vllm") == "0.31.0"
    assert not torch.cuda.is_available()
    raw = json.loads((path / "config.json").read_text())["quantization_config"]
    QuantizationConfig.model_validate(raw)
    scheme = QuantizationScheme.model_validate(raw["config_groups"]["group_0"])
    parsed = CompressedTensorsConfig.from_config(raw)
    args = parsed.target_scheme_map["Linear"]["weights"]
    assert args.num_bits == 4 and args.group_size == 128 and args.symmetric
    assert parsed.target_scheme_map["Linear"]["input_activations"] is None
    assert args.actorder == "static"
    state = load_file(str(path / "model.safetensors"), device="cpu")
    reference = load_file(str(path / "reference_ct019.safetensors"), device="cpu")
    checks = []
    for name in state:
        if name.endswith(".weight_packed"):
            prefix = name[:-len(".weight_packed")]
            local = {k[len(prefix) + 1:]: v for k, v in state.items() if k.startswith(prefix + ".")}
            shape = tuple(local["weight_shape"].tolist())
            assert local["weight_packed"].dtype == torch.int32
            assert tuple(local["weight_packed"].shape) == (shape[0], shape[1] // 8)
            assert tuple(local["weight_scale"].shape) == (shape[0], shape[1] // 128)
            actual = PackedQuantizationCompressor.decompress(local, scheme)["weight"]
            assert torch.equal(actual, reference[prefix + ".weight"]), prefix
            checks.append({"module": prefix, "weight_shape": list(shape),
                "packed_shape": list(local["weight_packed"].shape),
                "packed_dtype": str(local["weight_packed"].dtype),
                "scale_shape": list(local["weight_scale"].shape),
                "scale_dtype": str(local["weight_scale"].dtype), "bitwise_dequantization_match": True})
    assert len(checks) == 7 and "lm_head.weight" in state
    evidence = {"status": "cpu_serialization_compatibility_verified",
        "scope": "random tiny Qwen2 model; synthetic tokens; CPU only; no project result",
        "gpu_kernels_tested": False, "real_checkpoint_tested": False,
        "test_dataset_accessed": False, "seed": 13,
        "producer_versions": json.loads((path / "producer_versions.json").read_text()),
        "consumer_versions": {k: version(k) for k in ("vllm", "compressed-tensors", "torch", "transformers")},
        "recipe": {"targets": "Linear", "scheme": "W4A16", "ignore": ["lm_head"],
                   "group_size": 128, "actorder": "static"},
        "model_config": {"hidden_size": 128, "intermediate_size": 256, "num_hidden_layers": 1,
                         "num_attention_heads": 2, "num_key_value_heads": 1, "vocab_size": 64},
        "calibration": {"source": "synthetic random token ids", "samples": 4, "tokens_per_sample": 16},
        "vllm_config_parse": "passed", "ct017_config_parse": "passed", "lm_head_unquantized": True,
        "module_checks": checks}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["produce", "consume"])
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "results/compatibility/quantization_cpu.json")
    options = parser.parse_args()
    if options.phase == "produce":
        produce()
    elif options.checkpoint is None:
        parser.error("consume requires the temporary --checkpoint printed by produce")
    else:
        consume(options.checkpoint, options.output)
