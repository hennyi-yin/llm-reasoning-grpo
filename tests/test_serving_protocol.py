import copy
import json

import pytest

from src.benchmark_common import (checkpoint_manifest, measured_trial, offline_speedup,
                                  summarize_trials, weight_files)
from src.bench_online import parse_weight_memory, validate_result
from src.quantize import validate_config


def trial(seconds):
    return measured_trial(seconds, [256] * 512)


def test_timed_outputs_cannot_silently_stop_early():
    with pytest.raises(ValueError, match="fixed-length"):
        measured_trial(1.0, [255] + [256] * 511)
    with pytest.raises(ValueError, match="elapsed"):
        measured_trial(float("nan"), [256] * 512)


def test_median_is_of_elapsed_time_and_three_trials_are_required():
    summary = summarize_trials([trial(2), trial(1), trial(10)])
    assert summary["median_seconds"] == 2
    assert summary["output_tokens_per_second"] == 512 * 256 / 2
    with pytest.raises(ValueError, match="three"):
        summarize_trials([trial(1)])


def test_speedup_uses_strongest_hf_batch_and_validates_prompt_identity():
    identity = {"model_weights_sha256": "weights", "prompt_ids_sha256": "ids",
                "prompt_tokens_sha256": "tokens", "protocol_sha256": "protocol",
                "num_prompts": 512, "output_tokens": 256, "status": "measured",
                "model_sha256": "full-model"}
    hf = {**identity, "batch_results": [
        {"batch_size": n, "status": "measured", "trials": [trial(s)] * 3}
        for n, s in [(1, 10), (8, 5), (16, 2)]] + [{"batch_size": 32, "status": "oom"}]}
    vllm = {**identity, "trials": [trial(1)] * 3}
    assert offline_speedup(hf, vllm) == 2
    changed = {**vllm, "prompt_tokens_sha256": "different"}
    with pytest.raises(ValueError, match="prompt_tokens"):
        offline_speedup(hf, changed)
    assert offline_speedup(None, vllm) is None


def test_checkpoint_accounting_excludes_tokenizer_and_optimizer(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"12345")
    (tmp_path / "tokenizer.json").write_bytes(b"unused" * 20)
    (tmp_path / "optimizer.pt").write_bytes(b"unused" * 20)
    assert [p.name for p in weight_files(tmp_path)] == ["model.safetensors"]
    assert checkpoint_manifest(tmp_path)["weight_bytes"] == 5


def test_checkpoint_index_rejects_parent_shard(tmp_path):
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"x": "../outside.safetensors"}}))
    with pytest.raises(ValueError, match="out-of-directory"):
        weight_files(tmp_path)


def online_fixture():
    return {"completed": 500, "failed": 0, "input_lens": [256] * 500,
            "output_lens": [256] * 500, "total_output_tokens": 500 * 256,
            "output_throughput": 100.0, "p50_ttft_ms": 10.0, "p99_ttft_ms": 20.0,
            "p50_tpot_ms": 1.0, "p99_tpot_ms": 2.0}


def test_online_result_requires_fixed_actual_lengths_and_no_failures():
    raw = online_fixture()
    assert validate_result(raw)["output_throughput"] == 100
    early = copy.deepcopy(raw)
    early["output_lens"][0] = 12
    with pytest.raises(ValueError, match="output_lens"):
        validate_result(early)
    failed = {**raw, "failed": 1}
    with pytest.raises(ValueError, match="500 requests"):
        validate_result(failed)


def test_loaded_weight_memory_is_not_total_process_memory():
    assert parse_weight_memory("GPU memory utilization 0.85; KV cache 3.2 GiB") is None
    assert parse_weight_memory("Loading model weights took 2.93 GiB") == 2.93
    assert parse_weight_memory("Model loading took 2.930 GiB memory and 1.5 seconds") == 2.93


def test_quantization_recipe_cannot_use_test_data_or_different_sample_count():
    config = {"num_calibration_samples": 256, "targets": "Linear", "scheme": "W4A16", "ignore": ["lm_head"]}
    validate_config(config)
    with pytest.raises(ValueError, match="num_calibration_samples"):
        validate_config({**config, "num_calibration_samples": 500})
