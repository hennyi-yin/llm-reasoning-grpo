import copy

import pytest

from src.bench_hf import completed_batches
from src.benchmark_common import digest, measured_trial, summarize_trials


def records():
    current = {"status": "running", "config": {"hf_batch_sizes": [1, 8, 16, 32], "seed": 42},
               "model_weights_sha256": "weights", "model_sha256": "full-model",
               "protocol_sha256": "frozen", "prompt_tokens_sha256": "tokens",
               "gpu": "RTX5070", "torch_version": "2.13.0", "batch_results": []}
    saved = copy.deepcopy(current)
    for size in (1, 8):
        saved["batch_results"].append({"batch_size": size, "status": "measured",
            "warmup": measured_trial(10, [256] * 512), "peak_allocated_bytes": 1234,
            **summarize_trials([measured_trial(t, [256] * 512) for t in (12, 10, 11)])})
    return saved, current


def test_resume_keeps_whole_groups_bit_for_bit_and_independently_owned():
    saved, current = records()
    original_digest = digest(saved["batch_results"])
    rows = completed_batches(saved, current)
    assert [r["batch_size"] for r in rows] == [1, 8]
    assert digest(rows) == original_digest
    rows[0]["trials"][0]["seconds"] = 99
    assert digest(saved["batch_results"]) == original_digest
    assert completed_batches(None, current) == []


@pytest.mark.parametrize("key", ["config", "model_weights_sha256", "model_sha256", "protocol_sha256",
                                "prompt_tokens_sha256", "gpu", "torch_version"])
def test_resume_refuses_different_model_inputs_settings_or_environment(key):
    saved, current = records()
    saved[key] = {"different": True} if key == "config" else "different"
    with pytest.raises(ValueError, match=key):
        completed_batches(saved, current)


@pytest.mark.parametrize("damage", ["missing_trial", "short_warmup", "wrong_rate", "wrong_summary",
                                    "unfinished", "reordered"])
def test_resume_cannot_accept_partial_or_inconsistent_measurements(damage):
    saved, current = records()
    row = saved["batch_results"][0]
    if damage == "missing_trial":
        row["trials"].pop()
    elif damage == "short_warmup":
        row["warmup"]["num_outputs"] = 511
    elif damage == "wrong_rate":
        row["trials"][0]["output_tokens_per_second"] *= 2
    elif damage == "wrong_summary":
        row["median_seconds"] = 1
    elif damage == "unfinished":
        row["status"] = "running"
    else:
        saved["batch_results"].reverse()
    with pytest.raises(ValueError):
        completed_batches(saved, current)


def test_resume_only_excludes_documented_cuda_out_of_memory():
    saved, current = records()
    saved["batch_results"].append({"batch_size": 16, "status": "oom", "error": "CUDA out of memory."})
    assert completed_batches(saved, current)[-1]["status"] == "oom"
    saved["batch_results"][-1]["error"] = "server timed out"
    with pytest.raises(ValueError, match="CUDA OOM"):
        completed_batches(saved, current)
