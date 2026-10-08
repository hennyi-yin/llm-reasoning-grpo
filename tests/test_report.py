import pytest

from src.benchmark_common import digest
from src.report import complete_accuracy, render_markdown, validate_provenance, weight_ratio


def test_partial_accuracy_never_becomes_resume_value():
    summary = {"accuracy": [{"model": "base", "task": "gsm8k", "n": 10, "accuracy": 1.0}]}
    assert complete_accuracy(summary, "base", "gsm8k") is None


def test_pending_report_keeps_every_resume_placeholder_pending():
    summary = {"status": "pending_gpu_execution", "accuracy": [], "comparisons": [],
               "offline": {"hf": None, "vllm": None}, "online": [], "quantization": None,
               "vllm_weight_memory_gib": {"bf16": None, "w4a16": None},
               "resume_values": {k: None for k in "ABCDEFG"}, "quantization_mcnemar_p": None}
    rendered = render_markdown(summary)
    assert "No accuracy, speedup, or memory reduction is claimed" in rendered
    assert "| Measured | Pending | Pending | Pending | Pending | Pending | Pending | Pending |" in rendered


def test_size_ratio_is_recalculated_from_weight_only_manifest():
    def manifest(size):
        entries = [{"name": "model.safetensors", "bytes": size, "sha256": "a" * 64}]
        return {"files": entries, "weights_sha256": digest(entries), "weight_bytes": size}
    quant = {"status": "measured", "calibration_split": "train",
             "calibration_ids": [f"train-{i}" for i in range(256)],
             "source": manifest(100), "compressed": manifest(40), "weight_size_ratio": 99}
    assert weight_ratio(quant) == 2.5
    quant["compressed"]["weight_bytes"] = 50
    with pytest.raises(ValueError, match="byte count"):
        weight_ratio(quant)


def test_accuracy_serving_quantization_cannot_mix_model_artifacts(tmp_path, monkeypatch):
    import src.report as report
    monkeypatch.setattr(report, "ROOT", tmp_path)
    monkeypatch.setattr(report, "read_optional", lambda path: None)
    import src.protocol
    monkeypatch.setattr(src.protocol, "verify_frozen", lambda: {})
    monkeypatch.setattr(report, "sha256_file", lambda path: "protocol")
    a = {"status": "measured", "protocol_sha256": "protocol", "model_sha256": "model-a"}
    b = {**a, "model_sha256": "model-b"}
    with pytest.raises(ValueError, match="different final model"):
        validate_provenance(a, b, None)


def test_accuracy_serving_quantization_cannot_mix_protocols(tmp_path, monkeypatch):
    import src.report as report
    monkeypatch.setattr(report, "ROOT", tmp_path)
    monkeypatch.setattr(report, "read_optional", lambda path: None)
    a = {"status": "measured", "protocol_sha256": "protocol-a", "model_sha256": "model"}
    b = {**a, "protocol_sha256": "protocol-b"}
    with pytest.raises(ValueError, match="frozen protocol"):
        validate_provenance(a, b, None)


def raw_fixture(tmp_path, monkeypatch):
    import hashlib
    import json
    import src.report as report
    import src.protocol
    from src.common import sha256_file
    monkeypatch.setattr(report, "ROOT", tmp_path)
    monkeypatch.setattr(report, "rooted", lambda path: tmp_path / path)
    monkeypatch.setattr(report, "read_optional", lambda path: None)
    monkeypatch.setattr(src.protocol, "verify_frozen", lambda: {})
    (tmp_path / "docs").mkdir()
    frozen = tmp_path / "docs/frozen_manifest.json"
    frozen.write_text("{}")
    identity = {"model": "base", "task": "gsm8k", "split": "test", "model_sha256": "model-a",
                "protocol_sha256": sha256_file(frozen)}
    raw = {"identity": identity, "problem": {"id": "test-1", "question": "Q", "gold": "7"},
           "generation": {"text": r"Answer: \boxed{7}", "token_ids": [1, 2], "finish_reason": "stop"}}
    path = tmp_path / "raw.json"
    path.write_text(json.dumps(raw))
    return {**identity, "id": "test-1", "correct": True, "has_boxed": True,
            "n_tokens": 2, "truncated": False, "question_sha256": hashlib.sha256(b"Q").hexdigest(),
            "raw_output_path": "raw.json", "raw_output_sha256": sha256_file(path),
            "extracted_answer": "7", "finish_reason": "stop"}


def write_csv_fixture(tmp_path, rows):
    import csv
    directory = tmp_path / "results"
    directory.mkdir(exist_ok=True)
    with (directory / "per_problem.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.mark.parametrize("field,value,message", [
    ("correct", False, "correctness"), ("n_tokens", 10, "token count"),
    ("extracted_answer", "8", "extracted answer"), ("finish_reason", "length", "finish reason")])
def test_csv_derived_measurements_must_match_raw_generations(tmp_path, monkeypatch, field, value, message):
    row = raw_fixture(tmp_path, monkeypatch)
    row[field] = value
    write_csv_fixture(tmp_path, [row])
    with pytest.raises(ValueError, match=message):
        validate_provenance(None, None, None)


def test_matching_csv_raw_measurements_are_accepted(tmp_path, monkeypatch):
    row = raw_fixture(tmp_path, monkeypatch)
    write_csv_fixture(tmp_path, [row])
    validate_provenance(None, None, None)
