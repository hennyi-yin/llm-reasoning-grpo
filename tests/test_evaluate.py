import json
import sys
from types import SimpleNamespace
import pytest
from src.evaluate import EvaluationStore, judge, run_final, validate_config


def test_math_verify_sees_only_final_box(monkeypatch):
    seen = []
    def parse(text, **kwargs):
        seen.append(text)
        return [text]
    monkeypatch.setitem(sys.modules, "math_verify", SimpleNamespace(
        LatexExtractionConfig=lambda **kwargs: object(), parse=parse,
        verify=lambda gold, pred: gold == pred))
    assert not judge("math500", "2", r"Earlier \boxed{2}; final \boxed{3}")
    assert seen == [r"\boxed{2}", r"\boxed{3}"]
    seen.clear()
    assert not judge("math500", "2", r"Earlier \boxed{2}; final \boxed{")
    assert seen == []


def identity():
    return {"model": "base", "task": "gsm8k", "split": "test",
            "model_sha256": "model-hash", "protocol_sha256": "protocol-hash"}


def items():
    return [{"id": "test-0", "question": "One plus one?", "gold": "2"},
            {"id": "test-1", "question": "Two plus two?", "gold": "4"}]


def test_completed_store_never_regenerates(tmp_path):
    store = EvaluationStore(tmp_path / "ledger", identity())
    calls = []
    def factory():
        def generate(batch):
            calls.append(len(batch))
            return [{"text": f"\\boxed{{{item['gold']}}}", "token_ids": [1, 2],
                     "finish_reason": "stop"} for item in batch]
        return generate
    first = store.generate(items(), factory, 1)
    assert calls == [1, 1] and all(row["correct"] for row in first)
    second = EvaluationStore(tmp_path / "ledger", identity()).generate(
        items(), lambda: pytest.fail("completed outputs must not reload the model"), 1)
    assert first == second


def test_partial_finished_chunks_resume_without_regeneration(tmp_path):
    store = EvaluationStore(tmp_path / "ledger", identity())
    store.initialize(items())
    # Simulate a crash after the first durable raw output but before clearing its
    # in-flight claim. Recovery may judge it; it must never sample it again.
    first = items()[0]
    raw = {"identity": identity(), "problem": first,
           "generation": {"text": r"\boxed{2}", "token_ids": [1], "finish_reason": "stop"}}
    store.raw_path(first["id"]).parent.mkdir(parents=True)
    store.raw_path(first["id"]).write_text(json.dumps(raw), encoding="utf-8")
    store.manifest["in_flight"] = [first["id"]]
    store._save()
    generated = []
    def factory():
        def generate(batch):
            generated.extend(item["id"] for item in batch)
            return [{"text": r"\boxed{4}", "token_ids": [2], "finish_reason": "length"}]
        return generate
    result = store.generate(items(), factory, 1)
    assert generated == ["test-1"]
    assert result[1]["truncated"]


def test_uncertain_inflight_is_never_retried(tmp_path):
    store = EvaluationStore(tmp_path / "ledger", identity())
    store.initialize(items())
    store.manifest["in_flight"] = ["test-0"]
    store._save()
    with pytest.raises(RuntimeError, match="reruns are prohibited"):
        store.generate(items(), lambda: pytest.fail("must not generate"), 1)


def test_claim_cannot_be_reused_for_different_model(tmp_path):
    EvaluationStore(tmp_path / "ledger", identity())
    altered = identity()
    altered["model_sha256"] = "changed-model"
    with pytest.raises(RuntimeError, match="different model/protocol"):
        EvaluationStore(tmp_path / "ledger", altered)


def test_committed_raw_tampering_rejected(tmp_path):
    store = EvaluationStore(tmp_path / "ledger", identity())
    store.generate(items(), lambda: lambda batch: [
        {"text": r"\boxed{0}", "token_ids": [1], "finish_reason": "stop"}
        for item in batch], 2)
    path = store.raw_path("test-0")
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["generation"]["text"] = r"\boxed{2}"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(RuntimeError, match="Committed raw generation has changed"):
        store.rows()


def test_final_gate_precedes_test_access(monkeypatch):
    import src.protocol
    import src.data
    def deny():
        raise RuntimeError("not eligible")
    monkeypatch.setattr(src.protocol, "require_final_eligibility", deny)
    monkeypatch.setattr(src.data, "load_items", lambda *args: pytest.fail("test accessed before eligibility"))
    with pytest.raises(RuntimeError, match="not eligible"):
        run_final("fake", model_label="base", task="gsm8k")


def test_final_completed_task_returns_without_dataset_or_model(monkeypatch, tmp_path):
    import src.evaluate as evaluation
    import src.protocol
    import src.data
    registry = {"models": {"base": {"path": "pinned", "revision": "a" * 40}}}
    final_identity = {**identity(), "registry_sha256": evaluation._digest(registry)}
    monkeypatch.setattr(evaluation, "ROOT", tmp_path)
    monkeypatch.setattr(evaluation, "EXPECTED_COUNTS", {"gsm8k": 2})
    monkeypatch.setattr(src.protocol, "require_final_eligibility", lambda: registry)
    monkeypatch.setattr(evaluation, "model_identity", lambda *args: {"same": "model"})
    monkeypatch.setattr(evaluation, "_identity", lambda *args: identity())
    monkeypatch.setattr(src.data, "load_items", lambda *args: pytest.fail("completed test must not access dataset"))
    monkeypatch.setattr(evaluation, "_make_generator", lambda *args: pytest.fail("completed test must not generate"))
    directory = tmp_path / "results/final_eval/base--gsm8k"
    store = EvaluationStore(directory, final_identity)
    store.generate(items(), lambda: lambda batch: [
        {"text": f"\\boxed{{{item['gold']}}}", "token_ids": [1], "finish_reason": "stop"}
        for item in batch], 2)
    output = evaluation.run_final("pinned", model_label="base", task="gsm8k")
    assert len(output) == 2 and all(row["correct"] for row in output)
    assert (tmp_path / "results/per_problem.csv").exists()


def test_decoding_limits_are_frozen():
    config = {"seed": 0, "temperature": 0., "top_p": 1., "top_k": -1,
              "n": 1, "dtype": "bfloat16", "numeric_tolerance": 1e-6,
              "max_tokens": {"gsm8k": 1024, "math500": 2048}}
    validate_config(config)
    config["max_tokens"]["gsm8k"] = 512
    with pytest.raises(ValueError):
        validate_config(config)
