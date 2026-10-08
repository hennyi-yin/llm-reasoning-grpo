import csv
import hashlib
import json
from pathlib import Path
import subprocess
import pytest

import src.pipeline as pipeline


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    monkeypatch.setattr(pipeline, "resolve", lambda path: Path(path) if Path(path).is_absolute() else tmp_path / path)
    monkeypatch.setattr(pipeline, "sha256_file", lambda path: hashlib.sha256(pipeline.resolve(path).read_bytes()).hexdigest())
    return tmp_path


def test_completed_stage_validates_evidence_before_skipping(project, monkeypatch):
    runner = pipeline.Pipeline("quant-python")
    runner.state["stages"]["memory"] = {"status": "complete"}
    checked = []
    monkeypatch.setattr(runner, "validate_stage", lambda name: checked.append(name))
    runner.stage("memory", lambda: pytest.fail("must reuse validated completion"))
    assert checked == ["memory"]


def test_completed_stage_with_missing_evidence_is_not_blindly_reused(project, monkeypatch):
    runner = pipeline.Pipeline("quant-python")
    runner.state["stages"]["memory"] = {"status": "complete"}
    def invalid(name):
        raise RuntimeError("missing evidence")
    monkeypatch.setattr(runner, "validate_stage", invalid)
    with pytest.raises(RuntimeError, match="missing evidence"):
        runner.stage("memory", lambda: pytest.fail("must not overwrite evidence"))


def test_quantization_committed_before_parent_crash_is_adopted(project, monkeypatch):
    put(project / "results/quantization.json", {"status": "measured"})
    runner = pipeline.Pipeline("quant-python")
    runner.state["stages"]["quantization"] = {"status": "running"}
    checked = []
    monkeypatch.setattr(runner, "validate_stage", lambda name: checked.append(name))
    runner.stage("quantization", lambda: pytest.fail("must not overwrite a quantized model"))
    assert checked == ["quantization"]
    assert runner.state["stages"]["quantization"]["recovered_from_complete_evidence"]


def test_operation_success_requires_valid_output(project, monkeypatch):
    runner = pipeline.Pipeline("quant-python")
    monkeypatch.setattr(pipeline.subprocess, "run", lambda *args, **kwargs: None)
    def invalid(name):
        raise RuntimeError("incomplete run")
    monkeypatch.setattr(runner, "validate_stage", invalid)
    with pytest.raises(RuntimeError, match="incomplete run"):
        runner.stage("sft_train", lambda: None)
    assert runner.state["stages"]["sft_train"]["status"] == "failed"


@pytest.mark.parametrize("stage, state", [
    ("grpo", {"global_step": 600, "epoch": .1}),
    ("sft", {"global_step": 872, "epoch": 2.0}),
])
def test_resume_suffix_steps_do_not_invalidate_complete_trainer_state(project, monkeypatch, stage, state):
    import src.protocol
    import src.select_checkpoint
    decision = {"grpo_overrides": {}}
    config = {"stage": stage, "output_dir": "checkpoints/run", "log_dir": "logs/run",
              "trainer": {"max_steps": 600, "num_train_epochs": 2}}
    monkeypatch.setattr(src.protocol, "require_memory_decision", lambda: decision)
    monkeypatch.setattr(pipeline, "load_config", lambda path: config)
    monkeypatch.setattr(src.select_checkpoint, "artifact_fingerprint", lambda path: {"weight": "hash"})
    put(project / "logs/run/run_manifest.json", {"config": config, "memory_decision": decision,
         "train_count": 6973, "dev_used_in_training": False})
    put(project / "logs/run/training_summary.json", {"steps_completed": 17})
    put(project / "checkpoints/run/trainer_state.json", state)
    (project / "checkpoints/run/checkpoint-600").mkdir()
    pipeline.validate_training("config")
    state["global_step"] = 599
    state["epoch"] = 1.9
    put(project / "checkpoints/run/trainer_state.json", state)
    with pytest.raises(RuntimeError, match="not completed"):
        pipeline.validate_training("config")


def test_resume_uses_newest_complete_checkpoint_not_partial_save(project, monkeypatch):
    config = {"output_dir": "checkpoints/sft", "log_dir": "logs/sft"}
    monkeypatch.setattr(pipeline, "load_config", lambda path: config)
    for step in (50, 100):
        directory = project / "checkpoints/sft" / f"checkpoint-{step}"
        put(directory / "trainer_state.json", {"global_step": step})
        put(directory / "adapter_config.json", {})
        (directory / "adapter_model.safetensors").write_bytes(b"weights")
        if step == 50:
            for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth"):
                (directory / name).write_bytes(b"state")
    runner = pipeline.Pipeline("quant-python")
    called = []
    monkeypatch.setattr(runner, "run", lambda *args, **kwargs: called.append(args))
    runner.train("sft_train", "configs/sft.yaml")
    assert called[0][-2:] == ("--resume", str(project / "checkpoints/sft/checkpoint-50"))


def memory_fixture(project, monkeypatch):
    import src.protocol
    settings = {"models_config": "models", "max_median_step_seconds": 180,
                "max_gpu_memory_fraction": .98,
                "plans": [{"plan": "A", "model_key": "large", "grpo_overrides": {"use_vllm": True}},
                          {"plan": "B", "model_key": "large", "grpo_overrides": {"use_vllm": False}}]}
    models = {"large": {"model_id": "Qwen/model", "revision": "a" * 40}}
    decision = {"status": "selected", "plan": "B", **models["large"],
                "grpo_overrides": {"use_vllm": False},
                "smoke_result": str(project / "results/smoke-custom/B/measurements.json")}
    evidence = {**decision, "status": "completed", "steps_completed": 20, "evaluated_accuracy": False,
                "median_step_seconds": 10, "nvml_device_peak_used_bytes": 800,
                "nvml_gpu_capacity_bytes": 1000}
    previous = {**models["large"], "plan": "A", "status": "oom", "evaluated_accuracy": False}
    put(Path(decision["smoke_result"]), evidence)
    summary = {"trials": [previous, evidence], "selected": decision}
    put(project / "results/smoke-custom/summary.json", summary)
    monkeypatch.setattr(src.protocol, "require_memory_decision", lambda: decision)
    monkeypatch.setattr(pipeline, "load_config", lambda path: settings if path == "configs/smoke.yaml" else models)
    return decision, summary


def test_plan_b_requires_successful_declared_training_only_smoke(project, monkeypatch):
    decision, _ = memory_fixture(project, monkeypatch)
    assert pipeline.validate_memory() == decision


def test_non_memory_runtime_failure_cannot_justify_plan_b(project, monkeypatch):
    _, summary = memory_fixture(project, monkeypatch)
    summary["trials"][0]["status"] = "runtime_error"
    put(project / "results/smoke-custom/summary.json", summary)
    with pytest.raises(RuntimeError, match="cannot justify fallback"):
        pipeline.validate_memory()


def test_recorded_overrides_cannot_differ_from_frozen_smoke_plan(project, monkeypatch):
    decision, _ = memory_fixture(project, monkeypatch)
    decision["grpo_overrides"] = {"use_vllm": True}
    with pytest.raises(RuntimeError, match="frozen candidate"):
        pipeline.validate_memory()


def final_fixture(project, monkeypatch, complete=True):
    import src.evaluate as evaluation
    import src.protocol
    import src.data
    registry = {"models": {"base": {"path": "pinned", "revision": "a" * 40}}}
    identity = {"model": "base", "task": "gsm8k", "split": "test",
                "model_sha256": "model", "protocol_sha256": "protocol"}
    monkeypatch.setattr(src.protocol, "require_final_eligibility", lambda: registry)
    monkeypatch.setattr(evaluation, "_identity", lambda *args: dict(identity))
    monkeypatch.setattr(evaluation, "EXPECTED_COUNTS", {"gsm8k": 2})
    monkeypatch.setattr(evaluation, "ROOT", project)
    monkeypatch.setattr(src.data, "load_items", lambda *args: pytest.fail("must not access test data"))
    identity["registry_sha256"] = evaluation._digest(registry)
    directory = project / "results/final_eval/base--gsm8k"
    store = evaluation.EvaluationStore(directory, identity)
    items = [{"id": "test-0", "question": "1+1?", "gold": "2"},
             {"id": "test-1", "question": "2+2?", "gold": "4"}]
    if complete:
        store.generate(items, lambda: lambda batch: [
            {"text": f"\\boxed{{{item['gold']}}}", "token_ids": [1], "finish_reason": "stop"}
            for item in batch], 2)
    else:
        store.initialize(items)
    return store


def test_completed_final_preflight_reads_raw_without_test_access(project, monkeypatch):
    final_fixture(project, monkeypatch)
    assert len(pipeline.final_preflight("base", "gsm8k", require_complete=True)) == 2


def test_deleted_committed_final_output_cannot_be_regenerated(project, monkeypatch):
    store = final_fixture(project, monkeypatch)
    store.raw_path("test-0").unlink()
    with pytest.raises(RuntimeError, match="regeneration is prohibited"):
        pipeline.final_preflight("base", "gsm8k")


def test_interrupted_uncertain_final_generation_blocks_before_subprocess(project, monkeypatch):
    store = final_fixture(project, monkeypatch, complete=False)
    store.manifest["in_flight"] = ["test-0"]
    store._save()
    with pytest.raises(RuntimeError, match="reruns are prohibited"):
        pipeline.final_preflight("base", "gsm8k")


def test_completed_final_raw_can_repair_missing_aggregate_csv_without_generation(project, monkeypatch):
    final_fixture(project, monkeypatch)
    runner = pipeline.Pipeline("quant-python")
    runner.validate_stage("final_base_gsm8k")
    with (project / "results/per_problem.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2 and all(row["correct"] == "True" for row in rows)


def test_commit_preserves_unrelated_user_staging(project):
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=project, text=True).strip()
    git("init", "-q")
    git("config", "user.name", "Test")
    git("config", "user.email", "test@example.invalid")
    user = project / "user.txt"
    user.write_text("original", encoding="utf-8")
    git("add", "user.txt")
    git("commit", "-q", "-m", "initial")
    user.write_text("user staged edit", encoding="utf-8")
    git("add", "user.txt")
    (project / "stage.txt").write_text("stage artifact", encoding="utf-8")
    pipeline.commit("stage result", ["stage.txt"])
    assert git("show", "HEAD:user.txt") == "original"
    assert git("diff", "--cached", "--name-only") == "user.txt"
    assert git("show", "HEAD:stage.txt") == "stage artifact"
