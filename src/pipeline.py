"""Run the measured project sequentially, retaining stage logs and local commits."""
from __future__ import annotations

import argparse
import copy
import csv
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
import math
from pathlib import Path
import subprocess
import sys
import re

from .common import ROOT, load_config, resolve, write_json, sha256_file


def now():
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def pipeline_lock():
    import fcntl
    path = ROOT / "logs/pipeline.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("The project pipeline is already running") from error
        yield


def commit(message, paths):
    existing = [name for name in paths if resolve(name).exists()]
    if not existing:
        return
    subprocess.run(["git", "add", "--", *existing], cwd=ROOT, check=True)
    changed = subprocess.run(["git", "diff", "--cached", "--quiet", "--", *existing], cwd=ROOT).returncode
    if changed not in (0, 1):
        raise RuntimeError("Could not inspect staged project changes")
    if changed == 1:
        # Preserve unrelated user-staged files; only the stated stage paths go
        # into this commit, even when the index was already nonempty.
        subprocess.run(["git", "commit", "--only", "-m", message, "--", *existing], cwd=ROOT, check=True)


def read_json(path):
    return json.loads(resolve(path).read_text(encoding="utf-8"))


def relative(path):
    return resolve(path).resolve().relative_to(ROOT.resolve()).as_posix()


TRAIN_CONFIGS = {"sft": "configs/sft.yaml", "grpo": "configs/grpo.yaml",
                 "ablation": "configs/grpo_no_sft.yaml"}
MERGED_OUTPUTS = {"sft": "checkpoints/sft_merged", "grpo": "models/final-bf16",
                  "ablation": "models/ablation-bf16"}


def validate_memory():
    from .protocol import require_memory_decision
    decision = require_memory_decision()
    settings = load_config("configs/smoke.yaml")
    models = load_config(settings["models_config"])
    entries = settings["plans"]
    match = next((entry for entry in entries if entry["plan"] == decision["plan"]), None)
    if not match:
        raise RuntimeError("R12 selected an undeclared plan")
    declared = models[match["model_key"]]
    if (decision["model_id"] != declared["model_id"] or decision["revision"] != declared["revision"]
            or decision.get("grpo_overrides") != match["grpo_overrides"]):
        raise RuntimeError("R12 decision differs from the frozen candidate")
    evidence = read_json(decision["smoke_result"])
    if (evidence.get("status") != "completed" or evidence.get("steps_completed") != 20
            or evidence.get("evaluated_accuracy") is not False
            or any(evidence.get(key) != decision[key] for key in ("plan", "model_id", "revision", "grpo_overrides"))):
        raise RuntimeError("R12 lacks a successful 20-step training-only smoke result")
    duration = evidence.get("median_step_seconds")
    peak, capacity = evidence.get("nvml_device_peak_used_bytes"), evidence.get("nvml_gpu_capacity_bytes")
    if (not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0
            or duration > settings["max_median_step_seconds"]
            or not isinstance(peak, int) or not isinstance(capacity, int) or capacity <= 0
            or peak <= 0 or peak > settings["max_gpu_memory_fraction"] * capacity):
        raise RuntimeError("R12 evidence fails the frozen memory/time limits")
    summary = read_json(resolve(decision["smoke_result"]).parent.parent / "summary.json")
    trials = summary.get("trials", [])
    selected_index = next(i for i, entry in enumerate(entries) if entry["plan"] == decision["plan"])
    if len(trials) != selected_index + 1 or summary.get("selected") != decision:
        raise RuntimeError("R12 does not follow the first-feasible-plan selection rule")
    for index, trial in enumerate(trials):
        plan = entries[index]
        model = models[plan["model_key"]]
        if (trial.get("plan") != plan["plan"] or trial.get("model_id") != model["model_id"]
                or trial.get("revision") != model["revision"] or trial.get("evaluated_accuracy") is not False):
            raise RuntimeError("R12 fallback trial identity is invalid")
        if index < selected_index and trial.get("status") not in {
            "oom", "too_slow", "insufficient_headroom", "timed_out"
        }:
            raise RuntimeError("A runtime error or successful earlier trial cannot justify fallback")
    return decision


def resolved_training_config(config_path, decision):
    config = copy.deepcopy(load_config(config_path))
    if config["stage"] == "grpo":
        config["trainer"].update(decision["grpo_overrides"])
    return config


def validate_training(config_path):
    from .protocol import require_memory_decision
    from .select_checkpoint import artifact_fingerprint
    decision = require_memory_decision()
    config = resolved_training_config(config_path, decision)
    manifest = read_json(resolve(config["log_dir"]) / "run_manifest.json")
    if manifest.get("config") != config or manifest.get("memory_decision") != decision:
        raise RuntimeError("Recorded training config/model differs from the current run")
    if manifest.get("train_count") != 6973 or manifest.get("dev_used_in_training") is not False:
        raise RuntimeError("Training provenance does not establish the frozen training partition")
    report = read_json(resolve(config["log_dir"]) / "training_summary.json")
    if not isinstance(report.get("steps_completed"), int) or report["steps_completed"] < 0:
        raise RuntimeError("Training has no valid run summary")
    output = resolve(config["output_dir"])
    state = read_json(output / "trainer_state.json")
    # A resumed callback only measures the remaining suffix. The Trainer state
    # records the complete optimizer progress and is the completion authority.
    if config["stage"] == "grpo":
        if state.get("global_step", -1) < config["trainer"]["max_steps"]:
            raise RuntimeError("GRPO has not completed the configured optimizer steps")
    elif state.get("epoch", -1) < config["trainer"]["num_train_epochs"] - 1e-6:
        raise RuntimeError("SFT has not completed the configured epochs")
    artifact_fingerprint(output / "final_adapter")
    checkpoints = [p for p in output.glob("checkpoint-*") if re.fullmatch(r"checkpoint-\d+", p.name)]
    if not checkpoints:
        raise RuntimeError("Completed training has no selectable checkpoint")


def validate_selection(config_path):
    from .data import load_split
    from .protocol import require_memory_decision
    from .select_checkpoint import artifact_fingerprint, checkpoint_step
    from .training import resolve_model
    from .stats import parse_bool
    decision = require_memory_decision()
    config = load_config(config_path)
    selection = read_json(config["selection"]["output_path"])
    candidates = selection.get("candidates", [])
    if (selection.get("split") != "dev" or selection.get("task") != "gsm8k"
            or selection.get("test_used") is not False or not candidates
            or selection.get("selection_rule") != "maximum_dev_correct_then_earliest_step"):
        raise RuntimeError("Checkpoint selection is missing dev-only provenance")
    if len({row["checkpoint"] for row in candidates}) != len(candidates):
        raise RuntimeError("Checkpoint selection includes duplicate candidates")
    saved_checkpoints = {path.resolve() for path in resolve(config["output_dir"]).glob("checkpoint-*")
                         if re.fullmatch(r"checkpoint-\d+", path.name)}
    if {resolve(row["checkpoint"]).resolve() for row in candidates} != saved_checkpoints:
        raise RuntimeError("Dev selection did not evaluate every saved checkpoint")
    selected = max(candidates, key=lambda row: (row["correct"], -row["step"]))
    if any(selection.get(key) != selected.get(key) for key in
           ("checkpoint", "step", "correct", "total", "accuracy", "artifact_sha256")):
        raise RuntimeError("Chosen checkpoint violates the frozen accuracy/tie rule")
    base, revision = resolve_model(config, decision)
    expected_ids = {f"train-{i}" for i in load_split()["dev"]}
    protocol = sha256_file("docs/frozen_manifest.json")
    for candidate in candidates:
        path = resolve(candidate["checkpoint"])
        if path.parent.resolve() != resolve(config["output_dir"]).resolve():
            raise RuntimeError("Selected checkpoint is outside this training run")
        if (candidate.get("split") != "dev" or candidate.get("total") != 500
                or candidate.get("base_model") != base or candidate.get("revision") != revision
                or checkpoint_step(path) != candidate.get("step")
                or candidate.get("eval_config_sha256") != sha256_file(config["selection"]["eval_config"])
                or candidate.get("artifact_sha256") != artifact_fingerprint(path)):
            raise RuntimeError("Checkpoint selection evidence differs from its artifacts")
        with resolve(candidate["raw_predictions"]).open(newline="", encoding="utf-8") as stream:
            predictions = list(csv.DictReader(stream))
        if (len(predictions) != 500 or {row["id"] for row in predictions} != expected_ids
                or any(row["split"] != "dev" or row["task"] != "gsm8k"
                       or row["protocol_sha256"] != protocol for row in predictions)):
            raise RuntimeError("Checkpoint selection did not use the exact frozen 500 dev IDs")
        correct = sum(parse_bool(row["correct"]) for row in predictions)
        if candidate.get("correct") != correct or candidate.get("accuracy") != correct / 500:
            raise RuntimeError("Checkpoint selection accuracy differs from its predictions")
        for row in predictions:
            raw_path = resolve(row["raw_output_path"])
            if sha256_file(raw_path) != row["raw_output_sha256"]:
                raise RuntimeError("Dev raw generation changed after checkpoint selection")
    return selection


def validate_merge(config_path, output):
    from .benchmark_common import checkpoint_manifest
    from .protocol import require_memory_decision
    from .training import resolve_model
    decision = require_memory_decision()
    config = load_config(config_path)
    selected = validate_selection(config_path)
    path = resolve(output)
    checkpoint_manifest(path)
    if not (path / "config.json").is_file() or not (path / "tokenizer_config.json").is_file():
        raise RuntimeError("Merged checkpoint config/tokenizer is missing")
    manifest = read_json(path / "merge_manifest.json")
    base, revision = resolve_model(config, decision)
    if (manifest.get("adapter") != selected["checkpoint"] or manifest.get("adapter_sha256") != selected["artifact_sha256"]
            or manifest.get("selection") != selected or manifest.get("base") != base
            or manifest.get("base_revision") != revision or manifest.get("dtype") != "bfloat16"
            or manifest.get("selected_model_id") != decision["model_id"]
            or manifest.get("selected_revision") != decision["revision"]):
        raise RuntimeError("Merged checkpoint lineage differs from dev-selected artifacts")


def validate_quantization():
    from .benchmark_common import checkpoint_manifest, evaluation_model_sha256
    from .data import load_split
    metadata = read_json("results/quantization.json")
    config = load_config("configs/quantize.yaml")
    expected_ids = [f"train-{i}" for i in load_split()["train"][:256]]
    if (metadata.get("status") != "measured" or metadata.get("config") != config
            or metadata.get("calibration_split") != "train" or metadata.get("calibration_ids") != expected_ids
            or metadata.get("protocol_sha256") != sha256_file("docs/frozen_manifest.json")):
        raise RuntimeError("Quantization evidence differs from the fixed calibration protocol")
    for name, path, identity in (("source", config["source"], "source_model_sha256"),
                                 ("compressed", config["output"], "compressed_model_sha256")):
        if (metadata.get(name) != checkpoint_manifest(path)
                or metadata.get(identity) != evaluation_model_sha256(path)):
            raise RuntimeError("Quantized/source model artifacts differ from measured weights")


def final_preflight(label, task, *, require_complete=False):
    """Never regenerate any previously committed or uncertain final output."""
    from .evaluate import EvaluationStore, EXPECTED_COUNTS, _identity, _digest
    from .protocol import require_final_eligibility
    registry = require_final_eligibility()
    spec = registry["models"][label]
    expected = _identity(label, spec["path"], spec.get("revision"), None, task, "test")
    expected["registry_sha256"] = _digest(registry)
    directory = ROOT / "results/final_eval" / f"{label}--{task}"
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        if require_complete:
            raise RuntimeError("A completed final pipeline stage has no durable evaluation ledger")
        return
    manifest = read_json(manifest_path)
    if manifest.get("identity") != expected:
        raise RuntimeError("Final evaluation identity changed after its original claim")
    store = EvaluationStore(directory, expected)
    for problem_id, digest in manifest.get("raw_sha256", {}).items():
        path = store.raw_path(problem_id)
        if not path.is_file() or sha256_file(path) != digest:
            raise RuntimeError("Previously committed final raw output is missing/changed; test regeneration is prohibited")
    # This only resolves fully committed in-flight batches; any uncertain output
    # fails closed before a subprocess could read test data or load a model.
    store.recover()
    if manifest.get("status") == "complete" or require_complete:
        if (not store.complete_raw() or len(store.manifest["items"]) != EXPECTED_COUNTS[task]):
            raise RuntimeError("Completed final evaluation has incomplete raw evidence; reruns are prohibited")
        return store.rows()


class Pipeline:
    def __init__(self, quant_python):
        self.path = ROOT / "results/pipeline_state.json"
        self.state = json.loads(self.path.read_text()) if self.path.exists() else {"version": 1, "stages": {}}
        self.quant_python = str(quant_python)

    def validate_stage(self, name):
        if name == "memory":
            validate_memory()
        elif name == "quantization_compatibility":
            evidence = read_json("results/compatibility/quantization_gpu.json")
            if (evidence.get("status") != "synthetic_gpu_kernel_check_passed"
                    or evidence.get("real_checkpoint_tested") is not False
                    or evidence.get("test_dataset_accessed") is not False
                    or evidence.get("output_tokens") != 4
                    or evidence.get("versions", {}).get("vllm") != "0.31.0"):
                raise RuntimeError("Synthetic W4A16 GPU compatibility check is incomplete")
        elif name.endswith("_train"):
            validate_training(TRAIN_CONFIGS[name.removesuffix("_train")])
        elif name.endswith("_select"):
            validate_selection(TRAIN_CONFIGS[name.removesuffix("_select")])
        elif name.endswith("_merge"):
            stage = name.removesuffix("_merge")
            validate_merge(TRAIN_CONFIGS[stage], MERGED_OUTPUTS[stage])
        elif name == "quantization":
            validate_quantization()
        elif name in {"offline_hf", "offline_vllm"}:
            from .benchmark_common import evaluation_model_sha256, checkpoint_manifest, summarize_trials, offline_speedup
            config = load_config("configs/bench.yaml")
            record = read_json(f"results/{name}.json")
            if (record.get("status") != "measured" or record.get("config") != config
                    or record.get("model_sha256") != evaluation_model_sha256(config["model"])
                    or record.get("model_weights_sha256") != checkpoint_manifest(config["model"])["weights_sha256"]
                    or record.get("protocol_sha256") != sha256_file("docs/frozen_manifest.json")):
                raise RuntimeError("Offline serving evidence does not match this final model/protocol")
            if name == "offline_hf":
                batches = record.get("batch_results", [])
                if [batch["batch_size"] for batch in batches] != [1, 8, 16, 32]:
                    raise RuntimeError("HF did not try every prescribed baseline batch size")
                if any(batch.get("status") not in {"measured", "oom"} for batch in batches):
                    raise RuntimeError("HF has an invalid baseline attempt")
                good = [batch for batch in batches if batch["status"] == "measured"]
                if not good:
                    raise RuntimeError("No HF baseline completed")
                for batch in good:
                    summarize_trials(batch["trials"])
            else:
                summarize_trials(record["trials"])
            if all(resolve(f"results/{engine}.json").exists() for engine in ("offline_hf", "offline_vllm")):
                offline_speedup(read_json("results/offline_hf.json"), read_json("results/offline_vllm.json"))
        elif name in {"online_bf16", "online_w4a16"}:
            from .report import online_rows
            from .benchmark_common import evaluation_model_sha256, checkpoint_manifest
            precision = name.removeprefix("online_")
            rows, _ = online_rows(precision)
            record = read_json(f"results/{name}.json")
            config = load_config("configs/bench.yaml")
            model = config["model"] if precision == "bf16" else load_config("configs/quantize.yaml")["output"]
            if (len(rows) != (3 if precision == "bf16" else 2) or record.get("config") != config
                    or record.get("model_sha256") != evaluation_model_sha256(model)
                    or record.get("model_weights_sha256") != checkpoint_manifest(model)["weights_sha256"]
                    or record.get("protocol_sha256") != sha256_file("docs/frozen_manifest.json")):
                raise RuntimeError("Online serving evidence does not match this model/protocol")
        elif name == "registry":
            self.validate_registry()
        elif name.startswith("final_"):
            label, task = name.removeprefix("final_").rsplit("_", 1)
            rows = final_preflight(label, task, require_complete=True)
            # The evaluator may have committed a complete task just before a
            # process died rebuilding the shared CSV. Rebuild from durable raw
            # outputs only; this performs no data access or generation.
            csv_path = ROOT / "results/per_problem.csv"
            saved = []
            if csv_path.is_file():
                with csv_path.open(newline="", encoding="utf-8") as stream:
                    saved = [row for row in csv.DictReader(stream) if row["model"] == label and row["task"] == task]
            expected = [{key: "" if value is None else str(value) for key, value in row.items()} for row in rows]
            if sorted(saved, key=lambda row: row["id"]) != sorted(expected, key=lambda row: row["id"]):
                from .evaluate import _refresh_final_csv
                _refresh_final_csv()
        elif name == "report":
            from .report import assemble, render_markdown
            summary = read_json("results/summary.json")
            actual = assemble()
            for key in actual:
                if summary.get(key) != actual[key]:
                    raise RuntimeError(f"Saved report differs from measured evidence: {key}")
            if summary.get("status") != "measured":
                raise RuntimeError("Saved final report is incomplete")
            expected = render_markdown(summary)
            if resolve("RESULTS.md").read_text(encoding="utf-8") != expected:
                raise RuntimeError("RESULTS.md differs from its measured summary")
        else:
            raise ValueError(f"Unknown pipeline stage: {name}")

    def validate_registry(self):
        from .protocol import require_final_eligibility
        registry = require_final_eligibility()
        selections = read_json(registry["selection_manifest"])
        for stage in TRAIN_CONFIGS:
            if selections.get(stage) != validate_selection(TRAIN_CONFIGS[stage]):
                raise RuntimeError("Final selection bundle differs from the selected dev checkpoints")
            if registry["models"][stage]["path"] != MERGED_OUTPUTS[stage]:
                raise RuntimeError("Final model path differs from the selected merged artifact")
            validate_merge(TRAIN_CONFIGS[stage], MERGED_OUTPUTS[stage])
        if registry["models"]["quantized"]["path"] != load_config("configs/quantize.yaml")["output"]:
            raise RuntimeError("Final quantized registry path differs from the measured checkpoint")
        validate_quantization()
        commits = subprocess.check_output(["git", "log", "--reverse", "--format=%H", "--",
                                            "configs/final_models.json", registry["selection_manifest"]],
                                           cwd=ROOT, text=True).splitlines()
        if not commits:
            raise RuntimeError("Commit the selected final artifacts before accessing test data")
        for path in ("configs/final_models.json", registry["selection_manifest"]):
            original = subprocess.check_output(["git", "show", f"{commits[0]}:{path}"], cwd=ROOT)
            if original != resolve(path).read_bytes():
                raise RuntimeError("Final model registry/selection bundle changed after its original commit")

    def stage(self, name, operation):
        record = self.state["stages"].get(name, {})
        if record.get("status") == "complete":
            self.validate_stage(name)
            print(f"Use completed stage: {name}", flush=True)
            return
        # A subprocess can finish and write its complete evidence before the
        # parent records the stage as complete. Reuse only validated results.
        metadata_path = {"quantization": "results/quantization.json",
                         "offline_hf": "results/offline_hf.json",
                         "offline_vllm": "results/offline_vllm.json",
                         "online_bf16": "results/online_bf16.json",
                         "online_w4a16": "results/online_w4a16.json"}.get(name)
        recover_complete = (metadata_path and resolve(metadata_path).is_file()
                            and read_json(metadata_path).get("status") == "measured")
        if name.startswith("final_"):
            label, task = name.removeprefix("final_").rsplit("_", 1)
            manifest = ROOT / "results/final_eval" / f"{label}--{task}" / "manifest.json"
            recover_complete = manifest.is_file() and read_json(manifest).get("status") == "complete"
        if recover_complete:
            self.validate_stage(name)
            self.state["stages"][name] = {**record, "status": "complete", "ended_at": now(),
                                           "recovered_from_complete_evidence": True}
            write_json(self.path, self.state)
            print(f"Recover completed evidence: {name}", flush=True)
            return
        self.state["stages"][name] = {"status": "running", "started_at": now(), "pid": os.getpid()}
        write_json(self.path, self.state)
        print(f"Start stage: {name}", flush=True)
        try:
            operation()
            self.validate_stage(name)
        except BaseException as error:
            self.state["stages"][name].update(status="failed", error=str(error), ended_at=now())
            write_json(self.path, self.state)
            subprocess.run([sys.executable, "-m", "src.report", "--update-readme"], cwd=ROOT)
            raise
        self.state["stages"][name].update(status="complete", ended_at=now())
        write_json(self.path, self.state)

    def run(self, name, module, *arguments, python=None):
        path = ROOT / "logs/pipeline" / f"{name}.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        command = [python or sys.executable, "-u", "-m", module, *map(str, arguments)]
        print("Run " + " ".join(command), flush=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(f"\nStarted {now()}\n")
            stream.flush()
            process = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        if process.returncode:
            tail = path.read_text(encoding="utf-8", errors="replace").splitlines()[-20:]
            print("\n".join(tail), flush=True)
            raise RuntimeError(f"{name} failed with exit {process.returncode}; inspect {path}")

    def train(self, name, config_path):
        config = load_config(config_path)
        summary = resolve(config["log_dir"]) / "training_summary.json"
        if summary.exists():
            validate_training(config_path)
            print(f"Use completed training evidence: {summary}", flush=True)
            return
        output = resolve(config["output_dir"])
        from .select_checkpoint import artifact_fingerprint
        candidates = []
        for path in output.glob("checkpoint-*"):
            if not re.fullmatch(r"checkpoint-\d+", path.name) or not (path / "trainer_state.json").is_file():
                continue
            try:
                artifact_fingerprint(path)
                checkpoint_state = read_json(path / "trainer_state.json")
                if checkpoint_state.get("global_step") != int(path.name.split("-")[-1]):
                    continue
                if not all((path / filename).is_file() for filename in ("optimizer.pt", "scheduler.pt", "rng_state.pth")):
                    continue
            except (ValueError, FileNotFoundError, json.JSONDecodeError):
                continue
            candidates.append(path)
        manifest_path = resolve(config["log_dir"]) / "run_manifest.json"
        if manifest_path.exists():
            from .protocol import require_memory_decision
            recorded = read_json(manifest_path)
            expected = resolved_training_config(config_path, require_memory_decision())
            if recorded.get("config") != expected:
                raise RuntimeError("Cannot resume a training run under a changed configuration")
        resume = sorted(candidates, key=lambda p: int(p.name.split("-")[-1]))[-1] if candidates else None
        arguments = ["--config", config_path]
        if resume:
            arguments += ["--resume", str(resume)]
        self.run(name, "src.train_sft" if name == "sft_train" else "src.train_grpo", *arguments)

    def memory(self):
        from .protocol import verify_frozen, require_memory_decision
        verify_frozen()
        decision_path = ROOT / "configs/memory_decision.json"
        if not decision_path.exists():
            attempt = "results/smoke-pipeline"
            suffix = 1
            while resolve(attempt).exists():
                attempt = f"results/smoke-pipeline-{suffix}"
                suffix += 1
            self.run("memory", "src.smoke", "--config", "configs/smoke.yaml", "--output-dir", attempt,
                     "--record-decision")
        decision = json.loads(decision_path.read_text())
        if decision.get("status") != "selected":
            raise RuntimeError("R12 did not select a feasible model")
        # Older smoke processes may have started before relative evidence paths
        # were introduced. Normalize references only before the first commit.
        committed = subprocess.run(["git", "cat-file", "-e", "HEAD:configs/memory_decision.json"],
                                   cwd=ROOT, capture_output=True).returncode == 0
        if not committed:
            evidence = resolve(decision["smoke_result"])
            portable = relative(evidence)
            if decision["smoke_result"] != portable:
                decision["smoke_result"] = portable
                write_json(decision_path, decision)
                summary_path = evidence.parent.parent / "summary.json"
                summary = read_json(summary_path)
                summary["selected"] = decision
                write_json(summary_path, summary)
                write_json(evidence.parent.parent / "proposed_memory_decision.json", decision)
        protocol_path = ROOT / "docs/protocol.md"
        marker = "Pending: no model-size decision has been made and no dev/test evaluation is permitted."
        text = protocol_path.read_text(encoding="utf-8")
        if marker in text:
            result = json.loads(resolve(decision["smoke_result"]).read_text())
            replacement = (f"Selected plan {decision['plan']}: `{decision['model_id']}@{decision['revision']}`.\n"
                           f"20 training-only steps; median step {result['median_step_seconds']:.3f}s; "
                           f"device peak {result['nvml_device_peak_used_bytes'] / 2**30:.3f}GiB.\n"
                           f"Evidence: `{decision['smoke_result']}`. No accuracy was inspected.\n")
            protocol_path.write_text(text.replace(marker, replacement), encoding="utf-8")
        evidence_root = resolve(decision["smoke_result"]).parent.parent
        related_logs = []
        for config_path in evidence_root.glob("*/resolved_config.json"):
            log_dir = read_json(config_path)["log_dir"]
            if resolve(log_dir).is_dir():
                related_logs.append(relative(log_dir))
        commit("Record measured R12 memory choice before model selection", ["configs/memory_decision.json",
               "docs/protocol.md", "results/smoke", "results/smoke-attempt2", "results/smoke-attempt3", "results/smoke-attempt4",
               relative(evidence_root), *related_logs, "src/common.py", "src/smoke.py", "src/training.py", "src/train_grpo.py",
               "configs/grpo.yaml", "configs/grpo_no_sft.yaml"])
        require_memory_decision()

    def merge(self, name, config, output):
        if (resolve(output) / "merge_manifest.json").exists():
            validate_merge(config, output)
            return
        self.run(name, "src.merge", "--config", config, "--output", output)

    def registry(self):
        from .protocol import require_memory_decision, require_final_eligibility
        decision = require_memory_decision()
        selections = {stage: json.loads(resolve(f"results/selected_{stage}.json").read_text())
                      for stage in ("sft", "grpo", "ablation")}
        write_json("results/selection.json", selections)
        registry = {"models": {"base": {"path": decision["model_id"], "revision": decision["revision"]},
                    "sft": {"path": "checkpoints/sft_merged"}, "grpo": {"path": "models/final-bf16"},
                    "ablation": {"path": "models/ablation-bf16"}, "quantized": {"path": "models/final-w4a16"}},
                    "selection_manifest": "results/selection.json"}
        target = ROOT / "configs/final_models.json"
        if target.exists() and json.loads(target.read_text()) != registry:
            raise RuntimeError("The existing final registry cannot be replaced")
        write_json(target, registry)
        require_final_eligibility()
        commit("Record selected artifacts before once-only final testing", ["configs/final_models.json", "results/selection.json"])

    def final_eval(self):
        self.validate_registry()
        registry = load_config("configs/final_models.json")
        for label, spec in registry["models"].items():
            for task in ("gsm8k", "math500"):
                arguments = ["--model", spec["path"], "--model-label", label, "--task", task, "--split", "test"]
                if spec.get("revision"):
                    arguments += ["--revision", spec["revision"]]
                final_preflight(label, task)
                self.stage(f"final_{label}_{task}", lambda label=label, task=task, arguments=arguments:
                           self.run(f"final_{label}_{task}", "src.evaluate", *arguments))

    def execute(self, through):
        self.stage("memory", self.memory)
        if through == "memory":
            return
        self.stage("quantization_compatibility", lambda:
                   self.run("quantization_compatibility", "scripts.check_quant_gpu", "--quant-python", self.quant_python))
        self.stage("sft_train", lambda: self.train("sft_train", "configs/sft.yaml"))
        self.stage("sft_select", lambda: self.run("sft_select", "src.select_checkpoint", "--config", "configs/sft.yaml", "--reuse"))
        self.stage("sft_merge", lambda: self.merge("sft_merge", "configs/sft.yaml", "checkpoints/sft_merged"))
        commit("Train completion-only LoRA SFT and select using dev", ["src/train_sft.py", "src/select_checkpoint.py",
               "src/merge.py", "configs/sft.yaml", "tests/test_training.py", "logs/sft", "results/dev/sft",
               "results/selected_sft.json", "results/pipeline_state.json"])
        if through == "sft":
            return
        for stage, config in (("grpo", "configs/grpo.yaml"), ("ablation", "configs/grpo_no_sft.yaml")):
            self.stage(f"{stage}_train", lambda stage=stage, config=config: self.train(f"{stage}_train", config))
            self.stage(f"{stage}_select", lambda stage=stage, config=config:
                       self.run(f"{stage}_select", "src.select_checkpoint", "--config", config, "--reuse"))
        self.stage("grpo_merge", lambda: self.merge("grpo_merge", "configs/grpo.yaml", "models/final-bf16"))
        self.stage("ablation_merge", lambda: self.merge("ablation_merge", "configs/grpo_no_sft.yaml", "models/ablation-bf16"))
        commit("Run matched GRPO and no-SFT ablation with dev-only selection", ["logs/grpo", "logs/grpo_no_sft",
               "results/dev/grpo", "results/dev/grpo_no_sft", "results/selected_grpo.json", "results/selected_ablation.json",
               "results/pipeline_state.json"])
        if through == "training":
            return
        self.stage("quantization", lambda: self.run("quantization", "src.quantize", python=self.quant_python))
        for name, module, arguments in (("offline_hf", "src.bench_hf", []), ("offline_vllm", "src.bench_vllm", []),
                                        ("online_bf16", "src.bench_online", ["--precision", "bf16"]),
                                        ("online_w4a16", "src.bench_online", ["--precision", "w4a16"])):
            self.stage(name, lambda name=name, module=module, arguments=arguments: self.run(name, module, *arguments))
        commit("Measure serving throughput, latency and GPTQ weight size", ["src/bench_hf.py", "src/bench_vllm.py",
               "src/bench_online.py", "src/benchmark_common.py", "src/quantize.py", "src/report.py", "configs/bench.yaml",
               "configs/quantize.yaml", "results/serve", "results/offline_hf.json", "results/offline_vllm.json",
               "results/online_bf16.json", "results/online_w4a16.json", "results/quantization.json", "logs", "scripts",
               "requirements-quant.in", "requirements-quant-lock.txt", "tests/test_serving_protocol.py", "tests/test_report.py"])
        if through == "serving":
            return
        self.stage("registry", self.registry)
        self.final_eval()
        self.stage("report", lambda: self.run("report", "src.report", "--update-readme"))
        summary = load_config("results/summary.json")
        if summary["status"] != "measured":
            raise RuntimeError("Final report is incomplete; do not claim project completion")
        commit("Report once-only final evaluation, statistics and measured resume values", ["README.md", "RESULTS.md",
               "results", "figures", "logs", "src/pipeline.py"])
        print((ROOT / "RESULTS.md").read_text(encoding="utf-8"))
        subprocess.run(["git", "log", "--oneline"], cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--through", choices=["memory", "sft", "training", "serving", "final"], default="final")
    parser.add_argument("--quant-python", default=str(Path.home() / ".venvs/llm-reasoning-grpo-quant/bin/python"))
    args = parser.parse_args()
    if os.name != "posix":
        raise RuntimeError("Run the GPU pipeline inside WSL2 Linux")
    with pipeline_lock():
        Pipeline(args.quant_python).execute(args.through)


if __name__ == "__main__":
    main()
