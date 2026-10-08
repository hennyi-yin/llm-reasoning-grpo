"""Sequential training-only R12 smoke trials, with NVML process-tree and GPU peaks."""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone

from .common import ROOT, load_config, write_json
from .training import assert_frozen, package_versions, project_path


class NvmlMonitor:
    """Include vLLM worker processes and non-PyTorch allocations in GPU measurements."""
    def __init__(self, gpu_index: int, interval: float):
        import pynvml
        self.nvml = pynvml
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        self.capacity = int(info.total)
        self.baseline_used = int(info.used)
        self.gpu_name = pynvml.nvmlDeviceGetName(self.handle)
        if isinstance(self.gpu_name, bytes):
            self.gpu_name = self.gpu_name.decode()
        self.interval = interval
        self.pid = None
        self.device_peak = int(info.used)
        self.process_peak = None
        self.process_ids: set[int] = set()
        self.observed_other_process_ids: set[int] = set()
        self.process_query_errors: set[str] = set()
        self.samples = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self, pid: int):
        self.pid = pid
        self._thread.start()

    def _sample(self):
        import psutil
        pids = {self.pid}
        try:
            pids.update(item.pid for item in psutil.Process(self.pid).children(recursive=True))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        self.process_ids.update(pids)
        info = self.nvml.nvmlDeviceGetMemoryInfo(self.handle)
        self.device_peak = max(self.device_peak, int(info.used))
        self.samples += 1
        by_pid = {}
        for query in (self.nvml.nvmlDeviceGetComputeRunningProcesses,
                      self.nvml.nvmlDeviceGetGraphicsRunningProcesses):
            try:
                for item in query(self.handle):
                    memory = getattr(item, "usedGpuMemory", None)
                    if memory is not None and 0 <= memory < 2**63:
                        by_pid[item.pid] = max(by_pid.get(item.pid, 0), int(memory))
                    if item.pid not in pids:
                        self.observed_other_process_ids.add(item.pid)
            except self.nvml.NVMLError as exc:
                self.process_query_errors.add(str(exc))
        if any(pid in by_pid for pid in pids):
            total = sum(value for pid, value in by_pid.items() if pid in pids)
            self.process_peak = max(self.process_peak or 0, total)

    def _loop(self):
        while not self._stop.is_set():
            try:
                self._sample()
            except self.nvml.NVMLError as exc:
                self.process_query_errors.add(str(exc))
            self._stop.wait(self.interval)

    def stop(self) -> dict:
        self._stop.set()
        self._thread.join(timeout=max(2.0, self.interval * 2))
        result = {
            "gpu": self.gpu_name, "nvml_gpu_capacity_bytes": self.capacity,
            "nvml_device_baseline_used_bytes": self.baseline_used,
            "nvml_device_peak_used_bytes": self.device_peak,
            "nvml_process_tree_peak_used_bytes": self.process_peak,
            "nvml_process_tree_ids": sorted(self.process_ids),
            "nvml_other_process_ids": sorted(self.observed_other_process_ids),
            "nvml_process_query_errors": sorted(self.process_query_errors),
            "nvml_samples": self.samples,
            "memory_selection_metric": "nvml_device_peak_used_bytes",
            "memory_measurement_note": "Device total includes display/other processes; process-tree peak includes vLLM workers where NVML exposes per-process memory.",
        }
        self.nvml.nvmlShutdown()
        return result


def _kill_process_tree(process: subprocess.Popen):
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=15)
        except ProcessLookupError:
            return
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
    else:
        import psutil
        try:
            children = psutil.Process(process.pid).children(recursive=True)
        except psutil.NoSuchProcess:
            children = []
        for child in reversed(children):
            child.kill()
        process.kill()
        process.wait()


def run_trial(settings: dict, candidate: dict, output_dir) -> dict:
    plan = candidate["plan"]
    trial_dir = output_dir / plan
    trial_dir.mkdir(parents=True, exist_ok=False)
    config = copy.deepcopy(load_config(project_path(settings["grpo_config"])))
    config["output_dir"] = str(project_path(settings["checkpoint_dir"]) / output_dir.name / plan)
    config["log_dir"] = str(project_path(settings["log_dir"]) / output_dir.name / plan)
    config_file = trial_dir / "resolved_config.json"
    candidate_file = trial_dir / "candidate.json"
    write_json(config_file, config)
    write_json(candidate_file, candidate)
    monitor = NvmlMonitor(settings["gpu_index"], settings["poll_interval_seconds"])
    start = time.perf_counter()
    timeout = False
    with (trial_dir / "console.log").open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(
            [sys.executable, "-u", "-m", "src.train_grpo", "--config", str(config_file),
             "--smoke-candidate", str(candidate_file)], cwd=ROOT, stdout=stream,
            stderr=subprocess.STDOUT, start_new_session=os.name == "posix",
        )
        monitor.start(process.pid)
        try:
            process.wait(timeout=settings["process_timeout_seconds"])
        except subprocess.TimeoutExpired:
            timeout = True
            _kill_process_tree(process)
        finally:
            measurements = monitor.stop()
    console = (trial_dir / "console.log").read_text(encoding="utf-8", errors="replace")
    if timeout:
        status = "timed_out"
    elif process.returncode:
        status = "oom" if re.search(r"(?:CUDA|GPU|torch\.OutOfMemoryError).*out.of.memory|CUDA out of memory|No available memory for the cache blocks|free memory.*less than desired", console, re.I) else "runtime_error"
    else:
        summary_path = project_path(config["log_dir"]) / "training_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        measurements.update(summary)
        if summary["steps_completed"] != 20:
            raise RuntimeError(f"Plan {plan} did not complete all 20 smoke optimizer steps.")
        status = "completed"
        if summary["median_step_seconds"] > settings["max_median_step_seconds"]:
            status = "too_slow"
        elif measurements["nvml_device_peak_used_bytes"] > settings["max_gpu_memory_fraction"] * measurements["nvml_gpu_capacity_bytes"]:
            status = "insufficient_headroom"
    record = {**candidate, **measurements, "status": status, "returncode": process.returncode,
              "elapsed_seconds": time.perf_counter() - start, "evaluated_accuracy": False,
              "timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "package_versions": package_versions(), "console_log": str(trial_dir / "console.log"),
              "selection_thresholds": {key: settings[key] for key in
                  ("max_median_step_seconds", "max_gpu_memory_fraction", "process_timeout_seconds")}}
    write_json(trial_dir / "measurements.json", record)
    return record


def run(config_path: str, *, output_dir: str | None = None, record_decision: bool = False) -> dict:
    assert_frozen()
    settings = load_config(project_path(config_path))
    if settings["max_steps"] != 20:
        raise ValueError("R12 requires exactly 20 smoke steps per trial.")
    decision_path = project_path("configs/memory_decision.json")
    if decision_path.exists():
        raise RuntimeError("An R12 decision already exists; model choice cannot be changed after evaluation.")
    if any(project_path("results/final_eval").glob("**/manifest.json")) or any(project_path("results/dev").glob("**/*.csv")):
        raise RuntimeError("R12 must be decided before any dev or test evaluation.")
    directory = project_path(output_dir or settings["output_dir"])
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"Preserve prior smoke evidence; choose a new --output-dir: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    models = load_config(project_path(settings["models_config"]))
    history = []
    selected = None
    for entry in settings["plans"]:
        candidate = {"plan": entry["plan"], **models[entry["model_key"]],
                     "grpo_overrides": entry["grpo_overrides"]}
        if not re.fullmatch(r"[0-9a-f]{40}", candidate.get("revision", "")):
            raise ValueError("Every smoke candidate requires a pinned model revision.")
        print(f"R12 plan {entry['plan']}: {candidate['model_id']}, 20 training-only steps", flush=True)
        result = run_trial(settings, candidate, directory)
        history.append(result)
        write_json(directory / "summary.json", {"trials": history, "selected": selected})
        print(f"Plan {entry['plan']}: {result['status']}; median step {result.get('median_step_seconds')} s; NVML peak {result['nvml_device_peak_used_bytes'] / 2**30:.2f} GiB", flush=True)
        if result["status"] == "completed":
            selected = {"status": "selected", "plan": candidate["plan"],
                        "model_id": candidate["model_id"], "revision": candidate["revision"],
                        "grpo_overrides": candidate["grpo_overrides"],
                        "smoke_result": (directory / candidate["plan"] / "measurements.json").relative_to(ROOT).as_posix(),
                        "decided_before_evaluation": True,
                        "selection_rule": "first_plan_meeting_predeclared_memory_and_step_time_limits",
                        "decided_at_utc": datetime.now(timezone.utc).isoformat()}
            break
        if result["status"] == "runtime_error":
            raise RuntimeError(f"Plan {candidate['plan']} failed for a non-memory reason. Diagnose {directory / candidate['plan'] / 'console.log'} before deciding R12.")
    summary = {"trials": history, "selected": selected, "evaluated_accuracy": False}
    write_json(directory / "summary.json", summary)
    if selected:
        write_json(directory / "proposed_memory_decision.json", selected)
        if record_decision:
            write_json(decision_path, selected)
            print("R12 decision recorded. Commit configs/memory_decision.json and docs/protocol.md before training or evaluation.")
    else:
        raise RuntimeError("No A/B/C plan met the frozen memory/time thresholds; the local experiment needs additional GPU capacity.")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--output-dir")
    parser.add_argument("--record-decision", action="store_true")
    args = parser.parse_args()
    run(args.config, output_dir=args.output_dir, record_decision=args.record_decision)


if __name__ == "__main__":
    main()
