"""Run the pilot sequentially with durable status and a GPU memory guard."""

import argparse
from datetime import datetime, timezone
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from src.common import load_config, write_json, ROOT
from src.math_pilot_data import CONFIG, OUT, WORK


def gpu_memory():
    result = subprocess.check_output(
        ["nvidia-smi", "--id=0", "--query-gpu=memory.used,memory.total",
         "--format=csv,noheader,nounits"], text=True, timeout=5)
    used, total = (int(x.strip()) for x in result.strip().split(","))
    return used, total


def stop(proc):
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()


def run(resume=None):
    state_path = OUT / "math_status.json"
    cfg = load_config(CONFIG)["training"]
    if (OUT / "math_dev_comparison.json").is_file() and not (WORK / "dev_pilot.json").is_file():
        raise FileExistsError("Saved report without local generations. Use a separate LLM_MATH_PILOT_OUT.")
    state = {"status": "running", "pid": os.getpid(), "stage": "startup", "completed": []}

    def publish(**kwargs):
        state.update(kwargs, updated_utc=datetime.now(timezone.utc).isoformat())
        write_json(state_path, state)

    def stage(name, *args, watch=False):
        log = WORK / f"{name}.console.log"
        publish(stage=name, log=str(log.relative_to(ROOT) if log.is_relative_to(ROOT) else log))
        over_since, last_status, peak = None, 0, 0
        with log.open("ab") as output:
            output.write(f"\n--- {state['updated_utc']} ---\n".encode())
            output.flush()
            p = subprocess.Popen([sys.executable, "-u", "-m", "src.math_pilot", *args],
                                 cwd=ROOT, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                while p.poll() is None:
                    if watch:
                        used, total = gpu_memory()
                        peak = max(used, peak)
                        now = time.monotonic()
                        if now - last_status >= 5:
                            publish(gpu_used_mib=used, gpu_total_mib=total, gpu_peak_mib=peak)
                            last_status = now
                        if used / total >= cfg["max_gpu_used_fraction"]:
                            over_since = over_since if over_since is not None else now
                            if now - over_since >= cfg["memory_grace_seconds"]:
                                raise RuntimeError(f"GPU memory guard stopped training: {used}/{total} MiB")
                        else:
                            over_since = None
                    time.sleep(1)
            finally:
                stop(p)
        if p.returncode:
            raise RuntimeError(f"{name} exited with {p.returncode}; see {log}")
        state["completed"].append(name)
        publish()

    def interrupted(signum, frame):
        raise InterruptedError(f"Interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        if not (OUT / "math_split.json").exists() or not all(
                (WORK / f"math_{split}.json").is_file() for split in ("screen", "dev", "confirmation")):
            stage("prepare", "prepare")
        if not (OUT / "math_screen.json").exists():
            stage("screen", "screen")
        if load_config(OUT / "math_screen.json")["status"] != "ready":
            publish(status="gate_failed", stage="screen")
            return
        if not (WORK / "dev_base.json").exists():
            stage("dev_base", "eval", "--label", "base")
        if not (OUT / "math_train_summary.json").exists():
            stage("train", "train", *(("--resume", resume) if resume else ()), watch=True)
        if not (WORK / "dev_pilot.json").exists():
            stage("dev_pilot", "eval", "--label", "pilot")
        stage("compare_dev", "compare")
        if load_config(OUT / "math_dev_comparison.json")["decision"] == "confirm":
            for label in ("base", "pilot"):
                if not (WORK / f"confirmation_{label}.json").exists():
                    stage(f"confirmation_{label}", "eval", "--split", "confirmation", "--label", label)
            stage("compare_confirmation", "compare", "--split", "confirmation")
        publish(status="complete", stage="finished")
    except BaseException as exc:
        publish(status="failed", error=str(exc))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", type=Path, help="Explicit WSL path to a pilot checkpoint")
    args = parser.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    with (WORK / "runner.lock").open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(str(args.resume.resolve()) if args.resume else None)
