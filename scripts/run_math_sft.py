"""Run the matched SFT experiment without opening the confirmation set early."""

import argparse
import fcntl
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

from scripts.run_math_pilot import gpu_memory, stop
from src.common import ROOT, load_config, write_json
from src.math_sft import ARMS, CONFIG, OUT


def run(probe=False):
    cfg = load_config(CONFIG)
    state = {"status": "running", "completed": []}

    def publish(**values):
        state.update(values, updated_utc=datetime.now(timezone.utc).isoformat())
        write_json(OUT / ("probe_status.json" if probe else "status.json"), state)

    def stage(name, args, output, *, gpu=False):
        if output.is_file():
            state["completed"].append(name)
            return
        logfile = OUT / f"{name}.console.log"
        publish(stage=name, log=logfile.name, gpu_peak_mib=None)
        peak, started, over, last = 0, time.monotonic(), None, 0
        with logfile.open("ab") as log:
            log.write((f"\n--- {state['updated_utc']} ---\n").encode())
            log.flush()
            p = subprocess.Popen([sys.executable, "-u", "-m", "src.math_sft", *args],
                                 cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                while p.poll() is None:
                    now = time.monotonic()
                    if now - started > 1800:
                        raise TimeoutError(f"{name} exceeded its 30-minute stage budget")
                    if gpu:
                        used, total = gpu_memory()
                        peak = max(peak, used)
                        if now - last >= 5:
                            publish(gpu_peak_mib=peak, gpu_used_mib=used, gpu_total_mib=total)
                            last = now
                        if used / total >= cfg["max_gpu_used_fraction"]:
                            over = now if over is None else over
                            if now - over >= cfg["memory_grace_seconds"]:
                                raise RuntimeError(f"GPU memory guard: {used}/{total} MiB")
                        else:
                            over = None
                    time.sleep(1)
            finally:
                stop(p)
        if p.returncode or not output.is_file():
            raise RuntimeError(f"{name} did not complete; see {logfile}")
        state["completed"].append(name)
        if gpu:
            write_json(OUT / f"{name}_memory.json", {"peak_device_memory_mib_sampled": peak})
        publish()

    def interrupted(signum, _frame):
        raise InterruptedError(f"Interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        if probe:
            stage("dev_scale_probe", ["probe-scale"], OUT / "dev_scale_probe.json", gpu=True)
            publish(status="complete", stage="finished", decision="exploratory_only")
            return
        stage("dev_base", ["eval"], OUT / "dev_base.json")
        for label in ("short", "worked"):
            checkpoints = sorted((OUT / label).glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
            resume = next((p for p in reversed(checkpoints) if all((p / f).is_file() for f in
                          ("trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth",
                           "adapter_model.safetensors"))), None)
            args = ["train", "--arm", label]
            if resume:
                args.extend(["--resume", str(resume)])
            stage(f"train_{label}", args, OUT / f"{label}_train.json", gpu=True)
            stage(f"dev_{label}", ["eval", "--arm", label], OUT / f"dev_{label}.json", gpu=True)
        stage("compare_dev", ["compare"], OUT / "dev_comparison.json")
        decision = load_config(OUT / "dev_comparison.json")["decision"]
        if decision == "confirm":
            for label in ARMS:
                stage(f"confirmation_{label}", ["eval", "--split", "confirmation", "--arm", label],
                      OUT / f"confirmation_{label}.json", gpu=True)
            stage("compare_confirmation", ["compare", "--split", "confirmation"], OUT / "confirmation_comparison.json")
            decision = load_config(OUT / "confirmation_comparison.json")["decision"]
        publish(status="complete", stage="finished", decision=decision)
    except BaseException as exc:
        publish(status="failed", error=str(exc))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-scale", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "runner.lock").open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args.probe_scale)
