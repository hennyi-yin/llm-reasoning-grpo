"""Sequential, monitored jobs for the independent evaluation and data pilot."""

import argparse
from datetime import datetime, timezone
import fcntl
import signal
import subprocess
import sys
import time

from scripts.run_math_pilot import gpu_memory, stop
from src.common import ROOT, write_json
from src.math_independent import REPORT, WORK


def run(directory, stages):
    state = {"status": "running", "completed": []}

    def publish(**values):
        state.update(values, updated_utc=datetime.now(timezone.utc).isoformat())
        write_json(directory / "status.json", state)

    def interrupted(signum, frame):
        raise InterruptedError(f"Interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        for name, module, args, output, watch in stages:
            if output.is_file():
                state["completed"].append(name)
                continue
            logfile = directory / f"{name}.console.log"
            publish(stage=name, log=logfile.name, gpu_peak_mib=None)
            peak, over, last = 0, None, 0
            with logfile.open("ab") as log:
                log.write(f"\n--- {state['updated_utc']} ---\n".encode())
                log.flush()
                child = subprocess.Popen([sys.executable, "-u", "-m", module, *args], cwd=ROOT,
                                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    while child.poll() is None:
                        now = time.monotonic()
                        if watch:
                            used, total = gpu_memory()
                            peak = max(used, peak)
                            over = ((now if over is None else over) if used / total >= .96 else None)
                            if over is not None and now - over >= 12:
                                raise RuntimeError(f"GPU memory guard stopped {name}: {used}/{total} MiB")
                        if now - last >= 10:
                            publish(**({"gpu_peak_mib": peak, "gpu_used_mib": used} if watch else {}))
                            last = now
                        time.sleep(1)
                finally:
                    stop(child)
            if child.returncode or not output.is_file():
                raise RuntimeError(f"{name} failed (exit {child.returncode}); see {logfile}")
            if watch:
                write_json(directory / f"{name}_memory.json", {"peak_device_memory_mib": peak})
            state["completed"].append(name)
            publish()
        publish(status="complete", stage="finished")
    except BaseException as exc:
        publish(status="failed", error=str(exc))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("independent", "self-generate", "self-compare", "serving", "multisample"))
    args = parser.parse_args()
    if args.mode == "independent":
        module = "src.math_independent"
        directory = WORK
        stages = [("prepare", module, ["prepare"], REPORT / "protocol.json", False)]
        for arm in ("base", "candidate"):
            stages.append((f"generate_{arm}", module, ["generate", "--arm", arm], WORK / f"{arm}_complete.json", True))
            stages.append((f"judge_{arm}", module, ["judge", "--arm", arm], WORK / f"judged_{arm}.json", False))
        stages.append(("compare", module, ["compare"], REPORT / "comparison.json", False))
    elif args.mode == "serving":
        from src.math_extension_serving import stages
        directory, stages = stages()
    elif args.mode == "multisample":
        from src.math_multisample import ARMS, REPORT as MULTISAMPLE_REPORT, WORK as MULTISAMPLE_WORK
        module = "src.math_multisample"
        directory = MULTISAMPLE_WORK
        stages = [("prepare", module, ["prepare"], MULTISAMPLE_REPORT / "protocol.json", False)]
        for arm in ARMS:
            stages.append((f"generate_{arm}", module, ["generate", "--arm", arm],
                           directory / f"{arm}_complete.json", True))
        for arm in ARMS:
            stages.append((f"judge_{arm}", module, ["judge", "--arm", arm],
                           directory / f"judged_{arm}.json", False))
        stages.append(("compare", module, ["compare"], MULTISAMPLE_REPORT / "comparison.json", False))
    else:
        from src.math_self_sft import stages
        directory, stages = stages(args.mode)
    directory.mkdir(parents=True, exist_ok=True)
    with (WORK.parent / "math_extension.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(directory, stages)


if __name__ == "__main__":
    main()
