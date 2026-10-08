"""Run the official vLLM 0.31.0 serving CLI and retain its detailed raw results."""
from __future__ import annotations

import argparse
from importlib.metadata import version
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import time
import urllib.request

from .benchmark_common import checkpoint_manifest, evaluation_model_sha256, rooted
from .common import ROOT, load_config, sha256_file, write_json


def validate_result(raw: dict) -> dict:
    if raw.get("completed") != 500 or raw.get("failed", 0) != 0:
        raise ValueError("Online benchmark did not complete all 500 requests")
    for key in ("input_lens", "output_lens"):
        lengths = raw.get(key, [])
        if len(lengths) != 500 or any(n != 256 for n in lengths):
            raise ValueError(f"Online benchmark requires exactly 500 x 256 {key}")
    if raw.get("total_output_tokens") != 500 * 256:
        raise ValueError("Online output count does not match the fixed-work protocol")
    required = ["output_throughput", "p50_ttft_ms", "p99_ttft_ms",
                "p50_tpot_ms", "p99_tpot_ms"]
    import math
    if any(not isinstance(raw.get(k), (float, int)) or not math.isfinite(raw[k])
           or raw[k] < 0 for k in required):
        raise ValueError("Missing or invalid online throughput/latency statistics")
    return {k: raw[k] for k in required}


def parse_weight_memory(log: str) -> float | None:
    # This log identifies loaded weights. Do not substitute process or KV-cache memory.
    matches = re.findall(r"(?:Loading model weights took|Model loading took)\s+([0-9.]+)\s+GiB", log)
    return float(matches[-1]) if matches else None


def run(config: dict, precision: str, port: int) -> dict:
    from .protocol import require_memory_decision, verify_frozen
    verify_frozen()
    require_memory_decision()
    if version("vllm") != "0.31.0":
        raise RuntimeError("The verified serving CLI is pinned to vllm==0.31.0")
    online = config["online"]
    for key, expected in {"num_prompts": 500, "input_tokens": 256, "output_tokens": 256,
                          "temperature": 0.0, "random_range_ratio": 0.0}.items():
        if online[key] != expected:
            raise ValueError(f"Online serving protocol requires {key}={expected}")
    model = rooted(config["model"] if precision == "bf16" else
                   load_config("configs/quantize.yaml")["output"])
    manifest = checkpoint_manifest(model)
    model_sha256 = evaluation_model_sha256(model)
    protocol_sha256 = sha256_file(ROOT / "docs/frozen_manifest.json")
    binary = shutil.which("vllm")
    if not binary:
        raise RuntimeError("vllm executable is absent from the current environment")
    with socket.socket() as check:
        check.bind(("127.0.0.1", port))
    folder = ROOT / "results/serve"
    folder.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    log_path = ROOT / "logs" / f"vllm-serve-{precision}-{stamp}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    server_cmd = [binary, "serve", str(model), "--host", "127.0.0.1", "--port", str(port),
        "--served-model-name", f"reasoning-{precision}", "--dtype", "bfloat16",
        "--gpu-memory-utilization", str(config["vllm_gpu_memory_utilization"]),
        "--max-model-len", str(config["vllm_max_model_len"]), "--seed", str(config["seed"]),
        "--no-enable-prefix-caching"]
    records = []
    with log_path.open("w", encoding="utf-8") as log:
        server = subprocess.Popen(server_cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                  start_new_session=(os.name != "nt"))
        try:
            deadline = time.monotonic() + 600
            while True:
                if server.poll() is not None:
                    raise RuntimeError(f"vLLM server exited; inspect {log_path}")
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                        if r.status == 200:
                            break
                except (OSError, TimeoutError):
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError(f"vLLM server readiness timed out; inspect {log_path}")
                time.sleep(1)
            concurrencies = online[f"{precision}_concurrency"]
            if concurrencies != ([1, 8, 32] if precision == "bf16" else [1, 32]):
                raise ValueError("Concurrency scan does not follow the serving protocol")
            for concurrency in concurrencies:
                filename = f"{precision}-c{concurrency}-{stamp}.json"
                command = [binary, "bench", "serve", "--model", str(model),
                    "--served-model-name", f"reasoning-{precision}", "--port", str(port),
                    "--dataset-name", "random", "--random-input-len", "256",
                    "--random-output-len", "256", "--random-range-ratio", "0.0",
                    "--num-prompts", "500", "--max-concurrency", str(concurrency),
                    "--ignore-eos", "--temperature", "0", "--seed", str(config["seed"]),
                    "--num-warmups", str(online["warmup_requests"]),
                    "--percentile-metrics", "ttft,tpot,e2el", "--metric-percentiles", "50,99",
                    "--save-result", "--save-detailed", "--result-dir", str(folder),
                    "--result-filename", filename, "--metadata", f"precision={precision}",
                    "dataset_name=random", "random_input_len=256", "random_output_len=256",
                    "random_range_ratio=0.0", "temperature=0", "ignore_eos=true",
                    "vllm_version=0.31.0", f"model_weights_sha256={manifest['weights_sha256']}",
                    f"model_sha256={model_sha256}", f"protocol_sha256={protocol_sha256}"]
                subprocess.run(command, cwd=ROOT, check=True)
                path = folder / filename
                raw = json.loads(path.read_text(encoding="utf-8"))
                metrics = validate_result(raw)
                records.append({"concurrency": concurrency, "raw_result": str(path.relative_to(ROOT)),
                                "raw_sha256": sha256_file(path), **metrics})
        finally:
            if os.name != "nt":
                # Stop this launched server's worker group without affecting
                # another user's vLLM process on the same machine.
                try:
                    os.killpg(server.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            else:
                server.terminate()
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                if os.name != "nt":
                    os.killpg(server.pid, signal.SIGKILL)
                else:
                    server.kill()
                server.wait()
    metadata = {"status": "measured", "precision": precision, "server_command": server_cmd,
        "model_manifest": manifest, "model_weights_sha256": manifest["weights_sha256"],
        "model_sha256": model_sha256, "protocol_sha256": protocol_sha256,
        "vllm_version": "0.31.0", "config": config, "results": records,
        "server_log": str(log_path.relative_to(ROOT)), "server_log_sha256": sha256_file(log_path),
        "weight_memory_gib": parse_weight_memory(log_path.read_text(encoding="utf-8"))}
    write_json(f"results/online_{precision}.json", metadata)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/bench.yaml")
    parser.add_argument("--precision", choices=["bf16", "w4a16"], default="bf16")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    print(json.dumps(run(load_config(args.config), args.precision, args.port), indent=2))


if __name__ == "__main__":
    main()
