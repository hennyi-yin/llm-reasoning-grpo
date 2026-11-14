"""Gated streaming latency and throughput comparison on real MATH prompts."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import random
import signal
import socket
import subprocess
import time
from urllib.request import Request, urlopen

from scripts.run_math_pilot import stop
from .common import load_config, sha256_file, write_json
from .math_independent import ADAPTER, LABELS, REPORT, WORK as EVAL_WORK, adapter_hashes
from .math_pilot import local_base
from .prompts import build_messages

WORK = EVAL_WORK.parent / "math_extension_serving"


def stages():
    if not load_config(REPORT / "comparison.json")["serving_gate"]:
        raise RuntimeError("Independent results do not authorize the serving comparison")
    module = "src.math_extension_serving"
    return WORK, [(f"serve_{a}", module, ["bench", "--arm", a], WORK / f"{a}.json", True) for a in LABELS] + [
        ("serve_compare", module, ["compare"], REPORT / "serving.json", False)]


def request(url, label, q, g):
    body = json.dumps({"model": label, "messages": build_messages(q["question"]), "stream": True,
                       "stream_options": {"include_usage": True}, "max_tokens": g["max_tokens"],
                       "temperature": 0, "seed": g["seed"], "top_p": 1}).encode()
    start = time.perf_counter()
    first, usage, finish = None, None, None
    with urlopen(Request(url + "/v1/chat/completions", data=body,
                         headers={"Content-Type": "application/json"}), timeout=240) as response:
        for line in response:
            if not line.startswith(b"data: "):
                continue
            data = line[6:].strip()
            if data == b"[DONE]":
                break
            event = json.loads(data)
            for choice in event.get("choices", []):
                if first is None and choice.get("delta", {}).get("content"):
                    first = time.perf_counter()
                finish = choice.get("finish_reason") or finish
            if event.get("usage"):
                usage = event["usage"]
    if first is None or usage is None or finish not in {"stop", "length"}:
        raise RuntimeError(f"Incomplete measured response: {q['id']}")
    return {"id": q["id"], "e2e_ms": (time.perf_counter()-start)*1000, "ttft_ms": (first-start)*1000,
            "tokens": usage["completion_tokens"], "finish_reason": finish}


def bench(arm):
    if not load_config(REPORT / "comparison.json")["serving_gate"]:
        raise RuntimeError("Serving gate failed")
    plan = load_config(REPORT / "protocol.json")
    if plan["adapter"] != adapter_hashes() or plan["items_sha256"] != sha256_file(EVAL_WORK / "items.json"):
        raise ValueError("Serving inputs differ from the test evaluation")
    g = plan["policy"]["generation"]
    rows = load_config(EVAL_WORK / "items.json")["rows"]
    chosen = random.Random(142).sample(rows, 36)
    warm, prompts = chosen[:4], chosen[4:]
    base, _ = local_base()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    command = ["vllm", "serve", base, "--host", "127.0.0.1", "--port", str(port), "--served-model-name", "base",
               "--dtype", "bfloat16", "--seed", str(g["seed"]), "--max-model-len", str(g["max_model_len"]),
               "--max-num-seqs", str(g["max_num_seqs"]), "--gpu-memory-utilization", str(g["gpu_memory_utilization"]),
               "--enforce-eager", "--no-enable-flashinfer-autotune", "--enable-lora", "--max-lora-rank", "16",
               "--no-enable-prefix-caching"]
    if arm == "candidate":
        command += ["--lora-modules", f"candidate={ADAPTER}"]
    results = {}
    def interrupted(signum, _frame):
        raise InterruptedError(f"Serving benchmark stopped by signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    with (WORK / f"{arm}_server.log").open("ab") as log:
        p = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            deadline = time.monotonic() + 150
            while True:
                if p.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError(f"Serving endpoint did not start; see {arm}_server.log")
                try:
                    with urlopen(url + "/health", timeout=2) as response:
                        if response.status == 200:
                            break
                except OSError:
                    time.sleep(1)
            for q in warm:
                request(url, arm, q, g)
            for c in (1, 8):
                start = time.perf_counter()
                with ThreadPoolExecutor(max_workers=c) as pool:
                    measured = list(pool.map(lambda q: request(url, arm, q, g), prompts))
                results[str(c)] = {"elapsed_s": time.perf_counter()-start, "requests": measured}
                print(f"serving {arm} concurrency {c}: {len(measured)} requests complete", flush=True)
        finally:
            stop(p)
    write_json(WORK / f"{arm}.json", results)


def compare():
    import numpy as np

    output = {a: load_config(WORK / f"{a}.json") for a in LABELS}
    summary = {}
    for c in ("1", "8"):
        reference_ids = [x["id"] for x in output["base"][c]["requests"]]
        summary[c] = {}
        for a in LABELS:
            data = output[a][c]
            rows = data["requests"]
            if [x["id"] for x in rows] != reference_ids or len(rows) != 32:
                raise ValueError("Serving workload differs")
            summary[c][a] = {
                "e2e_ms_p50_p95": np.percentile([x["e2e_ms"] for x in rows], [50, 95]).tolist(),
                "ttft_ms_p50_p95": np.percentile([x["ttft_ms"] for x in rows], [50, 95]).tolist(),
                "requests_per_second": len(rows)/data["elapsed_s"],
                "output_tokens_per_second": sum(x["tokens"] for x in rows)/data["elapsed_s"],
                "mean_output_tokens": sum(x["tokens"] for x in rows)/len(rows),
                "truncated": sum(x["finish_reason"] == "length" for x in rows),
            }
    write_json(REPORT / "serving.json", {"summary": summary, "measurements": output})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("bench", "compare"))
    parser.add_argument("--arm", choices=LABELS, default="base")
    args = parser.parse_args()
    if args.stage == "bench":
        bench(args.arm)
    else:
        compare()
