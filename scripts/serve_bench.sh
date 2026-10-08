#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# The wrapper manages and stops only its own server, checks vLLM 0.31.0,
# and invokes the official `vllm bench serve` CLI with the frozen workload.
python -m src.bench_online "$@"
