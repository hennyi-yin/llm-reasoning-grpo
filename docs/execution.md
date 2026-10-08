# Execution record

The pipeline runs inside WSL2 Ubuntu24.04 on the actual RTX5070. It must finish its GPU
experiments before this repository can claim accuracy/speedup results.

Preparation froze dataset/model revisions and split IDs. `170fa2b` is the dedicated evaluation
freeze commit. Every later runtime repair leaves the frozen evaluator, parser, reward,
statistics and `configs/eval.yaml` byte-identical.

Startup repairs before any parameter update or accuracy evaluation:

1. Smoke attempt1: JSON scientific notation was parsed through YAML as text. The common
   JSON loader now uses `json.load`; evidence remains in `results/smoke/`.
2. Smoke attempt2: Transformers5.17 removed `logging_dir` and `save_safetensors` constructor
   fields. Training uses its own JSONL log callback and current supported arguments.
   Evidence remains in `results/smoke-attempt2/`.
3. Smoke attempt3: Triton compilation required a C compiler. Installed Ubuntu
   `build-essential`, `python3-dev`, `pkg-config`; evidence remains in `results/smoke-attempt3/`.
4. Smoke attempt4: uses identical R12 selection thresholds and model candidates after those repairs.
   PlanA completed20steps (median13.408s), but device peak12,755,021,824bytes exceeded
   the predeclared98%headroom threshold. PlanB was interrupted before a decision to apply
   a memory-only microbatch repair consistently to both GRPO variants.
5. Smoke attempt5: reduce GRPO microbatch8→2 and increase accumulation4→16. Generation batch
   remains32, group size8, learning rate/rewards/sampling/600steps unchanged. The original
   frozen A/B/C candidates and selection thresholds remain unchanged. No dev/test score
   has been inspected; both main and ablation use the same updated microbatch.

Attempt5 selected plan B after A failed the unchanged headroom threshold. B completed all
20 steps, with median 64.913 seconds and device peak 12,131,897,344 bytes (11.299 GiB,
94.626% of detected capacity). The model remains Qwen2.5-1.5B-Instruct. The R12 decision
and successful evidence were committed as `d4a8a47` before any dev selection.

Before SFT, vLLM generated four tokens from the synthetic W4A16 checkpoint on the RTX5070
(compute capability 12.0). `results/compatibility/quantization_gpu.json` records that check.
It establishes that this quantization kernel can execute on the device; it does not measure
the real project's accuracy, throughput, or weight-size reduction.

The local CPU CI replication used exactly the workflow's six primary dependencies in a fresh
environment without torch, TRL, transformers or vLLM: 104 tests passed and the optional training
integration test was skipped. This verifies the local workflow command, not a remote Actions run.
During formal SFT, its completed first-epoch adapter (`checkpoint-436`) was read through
vLLM0.31.0's native PEFT loader on CPU: rank16, 196 layers, all tensors on CPU. No answer
generation or dataset access occurred. `scripts/check_lora_adapter_cpu.py` repeats this check
and retains adapter hashes under `results/compatibility/`.

No test scores were used for these repairs. Smoke correctness rewards are retained in the
training logs but model-size selection reads only runtime success, GPU peak and median step time.

## Run and recover

Activate `~/.venvs/llm-reasoning-grpo`, then use `python -m src.pipeline` from the repository.
The quantization interpreter defaults to `~/.venvs/llm-reasoning-grpo-quant/bin/python`.
On Windows, `scripts/run_project.ps1` starts this WSL pipeline and holds a temporary
system-awake request while it runs, then releases it. It does not alter power settings
or keep the display on. Host status is recorded in `logs/windows_runner.json`.
The pipeline serializes GPU jobs, commits the measured R12 decision before dev access,
selects each adapter on dev, merges selected models, runs serving/calibration, then claims the
five final models on both test tasks once. It never pushes to a remote.

`results/pipeline_state.json` and `logs/pipeline/` record actual stages. Completed final
outputs are read without generation. Keep every final ledger and raw output; deleting them
does not authorize another test run. A crash with uncertain in-flight generations stops
the evaluation instead of resampling those problems. Use the guarded pipeline as the
supported final evaluation/recovery entry point.

Inspect `RESULTS.md` and `results/summary.json` for evidence-derived values. A pending/null
entry is an unmeasured result, not a zero score. Public repository publication, pinning and
model Release uploads have not occurred during this local implementation.
