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

The local CPU CI replication used exactly the workflow's five primary dependencies in a fresh
environment without torch, TRL, transformers or vLLM: 104 tests passed and the optional training
integration test was skipped. This verifies the local workflow command, not a remote Actions run.
During formal SFT, its completed first-epoch adapter (`checkpoint-436`) was read through
vLLM0.31.0's native PEFT loader on CPU: rank16, 196 layers, all tensors on CPU. No answer
generation or dataset access occurred. `scripts/check_lora_adapter_cpu.py` repeats this check
and retains adapter hashes under `results/compatibility/`.

No test scores were used for these repairs. Smoke correctness rewards are retained in the
training logs but model-size selection reads only runtime success, GPU peak and median step time.

## Recovery after completed SFT

Both SFT epochs completed (872 optimizer steps). The first dev worker then failed during
vLLM engine startup, before any dev answer generation: FlashInfer could not find `nvcc`.
The existing dev ledger has no in-flight or committed generations and is retained for recovery.
The pip-installed compiler also used CUDA 13.4 with runtime headers from CUDA 13.0, which
FlashInfer's CCCL compatibility check rejects. Pin compiler, CRT and NVVM to 13.0.88,
retain the torch runtime at 13.0.96, and source `scripts/runtime_env.sh` to expose the toolkit.
`scripts/check_cuda_runtime.py` compiles and executes a synthetic sampler with no dataset
access; its evidence is recorded in `results/compatibility/cuda_sampler_gpu.json` on success.
All frozen files remain identical. The complete test suite passed 105 tests before recovery.

Recovery succeeded: both SFT checkpoints generated all 500 dev answers, with no lost
in-flight generations. Checkpoint-436 scored 369/500 (73.8%) and checkpoint-872 scored
340/500 (68.0%); the frozen rule selected checkpoint-436. The selected adapter was
merged and these measurements committed as `8ac51b4`. The formal 600-step GRPO run
started at 2026-10-08T22:30:48Z and completed its first ten optimizer steps successfully.
Its first logged loss and gradient norm were finite; no completion in that logged window
was truncated. This is startup evidence, not evidence of final improvement.

## Resume after requested pause

The previous user-requested pause stopped main GRPO after optimizer step 69; the last
durable checkpoint was step 50. `results/pause.json` remains the historical pause record.
On 2026-10-09T04:52:10Z, the full pipeline resumed from `checkpoints/grpo/checkpoint-50`.
Both adapter hashes matched that pause record, the frozen protocol verified, the selected
SFT merge passed lineage validation, and all 107 tests passed before recovery.
Optimizer, scheduler and RNG state are restored from the saved checkpoint. Unsaved updates
51–69 are discarded and those steps run again; earlier log entries are retained as evidence
of the interrupted attempt. The configured 600 optimizer steps, main/ablation parameters,
training partition, and frozen evaluation protocol are unchanged. Final test evaluation
had not started before this recovery.

The first resume attempt failed before its first optimizer update: PEFT promotes restored
LoRA weights to fp32; optimizer loading consequently promotes the saved bf16 moments,
while TRL's continuous-batching generation later casts the model back to bf16. Adam's
foreach update then rejects the mixed dtypes. Both the saved adapter and Adam moment
tensors were verified to be bf16. The project trainer now restores the generation dtype
immediately before loading optimizer state, retaining the saved moments' original dtype.
An offline PEFT/Adam regression test verifies that the next resumed update is bit-identical
to the uninterrupted update. The repair does not modify checkpoint files or hyperparameters.
The plotted training path now excludes discarded unsaved log suffixes after a rollback;
the complete original JSONL logs remain intact.

The repaired full training environment passed all 109 tests. A newly created CPU environment
using exactly the workflow's five primary dependencies passed 107 tests and skipped the two
optional training integrations; both CLI help checks also passed. This environment contains
no torch or training packages. `results/compatibility/cpu_ci_resume.json` and its JUnit XML
retain the source commit and evidence; the earlier CPU CI record is preserved separately.
This is a local workflow replication, not a remote GitHub Actions run. Main GRPO successfully
resumed and wrote complete checkpoints at steps 100, 150, 200, 250, 300 and 350.

## Run and recover

Activate `~/.venvs/llm-reasoning-grpo`, source `scripts/runtime_env.sh`, then use
`python -m src.pipeline` from the repository. The runtime setup discovers pip's bundled
CUDA toolkit, exports its compiler/library paths and limits JIT build parallelism to two jobs.
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
