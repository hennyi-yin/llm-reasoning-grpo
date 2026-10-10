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

## Completed training and quantization recovery

Main GRPO completed all 600 steps and evaluated every saved checkpoint on dev. The
frozen rule selected checkpoint-600: 382/500 (76.4%). The matched no-SFT run also
completed 600 steps and all dev candidates; checkpoint-400 and checkpoint-450 both
scored 434/500 (86.8%), so the earlier checkpoint-400 was selected. Both selected
adapters were merged. These are dev results, not final test results.

The first GPTQ attempt calibrated all 256 training questions and wrote compressed
weights, then failed while serializing tokenizer BatchEncoding objects for provenance.
No final test generation had started. The failed model directory was preserved as
`models/final-w4a16-failed-20261009T143745Z`, and its original log as
`logs/pipeline/quantization-attempt1.log`. Calibration records now become plain
mappings with unchanged token lists, and their digest is validated before GPU work.
A regression check verifies that the provenance digest describes the exact Dataset
records. Recovery reruns calibration from the same dev-selected main model; it does
not change the source, recipe, sample IDs or frozen protocol.

Recovery passed all 110 tests in the training environment and 107 tests in the fresh
CPU workflow environment (three optional training integrations skipped). GPTQ then
completed successfully: BF16 weight files total 3,087,467,144 bytes and W4A16 weight
files total 1,142,692,360 bytes, a ratio of 2.7019233278. This is serialized weight
size, not total serving VRAM. Before serving finished, CPU-only package inventory
checks verified that all 221 main-environment packages and all 85 quantization packages
exactly match their respective lockfiles, with no missing or extra packages.
`results/compatibility/current_training_lock.json` and `current_quantization_lock.json`
record these checks without creating a CUDA context or accessing any dataset.

## User monitoring and final verification

The user requested continued supervision in this conversation and asked how to monitor
the hidden Windows/WSL runner. `scripts/watch_project.ps1` now shows the runner's live
process, current pipeline stage and recent log lines without starting or stopping work.
A one-shot invocation was checked against the running HF benchmark. Instructions are
in `docs/monitoring.md`. During formal HF timing, checks are limited to lightweight
process, NVML and file reads; full tests and table re-derivation are deferred.

`scripts.check_saved_report_cpu` provides a read-only final verification entry point:
it reconstructs tables from saved raw answers, checks their hashes and frozen judgments,
and compares the exact summary and Markdown against the completed report. Its CLI was
checked in the independent CPU workflow environment; complete-data verification remains
pending until final evaluation finishes. It does not regenerate answers or overwrite
the report. The final verification record must describe the actual completed check.

Release loading instructions in `docs/release_artifacts.md` explicitly resolve the
main GRPO adapter against the selected merged SFT model. Its adapter configuration
retains the original machine's absolute path as provenance; that configuration and
the saved checkpoint hashes are not rewritten. Ready merged artifacts can be loaded
directly after extraction into an empty directory.

## HF benchmark interruption and host restart

The original Windows runner and WSL jobs disappeared during the first measured
batch-16 trial on 2026-10-09. The cause was not established. The user subsequently
reported restarting Windows; the next inspection found WSL stopped and no project
job running. The old `windows_runner.json` still said `running`, which was stale
process state, not evidence of a live job.

The completed batch-1 and batch-8 groups are preserved in
`results/offline_hf-interrupted-20261009T2041Z.json` with SHA-256
`1980fcde0e74c87dbfd95a9534826f15948dbad01c9076949324e42e61fbdb06`.
The original log is also preserved as `logs/pipeline/offline_hf-attempt1.log`.
No final test evaluation had started. `results/interruption-20261009T2041Z.json`
records the observation before the user-reported host restart.

HF recovery now validates the exact saved model manifest, frozen protocol,
configuration, GPU and Torch version, and training-prompt token identities before
retaining complete batch groups. It checks each group's full 512-prompt warmup,
three measured trials, fixed 256-token output counts, rates, median and peak
allocation. The saved groups are copied without changing their measurements.
Batch 16 and 32 each require a fresh full warmup and three complete trials; the
unfinished batch-16 trial contributes no timing result. The timed generation code
and frozen serving settings are unchanged.

Recovery passed 125 tests in the main environment and 122 tests in the minimal
CPU environment (three optional training integrations skipped).
`results/compatibility/hf_resume_preflight.json` records the actual saved-group
and current-artifact validation before resuming. Environment metadata was refreshed
before any resumed timing; its prior version is preserved in
`results/environment-before-cuda-repair.json`.

A bounded Windows event-log inspection later found black-screen diagnostics
around 16:39–16:41 Eastern, including RDP/DWM records and LiveKernelEvent codes
1a8 and 1b8. The selected event fields are saved in
`results/compatibility/host_display_interruption_events.json`; no memory dump was
opened and no system configuration was changed. Microsoft describes
[1a8](https://learn.microsoft.com/en-us/windows-hardware/drivers/debugger/bug-check-0x1a8--video-dxgkrnl-black-screen-livedump)
and [1b8](https://learn.microsoft.com/en-us/windows-hardware/drivers/debugger/bug-check-0x1b8--video-miniport-black-screen-livedump)
as live dumps for black-screen diagnostics, including hotkey-triggered captures.
These records establish that display diagnostics occurred near the interruption;
they do not establish the cause of the disappearing project/WSL processes or a
hardware failure. Recovery therefore retains the interruption cause as unknown.

## Final dataset-loading recovery

The resumed pipeline completed all serving measurements and nine final model/task
evaluations. The quantized MATH-500 stage stopped on 2026-10-09 at 23:26 UTC when
the Hub connection became unreachable while loading the pinned dataset. Its
durable ledger contained `items: null`, no in-flight claims and no raw outputs;
the exception occurred before generator construction. The failure log and original
ledger are preserved as `logs/pipeline/final_quantized_math500-attempt1.log` and
`results/recovery/quantized-math500-manifest-before-network.json`.

On 2026-10-10 the network connection was available again. A read-only preflight
verified the unchanged frozen protocol, an empty quantized MATH-500 ledger, and
the exact question hashes, gold-answer hashes and order of all 500 reloaded items
against the four completed MATH-500 ledgers. The evidence is saved in
`results/recovery/quantized_math500_network_preflight.json`. Recovery continues
only this not-yet-generated group; the nine completed groups retain their original
answers. No frozen source, model registry, dataset revision or evaluation setting
was changed. Before continuation, 125 main-environment tests and 122 minimal CPU
tests passed, with three optional integrations skipped in the CPU environment.

The report now renders the quantization McNemar probability with six significant
digits, matching its comparison table, so a measured small positive probability
is not printed as `0.000000`. This presentation change does not change statistics
or any frozen evaluator behavior.

## Final measured report and publication preparation

On 2026-10-10 the retained nine model/task ledgers passed their recovery checks.
The remaining quantized MATH-500 group then generated its first 500 answers;
its ledger completed with no in-flight claims. The final CSV contains 9,095
unique model/task/problem records, and the generated summary status is `measured`.
Quantized MATH-500 accuracy is 111/500 (22.20%), with CI [18.60%, 25.80%].
The original frozen manifest and final model registry remain unchanged.

The user explicitly requested GitHub repository creation during this continuation.
The existing authenticated account is `hennyi-yin`; the public repository is
https://github.com/hennyi-yin/llm-reasoning-grpo and is pinned on that profile.
Publishing is separate from local experiment execution and archive generation.
The release publisher requires a complete report, matching CPU reproduction proof,
a pushed immutable tag, and successful CPU CI; it uploads without overwriting
existing assets and verifies each remote SHA256 before publishing the draft.
The original local archive manifest is preserved, while public upload receipts
are recorded separately in `results/publication.json`.
