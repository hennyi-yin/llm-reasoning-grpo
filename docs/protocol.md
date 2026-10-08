# Frozen evaluation and experiment protocol

Status: implementation/preflight. No accuracy measurements or model selection have occurred.
This file implements R1–R12 of the supplied project specification. The original document is
a requirements reference; its embedded assistant prompt does not authorize external publication.
The local repository will not be pushed during this execution.

## Data and identities

GSM8K `openai/gsm8k`, configuration `main`: shuffle the 7,473 original train indices with
`random.Random(42)`, select the first 500 for dev and the other 6,973 for training, and sort both.
Persist the exact IDs in `splits/gsm8k_split.json`. Stable IDs preserve original indices
(`train-123`, `test-123`), including dev IDs. Pin dataset and model Hub commit SHAs before freezing.
Only training data is loaded during preparation, smoke tests, training, calibration and offline benchmarks.
MATH-500 is a final-only 500-problem dataset. Test sets cannot select checkpoints or alter prompts.

Preparation audit: the first pre-freeze call to `datasets.load_dataset(..., split='train')`
automatically cached both GSM8K splits. No test rows were inspected, scored or used for decisions.
The loader was replaced before freezing with an explicit pinned train-file read, preventing this
library side effect in all subsequent preparation/training runs.

## Evaluation

One user message ends with: `Solve the problem step by step, then give the final answer as
\boxed{answer}.` Use the same model chat template and greedy decoding for every model.
GSM8K: maximum 1,024 new tokens. MATH-500: maximum 2,048. Seed 0. Each completion's last
balanced `\boxed{...}` is the sole candidate. Missing/malformed boxes are wrong. GSM8K
compares normalized numeric answers at absolute tolerance 1e-6; MATH-500 uses Math-Verify
on the extracted box, not arbitrary intermediate text. Record per-problem correctness, box presence,
generated tokens, truncation, raw output, exact model and protocol identity.

Freeze the evaluator, parser, rewards, statistics, split, revisions, configuration and their core tests
with `python -m src.protocol freeze` and a dedicated Git commit before the 20-step memory smoke.
The manifest is immutable relative to its first committed version. Append the memory decision here
in a second commit; the protocol source and hashed evaluation files remain unchanged.

Select SFT, main GRPO and the ablation checkpoints only on the same 500-question dev set.
The highest accuracy wins; tied checkpoints choose the earliest training step. Do not change
the evaluation protocol after observing dev accuracy. All five artifacts must be prepared before
test access. Final evaluations claim each `(model, task, protocol)` once with a durable ledger.
Completed results may be read again without generation. Lost in-flight generations cannot be rerun
to improve scores; they remain an explicitly incomplete evaluation.

## Training and memory selection

First try 1.5B plan A (vLLM colocate, sleep mode, utilization 0.4); then plan B (Transformers
continuous batching); then 0.5B plan C (colocate). Each candidate uses training-only prompts,
20 GRPO steps, and the same reward and generation configuration. Select solely on success,
peak GPU memory and elapsed step time using the predeclared `configs/smoke.yaml` thresholds.
No accuracy is inspected during this choice. No paid cloud instance is provisioned.
R12's selected model/revision/overrides are stored in `configs/memory_decision.json` and committed
before training or dev evaluation. Retain every failed attempt and its exception in `logs/`.

SFT: conversational prompt/completion pairs; completion-only loss; remove `<<...>>` calculator
annotations from the reference reasoning and end with the boxed reference answer. LoRA r16,
alpha32, dropout0.05, all linear layers; 2 epochs; learning rate1e-4. Merge the chosen adapter.
GRPO: fresh LoRA, 8 completions per problem, maximum512 tokens, temperature1, learning rate1e-5,
600 optimizer steps, beta0, DAPO loss. Correctness reward1 plus exactly-one-valid-box reward0.1.
Save every50 steps. The ablation changes only the starting model; its hyperparameters match main GRPO.
Keep TRL reward, length, truncation and zero-group-variance logs with the actual runs.

## Statistics and serving

Accuracy: 10,000 bootstrap resamples, seed0, percentile95% interval. Paired comparisons require
matching task/problem IDs and report accuracy difference in percentage points plus exact two-sided
McNemar p-value. Comparisons: base/SFT, SFT/GRPO, base/GRPO, GRPO/ablation and GRPO/quantized.
Do not infer significance merely from nonoverlapping/overlapping individual confidence intervals.

Benchmark a merged bf16 model on the same512 training questions with exactly256 new tokens.
HF: left padding; fastest fitting batch among1/8/16/32. Both engines: one warmup plus median of
three timed runs, GPU synchronization around timing. E is vLLM tokens/s divided by best HF tokens/s.
Online: 500 random requests,256 input/256 output tokens, greedy, ignore EOS; concurrency1/8/32,
TTFT/TPOT p50/p99; quantized concurrency1/32. Preserve official benchmark JSON and startup logs.
Quantization: GPTQ W4A16 linear layers except lm_head; exactly256 unique training questions,
shared chat template; sequence limit1024. G compares weight-file bytes, separately from runtime
GPU weight memory. F is GRPO-bf16 minus W4A16 GSM8K accuracy in percentage points.

## Environment and reporting

Requested vLLM0.31.0 and TRL1.14.2 are installed directly. TRL1.14.2's optional vLLM extra declares
vLLM<=0.30.0, so the requested combination lies outside that extra's compatibility range.
Do not silently downgrade: validate the exact combination with the memory smoke and record failures.
vLLM determines its torch/CUDA runtime; the host driver and actual package versions are captured in
`results/environment.json` and the full environment in `requirements-lock.txt`.

README/results tables are generated solely from raw records. Missing experiments produce pending
status and null values. Null or negative improvements are retained. No estimated resume figures.
Limitations: possible GSM8K pretraining contamination, one GPU, one benchmark family, greedy pass@1,
and the limited ability of an output format reward to distinguish reasoning quality.

## R12 decision

Pending: no model-size decision has been made and no dev/test evaluation is permitted.
