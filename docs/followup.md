# Follow-up: answer format and small MATH GRPO

This is a separate exploratory protocol. It does not overwrite v1's ledgers,
evaluator, public model artifacts, or published benchmark numbers.

## Saved GSM8K answers

The original no-SFT comparison used a mandatory boxed final answer. The
follow-up verifies each raw file against the v1 ledger and applies a shared
final-answer rule to the same 1,319 pairs. It accepts a boxed answer, an explicit
final statement, or a standalone terminal result. It never searches the
derivation for a value matching the reference. Ambiguous records were reviewed
from a completion-only queue, with model labels and reference answers hidden.
The recorded choices are single-reviewer extractions of what the model stated;
`null` means no unambiguous completed answer was available.

Run the regrade from the repo:

```bash
python -m scripts.regrade_gsm8k --overrides results/followup/gsm8k/manual_answers.json
```

The summary is `results/followup/gsm8k/summary.json`. The script regenerates
per-response decisions and the review queue locally. Pair differences and
intervals are descriptive because this is a post-hoc regrade.

| Scoring | Base | GRPO without SFT | Paired gain (95% bootstrap interval) |
|---|---:|---:|---:|
| v1 strict boxed | 910/1,319 (68.99%) | 981/1,319 (74.37%) | +5.38 pp [3.11, 7.73] |
| Shared final-answer extraction + review | 977/1,319 (74.07%) | 981/1,319 (74.37%) | +0.30 pp [−1.82, 2.35] |

The second row used 117 recorded review decisions. There were 101 gains and
97 losses after regrading (paired exact McNemar p = 0.831, descriptive).
The regrade recovered 67 correct answers for the base model and changed none
of the GRPO judgments. Most of the original measured gain does not survive
this scoring change.
The remaining four-answer difference is inconclusive; this comparison does
not grade the reasoning steps or identify a causal mechanism.

## MATH pilot

- Start from the same pinned Qwen2.5-1.5B-Instruct checkpoint, without SFT.
- Use only the seven training parquet files from
  [EleutherAI/hendrycks_math](https://huggingface.co/datasets/EleutherAI/hendrycks_math)
  at the revision in `configs/math_pilot.yaml`. Deduplicate by normalized
  question text and compare fingerprints against v1 MATH-500. This is an
  exact-text check, not a guarantee against paraphrases or pretraining exposure.
- Split before inference, stratified by subject: 128 screening questions, 196
  development questions, and 350 confirmation questions. All are from the
  public training split; the latter two are held out from pilot training.
- Sample eight completions per screening question. Require a resolved correct
  and wrong answer to select a prompt; record all-correct, all-wrong, mixed,
  unresolved, and truncated outcomes. This is a fixed screen with the base
  model, not online resampling. The binary correctness reward has no format
  bonus; truncated training completions are masked by the trainer.
- Run 50 optimizer steps with checkpoints every 10 steps. Evaluate the fixed
  final adapter and base on identical dev questions, prompts, greedy sampling,
  and token budgets. Report both permissive and strict boxed scoring, unresolved
  counts, truncation, and paired changes. Only evaluate on the reserved
  confirmation set if dev improves by at least 2 pp, neither unresolved rate
  exceeds 25%, and the unresolved-rate difference is no larger than 3 pp.
  That gate is a budget decision, not a significance test. The confirmation
  result is considered supported only if those checks hold on that set as well
  and the paired 95% bootstrap interval is strictly positive.
- Long-answer SFT remains a separate experiment. A correct numeric answer alone
  does not establish that a generated derivation is suitable for SFT.

The initial screen produced 1,024 completions: 23 groups were all correct, 31
all wrong, and 74 mixed. Of those, 72 had a resolved correct and wrong answer
without relying on clipped samples. The completion truncation rate was 7.42%.
All 50 optimizer steps completed on the RTX 5070, with a median of 57.8 seconds
per step and a sampled peak device memory of 10,290 MiB. At least one group
had reward variation in 48 of the 50 logged steps.

| Dev metric | Base | 50-step pilot | Paired gain (95% bootstrap interval) |
|---|---:|---:|---:|
| Shared final-answer scoring | 119/196 (60.71%) | 119/196 (60.71%) | 0.00 pp [−3.06, 3.06] |
| Strict boxed scoring | 119/196 (60.71%) | 120/196 (61.22%) | +0.51 pp [−2.55, 4.08] |
| Unresolved answers | 19 | 17 | — |
| Truncated responses | 18 | 16 | — |

Under the primary scoring rule, five answers became correct and five became
incorrect. The pilot did not meet its +2 pp gate, so the 350-question
confirmation set was not evaluated and training was not extended. This is a
single-seed, small-data pilot with no measured net gain; it does not establish
that MATH training or GRPO cannot help.

We reviewed the two unresolved outputs that were not truncated. The base
output stated the correct numeric answer in prose, which the extractor missed.
The pilot output had a boxed symbolic answer matching the reference, which the
strict judge accepted but the conservative parser did not resolve. Counting
both as correct in this separate sensitivity check yields 120/196 for each
model. This does not alter the original pilot scores or the stop decision.

To recompute the aggregate from the
[196 paired judgments](../results/followup/math_dev_paired.json) on a CPU:

```bash
python -m scripts.verify_math_pilot
```

Use `--local-raw` if the original local raw generations are available to
rejudge them and verify their checksums against the
[audit record](../results/followup/math_pilot_audit.json).

To reproduce the GPU run in a separate directory on the configured Windows/WSL
host (with the pinned model snapshot already cached):

```powershell
./scripts/run_math_pilot.ps1 -RunName replay
Get-Content results/followup/work/reproductions/replay/math_status.json
Get-Content results/followup/work/reproductions/replay/work/train/step_times.jsonl -Tail 5
```

The runner keeps the host awake during the job and watches GPU memory during
training. For an interrupted training run, use `-Resume` with a completed
checkpoint under the selected run's `work/math_adapter` directory. No existing v1 model files
are used as the pilot's output destination.

The runner writes `math_screen.json`, `math_train_summary.json`, and
`math_dev_comparison.json` under `results/followup/` as each stage finishes.
When the dev gate passes it also writes `math_confirmation_comparison.json`.
Unfiltered generations, stage logs, and adapters are kept in the local
`results/followup/work/` directory.
