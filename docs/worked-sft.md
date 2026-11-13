# Worked solutions versus final answers on MATH

This experiment asks whether worked references help a small model on held-out
math problems more than training on the same problems and their final answers.
It follows the [50-step GRPO pilot](followup.md), which found no net dev gain.
The original benchmark tables and checkpoints remain separate.

- **Training data:** sample 48 questions per subject from the same pinned
  [MATH training files](https://huggingface.co/datasets/EleutherAI/hendrycks_math).
  Exclude the previous screening, dev, and confirmation IDs and their normalized
  question texts, as well as normalized MATH-500 question texts. Require a
  single boxed answer, at least 120 worked completion tokens, and an intact
  sequence within 1,024 tokens. Exact-text deduplication does not rule out
  paraphrases or exposure during pretraining.
- **Reference review:** audit three fixed samples per subject before training.
  [Twenty of 21](../results/sft_followup/reference_review.json) pass a check of
  the substantive derivation; one depends on a missing figure and is excluded
  from both arms. This is a sampled review of human-provided references, not
  independent verification of all references or self-generated reasoning.
- **Comparison:** Qwen2.5-1.5B-Instruct base, answer-only SFT, and worked-reference
  SFT. The two adapters use the same 335 training problems, order, base
  revision, rank-16 LoRA setup, learning rate, and number of updates. The
  completion targets differ, so the experiment does not match supervised token
  counts or isolate reasoning from every other feature of the targets.
- **Evaluation:** use the same 196 held-out dev prompts and greedy generation
  with a 2,048-token cap. Reuse the checksum-verified baseline outputs when
  present; a fresh GPU replay regenerates them with the same settings. Freeze
  the new judge before seeing either SFT arm's answers: it additionally accepts
  exact matching mathematical final expressions and a narrow form of numeric
  conclusion in prose. Report the previous judge and strict boxed score as
  sensitivity checks. There is no inference about correctness of the derivation
  from a correct final answer alone.
- **Decision:** use the 350 reserved questions only if the worked-reference arm
  gains at least 2 percentage points over **both** base and answer-only on dev,
  improves over both on the previous judge, and meets the unresolved-answer
  limits in [the configuration](../configs/math_sft.yaml). A positive
  confirmation requires the same directional and coverage checks plus a
  strictly positive paired 95% bootstrap interval against each comparator.

The pipeline runs on the existing 12 GB RTX 5070 setup. It uses one epoch per
arm, checkpointing every 14 steps and a device-memory guard. Stage logs and
weights live under `results/followup/work/math_sft/`.

## Results and stopping point

Both arms finished 42 optimizer steps. The answer-only targets contain 4,801
tokens and the worked targets 86,527 (including end-of-sequence tokens). The
reported peak device memory during training was 10,497 MiB and 6,522 MiB,
respectively; these device-wide samples include other GPU allocations.

| Dev outcome (196 questions) | Base | Answer-only SFT | Worked-reference SFT |
|---|---:|---:|---:|
| Shared final-answer score | 120 (61.22%) | 31 (15.82%) | 69 (35.20%) |
| Previous judge | 119 | 31 | 69 |
| Strict boxed score | 119 | 31 | 68 |
| Unresolved | 18 | 2 | 47 |
| Truncated | 18 | 1 | 26 |
| Mean generated tokens | 656.6 | 23.9 | 423.8 |

Worked-reference SFT gained **19.39 pp** over answer-only SFT (paired 95%
interval: 11.73 to 27.04 pp), but lost **26.02 pp** relative to the untrained
base (paired 95% interval: −33.16 to −18.88 pp). Against base, eight answers
improved and 59 regressed. The same direction holds under both sensitivity
judges and across all seven subject strata. Even on the 136 questions with a
resolved, untruncated answer from all three models, the base got 97 correct
and worked-reference SFT got 68. That subset is descriptive because its
membership depends on model outputs.

The [post-hoc review](../results/sft_followup/dev_unresolved_review.json)
found seven additional correct final answers among all 21 complete but
unresolved worked-model outputs, including one with a sign error in the
derivation. This raises its sensitivity total to 76/196. Even granting credit
for all 47 unresolved outputs would only reach 116/196 against the baseline's
120/196; extraction failures alone cannot account for the regression.

The run stopped at the predefined dev gate and left the 350-question
confirmation set unused. These results rule out this SFT recipe as an upgrade
on this dev set; they do not establish whether a smaller update, different
reference mix, or training on verified self-generated solutions would help.
Both SFT recipes fit the GPU. The immediate bottleneck is retaining the
baseline's accuracy while changing supervision, with output-format and
truncation failures as additional issues. Enlarging this run or adding GRPO
would change multiple variables before any baseline-preserving SFT result.

In a separate, post-hoc dev diagnostic, we reused the exact worked-adapter
weights and reduced their inference contribution to one quarter
(LoRA alpha 32 → 8, rank 16). This recovered much of the loss: **114/196
(58.16%)**, versus 69/196 at full strength. It remained **6 answers below
base** (−3.06 pp; paired 95% interval −7.65 to +1.02 pp), with six gains and
12 losses against base. This makes update strength a concrete variable for
future experiments, but does not establish that weaker training would beat
base. This diagnostic neither selects a model for publication nor opens the
confirmation set. Its
[paired evidence](../results/sft_followup/dev_scale_probe.json) is kept
separate from the original stopping decision.

Recompute the report from the [paired outcomes](../results/sft_followup/dev_paired.json)
without a GPU:

```bash
python -m scripts.verify_math_sft
```

To reproduce from the project WSL environment, cache the pinned model and
training files first, then run:

```bash
source ~/.venvs/llm-reasoning-grpo/bin/activate
source scripts/runtime_env.sh
export LLM_MATH_SFT_OUT="$PWD/results/followup/work/math_sft_replay"
python -m src.math_sft prepare
python -m scripts.run_math_sft
# Optional exploratory replay after the dev gate stops:
python -m scripts.run_math_sft --probe-scale
```
