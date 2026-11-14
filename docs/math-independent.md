# Independent test and data-source follow-up

The small [low-rate SFT development experiment](conservative-sft.md) could not
resolve a difference of a few percentage points. This test fixes the already
trained, full-strength rank-16 adapter at 5e-6 and compares it against the
same pinned unadapted Qwen2.5-1.5B-Instruct base. No new training or checkpoint
selection occurs in this comparison.

## 1. Frozen independent test

Sample 1,400 questions with seed 20261113, proportional to subject and
difficulty after filtering the [MATH official test split](https://arxiv.org/abs/2103.03874)
at the [pinned dataset mirror](https://huggingface.co/datasets/EleutherAI/hendrycks_math).
Exclude normalized exact matches to the full MATH training split and the
500 previously used MATH-500 questions, duplicate questions, figure-dependent
items marked with Asymptote, and references without a single verifiable final
answer. The 350-question confirmation split of the prior pilot remains unused.
This is a project holdout; the test data is public and pretraining exposure
cannot be ruled out.

The protocol lists question IDs, normalized question hashes, model and adapter
identities, data and scorer hashes, and scoring policy. Both models use the
same prompts, eager BF16 vLLM, greedy decoding, and a 2,048-token generation cap.
Raw answers are written in separate 200-question shards before CPU grading.
Report paired bootstrap intervals with the same final-answer judge as the prior
study, plus the earlier judge and strict boxed scoring as sensitivity checks.

The source had 5,000 test rows. Exact normalized matches to the train split or
MATH-500 excluded 502 rows; the figure/reference filter excluded another 768.
The 1,400 questions were selected from the 3,730 remaining eligible unique
questions. No test outputs had been generated when the protocol was frozen.

| Independent test | Base | Low-rate SFT |
|---|---:|---:|
| Correct, all three scoring rules | 840/1,400 (60.00%) | 830/1,400 (59.29%) |
| Unresolved | 66 | 75 |
| Truncated | 61 | 68 |
| Mean output tokens | 594.37 | 598.16 |

The candidate gained 62 and lost 72 answers: **−0.71 pp**, paired bootstrap
95% interval **[−2.36, +0.93] pp**. This supports noninferiority at the
predeclared −3 pp margin, but does not support superiority. See
[the fixed protocol](../results/math_independent/protocol.json),
[paired judgments](../results/math_independent/paired.json), and
[runtime provenance](../results/math_independent/run.json).

## 2. Decisions and conditional serving benchmark

Noninferiority requires a primary paired 95% interval with lower bound above
−3 percentage points, unresolved rates of at most 25%, and at most 3 percentage
points more unresolved for the candidate. A worthwhile improvement also needs
at least +2 percentage points, a strictly positive lower bound, and positive
deltas under both sensitivity judges. Use the base by default unless that
improvement gate passes.

Serving is measured only when noninferiority holds and mean output token count
falls by at least 3%. Use a local vLLM streaming endpoint on identical real
MATH prompts at concurrency 1 and 8; record time to first visible text,
end-to-end p50 and p95, requests/s and output tokens/s. The 32-request runs
are small and their tail estimates are descriptive.

**Decision:** the candidate's mean output grew by 0.64%, so the serving gate
did not pass and no online latency or throughput result is claimed. The base
remains the default. The test says the accuracy loss is bounded at the chosen
margin, not that the adapter improves accuracy or serving.
For context, offline generation across the seven accuracy-evaluation batches
took 543 s for base and 898 s for the runtime LoRA. These sequential batch
timings exclude model startup and are not an online latency comparison.

## 3. Matched self-generated data pilot

Start from the same base weights, using the same reviewed questions in both
treatment arms. One receives worked references; the other receives solutions
sampled from the unadapted model without reference answers in the prompt.
Both use 5e-6, the same order, epochs, completion mask, and optimizer updates.
Supervised token count may differ and is reported separately.

Screen four generations per question for an intact, substantial completion
with a correct final answer. Before training, manually check key reasoning
steps of every admitted self-generated trace against the problem and
reference. Final-answer matching alone is insufficient. Use at most 32
approved pairs, and stop at the review gate if fewer than 24 pass. Compare
base and both arms on a newly fixed 350-question development split from the
official training data, disjoint from the training pool and previous splits.
This is a small pilot with selection toward problems the base can already
solve. A positive result against both controls would justify an independent
study before trying GRPO.

Of 56 training questions (224 sampled responses), 31 had a completion
passing the automatic checks. The assistant independently inspected key
steps of all 31 selected responses before evaluation; 25 passed and six
were rejected despite matching the reference answer. The
[review record](../results/math_self_sft/review.json) binds decisions to
the original queue hash and includes the checked step or reason for rejection.
The two arms use those same 25 questions. The original 350-question
confirmation set stays reserved; the new dev set also contains 350
questions, 50 per subject, with separate identities.
Both arms completed 12 optimizer steps. The supervised targets total
4,951 tokens per epoch for the references and 12,181 for self-generated
solutions. Equal updates therefore do not isolate source from target length.

| Fresh development set | Base | Reference SFT | Self-generated SFT |
|---|---:|---:|---:|
| Correct, all three scoring rules | 233/350 (66.57%) | 236/350 (67.43%) | 232/350 (66.29%) |
| Unresolved | 21 | 16 | 18 |
| Truncated | 20 | 16 | 18 |
| Mean output tokens | 630.79 | 620.87 | 629.64 |

Self-generated SFT changed accuracy by **−0.29 pp** versus base (12 gains,
13 losses; paired 95% interval **[−3.14, +2.57] pp**) and **−1.14 pp** versus
reference SFT (13 gains, 17 losses; **[−4.29, +2.00] pp**). Neither
comparison met its advancement rule. This small, single-seed experiment does
not establish a source effect; it also does not support expanding the run or
starting GRPO. The separate 1,400-question test was not used to score either
new SFT adapter. See [training receipts](../results/math_self_sft/training.json)
and [paired results](../results/math_self_sft/paired.json).

Both reports can be recomputed on CPU without generating answers:

```bash
python -m src.math_independent verify
python -m src.math_self_sft verify
```

On the configured Windows/WSL host:

```powershell
./scripts/run_math_extension.ps1 -Mode independent
./scripts/run_math_extension.ps1 -Mode serving    # only if its gate passes
./scripts/run_math_extension.ps1 -Mode self-generate
# Review the generated queue and record decisions before the next command.
./scripts/run_math_extension.ps1 -Mode self-compare
```

The runner saves state and logs under `results/followup/work/`. The original
frozen evaluation and released artifacts are unaffected.
