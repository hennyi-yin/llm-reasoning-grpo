# Reference-blind answer selection on MATH

The earlier [MATH multi-sample study](math-multisample.md) found a correct sample
on 166 of 200 questions for the 50-step no-SFT GRPO adapter. Plurality voting
returned a correct answer on 134 of those questions. This study asks whether
information available at selection time can close part of that gap with the
same eight generated outputs. The baseline uses the earlier implementation of
[self-consistency voting](https://arxiv.org/abs/2203.11171).

## Frozen design

- Reuse the GRPO arm of the previous 200-question run as the development set.
  A candidate is a group of samples with the same normalized answer key. Both
  methods return the earliest sample in the winning group and abstain if no key
  was extracted.
- Fit a pairwise logistic ranker on development questions that contain a
  verifiably correct representative and an incorrect representative. Give each
  informative question equal total weight. The features are log vote count,
  log scaled median output length, the fraction of group samples that finished
  before the cap, and indicators for numeric and word answer keys. Regularize
  toward log vote count alone.
- Compare three regularization strengths using five question-level
  cross-validation folds. Pick the best out-of-fold accuracy, with ties going
  to the strongest regularization, then refit on all 200 development questions.
  Development labels enter fitting and scoring. At test selection time, each
  method sees answer keys, token counts, and truncation flags. The scorer reads
  the reference after both choices have been made.
- Draw 300 fresh official MATH test questions proportionally by subject and
  difficulty. Exclude exact normalized matches to all official MATH training
  questions, the original MATH-500, the earlier 1,400-question test, and all 200
  development questions. Exclude duplicates, Asymptote figures, and references
  without one parseable boxed final answer. The older 350-question confirmation
  set remains reserved.
- Use the same pinned Qwen2.5-1.5B-Instruct and 50-step MATH GRPO adapter as the
  development run. Generate eight samples per question at temperature 0.7,
  top-p 1.0, and a 2,048-token cap. Consecutive questions use disjoint eight-seed
  ranges. Score both choices with the shared final-answer grader.
- The primary endpoint is the paired difference in test accuracy between the
  frozen ranker and plurality. Advancement requires a gain of at least 2
  percentage points and a positive lower bound of the paired 95% question
  bootstrap interval (10,000 replicates). Report changed choices, corrections,
  regressions, and pass@8 from the same generation budget.

The [frozen protocol](../results/math_selector/protocol.json) records the fitted
coefficients, model and code hashes, fit results, inference settings, and test
question fingerprints. It was published before test scoring. Filtering left
2,130 eligible unique test rows. This project holdout comes from a public
benchmark that may have appeared in pretraining. The grader measures final-answer
correctness.

## Development result

| Rule | Correct out of 200 |
|---|---:|
| Plurality | 134 |
| Ranker, penalty 1.0 | 136 |
| Ranker, penalty 0.1 | 136 |
| Ranker, penalty 0.01 | 136 |

There were 103 informative development questions. The frozen ranker uses
penalty 1.0. With eight samples and the configured length cap, the fitted vote
weight dominates the largest possible difference from the other features for
unequal vote counts. The rule therefore acts as a tie-breaker in this test.
This limits the experiment to recovering correct answers among the top-voted
groups.

## Independent test result

| Rule or diagnostic | Correct out of 300 | Accuracy |
|---|---:|---:|
| Plurality | 206 | 68.67% |
| Frozen ranker | 206 | 68.67% |
| Best possible only among top-voted groups | 211 | 70.33% |
| At least one correct sample in eight | 230 | 76.67% |

The ranker changed 26 selections, fixing four wrong answers and introducing
four errors. The paired accuracy difference was **0.00 percentage points**
(95% bootstrap interval **[−2.00, +2.00]**). The ranker fell short of the frozen
advancement gate. Both rules used the same 2,400 completions. Generation used
1,375,763 output tokens with 55 length-cap stops and took 24.16 minutes on the
RTX 5070.

The test contained 37 questions with multiple keys tied for most votes. For
the post-hoc ceiling, count a question when at least one top-voted group's
earliest sample passes the grader. The ceiling is five questions above baseline
(+1.67 points), below the study's +2-point practical threshold on these samples.
On another 19 questions a correct candidate group's representative existed at a
lower vote count. A candidate verifier that can select lower-vote answers
would test a larger part of the observed gap.

See the [paired test summary](../results/math_selector/comparison.json),
[sample-level judgments](../results/math_selector/paired.json), and
[generation receipt](../results/math_selector/run.json).

## Reproduction

```bash
python -m src.math_selector_experiment verify
```

The CPU check refits the ranker from the public development ledger, recomputes
both selections from the public test keys and token counts, and checks the saved
paired results. The generation stage requires the cached model, audited adapter,
pinned dataset, and the frozen local inputs. On the experiment host, the runner
resumes from the last completed shard.

```powershell
./scripts/run_math_extension.ps1 -Mode selector
Get-Content results/followup/work/math_selector/status.json
```
