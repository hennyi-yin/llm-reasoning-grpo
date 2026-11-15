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
question fingerprints. Filtering left 2,130 eligible unique test rows.

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
