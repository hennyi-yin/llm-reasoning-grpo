# Arithmetic checks for MATH answer selection

The [answer selection study](math-selector.md) found that the correct answer was
in a lower-vote candidate group on 19 of 300 questions. This follow-up tests
whether checking explicit computations can identify such candidates from the
same eight completions. Prior studies have used verifiers to choose among
generated mathematical solutions, including evaluations of intermediate
steps. [Training Verifiers to Solve Math Word Problems](https://arxiv.org/abs/2110.14168)
and [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050).

## Frozen design

- Use the prior 300-question answer selection test as development data.
  Recompute arithmetic features from those 2,400 hash-verified raw
  completions. Use final-answer labels only to choose the verification weight.
- Extract adjacent equality links from delimited math. Evaluate supported
  closed arithmetic with exact rational numbers, including arithmetic
  operators, braced fractions, small integer exponents, and exact square
  roots. Count repeated expressions once within a completion. Record true
  nontrivial equalities, unequal values, and true nontrivial equalities whose
  value matches the candidate's normalized numeric answer.
- Each answer group receives its log-base-two vote count plus the selected
  weight times its mean evidence score. A sample receives +1 if it has a
  checked link to its numeric answer and −1 if it has an unequal arithmetic
  link. Both flags may apply. For an unsupported expression the check
  contributes no evidence. Both methods return the first sample of the
  selected group and break exact rank ties in sample order.
- Select the weight with highest development accuracy among 0.5, 1, 2, and 4,
  breaking ties in favor of the smaller weight. The test uses this one frozen
  weight. Both selectors read completions without references. The shared
  final-answer grader reads references after both selections.
- Draw 300 official MATH test questions from the same pinned dataset,
  stratified by subject and difficulty. Exclude exact normalized matches to
  all official training rows, MATH-500, and the three earlier test cohorts of
  1,400, 200, and 300 questions. Exclude duplicate problems, Asymptote
  figures, infeasible prompts, and references without a single verifiable
  boxed answer. Use the same pinned Qwen2.5-1.5B-Instruct and 50-step no-SFT
  MATH GRPO adapter with eight samples, temperature 0.7, top-p 1.0, and a
  2,048-token cap. Each question receives its own range of eight child seeds.
- The primary metric is the paired final-answer accuracy difference versus
  ordinary plurality. The advancement criterion is a gain of at least two
  points and a positive lower bound of the paired 95% question-bootstrap
  interval. Report all corrected and newly wrong answers and the subset
  caused by selecting a lower-vote group.

The check covers numerical equality and can find a local false computation.
General algebra, logical deductions, applicability to the problem, and
context such as an abandoned attempt are outside this rule's coverage. An
equality matching an answer provides arithmetic evidence for that value.
Benchmark texts are public, and the holdout boundary concerns project use.

See the [frozen selection protocol](../results/math_arithmetic/protocol.json)
and the [development arithmetic features](../results/math_arithmetic/development.json).

## Development results

| Rule | Correct out of 300 |
|---|---:|
| Plurality | 206 |
| Arithmetic weight 0.5 | 207 |
| Arithmetic weight 1.0 | 207 |
| Arithmetic weight 2.0 | 205 |
| Arithmetic weight 4.0 | 195 |

The frozen test rule uses weight 0.5. Test filtering left 1,830 eligible
unique rows before drawing the 300-question holdout.
