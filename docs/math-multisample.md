# MATH answer coverage and voting

The [50-step no-SFT MATH GRPO pilot](followup.md) tied the unadapted
Qwen2.5-1.5B-Instruct on its development set. This experiment uses both saved
models to ask whether the adapter changes the chance that a correct final
answer appears among several attempts, and whether voting can select it.

## Protocol

- Freeze 200 questions from the official 5,000-row MATH test split before generating
  either model's outputs. Exclude normalized exact matches to the complete MATH
  training split, the earlier MATH-500 evaluation, and the 1,400 questions in the
  [previous independent test](math-independent.md). Deduplicate and require a single
  verifiable boxed reference, no Asymptote marker, and a prompt that fits the context
  budget. Sample proportionally by subject and difficulty using seed 314159.
- Compare the pinned unadapted base and the audited full-strength rank-16 adapter
  from the no-SFT MATH pilot. Use the same prompt, BF16 vLLM runtime, eight samples
  per question, temperature 0.7, top-p 1.0, and a 2,048-token output cap. Neither
  model receives the reference or a worked solution. Keep the previous 350-question
  confirmation set reserved.
- The primary metric is pass@8 under the shared final-answer scorer, which can
  grade well-formed unboxed answers. For pass@1, @2, and @4, average the probability
  of at least one correct answer in a uniformly drawn size-*k* subset of the eight
  outputs. Given *c* correct samples, that probability is
  `1 − C(8 − c, k) / C(8, k)`. Strict boxed pass@1 and @8 check sensitivity to the
  scoring rule.
- For voting, extract each sample's final answer and tally a normalized key without
  the reference. Normalize exact integer, decimal, and simple fraction equivalents,
  a restricted set of word answers, and whitespace or delimiter differences.
  Symbolic equivalence beyond these normalizations can split votes. Choose the
  earliest sampled answer among tied keys and count abstentions as incorrect.
  Grade the selected sample with the same final-answer scorer. A correct sample
  may exist even when voting misses it.
- Bootstrap the paired questions 10,000 times. Each resampled question retains
  both models' eight outputs. The interval describes uncertainty across sampled
  questions conditional on the generated outputs. The other metrics are exploratory.
  Report actual generated tokens and length-cap stops alongside accuracy.
- Both models use the same prompt seeds. Each question's eight vLLM child seeds
  are distinct. The prompt seed advances by one per question and vLLM adds the
  sample index, so neighboring questions reuse some seeds. Treat the pilot
  as conditional on this run. An independent replication should use a separate
  seed range for each question and repeat the experiment across seed schedules.

The [frozen protocol](../results/math_multisample/protocol.json) records question
fingerprints, model weights, pinned data, configuration, and scoring-code hashes.
Of the 5,000 test rows, 1,902 matched training or prior evaluations and 768
failed the reference/diagram filter. The remaining 2,330 questions were eligible
after deduplication.
The sample is new to this project. The public MATH dataset may have appeared in
base pretraining. The judge checks final-answer equivalence, so this study cannot
establish that a generated derivation is sound. The design draws on prior work
on [pass@k under RL with verifiable rewards](https://arxiv.org/abs/2504.13837)
and [self-consistency voting](https://arxiv.org/abs/2203.11171).

## Results

The run generated and scored 1,600 completions per model. Differences and paired
95% bootstrap intervals are in percentage points.

| Metric | Base | 50-step GRPO | Difference | Paired 95% interval |
|---|---:|---:|---:|---:|
| Pass@1 | 55.19% | 55.06% | −0.13 | [−2.00, +1.75] |
| Pass@2 | 66.46% | 65.93% | −0.54 | [−2.54, +1.45] |
| Pass@4 | 74.80% | 75.46% | +0.66 | [−1.66, +2.96] |
| **Pass@8 (primary)** | **80.50%** | **83.00%** | **+2.50** | **[−0.50, +6.00]** |
| Reference-blind vote@8 | 68.50% | 67.00% | −1.50 | [−5.51, +2.50] |
| Strict boxed pass@8 | 80.50% | 83.00% | +2.50 | [−0.50, +6.00] |

Both models produced at least one correct sample on 158 questions. GRPO covered
8 additional questions and base covered 3, leaving 31 uncovered by either.
The primary interval includes zero. Strict boxed scoring gave the same pass@8
counts and paired differences.

Voting selected a correct answer on 137 base questions and 134 GRPO questions.
The gap between observed coverage and vote accuracy was 24 questions for base
and 32 for GRPO. In an exploratory check of the eight GRPO-only coverage cases,
the vote recovered only one. Across all covered questions the correct key lost
on frequency 16 times for base and 22 for GRPO, and lost a tie 8 and 10 times,
respectively. These are failures of this voting procedure on the fixed samples.

| Generation cost | Base | 50-step GRPO |
|---|---:|---:|
| Mean output tokens per sample | 591.85 | 596.46 |
| Length-cap stops, of 1,600 | 36 (2.25%) | 45 (2.81%) |
| Unresolved final answers, of 1,600 | 45 | 54 |
| Offline generation minutes on RTX 5070 | 9.95 | 18.17 |

The small positive coverage estimate merits testing with new questions and
disjoint seed ranges if coverage is the research target. The vote results
provide no evidence of an end-to-end accuracy gain with this selector.
For a selection experiment, tune one reference-blind selector on development
outputs and evaluate the fixed selector on independently generated test
outputs. The observed coverage provides an upper bound for selectors that
return one of the eight samples under the same scorer.

See the [numerical comparison](../results/math_multisample/comparison.json),
[sample-level judgments](../results/math_multisample/paired.json), and
[run receipt](../results/math_multisample/run.json).

## Reproduction

```bash
python -m src.math_multisample verify
```

This check recomputes all comparisons from the public per-question sample
judgments and verifies their identities against the frozen protocol. Replaying
generation on the configured GPU host requires the cached model and dataset,
the local 50-step pilot adapter, and the frozen input rows. On that host, resume
from completed generation shards with:

```powershell
./scripts/run_math_extension.ps1 -Mode multisample
Get-Content results/followup/work/math_multisample/status.json
```

The public files contain extracted voting keys, correctness flags, and token
counts. Full sampled completions and the pilot adapter reside in the local
work directory.
