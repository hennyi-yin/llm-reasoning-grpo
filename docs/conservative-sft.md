# Lower-rate worked-reference SFT

The [previous matched SFT experiment](worked-sft.md) lost 26.02 percentage
points versus base on MATH dev at full LoRA strength. Using one quarter of
the same adapter at inference reduced the deficit to 3.06 points. This
follow-up tests whether a smaller update during training helps.

The sole intended training change is the peak learning rate: **5e-5 → 5e-6**.
Both runs start at the same pinned Qwen2.5-1.5B-Instruct weights and use the
same 335 worked-reference examples in the same order, one epoch, 42 steps,
rank-16 LoRA, and the same completion mask and token limits. The new adapter
is evaluated at full strength, on the same 196 prompts, with the same greedy
2,048-token limit, fixed primary judge, and both sensitivity judges. The
base outputs are reused after checking their identity. Reducing the learning
rate is not equivalent to multiplying an already trained LoRA at inference.

The **preservation check** requires the lower end of the paired bootstrap
95% interval for the primary delta to be above **−3 pp**, with an unresolved
fraction no greater than 25% in either arm and no increase greater than
3 pp for the candidate. A preservation pass on this previously inspected
development set is an engineering signal, not independent confirmation.

The **improvement gate** additionally requires a primary point estimate of
at least **+2 pp**, a strictly positive paired lower confidence bound, and
positive changes under both sensitivity judges. Only an improvement pass
allows the reserved 350-question set to be evaluated. There is one fixed
low-rate arm; checkpoint and learning-rate sweeps are outside this run.

## Results

| Same MATH development questions | Base | Original worked SFT | Lower-rate worked SFT |
|---|---:|---:|---:|
| Primary judge, out of 196 | 120 (61.22%) | 69 (35.20%) | 118 (60.20%) |
| Previous judge | 119 | 69 | 118 |
| Strict boxed judge | 119 | 68 | 118 |
| Unresolved | 18 | 47 | 12 |
| Truncated | 18 | 26 | 12 |

The lower-rate model gained six answers and lost eight against base
(**−1.02 pp**, paired 95% bootstrap interval **[−4.59, +2.55] pp**).
On the 174 questions where both responses were complete and resolved, base
scored 120 and the candidate 117; this is descriptive because the outputs
determine membership. The training manifests differ only in learning rate
and output directory. Both arms used the same 42 optimizer updates. Thus
the fixed lower-rate recipe avoided most of the large observed regression,
but the interval still includes losses beyond the preservation margin as
well as small gains. It passed neither gate. The 350-question confirmation
set was not used.

These are single-seed comparisons on a repeatedly inspected development
set. Further small gains should be measured on a fresh, larger evaluation
split, with a candidate fixed before evaluation. With discordance near
14/196, a normal-approximation power calculation gives roughly 1,400 paired
questions for detecting +2 pp at 80% power and a two-sided 5% threshold,
assuming the discordance rate transfers. This is a planning estimate, not
a requirement or a power guarantee for another model or dataset.

The [paired dev ledger](../results/sft_conservative/dev_paired.json) and
[training record](../results/sft_conservative/run.json) can be checked on CPU:

```bash
python -m src.math_sft_conservative verify
```

## Reproduction

Run on the configured Windows/WSL host:

```powershell
./scripts/run_math_sft.ps1 -Conservative
Get-Content results/followup/work/math_sft_conservative/status.json
```

The runner uses a separate directory, keeps the host awake, and resumes from
a complete checkpoint if restarted. It reuses the checked training set if
available; a fresh replay reconstructs it from the cached pinned source.
