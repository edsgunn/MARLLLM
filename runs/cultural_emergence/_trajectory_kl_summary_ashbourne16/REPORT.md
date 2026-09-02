# Trajectory-level KL — Ashbourne GC 16-agent (longer horizon, 2× population)

**Setup.** Same construction as the Inner Loop trajectory-KL measurement. Five Ashbourne GC 16-agent populations (seeds 1–5), per-trajectory log-likelihood ratio summed across all 16 character slots, averaged across A-sourced trajectories.

**Coverage.** Full 5-seed panel at iters 25, 50, 75, 100, 125, 150 — 20 ordered pairs at every measured iter (cleanest panel of the three datasets so far).

**Compute.** SLURM array 4825303, 12 single-GPU tasks, ~4 min each (twice Inner Loop's per-task time owing to 16 vs 8 characters).

## Headline trajectory ([fig_kl_trajectory.png](fig_kl_trajectory.png))

| iter | mean cross-pop $\hat{D}$ per token | min | max | n ordered pairs |
|-----:|---:|---:|---:|---:|
|  25 | **0.0081** | −0.014 | +0.035 | 20 |
|  50 | 0.0404     | −0.052 | +0.134 | 20 |
|  75 | 0.1450     | +0.038 | +0.385 | 20 |
| 100 | 0.1609     | +0.058 | +0.297 | 20 |
| 125 | 0.2298     | +0.113 | +0.545 | 20 |
| 150 | **0.3323** | +0.157 | +0.536 | 20 |

Within-population diagonal: 0.000000 across all (seed, iter).

**Iter-25 baseline 0.0081 nats/token** — slightly higher than the 8-agent datasets (Inner Loop 0.0014, Margin Notes 0.0012), as expected: with 16 character slots contributing variance, the noise floor scales modestly with population size. Still small compared to the iter-150 value (×40 above baseline).

**Monotone growth across all six checkpoints**, no plateau visible. The growth rate slows a bit between iter 75 → 100 (0.145 → 0.161) but resumes between 100 → 125 → 150 (0.161 → 0.230 → 0.332). The iter-150 increment over iter 125 is +0.10 — still substantial.

**Strictly positive off-diagonal minimums by iter 75 onward.** Unlike Margin Notes (which had min ≤ 0 at every iter) or Inner Loop's iter-100 entry, *every* (source, target) pair from iter 75 onward has $\hat{D}(\text{source} \to \text{target}) > 0$. The cross-population divergence is universal, not localised to a few pairs.

## Per-character correlation with end-KL-from-base ([kl_vs_endkl_scatter.png](kl_vs_endkl_scatter.png))

**r = −0.779 across n = 80 (seed, char) pairs.**

This is the cleanest replication of Inner Loop's r = −0.766 (n = 32) at 2.5× the sample size. The negative-correlation finding is robust:

- Inner Loop 8-agent: r = −0.766 (n = 32)
- Ashbourne 16-agent: r = −0.779 (n = 80)
- Margin Notes 8-agent: r = −0.057 (n = 32), or r = −0.343 restricted to coherent-only (n = 24) — weakened by sparse committed-stratum sample

The two clean datasets give nearly identical correlation strength. Across 80 (seed, character) pairs in Ashbourne, **characters that drifted further from base have substantially smaller (often negative) per-character contributions to cross-population trajectory KL**. The qualitative finding from Inner Loop replicates with double the statistical leverage.

Regime composition for Ashbourne 16-agent is more balanced than either 8-agent dataset: 38 coherent, 31 mid-range, 5 committed, 3 policy-collapse, 3 text-degenerate. The mid-range stratum (31 agents — the largest single bin) plays a role here that didn't exist in Inner Loop, and its presence may explain why the correlation stays clean even though it's not driven exclusively by a small high-KL tail.

## Pairwise matrix at iter 150 ([fig_kl_matrix_iter150.png](fig_kl_matrix_iter150.png))

Full 5×5 with all off-diagonals well above zero (min 0.157). The matrix is moderately asymmetric ($\hat{D}(A\to B) \ne \hat{D}(B\to A)$ — expected for KL) and not all pairs are equidistant: some seed pairs have diverged ~3× more than others. No pair sits anywhere near the diagonal.

## Per-character decomposition ([fig_per_character_iter150.png](fig_per_character_iter150.png))

With 16 character slots the per-character variance is broader. Strong positive contributors dominate, with a few large-negative outliers driven by collapsed adapters in particular (seed, char) pairs.

## What this adds to the Inner Loop result

1. **Replication on a substrate with no shared structural conventions.** Ashbourne GC is a maths-chat substrate with no Python tool, no programming theme, completely different character roster. Same near-zero iter-25 baseline, same monotone growth, same r ≈ −0.78 with end-KL.

2. **Replication at 2× population size (16 vs 8 characters).** The trajectory-level construction scales — the divergence signal isn't a property of the 8-agent topology specifically.

3. **Full 5-seed panel across 6 iters.** The cleanest, most statistically tight version of the measurement. No checkpoint coverage gaps, 20 ordered pairs at every iter.

4. **Substantially larger end-of-training divergence** (0.332 vs 0.147 nats/token). Some of this is from the longer horizon (iter 150 vs 100); some is likely from the larger population having more degrees of freedom in which cultural divergence can accumulate. We don't have iter-100 measurements at 8 agents on Ashbourne or 16 agents on Inner Loop to disentangle "more iters" from "more agents."

## Caveats

- Per-trajectory aggregation dropped 70 trajectories (out of 600) that didn't have all 16 characters score (typically because one character didn't post in that episode). The remaining 530 give the headline number.
- Some agents in failure regimes (policy collapse, text degenerate) contribute large-magnitude noise to per-character decomposition. The full-sample correlation r = −0.779 is robust to dropping these (still strongly negative if restricted to coherent + mid_range + committed).

## Inputs

- Raw shards: [`runs/cultural_emergence/_trajectory_kl/4825303/`](../_trajectory_kl/4825303/) (9 530 rows)
- Trajectory-aggregated: [`per_traj_kl.csv`](per_traj_kl.csv) (530 rows)
- Pairwise: [`pairwise_kl_by_iter.csv`](pairwise_kl_by_iter.csv)
- Per-(seed, char): [`kl_vs_endkl.csv`](kl_vs_endkl.csv)
