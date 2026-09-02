# Trajectory-level KL — Margin Notes 8-agent (longer horizon)

**Setup.** Same construction as the Inner Loop trajectory-KL measurement. Five Margin Notes 8-agent populations (seeds 1–5), per-trajectory log-likelihood ratio scored under each pair of seeds' weights at each shared checkpoint, averaged across A-sourced trajectories to estimate $\hat{D}(p_A \to p_B)$.

**Coverage.** Iters 25, 50, 75 full 5-seed panel (20 ordered pairs). Iters 100, 125, 150 are 4-seed panels — seed 4 stops at iter 75 (12 pairs). Seed 1 alone continues to iter 300 but those aren't used since there's no partner.

**Compute.** SLURM array 4825302, 12 single-GPU tasks, ~2.5 min each.

## Headline trajectory ([fig_kl_trajectory.png](fig_kl_trajectory.png))

| iter | mean cross-pop $\hat{D}$ per token | min | max | n ordered pairs |
|-----:|---:|---:|---:|---:|
|  25 | **0.0012** | −0.002 | +0.005 | 20 |
|  50 | 0.0158 | −0.012 | +0.058 | 20 |
|  75 | 0.0679 | −0.026 | +0.212 | 20 |
| 100 | 0.0843 | −0.076 | +0.254 | 12 |
| 125 | 0.181  | −0.120 | +0.672 | 12 |
| 150 | **0.287** | −0.227 | +1.187 | 12 |

Within-population diagonal: 0.000000 across all (seed, iter).

**Iter-25 baseline 0.0012 nats/token** — replicates Inner Loop's ≈0 baseline. The cross-population divergence at iter 150 is **240× the iter-25 baseline**, and **2× the Inner Loop end value** at iter 100 (0.147).

**No plateau by iter 150.** The growth from 100 → 125 → 150 is 0.084 → 0.181 → 0.287 — still accelerating (in fact, the inter-iter increment at the latest gap is the largest yet). Parallel populations are still actively diverging at the latest measured checkpoint.

## Per-character correlation with end-KL-from-base ([kl_vs_endkl.png](kl_vs_endkl.png))

| Subset | r | n |
|---|---:|---:|
| All agents | −0.057 | 32 |
| Coherent-only | **−0.343** | 24 |
| Exclude policy_collapse + text_degenerate | −0.233 | 28 |

**The full-sample correlation is weak** because Margin Notes' regime distribution is dominated by coherent agents (24 of 32), with only 3 committed and a handful of failure-regime agents. Two seed-1 policy-collapse agents (Kai Dempsey: kl_pertok = −9.49 at end-KL=0.72; Renata Solberg: kl_pertok = −7.47 at end-KL=0.97) have extreme negative contributions paired with *low* end-KL, which dilutes the inverse relationship that drove Inner Loop's r = −0.766. Restricting to coherent agents (where the noise floor is small) recovers a meaningful negative correlation (r = −0.34). The qualitative story is the same as Inner Loop but the linear-correlation summary statistic is sensitive to outlier composition.

## Pairwise matrix at iter 150 ([fig_kl_matrix_iter150.png](fig_kl_matrix_iter150.png))

Heatmap over the 4 seeds (1–3, 5) that share iter 150. Off-diagonal entries range from −0.12 to +0.67 — heavily heterogeneous, far more so than Inner Loop's iter-100 matrix. Some seed pairs have diverged dramatically more than others. The wide max (1.19) at iter 150 is for a specific (source, target) pair; the asymmetry of the KL is substantial here.

## Per-character decomposition ([fig_per_character_iter150.png](fig_per_character_iter150.png))

Highly uneven across characters. The Margin Notes character roster differs from Inner Loop's, but the same pattern obtains: a subset of slots contribute the bulk of the divergence (positive contributions), while a small number of slots that have policy-collapsed in some seed contribute large negatives.

## What this adds to the Inner Loop result

1. **Replication of the iter-25 baseline.** A second substrate, with a different forum framing and a different character roster, gives the same near-zero baseline (0.0012 vs 0.0014). The clean baseline is a property of the trajectory-level construction, not of Inner Loop specifically.

2. **Replication of monotone growth.** Same shape: small at iter 25, ×100+ by the latest checkpoint.

3. **No plateau at 2× the iter horizon.** Inner Loop measured to iter 100 with growth still in progress. Margin Notes measures to iter 150 and the growth is *accelerating*, not plateauing. Whatever drives cross-population divergence keeps driving it well past the iter-100 mark.

4. **The end-KL correlation depends on regime composition.** Inner Loop has more committed agents (8 of 32); Margin Notes has only 3. With fewer committed agents the cross-pop KL is dominated by coherent agents, which all contribute positively, so the inverse relationship is weaker. The coherent-only subset still gives r = −0.34.

## Caveats

- Margin Notes seed 4 only reaches iter 75; iters 100–150 are 4-seed comparisons (12 ordered pairs, not 20).
- The two extreme-negative policy_collapse contributors in seed 1 are real artefacts of those agents' policies (their own adapters score their own outputs as very surprising, consistent with the policy-collapse regime). They are present in the data and reported, not filtered.
- Per-pair variance is high in this dataset (min/max range much wider than Inner Loop), reflecting genuine asymmetry across seed pairs. The mean-across-pairs trajectory is the appropriate headline; individual matrix entries are noisier.

## Inputs

- Raw per-char shards: [`runs/cultural_emergence/_trajectory_kl/4825302/`](../_trajectory_kl/4825302/) (3 936 rows)
- Trajectory-aggregated: [`per_traj_kl.csv`](per_traj_kl.csv) (492 rows)
- Pairwise: [`pairwise_kl_by_iter.csv`](pairwise_kl_by_iter.csv)
- Per-(seed, char): [`kl_vs_endkl.csv`](kl_vs_endkl.csv)
