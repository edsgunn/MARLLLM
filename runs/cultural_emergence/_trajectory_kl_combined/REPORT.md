# Trajectory-level cross-population KL — three-dataset consolidated report

Combined view of the trajectory-KL measurement across three multi-seed substrates. Single-dataset reports linked at the end.

## Summary table (mean cross-pop $\hat{D}$ per token, nats)

| iter | Inner Loop 8a | Margin Notes 8a | Ashbourne 16a |
|-----:|---:|---:|---:|
|  25  | 0.0014 | 0.0012 | 0.0081 |
|  50  | 0.0214 | 0.0158 | 0.0404 |
|  75  | 0.0945 | 0.0679 | 0.1450 |
| 100  | 0.1474 | 0.0843 | 0.1609 |
| 125  |   —    | 0.181  | 0.230  |
| 150  |   —    | **0.287** | **0.332** |

Inner Loop's measurement window ends at iter 100. Margin Notes and Ashbourne both extend to iter 150 (and still rising). Within-population diagonal is 0.000000 across every (seed, iter) in every dataset — confirms the scoring is numerically clean across all three.

### Trajectory figure: [fig_combined_trajectory.png](fig_combined_trajectory.png)

## End-KL correlation table

| Dataset | r (per-(seed, char)) | n | regime composition |
|---|---:|---:|---|
| Inner Loop 8-agent | **−0.766** | 32 | mostly coherent + ~25% committed |
| Ashbourne GC 16-agent | **−0.779** | 80 | 38 coherent + 31 mid_range + 5 committed + 6 fail |
| Margin Notes 8-agent | −0.057 (full) / −0.343 (coherent only) | 32 / 24 | 24 coherent + 3 committed + 5 fail |

### End-KL scatter figure: [fig_combined_kl_vs_endkl.png](fig_combined_kl_vs_endkl.png)

## Headline findings consolidated

1. **The trajectory-KL construction works in every dataset we tested.** Iter-25 baseline is ≈0 (0.001–0.008) in all three. Cross-population divergence grows monotonically over training. No plateau anywhere — the measurement is still increasing at the latest measured checkpoint in every dataset.

2. **Longer horizons give larger divergences without flattening.** Going from Inner Loop's iter 100 (0.147) to Margin Notes' iter 150 (0.287) and Ashbourne's iter 150 (0.332) shows further ~2× growth — and the inter-iter growth is *accelerating* at the latest points in both extended datasets. Cultural divergence does not appear to be approaching a ceiling at the iter horizons we can measure.

3. **The inverse correlation between per-character end-KL and trajectory-KL contribution replicates.** Inner Loop r = −0.766 (n = 32) and Ashbourne r = −0.779 (n = 80) are essentially identical. The Margin Notes weak correlation is explained by its regime composition (very few committed agents and two extreme policy-collapse outliers in seed 1 with low end-KL and large negative kl_pertok) — restricting to coherent agents still gives r = −0.34.

4. **The qualitative regime story is consistent:** characters that drifted far from base (committed regime) contribute *less* to cross-population divergence than characters that drifted little (coherent regime). Committed agents converge across seeds; coherent agents diverge. The paper's regime taxonomy — built around per-character KL drift — does not predict where cultural divergence lives in the trajectory distribution. The two quantities are inversely related.

5. **Larger populations show larger absolute divergence.** Ashbourne 16-agent produces larger end-of-training KL than Margin Notes 8-agent at the same iter (0.332 vs 0.287 at iter 150). Some of this is iter-horizon-confounded since the two substrates have different rate-of-growth; the comparison would be cleanest if we had Ashbourne 8-agent or Margin Notes 16-agent variants. The qualitative direction is consistent with more characters → more degrees of freedom in which to diverge.

## What this means for the paper

This is now the central empirical claim of the cultural-emergence section. The construction:

- Has a clean baseline (iter 25 ≈ 0)
- Has a strong, growing signal (×40–×240 over training)
- Replicates across two cleanly-comparable substrates (Inner Loop and Ashbourne) and a third with a noisier but consistent pattern (Margin Notes)
- Doesn't plateau at the iter horizons we tested — suggesting cultural divergence continues to develop with continued training
- Picks up something the existing per-character KL metric doesn't (and in fact inversely tracks)

The submitted draft's framing of "absorption" as the mechanism of cultural learning needs adjustment in light of these results: per-character KL absorption and trajectory-level cultural divergence are two different things, inversely related. The committed-stratum agents are converging across seeds (in policy space); the coherent agents are diverging. Cultural divergence is concentrated in the regime that *looks* like it learned the least by the existing per-character metric.

## Inputs

- Combined figures and table: this directory
- Per-dataset reports: [Inner Loop](../_trajectory_kl_summary/REPORT.md), [Margin Notes](../_trajectory_kl_summary_margin_notes/REPORT.md), [Ashbourne 16-agent](../_trajectory_kl_summary_ashbourne16/REPORT.md)
- Raw shards: `_trajectory_kl/{4824410, 4825302, 4825303}/`
- Scripts: [`analyze_trajectory_kl.py`](../../../scripts/analyze_trajectory_kl.py), [`analyse_trajectory_kl.py`](../../../scripts/analyse_trajectory_kl.py), [`combine_trajectory_kl_summaries.py`](../../../scripts/combine_trajectory_kl_summaries.py)
