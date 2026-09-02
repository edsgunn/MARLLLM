# Threshold-sweep + clustering rebuttal notes (2026-07-31)

Generated from `scripts/sweep_regime_thresholds_3d.py`; mechanical numbers in
`REPORT.md`, per-agent label distributions in `label_stability_per_agent.csv`.

Reviewer asks: (a) threshold sensitivity for the KL/perception-loss/entropy
regime labels, (b) an unsupervised clustering alternative, (c) uncertainty
over labels, (d) justify or rename the high-entropy branch of policy
collapse, (e) clarify the Inner Loop committed-agent count.

## Setup

Joint sweep over all five classification parameters — KL threshold
{0.5…2.0}, coherent perc-loss cut {1.5…1.9}, degenerate perc-loss cut
{1.9…2.2}, entropy lower bound {0.1…0.3}, entropy upper bound {1.5…1.9} —
2,850 valid combinations (a ±50–100% perturbation range around the paper
values; the paper setting is one grid point). Labels recomputed for the
Figure-1 agents (iter 200), all 220 labelled agents including the 15
seed-replicate populations, and the PMI-endpoint agents.

## (a) The PMI result survives every grid point

The committed-vs-coherent PMI shrinkage contrast (end-of-training labels,
pooled — same estimator as the submitted analysis) is **positive at 100% of
the 2,850 combinations in both datasets**:

- cross-env (56 agents): paper value +0.199 nats/token; grid median +0.185,
  IQR [+0.130, +0.237], minimum +0.028. Committed n ranges 2–17 over the grid.
- inner-loop seeds (40 agents): paper value +0.160; grid median +0.160, IQR
  [+0.160, +0.198], minimum +0.130. Committed n ranges 4–7.

The contrast never reverses, and is flat in the three parameters the reviewer
highlighted beyond KL (perc-degenerate cut, both entropy bounds — see
`fig_pmi_contrast_grid`). Its magnitude grows with the KL threshold (higher
cut = purer committed group), which is the direction the mechanism predicts.

## (b) Unsupervised alternative agrees

k-means / GMM on standardised (log10 KL, perception loss, entropy), no
thresholds:

- Inner-loop seeds: silhouette selects **k=2**, and the 2-cluster solution
  reproduces the committed/coherent split *exactly* (ARI = 1.00). Cluster
  contrast: high-KL cluster shrinkage +0.215 vs low-KL +0.055, one-sided
  Mann-Whitney p = 0.016.
- Pooled endpoints (96 agents): k=4; mean shrinkage is monotone in cluster
  mean KL (0.05 → 0.06 → 0.11 → 0.29, `fig_clustering_pmi`); alive-region
  high-KL vs low-KL cluster contrast +0.114 vs +0.057 (p = 0.12 — small n,
  same direction).
- Threshold-free continuous version (no labels at all, from
  `_absorption_threshold_sensitivity`): shrinkage regressed on end KL gives
  slope +0.046 nats/token per nat of KL, 95% CI [+0.026, +0.066], R² = 0.34
  (inner-loop seeds); same sign on cross-env with wider CI.

## (c) Label uncertainty, honestly

Median per-agent label stability across the grid is 0.83 (Figure-1 set) and
0.80 (replicates); 45%/39% of agents keep their label at ≥90% of settings.
The unstable mass is concentrated exactly where a threshold taxonomy predicts:
committed agents (median stability 0.58) trade labels with mid-range/coherent
as the KL and perc cuts move. Crucially, **flips stay inside the alive
region** — the dominant alternative label for every alive regime is another
alive regime, so condition-level conclusions do not move. Across all 2,850
settings, every condition with >=8 agents stays majority-alive (worst-case
alive fraction 0.75 for strathearn-16, >=0.80 elsewhere), while the
2-agent Strathearn condition never exceeds 0.50 alive; the small conditions
(2-4 agents) shift by at most one agent (+/-0.25-0.50) at extreme settings,
which is the granularity of n=2 (`composition_grid.csv`). The regime ordering
that the paper's narrative rests on holds at every grid point.

## (d) The high-entropy branch never fires

Of the 7 policy-collapse agents in the full labelled set (220 agents), **all
7 classify via entropy < 0.2** — near-deterministic generation, matching the
text's description. No agent in any run we have labelled exceeds entropy
1.59, so the entropy > 1.8 guard is empirically vacuous. Proposed text fix:
define policy collapse as the low-entropy condition (matching the
"near-deterministic" prose), keep the upper bound as a safety valve but state
explicitly that it never triggers in our data.

## (e) Inner Loop committed count

End-of-training committed agents across the 5 inner-loop seeds: **6
(run, character) instances from 5 unique characters** — seed 1 has 3, seeds
3/4/5 have 1 each, seed 2 has 0; Ravi Krishnamurthy is committed in two
seeds, which is why "5 agents" and "6 instances" both appear. The revision
should state both numbers with this definition.

## (f) Are the alive-regime labels arbitrary? Continuous response, bimodal occupancy

The reviewer's deeper worry — if absorption scales smoothly with KL, the
committed/coherent cut is an arbitrary bin — is half right, and conceding the
right half strengthens the paper. Two separable questions:

**Is the response step-like at the boundary? No — it is continuous.** Fitting
PMI shrinkage against end-of-training KL on the 73 alive-region agents
(`dose_response_models.csv`):

| subset | linear R2 (slope/nat) | step-at-KL=1 R2 | step+linear R2 (step coef) | LOOCV linear vs step |
|---|---|---|---|---|
| inner-loop seeds (n=40) | 0.34 (+0.046) | 0.20 | 0.34 (+0.006) | 0.0135 vs 0.0160 |
| pooled (n=73) | 0.16 (+0.029) | 0.13 | 0.17 (+0.051) | 0.0181 vs 0.0186 |
| cross-env (n=33) | 0.07 (+0.018) | 0.09 | 0.10 (+0.090) | 0.0283 vs 0.0255 |

On the seeds — the cleanest subset, single substrate, 5 replicates — the
linear model beats the step model on R2, AIC, and leave-one-out error, and
adding a step term to the linear model contributes nothing (coefficient
+0.006, R2 unchanged at 0.34). Within the committed group alone, higher KL
still predicts more shrinkage (r = 0.45, p = 0.14, n = 12): suggestive and
underpowered, but the direction a continuum predicts and a genuinely discrete
class would not require. Cross-env is ambiguous between the two models (n=33,
7 heterogeneous substrates, R2 <= 0.10 either way) — we do not claim it
discriminates.

**Does that make the labels arbitrary? No — the covariate is bimodal.** A 1-D
Gaussian mixture on log10 KL among alive agents (`kl_bimodality_gmm.csv`)
selects two components on the seeds (BIC 65.6 vs 74.7 for one component) and
pooled (141.7 vs 144.1), with modes at 0.13 / 2.84 nats (seeds) and 0.18 /
2.64 (pooled) — a >14x separation — and weights ~0.85/0.15 and ~0.76/0.24.
The sorted alive KL values show the valley directly: a dense run up to 0.95
nats, then a jump to 1.30, 1.50, 1.55, 1.93… The KL=1.0 threshold sits inside
that low-density gap, which is why k-means recovers the split exactly on the
seeds (ARI = 1.00) and why 86% of coherent agents keep their label across the
entire 2,850-point grid. Caveat to state: the cross-env subset alone prefers
k=1, so the bimodality evidence rests on the inner-loop seed replicates.

**What this means for the taxonomy.** The mechanism is continuous — absorption
per nat of policy movement — and the regimes describe *where training dynamics
deposit agents*: most stay near the reference policy, a minority travel far,
few stop in between. "Committed" names the upper mode of a bimodal outcome
distribution, not a mechanistically distinct phase. This reframing concedes
the reviewer's intuition, keeps every quantitative result (the contrast holds
at all 2,850 settings *because* the modes are well separated), replaces the
vulnerable claim (discrete regimes with mechanistic force) with the defensible
one (continuous dose-response plus bimodal occupancy), and explains the
measured label instability of committed agents (median 0.58) as boundary
geometry rather than a defect in the taxonomy.

Recommended text change: describe the taxonomy as a *descriptive partition of
a continuous, bimodally-occupied space*, and state the dose-response slope
(+0.046 [+0.026, +0.066] nats/token per nat, seeds) as the primary
mechanistic quantity, with regime labels as the readable summary.

## Suggested rebuttal paragraph

> We ran the requested sensitivity analysis as a joint sweep over all five
> classification parameters (KL, both perception-loss cuts, both entropy
> bounds; 2,850 combinations spanning ±50–100% around the paper values). The
> committed-vs-coherent PMI contrast is positive at 100% of grid points in
> both the cross-environment and inner-loop-seed datasets (minima +0.03 and
> +0.13 nats/token). An unsupervised alternative gives the same answer:
> silhouette-selected k-means on (log KL, perception loss, entropy) recovers
> the committed/coherent split exactly on the inner-loop seeds (ARI = 1.0;
> cluster-based contrast p = .016), and a fully threshold-free regression of
> PMI shrinkage on end-of-training KL yields slope +0.046 [+0.026, +0.066]
> nats/token per nat. Individual labels near boundaries do shift (median
> stability 0.8; committed agents trade labels with adjacent alive regimes),
> and we now report per-agent label distributions; no condition-level or
> mechanistic conclusion changes anywhere on the grid. On policy collapse:
> the reviewer is right that high entropy is not near-determinism — in our
> data the high-entropy branch never fires (all 7 policy-collapse agents have
> entropy < 0.2; max observed entropy anywhere is 1.59), and we have
> redefined the regime accordingly. Finally, the Inner Loop committed count:
> 6 committed (run, character) instances from 5 unique characters (one
> character commits in two seeds); we now state both numbers.
>
> We also want to address the deeper version of this concern — whether a
> threshold taxonomy is the right object at all. It is not, quite, and we have
> revised accordingly. PMI shrinkage responds to end-of-training KL
> continuously, not discontinuously at our boundary: on the seed replicates a
> linear model (slope +0.046 [+0.026, +0.066] nats/token per nat) beats a
> step-at-KL=1 model on R2, AIC, and leave-one-out error, and adding a step to
> the linear model contributes nothing (coefficient +0.006). What makes the
> labels non-arbitrary is not a discontinuity in the response but bimodality
> in the covariate: a mixture model on log KL among alive agents selects two
> components (modes 0.13 and 2.84 nats, a 14x separation) with a sparse valley
> where our threshold sits. We now present the taxonomy as a descriptive
> partition of a continuous, bimodally-occupied space — "committed" names the
> upper mode of an outcome distribution, not a distinct mechanistic phase —
> and report the dose-response slope as the primary mechanistic quantity. This
> is also why the contrast is so robust to threshold choice: the modes are far
> apart relative to any reasonable perturbation of the cut.

## Files

- `fig_pmi_contrast_grid.pdf/png` — ECDF of the contrast over the full grid +
  marginal sweeps for each of the five thresholds (entropy panels new).
- `fig_label_stability.pdf/png` — per-agent label-stability distributions.
- `fig_clustering_pmi.pdf/png` — cluster assignments and shrinkage-by-cluster.
- `fig_dose_response.pdf/png` — shrinkage vs KL with linear and step fits
  (left) and the log-KL marginal with the 2-component GMM (right).
- `pmi_contrast_grid.csv`, `composition_grid.csv`,
  `label_stability_per_agent.csv`, `clustering_*.csv`,
  `policy_collapse_agents.csv`, `dose_response_models.csv`,
  `kl_bimodality_gmm.csv`.

Regenerate everything with
`.venv/bin/python scripts/sweep_regime_thresholds_3d.py` (add `--dose-only`
for just the dose-response/bimodality section).
