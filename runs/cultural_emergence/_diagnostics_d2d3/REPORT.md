# Diagnostics for the trajectory-KL regime reinterpretation

Three diagnostics testing whether the trajectory-KL finding's *regime reinterpretation* ("committed agents converge across seeds, coherent agents diverge") is a substantive policy-level statement or a realisation-level artefact. Run on the three multi-seed datasets (Inner Loop 8-agent, Margin Notes 8-agent, Ashbourne GC 16-agent).

**Bottom line.** The headline cultural-divergence finding is **strengthened** — it now holds at the pure policy level, independent of which trajectories were sampled. But the **regime reinterpretation does not survive**: the strong negative correlation between per-character KL-from-base and cross-population contribution (r ≈ −0.77 at the realisation level) collapses to near-zero / slightly positive at the policy level. The negative per-character contributions that drove the reinterpretation are substantially artefacts of policy-collapse/committed-agent overconfidence. This lands in the spec's **third outcome**.

---

## Diagnostic 1 — coherent-only headline trajectory ([_diagnostics_d1_coherent_only/](../_diagnostics_d1_coherent_only/))

Re-aggregation of the existing trajectory-KL shards with per-character regime filtering. Three variants: `full` (all chars), `source_coherent` (chars coherent in the source population), `strict_coherent` (chars coherent in both source and target populations).

| Dataset | iter | full | source_coherent | strict_coherent |
|---|---:|---:|---:|---:|
| Inner Loop 8a | 25 | 0.0014 | 0.0020 | 0.0011 |
| | 100 | 0.147 | **0.207** | 0.065 |
| Margin Notes 8a | 25 | 0.0012 | 0.0005 | 0.0003 |
| | 150 | 0.287 | **0.342** | 0.121 |
| Ashbourne 16a | 25 | 0.0081 | 0.0068 | 0.0035 |
| | 150 | 0.331 | **0.439** | 0.184 |

**Findings:**
- **`source_coherent` strengthens the signal 20–40%** over the full-population headline at end of training, in all three datasets. Restricting to source characters that didn't drift far from base *increases* the cross-population divergence, because it removes the negative contributions of committed/collapsed source agents.
- **iter-25 baseline stays ≈0** across all three filter variants — the clean-baseline property of the construction is regime-independent.
- **`strict_coherent` is smaller than `full`**: requiring both source and target to be coherent restricts to pairs of low-drift adapters, which are closer to each other (and to base), so the divergence between them is smaller. This is the within-coherent-cluster divergence — a different quantity, not the headline.

The cleanest defensible headline is the **source-coherent** trajectory: it is larger, smoother, and free of the negative-contribution noise. Figure: [fig_d1_coherent_only.png](../_diagnostics_d1_coherent_only/fig_d1_coherent_only.png).

---

## Diagnostic 2 — entropy of negative-contribution agents ([fig_d2_scatter.png](fig_d2_scatter.png))

For each (seed, character), own-adapter entropy vs mean sibling-seed-adapter entropy, measured at the agent's own action-token positions. Negative `ent_diff` = own adapter is narrower (more confident) than siblings → the overconfidence that produces artefactual negative trajectory-KL contributions.

**Per-regime mean entropy difference (own − sibling) at end of training:**

| Regime | mean ent_diff (nats) | n |
|---|---:|---:|
| coherent_low_kl | **−0.001** | 89 |
| committed | −0.077 | 13 |
| mid_range | −0.058 | 32 |
| text_degenerate | −0.104 | 5 |
| policy_collapse | **−0.212** | 5 |

**Findings:**
- **Coherent agents have ent_diff ≈ 0** — no overconfidence artefact. Their contributions to cross-population KL are substantive.
- **Policy-collapse agents have the largest negative ent_diff (−0.21)** — exactly as predicted; the regime is *defined* by low policy entropy. Their extreme negative trajectory-KL contributions are largely artefactual.
- **`corr(kl_pertok, ent_diff) = +0.55`** (pooled, n=144): the more negative an agent's per-character trajectory-KL contribution, the more negative its entropy difference. The two extreme Margin Notes seed-1 outliers confirm this directly: Kai Dempsey (kl_pertok = −9.49) has own entropy 0.15 vs sibling 0.90 (ent_diff = −0.75); Renata Solberg (kl_pertok = −7.47) has ent_diff = −0.13.

**Conclusion:** the strongly-negative per-character contributions that dragged down the full-population trajectory KL are substantially an entropy-overconfidence artefact concentrated in the policy-collapse and text-degenerate regimes. These regimes should be carved out of any substantive cultural-divergence claim. Coherent agents are clean.

---

## Diagnostic 3 — direct policy-distance measurement ([fig_d3_trajectory.png](fig_d3_trajectory.png), [fig_d3_scatter.png](fig_d3_scatter.png))

Symmetric KL between every pair of seeds' per-character adapters, evaluated at matched contexts (action-token positions in held-out episodes), marginalising over the realisation. This is the policy-level distance, independent of what was sampled.

**Policy distance grows over training (baseline → end):**

| Dataset | iter-25 | end iter | end value |
|---|---:|---:|---:|
| Inner Loop 8a | 0.0092 | 100 | 0.585 |
| Margin Notes 8a | 0.0069 | 150 | 1.359 |
| Ashbourne 16a | 0.0503 | 150 | 1.349 |

**Finding 1 — the headline strengthens.** Policy distance grows from ≈0 at iter 25 to large values at end (×60–×200), in all three datasets, with no plateau. This is a *cleaner* statement of the cultural-divergence finding than the trajectory KL: it shows the **policies themselves** diverge across seeds, not merely the sampled trajectories. The realisation-independent measurement confirms the phenomenon.

**Finding 2 — the regime reinterpretation does NOT survive.** Correlation between per-character end-of-training KL-from-base and policy distance:

| Dataset | r (policy-level) | r (realisation-level, prior) |
|---|---:|---:|
| Inner Loop 8a | **+0.23** | −0.766 |
| Margin Notes 8a | **−0.01** | −0.057 |
| Ashbourne 16a | **+0.37** | −0.779 |

The strong negative correlation at the realisation level (r ≈ −0.77 in Inner Loop and Ashbourne) **collapses to near-zero or slightly positive at the policy level**. At the policy level, if anything, higher-KL agents are *slightly more* distant across seeds (positive r) — the intuitive direction (agents that moved more from base have more room to differ), and the *opposite* sign to the realisation-level correlation. The "committed agents converge in policy space" reading is not supported once the realisation-level artefacts are removed.

---

## Cross-diagnostic synthesis ([fig_cross_diagnostic.png](fig_cross_diagnostic.png))

Pooled across all (dataset, seed, character), n = 144:

| correlation | value | reading |
|---|---:|---|
| kl_pertok ↔ ent_diff | **+0.55** | negative trajectory-KL contributions ↔ own-adapter overconfidence (artefact) |
| kl_pertok ↔ policy_dist | −0.09 | realisation-level contribution barely related to actual policy distance |
| end_kl ↔ policy_dist | +0.19 | weak positive (more drift → slightly more policy distance) |
| policy_dist ↔ ent_diff | −0.12 | weak |

The three quantities do **not** align in the way the substantive convergence reading predicted. Instead: per-character trajectory-KL contribution is governed largely by entropy overconfidence (r = +0.55 with ent_diff), not by actual inter-seed policy distance (r = −0.09). The realisation-level regime correlation was a shadow of the entropy-artefact, not a substantive policy-level convergence.

---

## What the paper should claim

Following the spec's outcome map, this is the **third outcome**, with one upgrade:

1. **The cultural-divergence finding stands and is now stronger.** Cross-population divergence grows monotonically over training, replicated across three substrates, with a clean ≈0 baseline — and it holds at the *policy level* (Diagnostic 3), not just the realisation level. The realisation-independent policy-distance measurement is the cleanest version and should be cited as primary, with the trajectory KL as its realisation-level shadow.

2. **Drop the regime reinterpretation.** The "committed agents converge across seeds, coherent agents diverge" story was a realisation-level artefact. At the policy level the correlation with per-character KL-from-base is near-zero or slightly positive. The honest statement: *cross-population divergence develops over training and is distributed across character slots in ways that do not reduce to — and are not inversely predicted by — per-character KL from base; the realisation-level inverse correlation reflects entropy-overconfidence in collapsed/committed agents rather than substantive policy convergence.*

3. **Carve out the failure regimes.** Policy-collapse and text-degenerate agents contribute artefactual negatives (Diagnostic 2). Headline claims should use the source-coherent restriction (Diagnostic 1), which is both larger and clean.

4. **Suggested headline figure:** the D3 policy-distance trajectory ([fig_d3_trajectory.png](fig_d3_trajectory.png)) as the primary cultural-divergence result, with the D1 source-coherent trajectories as the realisation-level corroboration.

---

## Caveats

- Policy distance is evaluated at action-token positions in held-out episodes pooled across all five seeds' contexts; per-(char, iter) values average over ~5 sources × 4 episodes. This is the broad-context version; a single-canonical-source version would give similar trajectories with slightly different absolute values.
- D3 correlations use end-of-training iters that differ across datasets (IL=100, MN/Ash=150). The sign and magnitude are stable across iters (the policy-distance vs end-KL relationship is weak/positive at every measured iter, not just at the end).
- The two extreme Margin Notes seed-1 collapse outliers dominate the most-negative tail of kl_pertok; their removal further flattens the realisation-level correlation, reinforcing that it was artefact-driven.

## Inputs

- D1: [`_diagnostics_d1_coherent_only/`](../_diagnostics_d1_coherent_only/) — `combined_d1_summary.csv`, `fig_d1_coherent_only.png`
- D2+D3 raw shards: [`_policy_distance/4826830/`](../_policy_distance/4826830/) (48 110 rows)
- D2: `d2_entropy_diff_by_regime.csv`, `d2_joined_kl_pertok_entropy.csv`, `fig_d2_scatter.png`
- D3: `d3_policy_dist_per_iter.csv`, `d3_end_policy_dist_with_regime.csv`, `fig_d3_trajectory.png`, `fig_d3_scatter.png`
- Cross: `cross_diagnostic.csv`, `fig_cross_diagnostic.png`
- Scripts: [`analyse_trajectory_kl_coherent_only.py`](../../../scripts/analyse_trajectory_kl_coherent_only.py), [`analyze_policy_distance.py`](../../../scripts/analyze_policy_distance.py), [`analyse_policy_distance.py`](../../../scripts/analyse_policy_distance.py)
