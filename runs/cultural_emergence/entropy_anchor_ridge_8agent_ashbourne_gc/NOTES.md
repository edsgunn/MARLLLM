# entropy_anchor_ridge_8agent_ashbourne_gc

*cultural emergence — does a stronger KL anchor raise the safe ceiling on the
entropy bonus? **The family is the unit of inference.*** **RUNNING.**

|  |  |
|---|---|
| Configs | [`configs/cultural_emergence/entropy_anchor_ridge_8agent_ashbourne_gc/`](../../../configs/cultural_emergence/entropy_anchor_ridge_8agent_ashbourne_gc/) |
| Model | `Qwen2.5-7B-Instruct` |
| Population | 8 agents on `study_group_ashbourne_gc`, `extended_8` roster |
| Grid | (`kl_coef`, `beta`) ∈ {(2.0, 0.01), (2.0, 0.2), (2.0, 0.4), (5.0, 0.4)} × seeds {1, 2} |
| Held fixed | lr 5e-6, LoRA r64, 16 episodes/iter, 300 iterations, γ 0.99, `normalise_returns` true, α 1/1/1, `return_sign` +1 |
| Submitted | 2026-09-08, jobs 6404722–6404729, 08:00:00 each |
| **Differs from** | **[`../kl_beta_sweep_8agent_ashbourne_gc/`](../kl_beta_sweep_8agent_ashbourne_gc/NOTES.md)** — same substrate and rig; extends `kl_coef` beyond 0.5 for the first time, and gives each cell two seeds. |

## The question

Can policy entropy be made to rise without the perception loss collapsing?

## Why this grid, and not another sweep of the old one

The 20 existing cells were re-read on 2026-09-08
([the sweep's NOTES](../kl_beta_sweep_8agent_ashbourne_gc/NOTES.md), and
[`scripts/analyse_kl_beta_grid.py`](../../../scripts/analyse_kl_beta_grid.py)).
That re-reading answered the question **negatively** for everything already run,
and in doing so identified the one direction left untested:

- The entropy bonus is a **step function**. `beta` ≤ 0.05 moves entropy by at
  most 0.12 in either direction (9 cells); `beta` ≥ 0.10 moves it by at least
  +0.16 (11 cells). Complete separation, MWU exact p = 1.2 × 10⁻⁵. No `beta`
  raises entropy a little.
- Above the step, **entropy and perception loss are locked**: Spearman
  Δentropy vs Δperception = +0.853, p = 4.4 × 10⁻¹², n = 20.
- **But the anchor fights the bonus.** At `beta` = 0.10, Δentropy falls
  +1.04 → +0.92 → +0.79 → +0.16 as `kl_coef` goes 0 → 0.01 → 0.10 → 0.50.
  Monotone over four points; the duplicate pair agrees to 0.01.

If the anchor is what stops a large bonus from wrecking the text, the safe
`beta` should rise with `kl_coef`, and the target regime should sit on a diagonal
in (`kl_coef`, `beta`). `beta` = 0.2 already fails at `kl_coef` = 0.5 (Δentropy
+1.19, Δperception +0.71). **`kl_coef` above 0.5 has never been run.**

## Pre-registered readings

| | criterion |
|---|---|
| **Target regime** | Δentropy ≥ +0.30 **and** Δperception ≤ 0, on **both** seeds of a cell |
| **Ridge confirmed** | the target regime at `kl_coef` 2.0 or 5.0 where the same `beta` failed at 0.5 |
| **Ridge refuted** | every cell either collapses (Δperception > +0.4 on both seeds) or is pinned (Δentropy < +0.16) |

Ridge refuted is a real outcome, not a null: it would mean the anchor only ever
*suppresses* the bonus rather than licensing it, and that the uniform entropy
bonus is simply the wrong instrument for raising entropy on this substrate. That
closes the question.

## The cells

| cell | `kl_coef` | `beta` | role |
|---|---:|---:|---|
| `anchor_kl2p0_beta0p01` | 2.0 | 0.01 | **control.** `kl_coef` 2.0 is 4× the strongest anchor ever run, at the campaign-default bonus. Without it nothing at this `kl_coef` is attributable to `beta`. |
| `ridge_kl2p0_beta0p2` | 2.0 | 0.2 | **the key test.** This `beta` fails at `kl_coef` 0.5. |
| `ridge_kl2p0_beta0p4` | 2.0 | 0.4 | does the ceiling keep rising? Between the grid's 0.3 and 0.5, both catastrophic everywhere tried. |
| `ridge_kl5p0_beta0p4` | 5.0 | 0.4 | the extreme corner. If (2.0, 0.4) fails and this survives, the ridge has a slope. |

## Why two seeds per cell, and what cannot be read from one

The old grid contains an accidental exact duplicate — `kl0p5_beta0p1` was run
twice at the **same seed** — which calibrates the rig's noise floor:

| | Δ entropy | Δ perception | delivery |
|---|---:|---:|---:|
| spread, same config, same seed | 0.007 | **0.431** | **0.173** |

So **Δentropy is the low-noise observable** and carries the primary claim from a
single run. Δperception and delivery do not: differences below ~0.4 and ~0.17
respectively are unreadable from one run. Hence two seeds, and hence the target
regime is defined on *both* seeds.

## What this family will not settle

**The channel.** Delivery fell 0.98 → 0.50 in all 20 cells of the old grid, with
no trend in `beta` (ρ = −0.25, p = 0.28) or `kl_coef` (ρ = −0.16, p = 0.49) and
not one cell above 0.8. Regularisation does not reach the dark-room collapse
across a fifty-fold range in both knobs. Nothing here is expected to fix
delivery, and a delivery difference under 0.17 means nothing.

## How to read it when it lands

On Δentropy and Δperception, iterations 1–25 vs the last 25, both seeds shown,
with the well-formed post rate beside them. **Do not read `alive`**: across the
20 old cells it is uncorrelated with delivery (ρ = −0.10, p = 0.66) and it has
now reversed four readings.

Use `scripts/analyse_kl_beta_grid.py` for the window and the statistics; it will
need its `FAMILIES` dict pointed at this directory.

## Status

`RUNNING`. Nothing to cite yet.
