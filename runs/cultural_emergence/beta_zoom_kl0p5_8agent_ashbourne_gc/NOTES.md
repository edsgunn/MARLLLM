# beta_zoom_kl0p5_8agent_ashbourne_gc

*cultural emergence — a finer `beta` scan at fixed `kl_coef = 0.5`. **The family
is the unit of inference.***

|  |  |
|---|---|
| Configs | [`configs/cultural_emergence/beta_zoom_kl0p5_8agent_ashbourne_gc/`](../../../configs/cultural_emergence/beta_zoom_kl0p5_8agent_ashbourne_gc/) |
| Model | `Qwen2.5-7B-Instruct` |
| Population | 8 agents on `study_group_ashbourne_gc` |
| Grid | `beta` ∈ {0.05, 0.1, 0.2, 0.3} at `kl_coef` 0.5 |
| Ran | all four cells to iteration 300 |
| **Differs from** | **[`../kl_beta_sweep_8agent_ashbourne_gc/`](../kl_beta_sweep_8agent_ashbourne_gc/NOTES.md)** — same substrate and rig; fills in the `beta` gap between 0.01 and 0.5 at one KL coefficient. |

## What it was for

The coarse sweep left a factor-of-fifty gap in `beta` between the last safe
value (0.01) and the first catastrophic one (0.5), with only one point (0.1) in
between. This scan asks where the cliff actually is.

## Headline

**The cliff is between `beta` 0.05 and 0.2, and it is a cliff, not a slope.**
At 0.05 the population ends 0.75 alive with perception loss 1.75 — the best
result on this substrate anywhere in the campaign. At 0.2 it is 0.00 alive with
perception loss 2.87 and entropy 1.92. At 0.3 perception loss is 4.44 and
entropy 3.33. Two doublings of `beta` take the substrate from its best outcome
to complete failure.

## Core metrics

| `beta` | perc first | perc last | entropy last | KL last | alive @200 | alive @300 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 2.18 | **1.75** | 0.78 | 0.49 | 1.00 | 0.75 |
| 0.10 | 2.19 | 2.16 | 0.91 | 0.59 | 0.88 | 0.00 |
| 0.20 | 2.16 | 2.87 | 1.92 | 0.50 | 0.25 | 0.00 |
| 0.30 | 2.19 | 4.44 | 3.33 | 0.64 | 0.00 | 0.00 |

Entropy is the variable that moves: 0.78 → 0.91 → 1.92 → 3.33 across the scan,
while KL from base stays pinned near 0.5 by the KL anchor in every cell. **The
entropy bonus destroys populations without moving them away from the base
policy** — which is why the KL anchor cannot protect against it.

## The duplicate

The `beta = 0.1` cell here is the same configuration as the `(0.5, 0.1)` cell of
the coarse sweep — identical but for run name, output directory and GPU. It
ended at alive 0.00 where the sweep's copy ended at 0.875. The full comparison
is in [`../kl_beta_sweep_8agent_ashbourne_gc/NOTES.md`](../kl_beta_sweep_8agent_ashbourne_gc/NOTES.md);
the consequence for this scan is that **the 0.05-vs-0.1 step is inside the
noise and only the 0.05-vs-0.2 step is safe to read.**

## What this settles

- **The `beta` cliff sits below 0.2.** Every cell at 0.2 or above ends with
  alive 0.00 and entropy above 1.9. — *established*
- **The entropy bonus and the KL anchor act on different things.** All four
  cells hold KL from base at 0.46–0.64 while entropy ranges over a factor of
  four. — *established*, but see the correction below: they act on different
  *outcomes*, not on different *quantities*.
- **`beta = 0.05` at `kl_coef = 0.5` is the best-performing setting found on
  Ashbourne GC.** Perception loss 1.75, alive 1.00 at iteration 200. —
  *provisional*, single run, and see the duplicate above

## What it does not settle

Whether `beta = 0.05` beats `beta = 0.01`, which is the campaign default: the
two were never run against each other at the same `kl_coef`, and the run-to-run
spread would swamp the difference if they had been.

## Correction, 2026-09-08: the anchor does fight the bonus

This scan's headline claim — that **"the entropy bonus destroys populations
without moving them away from the base policy, which is why the KL anchor cannot
protect against it"** — was drawn from four cells at a single `kl_coef`, and
**with `kl_coef` varied it does not hold.** Re-reading the coarse sweep together
with this scan
([`../kl_beta_sweep_8agent_ashbourne_gc/NOTES.md`](../kl_beta_sweep_8agent_ashbourne_gc/NOTES.md),
§4 of the re-reading), the change in entropy at `beta` = 0.10 is:

| `kl_coef` | 0.00 | 0.01 | 0.10 | 0.50 |
|---|---:|---:|---:|---:|
| Δ entropy, first-25 → last-25 | +1.04 | +0.92 | +0.79 | +0.16 |

Monotone over four points, with this family's `beta` = 0.10 cell and the coarse
sweep's duplicate of it agreeing to 0.01. The anchor suppresses the entropy rise
sixfold. Both terms act only at ACT positions, so they compete for the same
distribution.

What survives from the original claim is narrower and still true: **the anchor
cannot protect at `beta` ≥ 0.20.** This family's (0.5, 0.2) cell ends at
Δentropy +1.19 and Δperception +0.71 despite the strongest anchor on the grid.
The cliff is real; it is the *reason* given for it that was wrong.

Also: this family's `alive` figures should not be cited without the delivery rate
beside them. The `beta` = 0.05 cell's "best result in the campaign" (alive 0.75,
perception 1.75) ends at delivery 0.474 — the channel had closed by half, like
every other cell in both families.

## Status

`CLOSED`. Cite the cliff. The best-cell claim is **withdrawn** — it rests on
`alive` at delivery 0.474. The mechanism claim is **corrected** above.

---
*Analysed retrospectively on 2026-09-01, corrected on 2026-09-08, from
`metrics.jsonl` and `config.json`.
Machine output stays local.*
