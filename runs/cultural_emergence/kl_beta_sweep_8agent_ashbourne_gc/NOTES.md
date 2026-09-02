# kl_beta_sweep_8agent_ashbourne_gc

*cultural emergence — the 4 × 4 regularisation sweep. **The family is the unit
of inference, not the cell.***

|  |  |
|---|---|
| Configs | [`configs/cultural_emergence/kl_beta_sweep_8agent_ashbourne_gc/`](../../../configs/cultural_emergence/kl_beta_sweep_8agent_ashbourne_gc/) |
| Model | `Qwen2.5-7B-Instruct` |
| Population | 8 agents on `study_group_ashbourne_gc`, `extended_8` roster |
| Grid | `kl_coef` ∈ {0, 0.01, 0.1, 0.5} × `beta` ∈ {0, 0.01, 0.1, 0.5} |
| Held fixed | lr 5e-6, LoRA r64, 16 episodes/iter, seed 1, 300 iterations |
| Ran | all 16 cells to iteration 300 |
| **Differs from** | **`run7_8agent_7B_study_group_ashbourne_gc`** — that run is the (0.05, 0.01) point, which is *not* on this grid. |

## What it was for

Both regularisers were set once, early, and never justified: `kl_coef` 0.05
anchors the policy to the frozen base, `beta` 0.01 adds an entropy bonus. The
sweep asks which of them is load-bearing and whether the campaign's default sits
anywhere sensible.

## Headline

**The entropy bonus is the dangerous knob and the default was set an order of
magnitude too high on it — in the safe direction, by luck.** Every cell with
`beta ≥ 0.1` ends degenerate; every cell with `beta = 0.5` ends catastrophically
so, with policy entropy above 6.5 nats and perception loss above 5.2. The KL
anchor, by contrast, does what it is supposed to and nothing more: raising it
from 0 to 0.5 pulls end-of-training KL from base down from ~1.2 to ~0.5 and
changes the alive fraction hardly at all.

**But the sweep cannot rank its own cells**, for a reason discovered by
accident — see below.

## Core metrics

Population means over the first and last 10% of the 300 logged iterations, and
the alive fraction at iterations 200 and 300:

| `kl_coef` | `beta` | perc first | perc last | entropy last | KL last | alive @200 | alive @300 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 0.0 | 2.18 | 2.15 | 0.70 | 1.22 | 0.88 | 0.50 |
| 0.0 | 0.01 | 2.18 | 1.97 | 0.73 | 1.50 | 0.88 | 0.50 |
| 0.0 | 0.1 | 2.17 | 2.56 | 1.80 | 1.75 | 0.12 | 0.00 |
| 0.0 | 0.5 | 2.17 | 5.69 | 6.79 | 1.97 | 0.00 | 0.00 |
| 0.01 | 0.0 | 2.15 | 1.97 | 0.77 | 1.24 | 0.88 | 0.75 |
| 0.01 | 0.01 | 2.17 | 2.20 | 0.80 | 1.22 | 0.88 | 0.00 |
| 0.01 | 0.1 | 2.18 | 2.56 | 1.67 | 1.38 | 0.62 | 0.00 |
| 0.01 | 0.5 | 2.17 | 5.67 | 6.87 | 2.26 | 0.00 | 0.00 |
| 0.1 | 0.0 | 2.16 | **1.65** | 0.62 | 0.95 | 0.88 | **0.88** |
| 0.1 | 0.01 | 2.18 | 1.97 | 0.75 | 0.91 | 0.88 | 0.50 |
| 0.1 | 0.1 | 2.16 | 2.46 | 1.55 | 1.45 | 0.88 | 0.00 |
| 0.1 | 0.5 | 2.16 | 5.23 | 6.63 | 2.13 | 0.00 | 0.00 |
| 0.5 | 0.0 | 2.20 | 2.10 | 0.68 | 0.48 | 0.88 | 0.12 |
| 0.5 | 0.01 | 2.17 | 1.95 | 0.70 | 0.60 | 0.88 | 0.75 |
| 0.5 | 0.1 | 2.18 | **1.71** | 0.91 | 0.46 | 1.00 | **0.88** |
| 0.5 | 0.5 | 2.17 | 6.42 | 6.54 | 1.02 | 0.00 | 0.00 |

Every cell starts at perception loss 2.16–2.20. That is the untrained model on
this substrate, and it is above the `text_degenerate` threshold of 2.0 — so a
cell that ends near 2.2 has not degraded, it has failed to learn.

## The accident that limits what this sweep can say

The `(0.5, 0.1)` cell was run **twice**: once here and once as part of
[`../beta_zoom_kl0p5_8agent_ashbourne_gc/`](../beta_zoom_kl0p5_8agent_ashbourne_gc/NOTES.md).
The two configs differ only in the run name, the output directory, and which of
the four GPUs the job landed on. Same seed, same hyperparameters, same code,
300 clean iterations each with no resume.

They agree closely to about iteration 150 and then separate:

| iterations | sweep cell | beta-zoom cell |
|---|---:|---:|
| 0–50 | 2.148 | 2.153 |
| 50–100 | 1.873 | 1.938 |
| 100–150 | 1.671 | 1.790 |
| 150–200 | 1.573 | 1.777 |
| 200–250 | 1.629 | 1.943 |
| 250–300 | **1.676** | **2.106** |

Alive at iteration 300: **0.875 against 0.000**. One of the two best cells on
this grid and one of its complete failures are the *same configuration*.

That is not a criticism of either run. It means the run-to-run spread at fixed
seed on this substrate is as large as the spread across most of the grid, so
**no ranking among the surviving cells is supported by one run each.** What
survives is the part of the signal that is much larger than that spread: the
`beta` effect.

## What this settles

- **`beta ≥ 0.1` destroys the population at every KL coefficient tried.** 8/8
  cells with `beta ≥ 0.1` end with alive 0.00, against 1/8 for `beta ≤ 0.01`.
  — *established*
- **`beta = 0.5` is catastrophic, not merely bad.** Entropy ends at 6.5–6.9 nats
  and perception loss at 5.2–6.4, five to eight times any healthy run. —
  *established*
- **The KL coefficient controls drift and little else.** End-of-training KL from
  base falls monotonically with `kl_coef` (≈1.2 → 1.2 → 0.9 → 0.5) while alive
  at iteration 200 is 0.88 in eleven of the twelve `beta ≤ 0.1` cells. —
  *likely*
- **This pipeline is not reproducible at fixed seed.** Two runs of one identical
  configuration ended at alive 0.875 and 0.000. — *established*, one direct
  demonstration
- **The campaign's default `beta = 0.01` sits inside the safe band**, and its
  `kl_coef = 0.05` sits between two grid points that both behave. — *likely*

## What it does not settle

**Which cell is best.** The two best cells on end-of-training perception loss —
`(0.1, 0.0)` at 1.65 and `(0.5, 0.1)` at 1.71 — differ by less than the gap
between the two runs of `(0.5, 0.1)`. Ranking the survivors needs replicates,
which is what
[`../kl_beta_seed_replicates_8agent_ashbourne_gc/`](../kl_beta_seed_replicates_8agent_ashbourne_gc/NOTES.md)
was built for and only partly delivers.

Also unsettled: whether `beta` between 0.01 and 0.1 has a usable optimum. The
zoom was run only at `kl_coef = 0.5`.

## Status

`CLOSED` as a screen. Cite the `beta` result; do not cite a cell ranking.

---
*Analysed retrospectively on 2026-09-01 from `metrics.jsonl` and `config.json`.
Machine output stays local.*
