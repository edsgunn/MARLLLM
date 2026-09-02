# B3 — substrate-relative perception thresholds

*Does `perception loss ≥ 2.0` measure breakdown, or substrate difficulty?*

|  |  |
|---|---|
| Script | [`scripts/tier_b/b3_substrate_relative_regimes.py`](../../../scripts/tier_b/b3_substrate_relative_regimes.py) |
| Inputs | iteration-1 records from all 108 live runs (LoRA at init, KL from base = 0) |
| Ran | 2026-09-01, no compute |

## What it was for

The classifier applies `perc ≥ 2.0 → text_degenerate` and `perc < 1.7 →
coherent` on every substrate. Those cuts were set by visual inspection at
iteration 200 on the Study Group substrates. This measures each substrate's
*untrained* perception loss and asks whether the cuts mean the same thing
everywhere.

## Headline

**On two substrates the untrained model is already classified degenerate.**

| substrate | untrained perception loss | runs | vs the cuts |
|---|---:|---:|---|
| `robotic_athanor` | **2.442** | 3 | above the 2.0 degenerate cut |
| `study_group_ashbourne_gc` | **2.196** | 46 | above the 2.0 degenerate cut |
| `ashbourne_school` | 1.588 | 6 | below both |
| `margin_notes` | 1.521 | 8 | below both |
| `study_group_strathearn_server` | 1.508 | 5 | below both |
| `conjecture_inn` | 1.434 | 2 | below both |
| `study_group` | 1.415 | 5 | below both |
| `study_group_strathearn` | 1.389 | 2 | below both |
| `inner_loop` | 1.351 | 6 | below both |

The spread is 1.09 nats — larger than the gap between the two cuts. So
`text_degenerate` conflates *has broken* with *never learned*, and which one it
means depends on which substrate you are on.

This is exactly the Robotic Athanor case. The paper reports "all eight agents
classified text-degenerate at every measured iteration" and reads it as degraded
action distributions. Robotic Athanor's untrained perception loss is 2.442; the
population's is 2.18 → 2.12 across 100 iterations. **They were below the
untrained baseline the whole time.**

## Core metrics

Reclassifying with `degenerate ≥ baseline`, `coherent < 0.85 × baseline`:

- per-agent label disagreement between conventions: **1,134 / 2,268 = 50.0%**
- at iteration 25, runs reading as 0% alive: **40/79 absolute, 17/79 relative**

The affected cells are almost all `study_group_ashbourne_gc` and
`robotic_athanor`, and almost all at early iterations — 32 runs flip from 0%
alive to 0.12–1.00 alive at iteration 25.

## What this settles

- **The absolute perception cut is not substrate-neutral.** Two of nine
  substrates start above the degenerate threshold. — *established*
- **`text_degenerate` at early iterations on those substrates means "has not
  learned yet".** — *established*
- **Half of all regime labels move under a differently-anchored convention.** —
  *established*

## What it does not settle

**Which convention is right.** The relative one is better named; the 0.85 factor
is as arbitrary as 1.7 was. The value here is the diagnosis and its size, not a
new set of blessed cuts. Choosing them properly is a calibration exercise this
does not attempt.

Note also the interaction with [B7](../b7_absorption_horizon/NOTES.md): the
window where the absolute cuts mislabel is the early window, which is exactly
the window a horizon-restricted absorption analysis has to use.

## Status

`CLOSED` as a diagnosis. The classifier has **not** been changed; every other
Tier-B analysis still uses the paper thresholds so it stays comparable.

---
*Analysed 2026-09-01. Machine output stays local.*
