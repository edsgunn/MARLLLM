# kl0p5_beta0p2

*one cell of the `beta` zoom at `kl_coef` 0.5 — `beta` 0.2 at `kl_coef` 0.5*

|  |  |
|---|---|
| Config | [`configs/cultural_emergence/beta_zoom_kl0p5_8agent_ashbourne_gc/kl0p5_beta0p2.yaml`](../../../../configs/cultural_emergence/beta_zoom_kl0p5_8agent_ashbourne_gc/kl0p5_beta0p2.yaml) |
| Model | `Qwen2.5-7B-Instruct` |
| Population | 8 agents on `study_group_ashbourne_gc` |
| Settings | kl_coef 0.5, β 0.2, lr 5e-06, LoRA r 64 |
| Ran to | iteration 300 of 300 · 12 checkpoints |
| **Differs from** | **the other cells of this family — see the family note** |

## What it was for

A single cell of the `beta` zoom at `kl_coef` 0.5. **The inference lives in the family note, not here**
— one population per cell cannot separate a cell from a seed on this substrate,
and this project has a direct demonstration that it cannot
([`../NOTES.md`](../NOTES.md)).

## Headline

Perception loss 2.16 → 2.87, policy entropy ending at
1.92, alive 0.25 at iteration 200 and
0.00 at 300.

Read this cell only against the rest of its family: [`../NOTES.md`](../NOTES.md).

## Core metrics

Population means over the first and last 10% of logged iterations (blocks of 30 of 300):

| metric | first block | last block |
|---|---:|---:|
| perception loss | 2.159 | 2.871 |
| KL from base | 0.015 | 0.504 |
| policy entropy | 0.740 | 1.921 |

Regime composition (paper thresholds; see `PROCESS.md`):

| iteration | n | alive | composition |
|---:|---:|---:|---|
| ~25 | 8 | 0.00 | text_degenerate 8 |
| ~100 | 8 | 1.00 | coherent_low_kl 2, committed 1, mid_range 5 |
| ~150 | 8 | 1.00 | mid_range 8 |
| ~200 | 8 | 0.25 | mid_range 2, text_degenerate 6 |
| ~300 | 8 | 0.00 | text_degenerate 8 |

## What this settles

- **Nothing on its own.** A cell is one draw.

## What it does not settle

Its own rank. See the family note for what the family does and does not
support.

## Status

`CLOSED` at iteration 300 as part of its family.

---
*Analysed retrospectively on 2026-09-01 from `metrics.jsonl` and `config.json`; see the note in `PROCESS.md` on what a back-filled note cannot tell you. Machine output stays local.*
