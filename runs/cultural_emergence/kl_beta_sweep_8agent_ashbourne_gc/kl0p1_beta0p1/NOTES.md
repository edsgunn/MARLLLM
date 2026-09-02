# kl0p1_beta0p1

*one cell of the 4 × 4 regularisation sweep — `kl_coef` 0.1, `beta` 0.1*

|  |  |
|---|---|
| Config | [`configs/cultural_emergence/kl_beta_sweep_8agent_ashbourne_gc/kl0p1_beta0p1.yaml`](../../../../configs/cultural_emergence/kl_beta_sweep_8agent_ashbourne_gc/kl0p1_beta0p1.yaml) |
| Model | `Qwen2.5-7B-Instruct` |
| Population | 8 agents on `study_group_ashbourne_gc` |
| Settings | kl_coef 0.1, β 0.1, lr 5e-06, LoRA r 64 |
| Ran to | iteration 300 of 300 · 12 checkpoints |
| **Differs from** | **the other cells of this family — see the family note** |

## What it was for

A single cell of the 4 × 4 regularisation sweep. **The inference lives in the family note, not here**
— one population per cell cannot separate a cell from a seed on this substrate,
and this project has a direct demonstration that it cannot
([`../NOTES.md`](../NOTES.md)).

## Headline

Perception loss 2.16 → 2.46, policy entropy ending at
1.55, alive 0.88 at iteration 200 and
0.00 at 300.

Read this cell only against the rest of its family: [`../NOTES.md`](../NOTES.md).

## Core metrics

Population means over the first and last 10% of logged iterations (blocks of 30 of 300):

| metric | first block | last block |
|---|---:|---:|
| perception loss | 2.161 | 2.464 |
| KL from base | 0.017 | 1.450 |
| policy entropy | 0.745 | 1.545 |

Regime composition (paper thresholds; see `PROCESS.md`):

| iteration | n | alive | composition |
|---:|---:|---:|---|
| ~25 | 8 | 0.00 | text_degenerate 8 |
| ~100 | 8 | 1.00 | coherent_low_kl 4, committed 1, mid_range 3 |
| ~150 | 8 | 0.88 | mid_range 7, text_degenerate 1 |
| ~200 | 8 | 0.88 | coherent_low_kl 2, mid_range 5, text_degenerate 1 |
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
