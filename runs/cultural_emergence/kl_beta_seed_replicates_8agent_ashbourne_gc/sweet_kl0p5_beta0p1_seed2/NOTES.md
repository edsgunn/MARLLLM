# sweet_kl0p5_beta0p1_seed2

*one cell of the seed replicates of two sweep cells — sweet kl0p5 beta0p1 seed2*

|  |  |
|---|---|
| Config | [`configs/cultural_emergence/kl_beta_seed_replicates_8agent_ashbourne_gc/sweet_kl0p5_beta0p1_seed2.yaml`](../../../../configs/cultural_emergence/kl_beta_seed_replicates_8agent_ashbourne_gc/sweet_kl0p5_beta0p1_seed2.yaml) |
| Model | `Qwen2.5-7B-Instruct` |
| Population | 8 agents on `study_group_ashbourne_gc` |
| Settings | kl_coef 0.5, β 0.1, lr 5e-06, LoRA r 64 |
| Ran to | iteration 300 of 300 · 12 checkpoints |
| **Differs from** | **the other cells of this family — see the family note** |

## What it was for

A single cell of the seed replicates of two sweep cells. **The inference lives in the family note, not here**
— one population per cell cannot separate a cell from a seed on this substrate,
and this project has a direct demonstration that it cannot
([`../NOTES.md`](../NOTES.md)).

## Headline

Perception loss 2.18 → 1.85, policy entropy ending at
0.92, alive 1.00 at iteration 200 and
0.88 at 300.

Read this cell only against the rest of its family: [`../NOTES.md`](../NOTES.md).

## Core metrics

Population means over the first and last 10% of logged iterations (blocks of 30 of 300):

| metric | first block | last block |
|---|---:|---:|
| perception loss | 2.181 | 1.850 |
| KL from base | 0.019 | 1.342 |
| policy entropy | 0.752 | 0.921 |

Regime composition (paper thresholds; see `PROCESS.md`):

| iteration | n | alive | composition |
|---:|---:|---:|---|
| ~25 | 8 | 0.00 | text_degenerate 8 |
| ~100 | 8 | 1.00 | coherent_low_kl 7, mid_range 1 |
| ~150 | 8 | 1.00 | coherent_low_kl 8 |
| ~200 | 8 | 1.00 | coherent_low_kl 8 |
| ~300 | 8 | 0.88 | mid_range 7, text_degenerate 1 |

## What this settles

- **Nothing on its own.** A cell is one draw.

## What it does not settle

Its own rank. See the family note for what the family does and does not
support.

## Status

`CLOSED` at iteration 300 as part of its family.

---
*Analysed retrospectively on 2026-09-01 from `metrics.jsonl` and `config.json`; see the note in `PROCESS.md` on what a back-filled note cannot tell you. Machine output stays local.*
