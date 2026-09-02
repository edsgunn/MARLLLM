# lowreg_kl0p0_beta0p01_seed4

*one cell of the seed replicates of two sweep cells — lowreg kl0p0 beta0p01 seed4*

|  |  |
|---|---|
| Config | [`configs/cultural_emergence/kl_beta_seed_replicates_8agent_ashbourne_gc/lowreg_kl0p0_beta0p01_seed4.yaml`](../../../../configs/cultural_emergence/kl_beta_seed_replicates_8agent_ashbourne_gc/lowreg_kl0p0_beta0p01_seed4.yaml) |
| Model | `Qwen2.5-7B-Instruct` |
| Population | 8 agents on `study_group_ashbourne_gc` |
| Settings | kl_coef 0, β 0.01, lr 5e-06, LoRA r 64 |
| Ran to | iteration 300 of 300 · 12 checkpoints |
| **Differs from** | **the other cells of this family — see the family note** |

## What it was for

A single cell of the seed replicates of two sweep cells. **The inference lives in the family note, not here**
— one population per cell cannot separate a cell from a seed on this substrate,
and this project has a direct demonstration that it cannot
([`../NOTES.md`](../NOTES.md)).

## Headline

Perception loss 2.19 → 1.28, policy entropy ending at
0.44, alive 0.88 at iteration 200 and
0.88 at 300.

Read this cell only against the rest of its family: [`../NOTES.md`](../NOTES.md).

## Core metrics

Population means over the first and last 10% of logged iterations (blocks of 30 of 300):

| metric | first block | last block |
|---|---:|---:|
| perception loss | 2.187 | 1.284 |
| KL from base | 0.021 | 0.967 |
| policy entropy | 0.750 | 0.444 |

Regime composition (paper thresholds; see `PROCESS.md`):

| iteration | n | alive | composition |
|---:|---:|---:|---|
| ~25 | 8 | 0.00 | text_degenerate 8 |
| ~100 | 8 | 1.00 | coherent_low_kl 4, mid_range 4 |
| ~150 | 8 | 0.88 | coherent_low_kl 6, mid_range 1, policy_collapse 1 |
| ~200 | 8 | 0.88 | coherent_low_kl 6, committed 1, policy_collapse 1 |
| ~300 | 8 | 0.88 | coherent_low_kl 6, committed 1, policy_collapse 1 |

## What this settles

- **Nothing on its own.** A cell is one draw.

## What it does not settle

Its own rank. See the family note for what the family does and does not
support.

## Status

`CLOSED` at iteration 300 as part of its family.

---
*Analysed retrospectively on 2026-09-01 from `metrics.jsonl` and `config.json`; see the note in `PROCESS.md` on what a back-filled note cannot tell you. Machine output stays local.*
