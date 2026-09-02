# run10_2agent_7B_margin_notes

*cultural emergence — 2-agent Margin Notes*

|  |  |
|---|---|
| Config | [`configs/cultural_emergence/run10_2agent_7B_margin_notes.yaml`](../../../configs/cultural_emergence/run10_2agent_7B_margin_notes.yaml) |
| Model | `Qwen2.5-7B-Instruct` |
| Population | 2 agents on `margin_notes` |
| Settings | kl_coef 0.05, β 0.01, lr 5e-06, LoRA r 64 |
| Ran to | iteration 4 of 300 — **truncated** · 0 checkpoints |
| **Differs from** | **`run10_2agent_7B_study_group_ashbourne_gc` — substrate only** |

## What it was for

The 2-agent Margin Notes cell of the population-size grid.

## Headline

**Four metric records. No usable data.** The run stopped almost immediately and
was never resubmitted, so Margin Notes has no 2-agent cell.

## Core metrics

Population means over the first and last 10% of logged iterations (blocks of 1 of 4):

| metric | first block | last block |
|---|---:|---:|
| perception loss | 1.375 | 1.493 |
| KL from base | 0.000 | 0.001 |
| policy entropy | 0.332 | 0.357 |

## What this settles

- **Nothing.**

## What it does not settle

The 2-agent Margin Notes cell, which is still missing from the population-size
grid.

## Status

`TRUNCATED` at iteration 4. Gap in the grid.

---
*Analysed retrospectively on 2026-09-01 from `metrics.jsonl` and `config.json`; see the note in `PROCESS.md` on what a back-filled note cannot tell you. Machine output stays local.*
