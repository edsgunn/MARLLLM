# run10_4agent_7B_margin_notes

*cultural emergence — 4-agent Margin Notes*

|  |  |
|---|---|
| Config | [`configs/cultural_emergence/run10_4agent_7B_margin_notes.yaml`](../../../configs/cultural_emergence/run10_4agent_7B_margin_notes.yaml) |
| Model | `Qwen2.5-7B-Instruct` |
| Population | 4 agents on `margin_notes` |
| Settings | kl_coef 0.05, β 0.01, lr 5e-06, LoRA r 64 |
| Ran to | iteration 44 of 300 — **truncated** · 1 checkpoints |
| **Differs from** | **`run10_4agent_7B_study_group_ashbourne_gc` — substrate only** |

## What it was for

The 4-agent Margin Notes cell of the population-size grid.

## Headline

**Truncated at iteration 44, alive and wholly coherent as far as it got.**
Perception loss 1.53 → 1.39, KL from base 0.05 — the population had not begun
to differentiate.

## Core metrics

Population means over the first and last 10% of logged iterations (blocks of 7 of 73):

| metric | first block | last block |
|---|---:|---:|
| perception loss | 1.532 | 1.390 |
| KL from base | 0.000 | 0.047 |
| policy entropy | 0.397 | 0.366 |

Regime composition (paper thresholds; see `PROCESS.md`):

| iteration | n | alive | composition |
|---:|---:|---:|---|
| ~25 | 4 | 1.00 | coherent_low_kl 4 |

## What this settles

- **Nothing.** Too short to reach the iterations where regimes separate.

## What it does not settle

The 4-agent Margin Notes cell, still effectively missing.

## Status

`TRUNCATED` at iteration 44. Gap in the grid.

---
*Analysed retrospectively on 2026-09-01 from `metrics.jsonl` and `config.json`; see the note in `PROCESS.md` on what a back-filled note cannot tell you. Machine output stays local.*
