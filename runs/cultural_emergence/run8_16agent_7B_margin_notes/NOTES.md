# run8_16agent_7B_margin_notes

*cultural emergence — 16-agent Margin Notes*

|  |  |
|---|---|
| Config | [`configs/cultural_emergence/run8_16agent_7B_margin_notes.yaml`](../../../configs/cultural_emergence/run8_16agent_7B_margin_notes.yaml) |
| Model | `Qwen2.5-7B-Instruct` |
| Population | 16 agents on `margin_notes` |
| Settings | kl_coef 0.05, β 0.01, lr 5e-06, LoRA r 64 |
| Ran to | iteration 300 of 300 · 12 checkpoints |
| **Differs from** | **`run7_8agent_7B_margin_notes` — 16 agents instead of 8** |

## What it was for

Whether the population-size effect seen on Ashbourne GC also holds on a
tool-bearing substrate.

## Headline

**Doubling the population helps here too, and does not prevent the late
decline.** Alive at iteration 300 is 0.62 against 0.50 for the 8-agent run on
the same substrate; alive at 200 is 0.88. Six of sixteen agents end in
`policy_collapse`.

## Core metrics

Population means over the first and last 10% of logged iterations (blocks of 30 of 307):

| metric | first block | last block |
|---|---:|---:|
| perception loss | 1.601 | 1.300 |
| KL from base | 0.016 | 0.692 |
| policy entropy | 0.528 | 0.408 |

Regime composition (paper thresholds; see `PROCESS.md`):

| iteration | n | alive | composition |
|---:|---:|---:|---|
| ~25 | 16 | 1.00 | coherent_low_kl 16 |
| ~100 | 16 | 1.00 | coherent_low_kl 11, mid_range 5 |
| ~150 | 16 | 1.00 | coherent_low_kl 15, committed 1 |
| ~200 | 16 | 0.88 | coherent_low_kl 12, committed 2, policy_collapse 2 |
| ~300 | 16 | 0.62 | coherent_low_kl 8, committed 2, policy_collapse 6 |

## What this settles

- **The population-size association is not specific to Ashbourne GC.** — *likely*

## What it does not settle

Whether 0.62 vs 0.50 is a real difference: it is one run against one run, on a
project where identical configurations have diverged by more than this.

## Status

`CLOSED` at iteration 300.

---
*Analysed retrospectively on 2026-09-01 from `metrics.jsonl` and `config.json`; see the note in `PROCESS.md` on what a back-filled note cannot tell you. Machine output stays local.*
