# run11_16agent_7B_margin_notes_2turns

*cultural emergence — 16 agents, two turns per episode (Margin Notes)*

|  |  |
|---|---|
| Config | [`configs/cultural_emergence/run11_16agent_7B_margin_notes_2turns.yaml`](../../../configs/cultural_emergence/run11_16agent_7B_margin_notes_2turns.yaml) |
| Model | `Qwen2.5-7B-Instruct` |
| Population | 16 agents on `margin_notes` |
| Settings | kl_coef 0.05, β 0.01, lr 5e-06, LoRA r 64 |
| Ran to | **no metrics written** |
| **Differs from** | **`run8_16agent_7B_margin_notes` — two turns per episode** |

## What it was for

The Margin Notes arm of the two-turn experiment.

## Headline

**No metrics were ever written.** The log shows the population starting and one
checkpoint trace at iteration 0, then nothing. The third arm of the two-turn
experiment does not exist.

## Core metrics

_No metrics were written._

## What this settles

- **Nothing.**

## What it does not settle

Whether the two-turn collapse is substrate-general. It has been shown on two
Study Group framings and never tested on a tool-bearing substrate.

## Status

`NO DATA`. Never resubmitted.

---
*Analysed retrospectively on 2026-09-01 from `metrics.jsonl` and `config.json`; see the note in `PROCESS.md` on what a back-filled note cannot tell you. Machine output stays local.*
