# run7_8agent_7B_inner_loop_seed2_v1

*cultural emergence — first attempt at an Inner Loop run, superseded*

|  |  |
|---|---|
| Model | `Qwen2.5-7B-Instruct` |
| Population | 8 agents on `inner_loop` |
| Settings | kl_coef 0.05, β 0.01, lr 5e-06, LoRA r 64 |
| Ran to | iteration 20 of 100 — **truncated** · 0 checkpoints |
| **Differs from** | **the `v1`-less run of the same name, which replaced it** |

## What it was for

A first attempt at this Inner Loop population, kept so that the job is findable
rather than a gap. Superseded by the run that carries the same name without the
`_v1` suffix.

## Headline

**Truncated early and superseded.** Alive and wholly `coherent_low_kl` as far
as it got, with KL from base below 0.17 — nothing had begun to differentiate.

## Core metrics

Population means over the first and last 10% of logged iterations (blocks of 2 of 20):

| metric | first block | last block |
|---|---:|---:|
| perception loss | 1.304 | 1.261 |
| KL from base | 0.001 | 0.003 |
| policy entropy | 0.594 | 0.579 |

Regime composition (paper thresholds; see `PROCESS.md`):

| iteration | n | alive | composition |
|---:|---:|---:|---|
| ~25 | 8 | 1.00 | coherent_low_kl 8 |

## What this settles

- **Nothing.**

## What it does not settle

Anything. Use the superseding run.

## Status

`SUPERSEDED`. Do not cite.

---
*Analysed retrospectively on 2026-09-01 from `metrics.jsonl` and `config.json`; see the note in `PROCESS.md` on what a back-filled note cannot tell you. Machine output stays local.*
