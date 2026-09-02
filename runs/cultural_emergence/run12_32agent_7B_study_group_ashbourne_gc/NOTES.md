# run12_32agent_7B_study_group_ashbourne_gc

*cultural emergence — 32 agents, the top of the population-size ladder*

|  |  |
|---|---|
| Config | [`configs/cultural_emergence/run12_32agent_7B_study_group_ashbourne_gc.yaml`](../../../configs/cultural_emergence/run12_32agent_7B_study_group_ashbourne_gc.yaml) |
| Model | `Qwen2.5-7B-Instruct` |
| Population | 32 agents on `study_group_ashbourne_gc` |
| Settings | kl_coef 0.05, β 0.01, lr 5e-06, LoRA r 64 |
| Ran to | iteration 300 of 300 · 12 checkpoints |
| **Differs from** | **`run8_16agent_7B_study_group_ashbourne_gc` — 32 agents on the `extended_32` roster, with episode length scaled to hold the per-post token budget fixed** |

## What it was for

Whether the population-size effect keeps going past 16 agents.

## Headline

**It does not keep going.** At iteration 200 the 32-agent population is 0.84
alive against 1.00 for 16 agents on the identical substrate, and at 300 it is
0.62 against 0.88. Twelve of thirty-two agents end in `policy_collapse`.
Perception loss does fall further than at any other size (1.92 → 1.31).

Read with the 16-agent run, this puts the population-size optimum **at or below
16 agents**, not at "more is better".

## Core metrics

Population means over the first and last 10% of logged iterations (blocks of 30 of 300):

| metric | first block | last block |
|---|---:|---:|
| perception loss | 1.918 | 1.307 |
| KL from base | 0.017 | 0.424 |
| policy entropy | 0.815 | 0.504 |

Regime composition (paper thresholds; see `PROCESS.md`):

| iteration | n | alive | composition |
|---:|---:|---:|---|
| ~25 | 32 | 1.00 | mid_range 32 |
| ~100 | 32 | 1.00 | coherent_low_kl 32 |
| ~150 | 32 | 0.91 | coherent_low_kl 28, committed 1, policy_collapse 3 |
| ~200 | 32 | 0.84 | coherent_low_kl 26, committed 1, policy_collapse 5 |
| ~300 | 32 | 0.62 | coherent_low_kl 18, committed 2, policy_collapse 12 |

## What this settles

- **The population-size benefit is not monotone.** 32 agents is worse than 16 on
  alive fraction at both iteration 200 and 300. — *provisional*, one run per size

## What it does not settle

Whether the 32-agent run is compute-limited rather than size-limited: episode
length was scaled to hold per-post tokens fixed, which means each agent gets
proportionally fewer updates per iteration. That confound is not controlled.

## Status

`CLOSED` at iteration 300.

---
*Analysed retrospectively on 2026-09-01 from `metrics.jsonl` and `config.json`; see the note in `PROCESS.md` on what a back-filled note cannot tell you. Machine output stays local.*
