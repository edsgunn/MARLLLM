# run7_8agent_7B_study_group_strathearn_server

*cultural emergence — Study Group, Strathearn server framing; one of the two Figure-1 substrates*

|  |  |
|---|---|
| Config | [`configs/cultural_emergence/run7_8agent_7B_study_group_strathearn_server.yaml`](../../../configs/cultural_emergence/run7_8agent_7B_study_group_strathearn_server.yaml) |
| Model | `Qwen2.5-7B-Instruct` |
| Population | 8 agents on `study_group_strathearn_server` |
| Settings | kl_coef 0.05, β 0.01, lr 5e-06, LoRA r 64 |
| Ran to | iteration 300 of 300 · 12 checkpoints |
| **Differs from** | **`run7_8agent_7B_study_group_ashbourne_gc` — environment description only; this is the second arm of the substrate × population-size grid** |

## What it was for

The 8-agent cell of the substrate × population-size grid, paired with
Ashbourne GC. Together with `run8_16agent`, `run10_4agent` and `run10_2agent`
these are the runs behind the population-size figures.

## Headline

**Alive through iteration 200 and losing a quarter of the population by 300.**
6 coherent, 2 committed at iteration 200; 6 coherent, 2 policy-collapse at 300.
The published evaluation point sits one step before the decline starts.

## Core metrics

Population means over the first and last 10% of logged iterations (blocks of 30 of 300):

| metric | first block | last block |
|---|---:|---:|
| perception loss | 1.442 | 1.647 |
| KL from base | 0.009 | 1.128 |
| policy entropy | 0.512 | 0.373 |

Regime composition (paper thresholds; see `PROCESS.md`):

| iteration | n | alive | composition |
|---:|---:|---:|---|
| ~25 | 8 | 1.00 | coherent_low_kl 8 |
| ~100 | 8 | 1.00 | coherent_low_kl 7, committed 1 |
| ~150 | 8 | 1.00 | coherent_low_kl 7, committed 1 |
| ~200 | 8 | 1.00 | coherent_low_kl 6, committed 2 |
| ~300 | 8 | 0.75 | coherent_low_kl 6, policy_collapse 2 |

## What this settles

- **The 8-agent Strathearn cell is healthy at the evaluation iteration.** 8/8
  alive at 200. — *established*
- **It is not healthy at 300.** 6/8. — *established*

## What it does not settle

Whether the decline is a property of this substrate or of the horizon. The
balanced panel across 44 cultural-emergence populations says the latter.

## Status

`CLOSED` at iteration 300. One of the eight runs in the iter-200 classification table.

---
*Analysed retrospectively on 2026-09-01 from `metrics.jsonl` and `config.json`; see the note in `PROCESS.md` on what a back-filled note cannot tell you. Machine output stays local.*
