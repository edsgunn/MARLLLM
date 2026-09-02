# run8_16agent_7B_study_group_strathearn_server

*cultural emergence — 16-agent Strathearn server; the second 16-agent cell*

|  |  |
|---|---|
| Config | [`configs/cultural_emergence/run8_16agent_7B_study_group_strathearn_server.yaml`](../../../configs/cultural_emergence/run8_16agent_7B_study_group_strathearn_server.yaml) |
| Model | `Qwen2.5-7B-Instruct` |
| Population | 16 agents on `study_group_strathearn_server` |
| Settings | kl_coef 0.05, β 0.01, lr 5e-06, LoRA r 64 |
| Ran to | iteration 300 of 300 · 12 checkpoints |
| **Differs from** | **`run8_16agent_7B_study_group_ashbourne_gc` — environment description only** |

## What it was for

The 16-agent Strathearn cell, completing the 2 × 4 substrate × population-size
grid at its largest size.

## Headline

**The same shape as its Ashbourne twin, one step behind.** 11 coherent, 3
committed, 2 policy-collapse at iteration 200 (alive 0.88), falling to 0.69 at
300. Perception loss barely moves (1.50 → 1.39) — this substrate starts much
easier than Ashbourne GC, which opens at 2.35.

## Core metrics

Population means over the first and last 10% of logged iterations (blocks of 30 of 300):

| metric | first block | last block |
|---|---:|---:|
| perception loss | 1.498 | 1.391 |
| KL from base | 0.015 | 0.749 |
| policy entropy | 0.564 | 0.437 |

Regime composition (paper thresholds; see `PROCESS.md`):

| iteration | n | alive | composition |
|---:|---:|---:|---|
| ~25 | 16 | 1.00 | coherent_low_kl 16 |
| ~100 | 16 | 1.00 | coherent_low_kl 15, committed 1 |
| ~150 | 16 | 0.94 | coherent_low_kl 14, committed 1, policy_collapse 1 |
| ~200 | 16 | 0.88 | coherent_low_kl 11, committed 3, policy_collapse 2 |
| ~300 | 16 | 0.69 | coherent_low_kl 7, committed 4, policy_collapse 5 |

## What this settles

- **Substrates differ in starting perception loss by ~0.85 nats**, so
  cross-substrate comparisons of absolute perception loss are meaningless
  without the baseline. — *established*

## What it does not settle

Whether Strathearn's earlier decline (0.88 at 200 vs 1.00 for Ashbourne) is
real or one draw. No seed replicates were run for Strathearn at any size.

## Status

`CLOSED` at iteration 300.

---
*Analysed retrospectively on 2026-09-01 from `metrics.jsonl` and `config.json`; see the note in `PROCESS.md` on what a back-filled note cannot tell you. Machine output stays local.*
