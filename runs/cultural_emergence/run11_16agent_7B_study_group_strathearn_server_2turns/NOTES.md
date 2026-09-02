# run11_16agent_7B_study_group_strathearn_server_2turns

*cultural emergence — 16 agents, two turns per episode (Strathearn)*

|  |  |
|---|---|
| Config | [`configs/cultural_emergence/run11_16agent_7B_study_group_strathearn_server_2turns.yaml`](../../../configs/cultural_emergence/run11_16agent_7B_study_group_strathearn_server_2turns.yaml) |
| Model | `Qwen2.5-7B-Instruct` |
| Population | 16 agents on `study_group_strathearn_server` |
| Settings | kl_coef 0.05, β 0.01, lr 5e-06, LoRA r 64 |
| Ran to | iteration 300 of 300 · 12 checkpoints |
| **Differs from** | **`run8_16agent_7B_study_group_strathearn_server` — two turns per episode instead of the default** |

## What it was for

Whether giving each agent two turns per episode — more action tokens per unit
of environment — changes what the population becomes.

## Headline

**Two turns per episode produces the sharpest late collapse in the campaign.**
Alive is 1.00 at iteration 200 and **0.38 at 300** on Ashbourne (10 of 16
agents in `policy_collapse`); Strathearn is worse still, 0.44 at 200 falling to
0.25. The single-turn 16-agent runs on the same substrates end at 0.88 and
0.69.

## Core metrics

Population means over the first and last 10% of logged iterations (blocks of 30 of 300):

| metric | first block | last block |
|---|---:|---:|
| perception loss | 1.096 | 0.809 |
| KL from base | 0.008 | 1.513 |
| policy entropy | 0.476 | 0.220 |

Regime composition (paper thresholds; see `PROCESS.md`):

| iteration | n | alive | composition |
|---:|---:|---:|---|
| ~25 | 16 | 1.00 | coherent_low_kl 16 |
| ~100 | 16 | 1.00 | coherent_low_kl 15, committed 1 |
| ~150 | 16 | 0.88 | coherent_low_kl 13, committed 1, policy_collapse 2 |
| ~200 | 16 | 0.44 | coherent_low_kl 3, committed 4, policy_collapse 9 |
| ~300 | 16 | 0.25 | coherent_low_kl 1, committed 3, policy_collapse 12 |

## What this settles

- **More action tokens per episode accelerates policy collapse.** Both 2-turn
  arms end far below their single-turn counterparts, and the failure mode is
  `policy_collapse` (entropy below 0.2) rather than text degeneration. —
  *likely*, one run per substrate

## What it does not settle

Whether the mechanism is the extra action tokens or the longer within-episode
context they produce. Nothing separates them here.

## Status

`CLOSED` at iteration 300. A negative result the campaign did not follow up.

---
*Analysed retrospectively on 2026-09-01 from `metrics.jsonl` and `config.json`; see the note in `PROCESS.md` on what a back-filled note cannot tell you. Machine output stays local.*
