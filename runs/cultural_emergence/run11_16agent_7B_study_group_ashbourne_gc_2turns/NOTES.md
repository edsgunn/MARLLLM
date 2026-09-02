# run11_16agent_7B_study_group_ashbourne_gc_2turns

*cultural emergence — 16 agents, two turns per episode (Ashbourne)*

|  |  |
|---|---|
| Config | [`configs/cultural_emergence/run11_16agent_7B_study_group_ashbourne_gc_2turns.yaml`](../../../configs/cultural_emergence/run11_16agent_7B_study_group_ashbourne_gc_2turns.yaml) |
| Model | `Qwen2.5-7B-Instruct` |
| Population | 16 agents on `study_group_ashbourne_gc` |
| Settings | kl_coef 0.05, β 0.01, lr 5e-06, LoRA r 64 |
| Ran to | iteration 300 of 300 · 12 checkpoints |
| **Differs from** | **`run8_16agent_7B_study_group_ashbourne_gc` — two turns per episode instead of the default** |

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
| perception loss | 1.759 | 1.431 |
| KL from base | 0.014 | 0.961 |
| policy entropy | 0.701 | 0.717 |

Regime composition (paper thresholds; see `PROCESS.md`):

| iteration | n | alive | composition |
|---:|---:|---:|---|
| ~25 | 16 | 1.00 | mid_range 16 |
| ~100 | 16 | 1.00 | coherent_low_kl 13, committed 3 |
| ~150 | 16 | 1.00 | coherent_low_kl 13, committed 3 |
| ~200 | 16 | 1.00 | coherent_low_kl 15, committed 1 |
| ~300 | 16 | 0.38 | coherent_low_kl 4, committed 2, policy_collapse 10 |

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
