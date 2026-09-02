# run7_8agent_7B_study_group

*cultural emergence — pre-shuffle runs, superseded*

|  |  |
|---|---|
| Model | `Qwen2.5-7B-Instruct` |
| Population | 8 agents on `various` |
| Settings | kl_coef 0.05, β 0.01, lr 5e-06, LoRA r 64 |
| Ran to | iteration 26 of 100 — **truncated** · 1 checkpoints |
| **Differs from** | **the current runs — these predate role shuffling in the population trainer** |

## What it was for

Early cultural-emergence runs made before agent-to-role assignment was shuffled
between episodes. Retained under `unshuffled/` because they are not comparable
with anything current, not because they are wrong.

## Headline

**Superseded: the one arm that looks healthy, and it stops at iteration 26.** With a fixed agent-to-role assignment, each agent sees
the same position in every episode, which is a different experiment from the
one the rest of the campaign ran.

## Core metrics

Population means over the first and last 10% of logged iterations (blocks of 2 of 26):

| metric | first block | last block |
|---|---:|---:|
| perception loss | 1.404 | 1.155 |
| KL from base | 0.000 | 0.004 |
| policy entropy | 0.529 | 0.462 |

Regime composition (paper thresholds; see `PROCESS.md`):

| iteration | n | alive | composition |
|---:|---:|---:|---|
| ~25 | 8 | 1.00 | coherent_low_kl 8 |

## What this settles

- **Nothing that is still citable.**

## What it does not settle

Anything. Use the shuffled reruns.

## Status

`SUPERSEDED` by the shuffled reruns. Do not cite.

---
*Analysed retrospectively on 2026-09-01 from `metrics.jsonl` and `config.json`; see the note in `PROCESS.md` on what a back-filled note cannot tell you. Machine output stays local.*
