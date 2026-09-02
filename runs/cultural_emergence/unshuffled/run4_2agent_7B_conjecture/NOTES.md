# run4_2agent_7B_conjecture

*cultural emergence — pre-shuffle runs, superseded*

|  |  |
|---|---|
| Model | `Qwen2.5-7B-Instruct` |
| Population | 2 agents on `various` |
| Settings | kl_coef 0.05, β 0.01, lr 5e-06, LoRA r 64 |
| Ran to | iteration 100 of 100 · 4 checkpoints |
| **Differs from** | **the current runs — these predate role shuffling in the population trainer** |

## What it was for

Early cultural-emergence runs made before agent-to-role assignment was shuffled
between episodes. Retained under `unshuffled/` because they are not comparable
with anything current, not because they are wrong.

## Headline

**Superseded: perception loss 2.34, entropy 2.71 — broken.** With a fixed agent-to-role assignment, each agent sees
the same position in every episode, which is a different experiment from the
one the rest of the campaign ran.

## Core metrics

Population means over the first and last 10% of logged iterations (blocks of 11 of 111):

| metric | first block | last block |
|---|---:|---:|
| perception loss | 1.096 | 1.815 |
| KL from base | 0.001 | 0.307 |
| policy entropy | 0.561 | 2.053 |

Regime composition (paper thresholds; see `PROCESS.md`):

| iteration | n | alive | composition |
|---:|---:|---:|---|
| ~25 | 2 | 1.00 | coherent_low_kl 2 |
| ~100 | 2 | 0.00 | policy_collapse 1, text_degenerate 1 |

## What this settles

- **Nothing that is still citable.**

## What it does not settle

Anything. Use the shuffled reruns.

## Status

`SUPERSEDED` by the shuffled reruns. Do not cite.

---
*Analysed retrospectively on 2026-09-01 from `metrics.jsonl` and `config.json`; see the note in `PROCESS.md` on what a back-filled note cannot tell you. Machine output stays local.*
