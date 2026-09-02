# run5_8agent_7B

*cultural emergence — pre-shuffle runs, superseded*

|  |  |
|---|---|
| Model | `Qwen2.5-7B-Instruct` |
| Population | 8 agents on `various` |
| Settings | kl_coef 0.05, β 0.01, lr 5e-06, LoRA r 64 |
| Ran to | iteration 25 of 100 — **truncated** · 1 checkpoints |
| **Differs from** | **the current runs — these predate role shuffling in the population trainer** |

## What it was for

Early cultural-emergence runs made before agent-to-role assignment was shuffled
between episodes. Retained under `unshuffled/` because they are not comparable
with anything current, not because they are wrong.

## Headline

**Superseded: all eight agents text-degenerate by iteration 25.** With a fixed agent-to-role assignment, each agent sees
the same position in every episode, which is a different experiment from the
one the rest of the campaign ran.

## Core metrics

Population means over the first and last 10% of logged iterations (blocks of 2 of 25):

| metric | first block | last block |
|---|---:|---:|
| perception loss | 2.348 | 2.268 |
| KL from base | 0.001 | 0.172 |
| policy entropy | 1.264 | 1.323 |

Regime composition (paper thresholds; see `PROCESS.md`):

| iteration | n | alive | composition |
|---:|---:|---:|---|
| ~25 | 8 | 0.00 | text_degenerate 8 |

## What this settles

- **Nothing that is still citable.**

## What it does not settle

Anything. Use the shuffled reruns.

## Status

`SUPERSEDED` by the shuffled reruns. Do not cite.

---
*Analysed retrospectively on 2026-09-01 from `metrics.jsonl` and `config.json`; see the note in `PROCESS.md` on what a back-filled note cannot tell you. Machine output stays local.*
