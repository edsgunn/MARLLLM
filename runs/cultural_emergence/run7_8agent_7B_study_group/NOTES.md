# run7_8agent_7B_study_group

*cultural emergence — 8-agent substrate set, Study Group (original description)*

|  |  |
|---|---|
| Config | [`configs/cultural_emergence/run7_8agent_7B_study_group.yaml`](../../../configs/cultural_emergence/run7_8agent_7B_study_group.yaml) |
| Model | `Qwen2.5-7B-Instruct` |
| Population | 8 agents on `study_group` |
| Settings | kl_coef 0.05, β 0.01, lr 5e-06, LoRA r 64 |
| Ran to | iteration 300 of 100 · 12 checkpoints |
| **Differs from** | **`run7_8agent_7B_inner_loop` — substrate only; Study Group is a maths forum with no tool** |

## What it was for

The tool-free counterpart to Inner Loop: whether a population needs an
execution side channel to produce committed agents, or whether conversation
alone is enough.

## Headline

**Conversation alone is enough.** Three of eight agents end `committed` and
four `coherent_low_kl` with a single policy collapse, and the population is
100% alive at iterations 100, 150 and 200. Perception loss drifts up mildly
(1.37 → 1.53) while KL reaches 1.29. Committed agents do not require a tool.

## Core metrics

Population means over the first and last 10% of logged iterations (blocks of 30 of 300):

| metric | first block | last block |
|---|---:|---:|
| perception loss | 1.374 | 1.528 |
| KL from base | 0.003 | 1.294 |
| policy entropy | 0.575 | 0.950 |

Regime composition (paper thresholds; see `PROCESS.md`):

| iteration | n | alive | composition |
|---:|---:|---:|---|
| ~25 | 8 | 1.00 | coherent_low_kl 8 |
| ~100 | 8 | 1.00 | coherent_low_kl 6, committed 2 |
| ~150 | 8 | 1.00 | coherent_low_kl 6, committed 2 |
| ~200 | 8 | 1.00 | coherent_low_kl 6, committed 2 |
| ~300 | 8 | 0.88 | coherent_low_kl 4, committed 3, policy_collapse 1 |

## What this settles

- **A tool-free conversation substrate produces committed agents at the same
  rate as a tool-bearing one.** 3/8 here against 3/8 in Inner Loop. — *likely*,
  one run each

## What it does not settle

Whether the commitment is the same phenomenon in both. Nothing here compares
what the committed agents in the two substrates actually do.

## Status

`CLOSED` at iteration 300, alive 0.88.

---
*Analysed retrospectively on 2026-09-01 from `metrics.jsonl` and `config.json`; see the note in `PROCESS.md` on what a back-filled note cannot tell you. Machine output stays local.*
