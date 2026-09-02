# run7_8agent_7B_inner_loop_seed3

*cultural emergence — Inner Loop seed replicate (seed3)*

|  |  |
|---|---|
| Config | [`configs/cultural_emergence/run7_8agent_7B_inner_loop_seed3.yaml`](../../../configs/cultural_emergence/run7_8agent_7B_inner_loop_seed3.yaml) |
| Model | `Qwen2.5-7B-Instruct` |
| Population | 8 agents on `inner_loop` |
| Settings | kl_coef 0.05, β 0.01, lr 5e-06, LoRA r 64 |
| Ran to | iteration 100 of 100 · 4 checkpoints |
| **Differs from** | **`run7_8agent_7B_inner_loop` — training seed only; identical config and roster** |

## What it was for

One of four seed replicates of the reference Inner Loop population. The set
exists to answer the question a single population cannot: how much of a run's
regime composition is the substrate and how much is the seed. It is the unit of
inference, not this run — read
[`../_seed_replicate_regimes/SUMMARY.md`](../_seed_replicate_regimes/SUMMARY.md).

## Headline

**7 coherent, 1 committed.** All five Inner Loop populations are 100%
alive with zero policy-collapse and zero text-degeneration events; what varies
across seeds is only how many agents commit (0–3).

## Core metrics

Population means over the first and last 10% of logged iterations (blocks of 10 of 100):

| metric | first block | last block |
|---|---:|---:|
| perception loss | 1.321 | 1.201 |
| KL from base | 0.001 | 0.261 |
| policy entropy | 0.597 | 0.712 |

Regime composition (paper thresholds; see `PROCESS.md`):

| iteration | n | alive | composition |
|---:|---:|---:|---|
| ~25 | 8 | 1.00 | coherent_low_kl 8 |
| ~100 | 8 | 1.00 | coherent_low_kl 7, committed 1 |

## What this settles

- **Which individual agent commits is seed-random, not a property of the
  character sheet.** Fleiss κ across seeds with the roster held fixed is 0.10.
  — *established* — `../_seed_replicate_regimes/SUMMARY.md`

## What it does not settle

Nothing on its own. A single seed replicate is one draw; the claims live in the
family summary.

## Status

`CLOSED`. Cite the family, not this run.

---
*Analysed retrospectively on 2026-09-01 from `metrics.jsonl` and `config.json`; see the note in `PROCESS.md` on what a back-filled note cannot tell you. Machine output stays local.*
