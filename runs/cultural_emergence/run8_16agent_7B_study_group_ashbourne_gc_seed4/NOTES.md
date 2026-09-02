# run8_16agent_7B_study_group_ashbourne_gc_seed4

*cultural emergence — 16-agent Ashbourne GC seed replicate (seed4)*

|  |  |
|---|---|
| Config | [`configs/cultural_emergence/run8_16agent_7B_study_group_ashbourne_gc_seed4.yaml`](../../../configs/cultural_emergence/run8_16agent_7B_study_group_ashbourne_gc_seed4.yaml) |
| Model | `Qwen2.5-7B-Instruct` |
| Population | 16 agents on `study_group_ashbourne_gc` |
| Settings | kl_coef 0.05, β 0.01, lr 5e-06, LoRA r 64 |
| Ran to | iteration 150 of 150 · 6 checkpoints |
| **Differs from** | **`run8_16agent_7B_study_group_ashbourne_gc` — training seed only, and a 150-iteration horizon instead of 300** |

## What it was for

One of four seed replicates of the 16-agent Ashbourne population, run so that
the population-size claim could be stated with the population as the unit of
analysis. The unit of inference is the set of five — see
[`../_seed_replicate_regimes/SUMMARY.md`](../_seed_replicate_regimes/SUMMARY.md).

## Headline

**Across the five populations the alive fraction is 0.95 ± 0.08 and none
collapses**, but composition is not identical: seed 5 produced no committed and
no coherent agents at all (81% mid-range, 19% text-degenerate) while seeds 1–4
reproduce the coherent-majority, committed-minority mix. That spread is the
honest population-level variance.

## Core metrics

Population means over the first and last 10% of logged iterations (blocks of 15 of 150):

| metric | first block | last block |
|---|---:|---:|
| perception loss | 2.368 | 1.801 |
| KL from base | 0.006 | 0.514 |
| policy entropy | 0.867 | 0.770 |

Regime composition (paper thresholds; see `PROCESS.md`):

| iteration | n | alive | composition |
|---:|---:|---:|---|
| ~25 | 16 | 0.00 | text_degenerate 16 |
| ~100 | 16 | 0.81 | mid_range 13, text_degenerate 3 |
| ~150 | 16 | 0.94 | coherent_low_kl 13, committed 1, mid_range 1, policy_collapse 1 |

## What this settles

- **Large-population non-collapse replicates.** 5/5 populations majority-alive,
  0/5 wholly collapsed. — *established* — `../_seed_replicate_regimes/SUMMARY.md`
- **Agents within a population are not independent.** Perception-loss ICC across
  seeds is 0.84 — a 16-agent population is worth about 6 independent agents. —
  *established* — same

## What it does not settle

Anything at 300 iterations: these stop at 150, and the parent run loses agents
between 200 and 300.

## Status

`CLOSED` at 150 iterations. Cite the family.

---
*Analysed retrospectively on 2026-09-01 from `metrics.jsonl` and `config.json`; see the note in `PROCESS.md` on what a back-filled note cannot tell you. Machine output stays local.*
