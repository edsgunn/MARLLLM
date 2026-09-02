# run10_2agent_7B_study_group_ashbourne_gc

*cultural emergence — population-size grid, 2 agents on ashbourne*

|  |  |
|---|---|
| Config | [`configs/cultural_emergence/run10_2agent_7B_study_group_ashbourne_gc.yaml`](../../../configs/cultural_emergence/run10_2agent_7B_study_group_ashbourne_gc.yaml) |
| Model | `Qwen2.5-7B-Instruct` |
| Population | 2 agents on `study_group_ashbourne_gc` |
| Settings | kl_coef 0.05, β 0.01, lr 5e-06, LoRA r 64 |
| Ran to | iteration 300 of 300 · 12 checkpoints |
| **Differs from** | **`run8_16agent_7B_study_group_ashbourne_gc` — population size only (2 agents instead of 16)** |

## What it was for

The small end of the substrate × population-size grid. These runs supply the
2- and 4-agent cells of the population-size figures.

## Headline

**Alive at the evaluation iteration, dead on either side of it.** This
2-agent population is `text_degenerate` at iterations 25 and 100, recovers to
1 committed + 1 coherent at 150–200, and ends at 300 with both agents in
`policy_collapse` (entropy 0.12). Perception loss falls 2.50 → 1.29 while
entropy falls 0.56 → 0.12 — it learns to predict its world by becoming
deterministic.

## Core metrics

Population means over the first and last 10% of logged iterations (blocks of 30 of 300):

| metric | first block | last block |
|---|---:|---:|
| perception loss | 2.501 | 1.293 |
| KL from base | 0.008 | 0.679 |
| policy entropy | 0.564 | 0.115 |

Regime composition (paper thresholds; see `PROCESS.md`):

| iteration | n | alive | composition |
|---:|---:|---:|---|
| ~25 | 2 | 0.00 | text_degenerate 2 |
| ~100 | 2 | 0.00 | text_degenerate 2 |
| ~150 | 2 | 1.00 | committed 1, mid_range 1 |
| ~200 | 2 | 1.00 | coherent_low_kl 1, committed 1 |
| ~300 | 2 | 0.00 | policy_collapse 2 |

## What this settles

- **Two-agent populations lock the policy without ruining the text.** Both
  2-agent runs end wholly `policy_collapse` — entropy below 0.2 with perception
  loss well inside the coherent band — which the trace verification confirms:
  the posts read fine and the underlying generation loops closing tags. —
  *established* — `../_population_figures/trace_verification_iter200.md`
- **The two substrates reach that state at different times**, so the iteration a
  small population is evaluated at decides what it looks like: at iteration 200
  the Ashbourne pair is alive and the Strathearn pair has been dead for 50
  iterations. — *established*
- **There are no seed replicates at 2 or 4 agents**, so these cells are single
  observations and the population-size claim has to be phrased as association.
  — *established*

## What it does not settle

Population size as a cause. 2-agent Ashbourne is alive at iteration 200 and
2-agent Strathearn is dead by 150, so substrate and size interact.

## Status

`CLOSED` at iteration 300. Cells of the population-size grid; not independently powered.

---
*Analysed retrospectively on 2026-09-01 from `metrics.jsonl` and `config.json`; see the note in `PROCESS.md` on what a back-filled note cannot tell you. Machine output stays local.*
