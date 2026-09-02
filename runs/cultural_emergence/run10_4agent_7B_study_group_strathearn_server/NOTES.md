# run10_4agent_7B_study_group_strathearn_server

*cultural emergence — population-size grid, 4 agents on strathearn*

|  |  |
|---|---|
| Config | [`configs/cultural_emergence/run10_4agent_7B_study_group_strathearn_server.yaml`](../../../configs/cultural_emergence/run10_4agent_7B_study_group_strathearn_server.yaml) |
| Model | `Qwen2.5-7B-Instruct` |
| Population | 4 agents on `study_group_strathearn_server` |
| Settings | kl_coef 0.05, β 0.01, lr 5e-06, LoRA r 64 |
| Ran to | iteration 300 of 300 · 12 checkpoints |
| **Differs from** | **`run8_16agent_7B_study_group_strathearn_server` — population size only (4 agents instead of 16)** |

## What it was for

The small end of the substrate × population-size grid. These runs supply the
2- and 4-agent cells of the population-size figures.

## Headline

**The same shape as its Ashbourne twin: 1.00 alive to iteration 150, 0.75
after.** Mean KL from base reaches 2.34, the highest of any run in the
population-size grid, driven by one committed agent in a population of four.

## Core metrics

Population means over the first and last 10% of logged iterations (blocks of 30 of 300):

| metric | first block | last block |
|---|---:|---:|
| perception loss | 1.474 | 0.981 |
| KL from base | 0.007 | 2.338 |
| policy entropy | 0.446 | 0.451 |

Regime composition (paper thresholds; see `PROCESS.md`):

| iteration | n | alive | composition |
|---:|---:|---:|---|
| ~25 | 4 | 1.00 | coherent_low_kl 4 |
| ~100 | 4 | 1.00 | coherent_low_kl 4 |
| ~150 | 4 | 1.00 | coherent_low_kl 4 |
| ~200 | 4 | 0.75 | coherent_low_kl 3, policy_collapse 1 |
| ~300 | 4 | 0.75 | coherent_low_kl 2, committed 1, policy_collapse 1 |

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
