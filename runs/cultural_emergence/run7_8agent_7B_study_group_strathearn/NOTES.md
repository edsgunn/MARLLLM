# run7_8agent_7B_study_group_strathearn

*cultural emergence — Study Group with the Strathearn roster*

|  |  |
|---|---|
| Config | [`configs/cultural_emergence/run7_8agent_7B_study_group_strathearn.yaml`](../../../configs/cultural_emergence/run7_8agent_7B_study_group_strathearn.yaml) |
| Model | `Qwen2.5-7B-Instruct` |
| Population | 8 agents on `study_group_strathearn` |
| Settings | kl_coef 0.05, β 0.01, lr 5e-06, LoRA r 64 |
| Ran to | iteration 300 of 100 · 12 checkpoints |
| **Differs from** | **`run7_8agent_7B_study_group` — roster only; same environment description, different eight characters** |

## What it was for

Whether regime composition follows the environment or the cast. The roster is
the only thing that changes.

## Headline

**Changing the cast changes almost nothing.** 5 coherent, 2 committed, 1
policy collapse at iteration 300 against 4/3/1 for the original roster, with
alive 1.00 at iterations 100–200 in both. Perception loss is flatter here
(1.38 → 1.30) but the population-level outcome is the same.

## Core metrics

Population means over the first and last 10% of logged iterations (blocks of 30 of 300):

| metric | first block | last block |
|---|---:|---:|
| perception loss | 1.381 | 1.298 |
| KL from base | 0.003 | 1.301 |
| policy entropy | 0.604 | 0.617 |

Regime composition (paper thresholds; see `PROCESS.md`):

| iteration | n | alive | composition |
|---:|---:|---:|---|
| ~25 | 8 | 1.00 | coherent_low_kl 8 |
| ~100 | 8 | 1.00 | coherent_low_kl 6, committed 2 |
| ~150 | 8 | 1.00 | coherent_low_kl 6, committed 2 |
| ~200 | 8 | 1.00 | coherent_low_kl 6, committed 2 |
| ~300 | 8 | 0.88 | coherent_low_kl 5, committed 2, policy_collapse 1 |

## What this settles

- **Regime composition is a property of the substrate, not of the particular
  characters in it.** — *likely* here; later promoted to *established* by the
  five-seed sets, where Fleiss κ across seeds with a fixed roster is ≈ 0.

## What it does not settle

Whether any *individual* character behaves consistently. That needs the seed
replicates, and the answer there is no.

## Status

`CLOSED` at iteration 300, alive 0.88.

---
*Analysed retrospectively on 2026-09-01 from `metrics.jsonl` and `config.json`; see the note in `PROCESS.md` on what a back-filled note cannot tell you. Machine output stays local.*
