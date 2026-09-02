# run7_8agent_7B_robotic_athanor

*cultural emergence — 8-agent substrate set, Robotic Athanor*

|  |  |
|---|---|
| Config | [`configs/cultural_emergence/run7_8agent_7B_robotic_athanor.yaml`](../../../configs/cultural_emergence/run7_8agent_7B_robotic_athanor.yaml) |
| Model | `Qwen2.5-7B-Instruct` |
| Population | 8 agents on `robotic_athanor` |
| Settings | kl_coef 0.05, β 0.01, lr 5e-06, LoRA r 64 |
| Ran to | iteration 100 of 100 · 4 checkpoints |
| **Differs from** | **`run6_2agent_7B_robotic_athanor` — 8 agents instead of 2** |

## What it was for

Whether the Robotic Athanor scenario — the most stylised of the substrates,
with personae that define a register rather than a topic — supports an
8-agent population.

## Headline

**All eight agents are `text_degenerate` for the whole run, and the reason is
that perception loss never moves.** It sits at 2.18 at the start and 2.12 at
the end, flat within noise, while KL from base climbs to 0.97. The population
is not degrading — it is *static* at a baseline that happens to be above the
degenerate threshold.

This matters because it corrects the reading in
[`_absorption_by_regime/REPORT.md`](../_absorption_by_regime/REPORT.md), which
describes these agents' action distributions as degrading "into low-entropy
gibberish". Entropy is 1.07 → 1.02, neither low nor falling. The honest
statement is that **Robotic Athanor agents never learn to predict their
substrate at all**, so absorption is undefined for them — which supports the
report's conclusion (the "LoRA capacity" explanation is unnecessary) by a
different route than the report gives.

## Core metrics

Population means over the first and last 10% of logged iterations (blocks of 10 of 104):

| metric | first block | last block |
|---|---:|---:|
| perception loss | 2.177 | 2.115 |
| KL from base | 0.001 | 0.970 |
| policy entropy | 1.068 | 1.025 |

Regime composition (paper thresholds; see `PROCESS.md`):

| iteration | n | alive | composition |
|---:|---:|---:|---|
| ~25 | 8 | 0.00 | text_degenerate 8 |
| ~100 | 8 | 0.00 | text_degenerate 8 |

## What this settles

- **Robotic Athanor is unusable as a substrate for this objective.** Perception
  loss is flat at its initial value across 100 iterations at both 2 and 8
  agents. — *established*, two runs
- **`text_degenerate` conflates "never learned" with "broke".** These agents
  are classified degenerate at iteration 25, before training could have broken
  anything. — *established*

## What it does not settle

*Why* the substrate is unlearnable. The candidates — persona register too far
from the base model, episode structure, tool absence — are not separated by
anything run here.

## Status

`CLOSED`. Cite for the substrate failure; do **not** cite the absorption numbers for these agents as evidence about absorption.

---
*Analysed retrospectively on 2026-09-01 from `metrics.jsonl` and `config.json`; see the note in `PROCESS.md` on what a back-filled note cannot tell you. Machine output stays local.*
