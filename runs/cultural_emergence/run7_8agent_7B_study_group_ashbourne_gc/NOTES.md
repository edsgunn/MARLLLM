# run7_8agent_7B_study_group_ashbourne_gc

*cultural emergence — Study Group, Ashbourne GC framing; one of the two Figure-1 substrates*

|  |  |
|---|---|
| Config | [`configs/cultural_emergence/run7_8agent_7B_study_group_ashbourne_gc.yaml`](../../../configs/cultural_emergence/run7_8agent_7B_study_group_ashbourne_gc.yaml) |
| Model | `Qwen2.5-7B-Instruct` |
| Population | 8 agents on `study_group_ashbourne_gc` |
| Settings | kl_coef 0.05, β 0.01, lr 5e-06, LoRA r 64 |
| Ran to | iteration 300 of 300 · 12 checkpoints |
| **Differs from** | **`run7_8agent_7B_study_group` — environment description only ('gc' framing instead of the original Study Group text)** |

## What it was for

The 8-agent Ashbourne cell of the substrate × population-size grid. It became
the reference substrate for the KL/β sweep, the migration experiments and the
sanctioning corpus.

## Headline

**This population is alive at the evaluation iteration and entirely
text-degenerate a hundred iterations later.** Alive is 0.88 at iteration 200
(2 coherent, 1 committed, 4 mid-range, 1 text-degenerate) and **0.00 at
iteration 300**, with all eight agents above the perception-loss threshold.
Perception loss falls 2.16 → 1.73 by iteration 131 and then climbs back to
2.36.

This single run reconciles two published tables that appear to contradict each
other: the iter-200 classification table lists this run as mostly alive, and
[`_absorption_by_regime/REPORT.md`](../_absorption_by_regime/REPORT.md) lists
it as 8/8 `text_degenerate`. Both are right — **they are reading different
iterations of the same run**, which is exactly the regime-instability caveat
that report raises.

## Core metrics

Population means over the first and last 10% of logged iterations (blocks of 31 of 315):

| metric | first block | last block |
|---|---:|---:|
| perception loss | 2.163 | 2.271 |
| KL from base | 0.016 | 1.325 |
| policy entropy | 0.738 | 0.801 |

Regime composition (paper thresholds; see `PROCESS.md`):

| iteration | n | alive | composition |
|---:|---:|---:|---|
| ~25 | 8 | 0.00 | text_degenerate 8 |
| ~100 | 8 | 1.00 | mid_range 8 |
| ~150 | 8 | 0.88 | coherent_low_kl 2, mid_range 5, policy_collapse 1 |
| ~200 | 8 | 0.88 | coherent_low_kl 2, committed 1, mid_range 4, text_degenerate 1 |
| ~300 | 8 | 0.00 | text_degenerate 8 |

## What this settles

- **A population can be alive at iteration 200 and wholly degenerate at 300.**
  — *established*, this run directly
- **End-of-training regime and evaluation-iteration regime are different
  measurements and must be labelled as such.** — *established*

## What it does not settle

What causes the reversal at ~iteration 150. KL from base peaks at 3.50 around
iteration 183 and falls back to 1.32, so it is not a monotone drift story.

## Status

`CLOSED` at iteration 300. Cite with the iteration attached, always.

---
*Analysed retrospectively on 2026-09-01 from `metrics.jsonl` and `config.json`; see the note in `PROCESS.md` on what a back-filled note cannot tell you. Machine output stays local.*
