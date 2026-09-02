# run9_8agent_7B_ashbourne_dual_env

*cultural emergence — one population, two environments*

|  |  |
|---|---|
| Config | [`configs/cultural_emergence/run9_8agent_7B_ashbourne_dual_env.yaml`](../../../configs/cultural_emergence/run9_8agent_7B_ashbourne_dual_env.yaml) |
| Model | `Qwen2.5-7B-Instruct` |
| Population | 8 agents on `study_group_ashbourne_gc + margin_notes_ashbourne` |
| Settings | kl_coef 0.05, β 0.01, lr 5e-06, LoRA r 64 |
| Ran to | iteration 300 of 300 · 12 checkpoints |
| **Differs from** | **`run7_8agent_7B_study_group_ashbourne_gc` — the same eight agents now episode in **two** environments instead of one** |

## What it was for

Whether one population can hold two substrates at once — the precondition for
anything like a culture that travels between contexts rather than being a
property of a single forum.

## Headline

**Splitting the population across two environments destroyed it.** All eight
agents are `text_degenerate` at iterations 25, 100 and 300, with a brief window
at 150–200 where two recover to `mid_range`. Perception loss *rises* 2.34 →
2.50 across the run. The single-environment run on the same roster and
substrate is 88% alive at iteration 200.

## Core metrics

Population means over the first and last 10% of logged iterations (blocks of 31 of 319):

| metric | first block | last block |
|---|---:|---:|
| perception loss | 2.340 | 2.498 |
| KL from base | 0.017 | 1.259 |
| policy entropy | 0.720 | 0.837 |

Regime composition (paper thresholds; see `PROCESS.md`):

| iteration | n | alive | composition |
|---:|---:|---:|---|
| ~25 | 8 | 0.00 | text_degenerate 8 |
| ~100 | 8 | 0.00 | text_degenerate 8 |
| ~150 | 8 | 0.25 | mid_range 2, text_degenerate 6 |
| ~200 | 8 | 0.25 | mid_range 2, text_degenerate 6 |
| ~300 | 8 | 0.00 | text_degenerate 8 |

## What this settles

- **Dual-environment training at this budget destroys the population.** —
  *likely*, one run plus one crashed attempt and one truncated long variant
- **The damage is present from the start**, not a late decline: alive is 0.00 at
  iteration 25 and never recovers above 0.25. This is a different failure from
  the late decline seen everywhere else. — *established*

## What it does not settle

Whether the cause is the environment split itself or the halved per-environment
token budget that comes with it. The `long_dual_env` variant was built to test
exactly that and never produced data.

## Status

`CLOSED`. Negative result; the dual-environment line was not pursued.

---
*Analysed retrospectively on 2026-09-01 from `metrics.jsonl` and `config.json`; see the note in `PROCESS.md` on what a back-filled note cannot tell you. Machine output stays local.*
