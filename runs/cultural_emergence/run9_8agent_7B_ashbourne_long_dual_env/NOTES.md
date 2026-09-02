# run9_8agent_7B_ashbourne_long_dual_env

*cultural emergence — dual environment with a longer token budget*

|  |  |
|---|---|
| Config | [`configs/cultural_emergence/run9_8agent_7B_ashbourne_long_dual_env.yaml`](../../../configs/cultural_emergence/run9_8agent_7B_ashbourne_long_dual_env.yaml) |
| Model | `Qwen2.5-7B-Instruct` |
| Population | 8 agents on `study_group_ashbourne_gc + margin_notes_ashbourne` |
| Settings | kl_coef 0.05, β 0.01, lr 5e-06, LoRA r 64 |
| Ran to | iteration 6 of 300 — **truncated** · 0 checkpoints |
| **Differs from** | **`run9_8agent_7B_ashbourne_dual_env` — longer episodes, to test whether the dual-env failure is a token-budget artefact** |

## What it was for

The control for the dual-environment failure: if halving the effective
per-environment budget is what broke the population, a longer episode should
recover it.

## Headline

**Six iterations of data. The control was never run.** Perception loss opens at
1.86 — already much better than the failed run's 2.34, which is suggestive —
but six records settle nothing.

## Core metrics

Population means over the first and last 10% of logged iterations (blocks of 1 of 6):

| metric | first block | last block |
|---|---:|---:|
| perception loss | 1.859 | 1.652 |
| KL from base | 0.000 | 0.001 |
| policy entropy | 0.551 | 0.530 |

## What this settles

- **Nothing.**

## What it does not settle

The question it was built for. The dual-environment failure remains
unattributed, and the cheap control is still outstanding.

## Status

`TRUNCATED` at iteration 6. **This is an open experiment, not a closed one** — see the open questions in `STATE.md`.

---
*Analysed retrospectively on 2026-09-01 from `metrics.jsonl` and `config.json`; see the note in `PROCESS.md` on what a back-filled note cannot tell you. Machine output stays local.*
