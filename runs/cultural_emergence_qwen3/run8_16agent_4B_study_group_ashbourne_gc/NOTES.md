# run8_16agent_4B_study_group_ashbourne_gc

*model generation — Qwen3-4B on the reference Ashbourne GC substrate, 16 agents*

|  |  |
|---|---|
| Config | [`configs/cultural_emergence_qwen3/run8_16agent_4B_study_group_ashbourne_gc.yaml`](../../../configs/cultural_emergence_qwen3/run8_16agent_4B_study_group_ashbourne_gc.yaml) |
| Model | `Qwen3-4B` |
| Population | 16 agents on `study_group_ashbourne_gc` |
| Settings | kl_coef 0.05, β 0.01, lr 5e-06, LoRA r 64 |
| Ran to | iteration 92 of 300 — **truncated** · 3 checkpoints |
| **Differs from** | **`runs/cultural_emergence/run8_16agent_7B_study_group_ashbourne_gc` — base model generation only (Qwen3 instead of Qwen2.5-7B)** |

## What it was for

Whether the forum campaign's results survive a model-generation change. Qwen3
brings native thinking and a different chat template, so this is a portability
check on the substrate and the pipeline as much as a scaling datapoint.

## Headline

**Qwen3 runs the substrate and drifts far less than Qwen2.5 does.** Perception
loss falls 2.34 → 1.72 in 92 iterations with the population fully
alive, but **KL from base ends at 0.03–0.19** across all four Qwen3 arms against
1.3–1.6 for the Qwen2.5-7B runs on the same substrate at the same iterations.

No Qwen3 agent reaches the `committed` regime in any of these runs. Whatever
produces committed agents on Qwen2.5-7B is not happening here at this horizon.

## Core metrics

Population means over the first and last 10% of logged iterations (blocks of 9 of 92):

| metric | first block | last block |
|---|---:|---:|
| perception loss | 2.338 | 1.721 |
| KL from base | 0.001 | 0.094 |
| policy entropy | 0.257 | 0.407 |

Regime composition (paper thresholds; see `PROCESS.md`):

| iteration | n | alive | composition |
|---:|---:|---:|---|
| ~25 | 16 | 0.00 | text_degenerate 16 |
| ~100 | 16 | 1.00 | coherent_low_kl 12, mid_range 4 |

## What this settles

- **The substrate and pipeline port to Qwen3 without pathology.** All four arms
  alive at every measured iteration. — *likely*
- **Qwen3 policies barely move from base under the same settings.** End KL
  0.03–0.19 against 1.3–1.6 for Qwen2.5-7B. — *likely*, four arms

## What it does not settle

Whether the difference is the model, the chat template, or the horizon — none of
these runs reaches iteration 150, and the Qwen2.5 runs are below KL 0.7 there
too. The comparison needs matched horizons and does not have them.

## Status

`TRUNCATED` at iteration 92 of 300. Usable as a portability check, not as a scaling result.

---
*Analysed retrospectively on 2026-09-01 from `metrics.jsonl` and `config.json`; see the note in `PROCESS.md` on what a back-filled note cannot tell you. Machine output stays local.*
