# vllm_smoke

*smoke test — two iterations to prove the vLLM rollout path works*

|  |  |
|---|---|
| Config | [`configs/smoke_test/vllm_smoke.yaml`](../../../configs/smoke_test/vllm_smoke.yaml) |
| Model | `Qwen2.5-1.5B-Instruct` |
| Population | 2 agents on `robotic_athanor` |
| Settings | kl_coef 0.05, β 0.01, lr 1e-05, LoRA r 64 |
| Ran to | iteration 2 of 2 · 0 checkpoints |
| **Differs from** | **nothing — it is not an experiment** |

## What it was for

Confirming end to end that the vLLM rollout path produces episodes, traces,
snapshots and a checkpoint, after the install recipe changed. Two iterations on
a 1.5B model.

## Headline

**The pipeline works.** Two iterations, 86 seconds, traces and a behavioural
snapshot written, checkpoint saved, "Training complete".

Its metrics are meaningless as measurements — perception loss 2.85, entropy
2.56, both agents nominally `text_degenerate` — because 1.5B on Robotic Athanor
for two iterations is not an experiment. **The regime classifier will happily
label a smoke test**, which is worth knowing before someone reads a table of
every run in the project.

## Core metrics

Population means over the first and last 10% of logged iterations (blocks of 1 of 2):

| metric | first block | last block |
|---|---:|---:|
| perception loss | 2.878 | 2.845 |
| KL from base | 0.000 | 0.001 |
| policy entropy | 2.862 | 2.558 |

## What this settles

- **The vLLM rollout path is functional end to end.** — *established*

## What it does not settle

Anything scientific.

## Status

`CLOSED`. Infrastructure check, not an experiment. Never cite its numbers.

---
*Analysed retrospectively on 2026-09-01 from `metrics.jsonl` and `config.json`; see the note in `PROCESS.md` on what a back-filled note cannot tell you. Machine output stays local.*
