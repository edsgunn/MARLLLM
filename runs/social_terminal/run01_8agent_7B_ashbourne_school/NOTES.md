# run01_8agent_7B_ashbourne_school

*social terminal — first training run on the new substrate*

|  |  |
|---|---|
| Config | [`configs/social_terminal/run01_8agent_7B_ashbourne_school.yaml`](../../../configs/social_terminal/run01_8agent_7B_ashbourne_school.yaml) |
| Model | `Qwen2.5-7B-Instruct` |
| Population | 8 agents on `ashbourne_school` |
| Settings | kl_coef 0.05, β 0.01, lr 5e-06, LoRA r 64 |
| Ran to | iteration 112 of 300 — **truncated** · 4 checkpoints |
| **Differs from** | **`runs/cultural_emergence/run10_4agent_7B_study_group_ashbourne_gc` — substrate only. Same model, LoRA rig, KL, β and per-turn token budget; the forum is replaced by three rooms with fog of war.** |

## What it was for

The first run on the social-terminal substrate: eight students moving between a
maths classroom, a common room and a library on a Wednesday afternoon, each
seeing only their current room and the agents in it. Everything else is matched
to a forum run so that the comparison is substrate-only.

## Headline

**The substrate runs, and it is the gentlest one in the project.** All eight
agents are `coherent_low_kl` at every logged iteration; perception loss falls
1.42 → 1.33 with KL from base at 0.25. Nothing pathological appears in 112
iterations.

It is also the run that started the memory problem: `vllm_gpu_mem_util` 0.25 got
it to iteration 112 before the job ended.

## Core metrics

Population means over the first and last 10% of logged iterations (blocks of 11 of 112):

| metric | first block | last block |
|---|---:|---:|
| perception loss | 1.492 | 1.264 |
| KL from base | 0.001 | 0.232 |
| policy entropy | 0.506 | 0.612 |

Regime composition (paper thresholds; see `PROCESS.md`):

| iteration | n | alive | composition |
|---:|---:|---:|---|
| ~25 | 8 | 1.00 | coherent_low_kl 8 |
| ~100 | 8 | 1.00 | coherent_low_kl 8 |

## What this settles

- **A fog-of-war spatial substrate trains without immediate pathology.** —
  *likely*

## What it does not settle

Anything about the horizon: 112 of 300 iterations, and this project's forum runs
look healthy at 112 too.

## Status

`TRUNCATED` at iteration 112. Superseded by `run06` as the reference run.

---
*Analysed retrospectively on 2026-09-01 from `metrics.jsonl` and `config.json`; see the note in `PROCESS.md` on what a back-filled note cannot tell you. Machine output stays local.*
