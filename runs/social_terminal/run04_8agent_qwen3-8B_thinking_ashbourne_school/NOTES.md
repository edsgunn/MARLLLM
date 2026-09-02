# run04_8agent_qwen3-8B_thinking_ashbourne_school

*social terminal — Qwen3-8B with native thinking*

|  |  |
|---|---|
| Config | [`configs/social_terminal/run04_8agent_qwen3-8B_thinking_ashbourne_school.yaml`](../../../configs/social_terminal/run04_8agent_qwen3-8B_thinking_ashbourne_school.yaml) |
| Model | `Qwen3-8B` |
| Population | 8 agents on `ashbourne_school` |
| Settings | kl_coef 0.05, β 0.01, lr 5e-06, LoRA r 64 |
| Ran to | iteration 300 of 300 · 12 checkpoints |
| **Differs from** | **`run02` — Qwen3-8B with native `<think>` instead of Qwen2.5-7B, token budget 384 → 1024, 4 episodes per iteration** |

## What it was for

Whether a model with native chain-of-thought behaves differently on this
substrate. The environment treats `<think>…</think>` as text outside a tool tag
— private to the actor and ignored — so no environment change was needed.

## Headline

**One of only two runs in this entire project that reached its configured
horizon in a healthy state.** 300 of 300 iterations, 6 `coherent_low_kl` and 2
`committed`, alive 1.00 at both iteration 200 and 300. Perception loss falls
1.90 → 0.91 as a block mean, declining across all six 50-iteration blocks
(1.84, 1.58, 1.46, 1.40, 1.27, 0.99) — one of ten runs in the project that never
reverse, and the largest fall of any of them at forum scale. Mean KL from base
reaches 0.57.

The contrast with the forum campaign is stark: across 44 cultural-emergence
populations the mean alive fraction falls from 0.86 at iteration 100 to 0.47 at
300, and **not one** improves after iteration 150. This run does not decline at
all.

## Core metrics

Population means over the first and last 10% of logged iterations (blocks of 30 of 300):

| metric | first block | last block |
|---|---:|---:|
| perception loss | 1.899 | 0.910 |
| KL from base | 0.006 | 0.565 |
| policy entropy | 0.310 | 0.674 |

Regime composition (paper thresholds; see `PROCESS.md`):

| iteration | n | alive | composition |
|---:|---:|---:|---|
| ~25 | 8 | 1.00 | coherent_low_kl 3, mid_range 5 |
| ~100 | 8 | 1.00 | coherent_low_kl 8 |
| ~150 | 8 | 1.00 | coherent_low_kl 8 |
| ~200 | 8 | 1.00 | coherent_low_kl 8 |
| ~300 | 8 | 1.00 | coherent_low_kl 6, committed 2 |

## What this settles

- **A native-thinking model on this substrate trains to 300 iterations without
  losing an agent.** — *provisional*, one run
- **The late decline is not universal across substrates.** Both completed
  social-terminal runs end alive; every completed forum run does not. —
  *likely*

## What it does not settle

Which of the three differences does the work — the model, the thinking tokens,
or the substrate. `run06` holds the substrate fixed with Qwen2.5-7B and also
survives, which points at the substrate; but that is two runs against one
comparison.

## Status

`CLOSED` at iteration 300. One of the project's two clean full-horizon runs.

---
*Analysed retrospectively on 2026-09-01 from `metrics.jsonl` and `config.json`; see the note in `PROCESS.md` on what a back-filled note cannot tell you. Machine output stays local.*
