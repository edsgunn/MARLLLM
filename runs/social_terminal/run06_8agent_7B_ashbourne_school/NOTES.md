# run06_8agent_7B_ashbourne_school

*social terminal — the reference run: complete turn-ending upgrade, full horizon*

|  |  |
|---|---|
| Config | [`configs/social_terminal/run06_8agent_7B_ashbourne_school.yaml`](../../../configs/social_terminal/run06_8agent_7B_ashbourne_school.yaml) |
| Model | `Qwen2.5-7B-Instruct` |
| Population | 8 agents on `ashbourne_school` |
| Settings | kl_coef 0.05, β 0.01, lr 5e-06, LoRA r 64 |
| Ran to | iteration 300 of 300 · 12 checkpoints |
| **Differs from** | **`run05` — `vllm_gpu_mem_util` 0.30 → 0.18, and every information-returning verb (including `<go>` and the self-closing forms) is now a generation stop-point** |

## What it was for

The clean version of the observation-principle upgrade, at a memory setting that
survives. This is the before/after against `run02` on the substrate upgrade, not
on hyperparameters.

## Headline

**The reference run for this substrate, and the second of the project's two
clean full-horizon runs.** 300 of 300 iterations; 6 `coherent_low_kl`, 1
`committed`, 1 `policy_collapse` at the end. Perception loss falls 1.60 → 1.01,
and over 50-iteration blocks it declines in five of six steps (1.55, 1.42, 1.43,
1.37, 1.21, 1.11) — the one reversal is 0.02 nats.

It is also the run that shows the alive fraction *rising* late: 0.75 at
iteration 200, 0.88 at 300 — one of only three runs anywhere in the project
where that happens, and the only one with more than one agent.

## Core metrics

Population means over the first and last 10% of logged iterations (blocks of 30 of 300):

| metric | first block | last block |
|---|---:|---:|
| perception loss | 1.593 | 1.080 |
| KL from base | 0.005 | 0.642 |
| policy entropy | 0.581 | 0.631 |

Regime composition (paper thresholds; see `PROCESS.md`):

| iteration | n | alive | composition |
|---:|---:|---:|---|
| ~25 | 8 | 1.00 | coherent_low_kl 7, mid_range 1 |
| ~100 | 8 | 0.88 | coherent_low_kl 6, mid_range 1, policy_collapse 1 |
| ~150 | 8 | 0.88 | coherent_low_kl 6, mid_range 1, policy_collapse 1 |
| ~200 | 8 | 0.75 | coherent_low_kl 5, committed 1, policy_collapse 2 |
| ~300 | 8 | 0.88 | coherent_low_kl 6, committed 1, policy_collapse 1 |

## What this settles

- **The complete turn-ending upgrade trains stably to 300 iterations.** —
  *likely*, one run
- **Perception loss keeps improving to iteration 300 on this substrate**, where
  every forum run has reversed by then. — *established* for this run
- **The late decline is substrate-dependent, not inherent to the objective.** —
  *likely*; two social-terminal runs at full horizon against 44 forum
  populations that all decline

## What it does not settle

Which part of the upgrade is responsible, and whether the substrate or the
upgrade is doing the work. `run02` on the same substrate without the upgrade
never reached iteration 100, so the two cannot be compared at the horizon that
matters.

## Status

`CLOSED` at iteration 300. **The reference run for the social-terminal substrate.**

---
*Analysed retrospectively on 2026-09-01 from `metrics.jsonl` and `config.json`; see the note in `PROCESS.md` on what a back-filled note cannot tell you. Machine output stays local.*
