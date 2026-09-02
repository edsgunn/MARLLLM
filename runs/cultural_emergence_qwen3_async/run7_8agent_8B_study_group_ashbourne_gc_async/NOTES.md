# run7_8agent_8B_study_group_ashbourne_gc_async

*throughput — asynchronous rollout with PPO-clipped action loss, Qwen3-8B*

|  |  |
|---|---|
| Config | [`configs/cultural_emergence_qwen3_async/run7_8agent_8B_study_group_ashbourne_gc_async.yaml`](../../../configs/cultural_emergence_qwen3_async/run7_8agent_8B_study_group_ashbourne_gc_async.yaml) |
| Model | `Qwen3-8B` |
| Population | 8 agents on `study_group_ashbourne_gc` |
| Settings | kl_coef 0.05, β 0.01, lr 5e-06, LoRA r 64 |
| Ran to | iteration 123 of 300 — **truncated** · 4 checkpoints |
| **Differs from** | **`runs/cultural_emergence_qwen3/run7_8agent_8B_study_group_ashbourne_gc` — a background thread collects rollouts into a bounded buffer while the trainer consumes them (`replay_buffer_size` 2, `max_staleness` 1), and the action loss becomes a PPO-clipped surrogate instead of REINFORCE** |

## What it was for

Phase 2 of the off-policy roadmap in `RLvr_rollout_optimisation_audit.md`:
overlap rollout and training, and bound the resulting staleness with a shallow
replay buffer and a PPO clip.

## Headline

**The async path is behaviourally indistinguishable from the synchronous one and
gets further in the same wall time.** Perception loss ends at 1.57 against
its synchronous sibling's 1.71 / 1.62, entropy and KL match to two decimals, and
this run reached iteration 123 where the sibling reached 102.

That is the result the change was made for: **one-step-stale off-policy rollouts
under a PPO clip cost nothing measurable here.**

## Core metrics

Population means over the first and last 10% of logged iterations (blocks of 12 of 123):

| metric | first block | last block |
|---|---:|---:|
| perception loss | 1.919 | 1.572 |
| KL from base | 0.001 | 0.085 |
| policy entropy | 0.388 | 0.652 |

Regime composition (paper thresholds; see `PROCESS.md`):

| iteration | n | alive | composition |
|---:|---:|---:|---|
| ~25 | 8 | 1.00 | mid_range 8 |
| ~100 | 8 | 1.00 | coherent_low_kl 8 |

## What this settles

- **Async rollout with `max_staleness` 1 and a PPO clip reproduces synchronous
  training on this substrate.** Perception loss, entropy and KL all match. —
  *likely*, two matched pairs

## What it does not settle

The size of the throughput win, which was never measured properly — comparing
iteration counts across runs that ended for different reasons is not a
benchmark. And whether the equivalence holds at deeper staleness, which is the
setting that would actually buy something.

## Status

`TRUNCATED` at iteration 123 of 300. Cite the equivalence, not a speedup number.

---
*Analysed retrospectively on 2026-09-01 from `metrics.jsonl` and `config.json`; see the note in `PROCESS.md` on what a back-filled note cannot tell you. Machine output stays local.*
