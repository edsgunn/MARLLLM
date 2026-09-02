# B8 — does any of this depend on the base model?

|  |  |
|---|---|
| Script | [`scripts/tier_b/b8_model_generation.py`](../../../scripts/tier_b/b8_model_generation.py) |
| Inputs | all `study_group_ashbourne_gc` runs at `kl_coef` 0.05, `beta` 0.01, compared at matched iterations |
| Ran | 2026-09-01, no compute |

## What it was for

Reviewer pd7C objected that Qwen2.5-7B-Instruct is not a "pure pre-training"
model. The rebuttal conceded the epistemics and offered no data. We have four
Qwen3 runs on the reference substrate. Compared at *matched* iterations, because
the Qwen3 runs are all truncated below iteration 125.

## Headline

**The high-drift stratum the paper's absorption result lives in does not appear
in Qwen3 at all.**

| model | iter | runs | mean KL | max KL | mean perc | alive | agents at KL ≥ 1 |
|---|---:|---:|---:|---:|---:|---:|---|
| Qwen2.5-7B | 25 | 11 | 0.040 | 0.290 | 2.205 | 0.18 | 0/150 |
| Qwen2.5-7B | 50 | 11 | 0.182 | 0.632 | 2.081 | 0.28 | 0/150 |
| Qwen2.5-7B | 75 | 11 | 0.383 | 3.479 | 1.908 | 0.39 | 11/150 |
| Qwen2.5-7B | 100 | 15 | **0.910** | **7.572** | 1.862 | 0.58 | **27/182** |
| Qwen3-4B | 100 | 3 | 0.144 | 0.350 | 1.654 | 1.00 | **0/32** |
| Qwen3-8B | 100 | 2 | 0.065 | 0.097 | 1.582 | 1.00 | **0/16** |

The gap widens with training — Qwen2.5-7B's mean KL is 1.0× / 2.0× / 3.2× / 6.3×
Qwen3-4B's at iterations 25/50/75/100, and 5.4× / 7.3× / 9.0× / 14.0× Qwen3-8B's.
**No Qwen3 agent ever reaches KL ≥ 1.0**, against 27 of 182 Qwen2.5-7B agents at
the same iteration.

At the same time the Qwen3 populations are **healthier**: alive 1.00 against
0.58, and lower perception loss (1.58–1.65 against 1.86).

## What this settles

- **Movement from base under this objective is strongly base-model dependent.**
  6–14× less on Qwen3 at matched iterations under identical settings. —
  *likely*
- **"Commitment" is not something the objective reliably produces; it is
  something Qwen2.5-7B-Instruct does under it** at these horizons. Every claim
  conditioned on that stratum inherits the dependence. — *likely*
- **The failure modes are also model-dependent.** Qwen3 populations stay fully
  alive where Qwen2.5-7B is at 0.58, so the decline is not inherent to the
  objective either. — *provisional*, 5 runs, short horizons

## What it does not settle

The Qwen3 runs stop at iterations 92–123, so they may develop a high-drift
stratum later. Against that: at iteration 75 Qwen2.5-7B already has 11/150
agents past KL 1.0 and Qwen3 has 0/32 with a maximum of 0.235, so it is not
simply less time.

And it says nothing about pd7C's actual point. Qwen3 is another
instruction-tuned model, not a different *kind* of prior. Bounding
model-specificity within a family is not the same as testing whether instruction
tuning matters, which needs an open-corpus base such as Olmo-3 and has never been
run.

## Status

`CLOSED`. Cite as a robustness bound and as a limitation on the commitment
result, not as an answer to the instruction-tuning objection.

---
*Analysed 2026-09-01. Machine output stays local.*
