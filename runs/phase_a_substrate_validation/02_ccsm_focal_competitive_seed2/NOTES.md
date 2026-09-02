# 02_ccsm_focal_competitive_seed2

*Phase A condition 2 — full CCSM, focal vs frozen partner, competitive prompt, seed 137*

|  |  |
|---|---|
| Config | [`configs/phase_a_substrate_validation/02_ccsm_focal_competitive_seed2.yaml`](../../../configs/phase_a_substrate_validation/02_ccsm_focal_competitive_seed2.yaml) |
| Model | `Qwen2.5-1.5B-Instruct` |
| Population | 2 agents on `deal_or_no_deal` |
| Settings | kl_coef 0.05, β 0.01, lr 1e-05, LoRA r 0 |
| Ran to | iteration 500 of 500 · 10 checkpoints |
| **Differs from** | **`02_ccsm_focal_competitive` — nominally the training seed only (42 vs 137)** |

## What it was for

The headline Phase A condition: the focal agent trains with the unified loss
against a partner frozen at its pretrained weights, so it faces a non-stationary
conversational partner without the credit-assignment problems of co-training.
The `_seed2` arms exist to estimate variance, and the cooperative/competitive
pair is the prompt-swap diagnostic the whole design turns on.

## Headline

**Surprise minimisation works, and the seed decides whether the policy survives
it.** Perception loss falls 1.09 → 0.44 (means over the first and last
10% of 500 iterations) — a large, real improvement in the agent's ability to
predict its own trajectory, with KL from base of only 0.58.

Policy entropy ends at **0.580**, so this arm
stays inside the coherent band.
Both seed-42 arms collapse; both seed-137 arms do not. Nothing but the seed
separates the cooperative pair.

## Core metrics

Population means over the first and last 10% of logged iterations (blocks of 50 of 500):

| metric | first block | last block |
|---|---:|---:|
| perception loss | 1.087 | 0.436 |
| KL from base | 0.291 | 0.576 |
| policy entropy | 4.142 | 0.580 |

Regime composition (paper thresholds; see `PROCESS.md`):

| iteration | n | alive | composition |
|---:|---:|---:|---|
| ~25 | 1 | 0.00 | policy_collapse 1 |
| ~100 | 1 | 0.00 | policy_collapse 1 |
| ~150 | 1 | 0.00 | policy_collapse 1 |
| ~200 | 1 | 0.00 | policy_collapse 1 |
| ~300 | 1 | 1.00 | coherent_low_kl 1 |

## What this settles

- **CCSM reduces perception loss substantially on this substrate.** ~1.1–1.4 →
  0.38–0.64 in all four arms. — *established*
- **The headline Phase A condition is seed-unstable at the level that matters.**
  Seed 42 ends with entropy ≤ 0.003 under both prompts; seed 137 ends at
  0.58–0.75. — *established*, four arms

## What it does not settle

**Phase A's actual claim.** No trained checkpoint from this condition was ever
scored: there is no `eval_final.json` and no `prompt_conditional_post.json`
anywhere under `runs/`. Criteria (a) *behaviour differs from the pretrained
baseline* and (b) *behaviour varies with the character prompt* are untested, and
(c) was close to unmeasurable — the baseline reaches agreement on 3–5% of
episodes.

## Status

`CLOSED` as training. **The evaluation that defines this condition was never run.**

---
*Analysed retrospectively on 2026-09-01 from `metrics.jsonl` and `config.json`; see the note in `PROCESS.md` on what a back-filled note cannot tell you. Machine output stays local.*
