# 01_pretrained_baseline_cooperative

*Phase A condition 1 — pretrained baseline, cooperative prompt. No training.*

|  |  |
|---|---|
| Config | [`configs/phase_a_substrate_validation/01_pretrained_baseline_cooperative.yaml`](../../../configs/phase_a_substrate_validation/01_pretrained_baseline_cooperative.yaml) |
| Model | `—` |
| Ran to | **no metrics written** |
| **Differs from** | **every other Phase A condition — this one does not train; it is the reference the others are supposed to beat** |

## What it was for

The reference behaviour of the untrained model on deal-or-no-deal. Phase A's
claim is that CCSM training produces behaviour that (a) differs from the
pretrained baseline, (b) varies with the character prompt, and (c) improves on
external metrics. This run is the (a) and (c) denominator.

## Headline

**The substrate has almost no headroom to measure, because the base model almost
never closes a deal.** Over 64 evaluation episodes the cooperative prompt reaches
agreement in **4.7%** of them (39.0 mean joint utility across the
agreeing episodes, Pareto efficiency 0.036).

An external metric that fires on 2–3 episodes in 64 cannot show an improvement
of any size that this project could have produced.

## Core metrics

_No metrics were written._

## What this settles

- **Qwen2.5-1.5B-Instruct reaches agreement on 4.7% of deal-or-no-deal
  episodes with no training.** n = 64. — *established*
- **Agreement rate is a near-floor metric on this substrate**, so Phase A's
  criterion (c) — improvement on environment-relevant external metrics — was
  close to unmeasurable before any training was run. — *established*

## What it does not settle

Whether a different external metric would have had headroom. Joint utility over
*all* episodes (1.83 cooperative, 0.95 competitive) is not floored in the same
way and was never used as the criterion.

## Status

`CLOSED`. This is the only Phase A condition that was ever evaluated.

---
*Analysed retrospectively on 2026-09-01 from `metrics.jsonl` and `config.json`; see the note in `PROCESS.md` on what a back-filled note cannot tell you. Machine output stays local.*
