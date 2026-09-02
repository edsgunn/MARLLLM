# 03_perception_only_competitive

*Phase A condition 3 — perception only (α_act = α_val = 0), competitive prompt*

|  |  |
|---|---|
| Config | [`configs/phase_a_substrate_validation/03_perception_only_competitive.yaml`](../../../configs/phase_a_substrate_validation/03_perception_only_competitive.yaml) |
| Model | `Qwen2.5-1.5B-Instruct` |
| Population | 2 agents on `deal_or_no_deal` |
| Settings | kl_coef 0.05, β 0.01, lr 1e-05, LoRA r 0 |
| Ran to | iteration 500 of 500 · 10 checkpoints |
| **Differs from** | **`02_ccsm_focal_competitive` — α_act and α_val set to 0, so only the next-token term on observations trains** |

## What it was for

Isolating the perception term. If perception-only reproduces the full loss's
behaviour, the policy-gradient term is doing nothing.

## Headline

**The perception term does the perception work; the action term does the
damage.** Perception loss falls 0.86 → 0.59 — as low as the full loss
reaches — while KL from base stays at **0.077**, four to ten times smaller
than the full-loss arms, and policy entropy ends at **2.02** instead of
collapsing toward zero.

The improvement in world-model quality is attributable to the perception term
alone. The movement of the *policy* — including condition 2's entropy collapse —
comes from the action term.

## Core metrics

Population means over the first and last 10% of logged iterations (blocks of 50 of 500):

| metric | first block | last block |
|---|---:|---:|
| perception loss | 0.859 | 0.587 |
| KL from base | 0.068 | 0.077 |
| policy entropy | 2.056 | 2.017 |

Regime composition (paper thresholds; see `PROCESS.md`):

| iteration | n | alive | composition |
|---:|---:|---:|---|
| ~25 | 1 | 1.00 | coherent_low_kl 1 |
| ~100 | 1 | 0.00 | policy_collapse 1 |
| ~150 | 1 | 0.00 | policy_collapse 1 |
| ~200 | 1 | 1.00 | coherent_low_kl 1 |
| ~300 | 1 | 0.00 | policy_collapse 1 |

## What this settles

- **Perception loss improvement does not require the policy-gradient term.** —
  *established*, two arms
- **The entropy collapse in the full loss is the action term's doing.** —
  *likely*

## What it does not settle

Whether the action term buys anything in exchange. That needs the external
metrics, which were never computed for any trained checkpoint. The two
conditions built to answer it — `04_action_only_*` and `05_symmetric_ccsm_*` —
were never run.

## Status

`CLOSED` as training. Not evaluated externally.

---
*Analysed retrospectively on 2026-09-01 from `metrics.jsonl` and `config.json`; see the note in `PROCESS.md` on what a back-filled note cannot tell you. Machine output stays local.*
