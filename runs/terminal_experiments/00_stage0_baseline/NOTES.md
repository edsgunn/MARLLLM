# 00_stage0_baseline

*terminal — Stage 0, the calibration run that gates every later stage*

|  |  |
|---|---|
| Config | [`configs/terminal_experiments/00_stage0_baseline.yaml`](../../../configs/terminal_experiments/00_stage0_baseline.yaml) |
| Model | `Qwen2.5-1.5B` |
| Population | 1 agents on `terminal (stage 0)` |
| Settings | kl_coef 0, β 0.01, lr 3e-05, LoRA r 0 |
| Ran to | iteration 500 of 500 · 10 checkpoints |
| **Differs from** | **every later terminal stage — this one has no shell and no action tokens at all; corpus text is streamed straight into the observation channel** |

## What it was for

The framework's own σ-routing check, and the only pre-registered pass/fail
criterion written anywhere in this project. Its config states it:

> loss should be indistinguishable from a continued pretraining run on the same
> corpus. If it diverges significantly upward, there is a bug in the σ_t
> routing, the REINFORCE path, or the loss computation — fix the framework
> before touching later stages.

Every token is σ=1, `action_token_budget` is forced to 0, and there are no
character prompts.

## Headline

**It passes, but only just, and the margin is small enough to be worth saying
plainly.** Perception loss over 500 iterations falls from 2.147 to 2.087 as a
block mean — about 0.06 nats, against a per-iteration standard deviation of
0.07–0.09. The criterion was *do not diverge upward*, and it does not; but
neither does it learn much.

`act_loss`, `mean_return`, `mean_advantage` and `entropy` are identically 0
throughout, which is correct for a stage with no action tokens and is worth
recording as the reference for what an inert action side looks like in
`metrics.jsonl`.

## Core metrics

Population means over the first and last 10% of logged iterations (blocks of 50 of 500):

| metric | first block | last block |
|---|---:|---:|
| perception loss | 2.147 | 2.087 |
| KL from base | 0.000 | 0.000 |
| policy entropy | 0.000 | 0.000 |

Regime composition (paper thresholds; see `PROCESS.md`):

| iteration | n | alive | composition |
|---:|---:|---:|---|
| ~25 | 1 | 0.00 | text_degenerate 1 |
| ~100 | 1 | 0.00 | text_degenerate 1 |
| ~150 | 1 | 0.00 | text_degenerate 1 |
| ~200 | 1 | 0.00 | text_degenerate 1 |
| ~300 | 1 | 0.00 | text_degenerate 1 |

## What this settles

- **σ routing and the loss path do not diverge on pure observation.** 500
  iterations, no upward drift. — *likely*
- **A stage with no action tokens logs `act_loss` and `entropy` as exactly
  zero.** That is the signature to check for when a later stage's action side is
  silently masked out. — *established*

## What it does not settle

Whether the framework *learns* on pure observation. A 0.06-nat improvement over
500 iterations, at a per-iteration sd of 0.07, is not a demonstration of
continued pretraining working — it is a demonstration that nothing is broken.
The comparison the criterion actually names — an ordinary continued-pretraining
run on the same corpus — was never run alongside it.

## Status

`CLOSED`. Passes its gate. The stages it gates were never completed.

---
*Analysed retrospectively on 2026-09-01 from `metrics.jsonl` and `config.json`; see the note in `PROCESS.md` on what a back-filled note cannot tell you. Machine output stays local.*
