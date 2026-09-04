# c11_discount_normalisation

*Tier C item C11 — three runs testing the two settings that shaped every result
in this project and that nobody chose. A fourth (PopArt) is queued.*

|  |  |
|---|---|
| Configs | [`configs/cultural_emergence/c11_discount_normalisation/`](../../../configs/cultural_emergence/c11_discount_normalisation/README.md) |
| Model | `Qwen2.5-7B-Instruct`, LoRA r64, 8 agents on `study_group_ashbourne_gc` |
| Ran to | iteration 300 of 300, all three · jobs 6280553, 6280551, 6284325 |
| **Differs from** | **`run7_8agent_7B_study_group_ashbourne_gc`** — γ, return normalisation, and the arm denominator. Nothing else. |

## What it was for

γ=0.99 is applied *positionally per token*, a 100-token horizon against a
384-token post, producing a 48× within-post credit gradient where γ=1 is flat.
It is not in the derived objective — `surprise_minimisation_derivation.md` §6.2
introduces it as variance reduction "at the cost of bias". Return normalisation
is free in RL, where only advantage *ordering* matters, but not here, where the
action term is summed with a supervised term in nats. Neither key was reachable
from any config before 2026-09-03.

## Headline

**The discount is a real but secondary effect; the normalisation is load-bearing
only because the two arms are badly mis-scaled.** γ=1 with normalisation kept
ends at `alive` 0.50 where the reference ends at 0.00 — a genuine improvement
from removing the within-post credit gradient. Removing normalisation instead
**destroys the run**: perception loss reaches **37**, entropy **0.00**, and
every agent emits nothing at all.

## Core metrics

| arm | alive @300 | perception loss | KL | entropy | well-formed posts @300 |
|---|---|---|---|---|---|
| γ=1, normalisation ON | **0.50** | 2.17 → 2.15 | 0.01 → 0.91 | 0.75 → 0.91 | 32/64 = 0.50 |
| γ=1, normalisation OFF | 0.00 | 2.13 → **37.92** | 0.03 → 2.97 | 0.74 → **0.00** | 2/64 = 0.03 |
| pure surrogate | 0.00 | 2.16 → **37.18** | 0.03 → 0.78 | 0.75 → **0.00** | **0/64 = 0.00** |
| REF `run7` (γ=0.99, norm ON) | 0.00 | 2.16 → 2.27 | 0.02 → 1.32 | 0.74 → 0.80 | 36/64 = 0.56 |

`alive` at 25 / 100 / 150 / 200 / 300:

| arm | | | | | |
|---|---|---|---|---|---|
| γ=1, norm ON | 0.00 | 0.62 | 0.75 | 0.75 | **0.50** |
| γ=1, norm OFF | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 |
| pure surrogate | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 |
| REF `run7` | 0.00 | 1.00 | 0.88 | 0.88 | 0.00 |

## The traces, at iteration 300

**Pure surrogate** — every agent, every turn, empty:

> **Beatrice Okafor:** *(nothing)* · **Imogen Carter:** *(nothing)* ·
> **Sam Pritchard:** *(nothing)* · **Hana Yilmaz:** *(nothing)*

62 of 64 turns deliver nothing. This is the dark-room outcome in its purest
observable form, and it is *worse* than C10's action-only arm (24–30 missing of
64), which is what the pre-registered prediction said would happen.

**γ=1 with normalisation** — the reference's mixed picture, not a collapse:

> **Sam Pritchard:** so what's everyone been up to this week? i swear the only
> reason anyone shows up for davies' meetings…
> **Hana Yilmaz:** hey sam, iz anyone else bored of the same-old-same-old?…
> **Priya Shah:** `?", ?</?`

## What this settles

- **The un-normalised arms are not a test of the derived objective.** The
  prediction written in their headers before submission was that at γ=1 the raw
  return is ~2300 against a perception loss of ~1.4, so the action arm enters
  the summed loss ~1600× larger, making it an action-only run in all but name.
  That is exactly what happened, and more severely than C10's action-only arm
  because γ=1 enlarges the return a further ~300×. — *established* — this family
  against `runs/cultural_emergence/c10_term_ablation/NOTES.md`
- **The arm denominator is irrelevant once the scale mismatch dominates.**
  Pure surrogate (obs denominator) and γ=1-no-norm (act denominator) differ by
  nothing that matters: perception 37.18 vs 37.92, 62 vs 58 missing turns. —
  *established* — same
- **Removing the discount alone helps survival.** `alive` 0.50 at iteration 300
  against the reference's 0.00, with perception, entropy and channel otherwise
  comparable. The 48× within-post credit gradient is doing some damage, but it
  is not the main driver. — *provisional*, one seed — same
- **Normalisation is currently load-bearing, and that is a statement about the
  mismatch rather than a vindication.** With the arms mis-scaled by ~1600×,
  standardising the return is the only thing holding the action term to a
  comparable magnitude. — *likely* — same

## What it does not settle

- **Whether the derived objective is trainable with a working baseline.** These
  runs have an *inert* baseline: with normalisation off, the value head
  regresses onto returns of ~2300 with plain MSE, which is the conditioning
  problem that motivated standardising the target originally. The PopArt arm
  (job 6310961) is the test, and until it reports, "the derived objective does
  not train" is **not** established — only "it does not train with a baseline
  that cannot fit it".
- **Anything at more than one seed.** Every arm here is seed 1. Given that this
  project's full loss is seed-unstable at the level that decides survival, the
  γ=1 improvement in particular needs replication before it is more than
  provisional.
- **Whether γ=1 helps for the reason claimed.** It removes the within-post
  gradient *and* lengthens the horizon from 100 tokens to the episode. Nothing
  here separates those.
