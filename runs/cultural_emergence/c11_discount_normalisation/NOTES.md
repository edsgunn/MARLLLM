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

**PopArt fixed the baseline completely and rescued nothing.** With it, the value
head's loss falls from 486,932 to **3.6** after one update and stays O(10) for
300 iterations; without it the same run's value loss *diverges* to
**145,106,008**. A seven-million-fold improvement in the fit — and the run still
ends with perception loss 39.6, entropy 0.00 and 7 well-formed posts in 64
turns.

That result is informative precisely because the mechanism worked. **The scale
mismatch is not a baseline problem.** Even a perfectly fitted baseline leaves an
advantage on the order of the return's *spread*, which PopArt measures directly
as σ = 1552 by iteration 300 — so the action arm still enters the summed loss
~1000× larger than the perception arm. Fixing what the baseline predicts does
not change the scale of what is left over.

**And γ=1 does not help after all.** The earlier version of this write-up said
it did, on `alive` 0.50 against the reference's 0.00. By well-formed post rate —
the measure `alive` has since been shown unable to carry
(`runs/cultural_emergence/c6_extended_horizon/NOTES.md`) — it is **0.50 against
the reference's 0.56**, i.e. no improvement and possibly slightly worse. That
claim is withdrawn.

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

## Did PopArt work? Two separate questions, two answers

| | with PopArt | without |
|---|---|---|
| value loss, iteration 1 | 486,932 | 419,047 |
| value loss, iteration 100 | **14.2** | 1,542,080 |
| value loss, iteration 300 | **20.8** | **145,106,008** |
| perception loss @300 | 39.60 | 37.18 |
| entropy @300 | 0.00 | 0.00 |
| well-formed posts @300 | 7/64 | 0/64 |

PopArt's own state tracks a genuinely moving target: μ 697 → 861 and
**σ 344 → 1552** over 300 iterations. The return's scale grows 4.5×, which is
the non-stationarity that motivated the whole question — and confirms that a
per-batch standardisation was papering over something real rather than nothing.

## What this settles

- **PopArt does what it claims.** The value head goes from diverging by eight
  orders of magnitude to a stable O(10) loss, with the value kept in the
  return's own units. — *established* — this arm against the same config
  without it
- **A working baseline is not sufficient to make the derived objective
  trainable.** Both pure-surrogate arms end degenerate; the one with a fitted
  baseline delivers 7 posts of 64 instead of 0, which is not a rescue. — *established* — same
- **The scale mismatch is an arm-weighting problem, not a baseline problem.**
  The advantage's scale is the return's spread (σ = 1552), which a baseline
  centres but cannot shrink. Any fix has to change the *relative weight* of the
  two arms — normalisation, an `alpha_act` set against the raw scale, or a
  return expressed as a mean in nats rather than a sum. — *established* — same
- ~~**Removing the discount alone helps survival.**~~ **Withdrawn.** It rested
  on `alive`; by delivery it is 0.50 against the reference's 0.56. — *withdrawn*
  — this arm

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
