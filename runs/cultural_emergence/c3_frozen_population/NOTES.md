# c3_frozen_population

*Tier C item C3 — three runs: one learner training against seven frozen
population members, two learners × two seeds.*

|  |  |
|---|---|
| Configs | [`configs/cultural_emergence/c3_frozen_population/`](../../../configs/cultural_emergence/c3_frozen_population/README.md) |
| Model | `Qwen2.5-7B-Instruct`, LoRA r64 on q_proj/v_proj |
| Population | 8 agents on `study_group_ashbourne_gc`, 7 frozen |
| Ran to | iteration 300 of 300, all three · jobs 6260985–6260987 |
| **Differs from** | **`run7_8agent_7B_study_group_ashbourne_gc`** — seven of the eight members receive no gradient. Nothing else. |

## What it was for

Every failure this project documents was measured in a **co-adapting**
population, leaving two readings unseparated: *the agents drove each other into
it*, and *surprise minimisation does this to any agent against any population*.
Here the environment cannot move, so any drift the learner shows is its own.

## Headline

**Inconclusive, and the design is why.** The three learners' own well-formed
post rates at iteration 300 are **0.00, 0.38 and 1.00** — the full range —
straddling the co-adapting reference's 0.56. With three runs and that spread, the family
separates nothing. Its own config header said two seeds because the full loss is
seed-unstable at exactly the level that decides survival; the result confirms
that instability rather than answering the question. **The frozen-population
control needs five seeds per learner, not one or two.**

## The measurement trap this family walked into

The population-level `tool/post_success` rate reads **0.86–0.97** for these runs
against `run7`'s 0.52, which looks like the channel staying open. It is not.
Seven of eight members are frozen at a fresh LoRA adapter — the identity — so
they present the fluent base model and deliver **56 of 56** posts in every run.
7/8 = 0.875 is the number, and the population metric is measuring the freeze.

The learner's own rate has to be counted from the traces:

| run | learner | learner's well-formed rate @300 | population |
|---|---|---|---|
| `learner_priya_seed1` | Priya Shah | 3/8 = **0.38** (2 unclosed, 3 missing) | 59/64 = 0.92 |
| `learner_priya_seed2` | Priya Shah | 0/8 = **0.00** (8 missing) | 56/64 = 0.88 |
| `learner_marcus_seed1` | Marcus Webb | 8/8 = **1.00** | 64/64 = 1.00 |
| `run7` (all eight learning) | — | 36/64 = 0.56 | — |

Taken from `tool_counts_per_agent` in the checkpoint traces, which counts turns
delivering a **closed** `<post>` block. Counting non-empty thread text instead
would have given 0.62/0.00/1.00 against 0.66 and hidden the difference: an
unclosed block still has its text extracted onto the thread, which is why
`post_unclosed` is a separate category. On `run8` the two measures differ by
more than a factor of two.

The same caution applies to every population-mean metric in these runs: with one
trainable member the "population mean" is one agent, and `pop/n_trainable` is
logged each iteration so the denominator is visible.

## Core metrics

Learner values over the first and last 10% of logged iterations:

| run | perception loss | KL from base | policy entropy | regime @300 |
|---|---|---|---|---|
| `learner_priya_seed1` | 2.20 → 2.53 | 0.003 → 0.50 | 0.71 → **0.00** | `text_degenerate` |
| `learner_priya_seed2` | 2.18 → 1.97 | 0.009 → 0.34 | 0.72 → 0.38 | `mid_range` |
| `learner_marcus_seed1` | 2.17 → 2.15 | 0.021 → **5.24** | 0.74 → 0.91 | `mid_range` |
| `run7`, all eight @300 | 2.16 → 2.27 | 0.016 → 1.32 | 0.74 → 0.80 | 8/8 `text_degenerate` |

## The traces, at iteration 300

Worth reading because they show what the aggregate hides. In
`learner_priya_seed1` the seven frozen members are holding a competent
mathematical conversation while the learner posts nothing:

> **Sam Pritchard:** alright, here we go. let n be the number of students in a
> group, and each student makes exactly two friends…
> **Hana Yilmaz:** sam — nice try but yikes! you forgot to prove you can't have
> more than 2 friends. what if the total number of friendships is odd?
> **Priya Shah:** *(empty)*

Against `run7` at the same iteration, where everyone has gone:

> **Priya Shah:** `?</?  ?</?`
> **Marcus Webb:** sorry...

## What this settles

- **The freeze mechanism works as specified.** Frozen members delivered 56/56 in
  every run, no gradient reached a frozen parameter (the trainer raises rather
  than continues, and all three ran to 300), and `pop/n_frozen` was 7
  throughout. — *established* — this family
- **A single agent minimising surprise against a fixed population can still
  collapse completely.** `learner_priya_seed2` stops delivering entirely and
  `seed1` reaches entropy 0.00. Co-adaptation is therefore **not necessary** for
  the failure. — *likely*, two of three runs — same
- **It is not sufficient to say co-adaptation is irrelevant either**, because
  `learner_marcus_seed1` ends at 1.00 delivery with entropy 0.91, which no
  agent in `run7` does. — *provisional*, one run — same

## What it does not settle

- **The question it was built for.** Whether the decline *needs* a moving
  population is unanswered: 0.00/0.62/1.00 against a reference of 0.66 is not a
  contrast. Five seeds per learner would make it one, and that is the
  recommended next submission for this item.
- **Whether the learner identity or the seed drives the spread.** Priya at two
  seeds gives 0.62 and 0.00; Marcus at one seed gives 1.00. The design cannot
  attribute that difference.
- **The solo-agent and static-transcript arms of C3**, which have neither code
  nor configs.
- **Whether perception loss is comparable across the frozen and co-adapting
  conditions.** The learner is predicting an un-adapted population, and the
  regime classifier's `perc_loss ≥ 2.0` threshold was calibrated on co-adapting
  ones. All three learners sit near that boundary, so their regime labels should
  be read with that in mind rather than as settled classifications.
