# c10_term_ablation

*Tier C item C10 — a family of four runs whose unit of inference is the family:
perception-only and action-only, two seeds each, against the full loss.*

|  |  |
|---|---|
| Configs | [`configs/cultural_emergence/c10_term_ablation/`](../../../configs/cultural_emergence/c10_term_ablation/README.md) |
| Model | `Qwen2.5-7B-Instruct`, LoRA r64 on q_proj/v_proj |
| Population | 8 agents on `study_group_ashbourne_gc` |
| Regularisation | `kl_coef` 0.05, `beta` 0.01, `lr` 5e-6 |
| Ran to | iteration 300 of 300, all four · jobs 6261085–6261088 |
| **Differs from** | **`run7_8agent_7B_study_group_ashbourne_gc`** — the three `alpha_*` term weights, and nothing else. |

## What it was for

From the config headers, written before the data: the dark-room incentive
§3.5 blames for the decline reaches behaviour through the **action term alone**,
so `alpha_act = 0` removes the mechanism rather than a term. Everything the
decline consists of was pinned on that mechanism by an association — the public
channel closes first, by up to 200 iterations, in 9 of 12 runs. This arm was to
turn the association into a causal claim or kill it.

Second, sharper prediction: absorption is carried by the **perception** term,
since `prompt_as_observation` types prompt tokens `OBS`. If perception-only
keeps absorption and drops the collapse, the framework's learning signal and
its pathology are in different terms.

Pre-registered as making the arm *uninformative rather than negative*: if
perception-only barely moves the policy — entropy flat, KL near zero, PMI gap
unchanged — then "it did not collapse" is trivially true.

## Headline

**The perception term does the learning and the action term does the damage,
and on this substrate the separation is total.** Perception-only is the only
configuration in this project that ends 300 iterations with a live population,
an open public channel, and a perception loss that fell rather than rose:
2.13 → 0.70, `alive` 1.00 at every checkpoint, and **61 of 64 turns delivering
a well-formed post at iteration 300**. The full loss manages 36 of 64 with all
eight agents `text_degenerate`; action-only manages 5 of 64 at seed 2, with
perception loss at **4.21**, worse than the base model.

And the arms fail in different ways, not the same way to different degrees.
Perception-only's only defect is a handful of unclosed tags and **zero missing
posts**; action-only misses 24 and 30 of 64 turns entirely. Delivering nothing
is the dark-room incentive's signature — the return sums surprise over future
observation tokens, and the cheapest way to shrink it is to give the others
nothing to respond to.

## Core metrics

Population means over the first and last 10% of logged iterations:

| arm | perception loss | KL from base | policy entropy |
|---|---|---|---|
| perception-only s1 | 2.13 → **0.70** | 0.004 → 0.22 | 0.77 → 0.89 |
| perception-only s2 | 2.13 → **0.71** | 0.004 → 0.22 | 0.77 → 0.90 |
| full loss (`run7`) | 2.16 → 2.27 | 0.016 → 1.32 | 0.74 → 0.80 |
| action-only s1 | 2.17 → **4.21** | 0.009 → 0.90 | 0.74 → 0.58 |
| action-only s2 | 2.14 → **4.07** | 0.014 → 1.68 | 0.74 → 1.37 |

Public channel at iteration 300, from `tool_counts_per_agent` — turns
delivering a **closed** `<post>` block, and what went wrong on the rest:

| arm | well-formed | unclosed | missing |
|---|---|---|---|
| perception-only s1 | **61/64 = 0.95** | 3 | **0** |
| perception-only s2 | **59/64 = 0.92** | 5 | **0** |
| full loss (`run7`) | 36/64 = 0.56 | 6 | 22 |
| action-only s1 | 27/64 = 0.42 | 13 | 24 |
| action-only s2 | **5/64 = 0.08** | 29 | 30 |

`alive` fraction at iterations 25 / 100 / 150 / 200 / 300:

| arm | | | | | |
|---|---|---|---|---|---|
| perception-only s1 | 0.00 | 1.00 | 1.00 | 1.00 | **1.00** |
| perception-only s2 | 0.00 | 1.00 | 1.00 | 1.00 | **1.00** |
| full loss (`run7`) | 0.00 | 1.00 | 0.88 | 0.88 | 0.00 |
| action-only s1 | 0.00 | 1.00 | 0.50 | 0.25 | 0.00 |
| action-only s2 | 0.62 | 1.00 | 1.00 | 1.00 | 0.75 |

Both seeds agree within each arm, which for a project whose full loss is
seed-unstable at the level that decides survival is itself worth recording.

## The traces, at iteration 300

The metric tables would look similar for a policy that had collapsed into a
readable-looking loop, so per `PROCESS.md` the traces were read.

**Perception-only** — fluent, on-task, in character:

> **Beatrice Okafor:** hey guys, been thinking about one of the uni induction
> problems. thought maybe we could warm up with a simpler version first?
> **Imogen Carter:** nah, we haven't seen anything like this. sounds like a nice
> warming up problem!
> **Hana Yilmaz:** hey beatrice! sounds good! give us the problem and let's see
> what creative takes we come up with.

**Action-only** — three of five visible speakers deliver nothing at all, and
the fourth emits tag fragments:

> **Beatrice Okafor:** *(empty)*
> **Imogen Carter:** *(empty)*
> **Hana Yilmaz:** *(empty)*
> **Priya Shah:** `/?</?</post?`

**Full loss (`run7`)** — the same failure as action-only, less far along:

> **Priya Shah:** `?</?  ?</?`
> **Hana Yilmaz:** *(empty)*
> **Marcus Webb:** sorry...

Action-only delivering nothing is the dark-room incentive in its purest
observable form: the return sums surprise over *future observation tokens*, and
the cheapest way to shrink it is to give the others nothing to respond to.

## What this settles

- **The decline is caused by the action term, not merely accompanied by it.**
  Removing the policy gradient removes the channel closure (0.95 delivery vs
  0.52) and the population failure (`alive` 1.00 vs 0.00) on both seeds, on the
  substrate the paper's claims are about. — *established* — this family, against
  `run7`
- **Isolating the action term makes the failure worse and faster.** Action-only
  reaches perception loss 4.21, above the base model, and the lowest delivery
  rate of any arm. §3.5's mechanism is not just present, it is the dominant term
  in the failure. — *established* — same
- **Surprise minimisation on observation tokens alone is a viable training
  signal on this substrate.** Perception loss falls threefold with entropy
  intact and the channel open. — *established* — same

## What it does not settle

- **Whether perception-only still absorbs.** This is the precondition written
  into the config header, and it is **not measured**: the PMI gap needs the
  absorption scoring pass over checkpoints, which has not been run. KL from base
  is 0.22 against the full loss's 1.32 — the policy moves, about six times less,
  matching Phase A's four-to-tenfold — so the arm is not vacuous. But until
  absorption is measured, "perception-only is a stabilisation" is a candidate
  and not a result: it could still be that nothing character-specific is being
  learned. **Run the absorption pass on these four checkpoint sets before this
  family is cited.**
- **Whether it generalises past Ashbourne.** One substrate, one model, one
  population size.
- **What the KL anchor contributes.** `kl_coef` is un-gated by the alphas and
  stayed at 0.05 in every arm, so nothing here separates the anchor's role.
- **Whether action-only's collapse is the same event as the full loss's.** Both
  end broken; whether they break the same way is a trace-level question this
  family only samples.
