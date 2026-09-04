# c6_extended_horizon

*Tier C item C6 — the two reference populations resumed past iteration 300.
Jobs 6260989 (8-agent → 900) and 6260988 (16-agent → 600).*

|  |  |
|---|---|
| Configs | [`configs/cultural_emergence/c6_extended_horizon/`](../../../configs/cultural_emergence/c6_extended_horizon/README.md) |
| Ran to | 900/900 and 600/600 · 8h39m and 9h12m |
| **Differs from** | **`run7_…_ashbourne_gc`** and **`run8_16agent_…_ashbourne_gc`** — the horizon, and nothing else. |

## What it was for

Whether cultural divergence keeps growing and whether the decline bottoms out.
Neither is visible at any horizon this project had run, and every
population-level claim rests on iteration 200 — a point chosen before the
decline was visible.

## Headline

**The recovery is not real. The populations go quiet, and the regime classifier
rewards them for it.** At iteration 900 the 8-agent reference scores `alive`
0.62 while delivering **5 well-formed posts out of 64 turns**. Four of the five
agents the classifier calls alive deliver **nothing at all**, and the one with
the *lowest* perception loss in the population — Beatrice Okafor at 0.49 — has
not posted once.

The mechanism is mechanical. Perception loss is computed over observation
tokens; as agents stop posting, the observation stream becomes mostly
deterministic scaffolding, which is trivially predictable, so perception loss
falls. And `perc_loss < 1.7` is the classifier's primary gate. Within this run,
delivery and perception loss correlate at **r = +0.85** across iterations
400–900 while `alive` moves the opposite way:

| iter | delivery | mean perception | alive |
|---|---|---|---|
| 400 | 0.422 | 2.34 | 0.12 |
| 500 | 0.281 | 2.04 | 0.38 |
| 600 | 0.266 | 1.60 | 0.50 |
| 700 | 0.234 | 1.44 | 0.62 |
| 900 | **0.078** | 1.44 | **0.62** |

So the question this family was built to answer — does the decline bottom out —
**is not answered**. What is answered is that `alive` cannot be used to answer
it.

## The thread at iteration 900

Thirteen of sixteen turns deliver nothing. One agent emits the `</post>` loop
this project already documented once:

> 0–4 *(five agents, nothing)*
> 5 **Sam Pritchard:** "Let's not dwell on the weather—can anyone solve this:
> prove that the sum of the first n odd numb…"
> 6 **Priya Shah:** `/n?</post?</?</?</?</?</?</?</?</?</?</?…`
> 7–12 *(six agents, nothing)*
> 13 **Sam Pritchard:** "nah, that's more convoluted than necessary…"
> 14–15 *(nothing)*

## Core metrics

`alive` across the extended horizon:

| run | 300 | 400 | 500 | 600 | 700 | 800 | 900 |
|---|---|---|---|---|---|---|---|
| 8-agent → 900 | 0.38 | **0.12** | 0.38 | 0.50 | 0.62 | 0.50 | **0.62** |
| 16-agent → 600 | 0.00 | **0.88** | 0.88 | 0.81 | — | — | — |

| run | perception loss | KL from base | entropy | channel |
|---|---|---|---|---|
| 8-agent → 900 | 2.43 → 1.55 | 1.26 → 2.09 | 0.72 → 0.60 | 0.54 → 0.15 |
| 16-agent → 600 | 1.77 → 1.69 | 1.95 → 1.78 | 0.58 → 1.13 | 0.40 → 0.23 |

## The void condition, checked

The configs pre-registered: *a step change at iteration 301 is code drift, not
dynamics, checkable by comparing iterations 295–300 against 301–306.* The
originals trained on 2026-05 code; these continue on current code.

| run | perception 295–300 | 301–306 | step | verdict |
|---|---|---|---|---|
| 8-agent | 2.170 | 2.437 | **12.3%** | within the 15% threshold |
| 16-agent | 2.363 | 1.913 | **19.0%** | **void** |

The 16-agent run's perception loss *improves* by 19% across the resume
boundary, which is the wrong direction for dynamics and the right one for a
code change. Its recovery to 0.88 at iteration 400 sits immediately after that
step and cannot be separated from it.

The 8-agent run passes, but 12.3% is not nothing, and the threshold was chosen
by me rather than derived. Its recovery is more credible than the 16-agent's
for a second reason: it develops over iterations 400–900, hundreds of
iterations past the boundary, rather than immediately at it.

## What this settles

- **`alive` is invalid as a health measure wherever delivery is low**, because
  its dominant input moves *with* the failure it is meant to detect. Across ten
  (run, iteration) points the correlation between delivery rate and `alive` is
  only **r = +0.23**, and the two reference runs invert: `run7` delivers 56% of
  turns and scores 0.00, `run8` delivers 27% and scores 0.88. — *established* —
  this run, and `runs/cultural_emergence/c2_population_size/NOTES.md`
- **The 8-agent population does not recover; it falls silent.** Delivery goes
  0.42 → 0.08 over iterations 400–900 while KL from base climbs 1.26 → 2.09. —
  *established* — this run
- **The 16-agent extension is void twice over** — the 19% boundary step, and
  now the same silence artefact at 0.156 delivery with `alive` 0.81. —
  *established* — same

## What it does not settle

- **Whether the decline bottoms out.** Unanswered, and not answerable with the
  current classifier. It needs a health measure whose inputs are not driven by
  the failure — delivery rate itself is the obvious candidate, and by that
  measure these runs decline monotonically and never recover.
- **How far the artefact reaches.** It bites wherever delivery is low, which
  includes the 16-agent reference at iteration 300 (0.266 delivery, `alive`
  0.88). Every population-size reading drawn off `alive` needs re-checking
  against delivery before it is cited. C10's and C11's conclusions are
  unaffected because both were reported with delivery alongside and the two
  agree there.
- **Whether the regime thresholds can be repaired** or whether the perception
  input has to be conditioned on delivery. Not attempted here.
