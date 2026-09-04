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

**The decline is not terminal. Both populations bottom out and then partly
recover** — the 8-agent from `alive` 0.12 at iteration 400 to 0.62 at 900, and
the 16-agent from 0.00 at 300 to 0.88 at 400 and 0.81 at 600. But the 16-agent
run **fails its own pre-registered void condition**, so only the 8-agent
trajectory can carry the claim.

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

- **The decline bottoms out and reverses on the 8-agent reference.** `alive`
  falls to 0.12 at iteration 400 and recovers to 0.62 by 900, with perception
  loss falling 2.43 → 1.55 throughout. — *likely*, one run, 12.3% boundary step
  — this run
- **Iteration 200–300 is near the trough, not a plateau.** Every
  population-level claim in this project is drawn in the worst region of a
  non-monotone trajectory. That does not make those claims wrong, but it makes
  the choice of evaluation point load-bearing in a way nobody intended. —
  *likely* — same
- **Divergence does not plateau.** KL from base continues to rise, 1.26 → 2.09
  over iterations 300–900, while the population is recovering — so recovery and
  drift-from-base are not the same axis. — *likely* — same

## What it does not settle

- **The 16-agent trajectory**, which is void by its own criterion. It should be
  re-run from scratch on current code rather than resumed if the 16-agent
  recovery is wanted.
- **Whether the recovery is real or a code artefact.** The honest reading of a
  12.3% step is that the 8-agent extension is *probably* continuous but not
  demonstrably so. The clean version is a from-scratch 900-iteration run on
  current code, which is also what would let the recovery be replicated at more
  than one seed.
- **What recovers.** The public channel does **not**: it falls 0.54 → 0.15 on
  the 8-agent run while `alive` rises. So the regime classifier's recovery and
  the channel's decline are separate phenomena, and the classifier's inputs
  (perception, KL, entropy) can improve while the substrate's actual
  communication keeps degrading. That is worth a closer look than this write-up
  gives it.
