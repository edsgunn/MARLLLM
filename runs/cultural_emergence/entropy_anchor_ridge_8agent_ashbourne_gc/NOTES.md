# entropy_anchor_ridge_8agent_ashbourne_gc

*cultural emergence — does a stronger KL anchor raise the safe ceiling on the
entropy bonus? **The family is the unit of inference.***

|  |  |
|---|---|
| Configs | [`configs/cultural_emergence/entropy_anchor_ridge_8agent_ashbourne_gc/`](../../../configs/cultural_emergence/entropy_anchor_ridge_8agent_ashbourne_gc/) |
| Model | `Qwen2.5-7B-Instruct` |
| Population | 8 agents on `study_group_ashbourne_gc`, `extended_8` roster |
| Grid | (`kl_coef`, `beta`) ∈ {(2.0, 0.01), (2.0, 0.2), (2.0, 0.4), (5.0, 0.4)} × seeds {1, 2} |
| Held fixed | lr 5e-6, LoRA r64, 16 episodes/iter, 300 iterations, γ 0.99, `normalise_returns` true, α 1/1/1, `return_sign` +1 |
| Ran | 2026-09-08, jobs 6404722–6404729, all 8 cells to iteration 300 |
| **Differs from** | **[`../kl_beta_sweep_8agent_ashbourne_gc/`](../kl_beta_sweep_8agent_ashbourne_gc/NOTES.md)** — same substrate and rig; extends `kl_coef` beyond 0.5 for the first time, and gives each cell two seeds. |

## The question

Can policy entropy be made to rise without the perception loss collapsing?

## Why this grid, and not another sweep of the old one

The 20 existing cells were re-read on 2026-09-08
([the sweep's NOTES](../kl_beta_sweep_8agent_ashbourne_gc/NOTES.md), and
[`scripts/analyse_kl_beta_grid.py`](../../../scripts/analyse_kl_beta_grid.py)).
That re-reading answered the question **negatively** for everything already run,
and in doing so identified the one direction left untested:

- The entropy bonus is a **step function**. `beta` ≤ 0.05 moves entropy by at
  most 0.12 in either direction (9 cells); `beta` ≥ 0.10 moves it by at least
  +0.16 (11 cells). Complete separation, MWU exact p = 1.2 × 10⁻⁵. No `beta`
  raises entropy a little.
- Above the step, **entropy and perception loss are locked**: Spearman
  Δentropy vs Δperception = +0.853, p = 4.4 × 10⁻¹², n = 20.
- **But the anchor fights the bonus.** At `beta` = 0.10, Δentropy falls
  +1.04 → +0.92 → +0.79 → +0.16 as `kl_coef` goes 0 → 0.01 → 0.10 → 0.50.
  Monotone over four points; the duplicate pair agrees to 0.01.

If the anchor is what stops a large bonus from wrecking the text, the safe
`beta` should rise with `kl_coef`, and the target regime should sit on a diagonal
in (`kl_coef`, `beta`). `beta` = 0.2 already fails at `kl_coef` = 0.5 (Δentropy
+1.19, Δperception +0.71). **`kl_coef` above 0.5 has never been run.**

## Pre-registered readings

| | criterion |
|---|---|
| **Target regime** | Δentropy ≥ +0.30 **and** Δperception ≤ 0, on **both** seeds of a cell |
| **Ridge confirmed** | the target regime at `kl_coef` 2.0 or 5.0 where the same `beta` failed at 0.5 |
| **Ridge refuted** | every cell either collapses (Δperception > +0.4 on both seeds) or is pinned (Δentropy < +0.16) |

Ridge refuted is a real outcome, not a null: it would mean the anchor only ever
*suppresses* the bonus rather than licensing it, and that the uniform entropy
bonus is simply the wrong instrument for raising entropy on this substrate. That
closes the question.

## The cells

| cell | `kl_coef` | `beta` | role |
|---|---:|---:|---|
| `anchor_kl2p0_beta0p01` | 2.0 | 0.01 | **control.** `kl_coef` 2.0 is 4× the strongest anchor ever run, at the campaign-default bonus. Without it nothing at this `kl_coef` is attributable to `beta`. |
| `ridge_kl2p0_beta0p2` | 2.0 | 0.2 | **the key test.** This `beta` fails at `kl_coef` 0.5. |
| `ridge_kl2p0_beta0p4` | 2.0 | 0.4 | does the ceiling keep rising? Between the grid's 0.3 and 0.5, both catastrophic everywhere tried. |
| `ridge_kl5p0_beta0p4` | 5.0 | 0.4 | the extreme corner. If (2.0, 0.4) fails and this survives, the ridge has a slope. |

## Why two seeds per cell, and what cannot be read from one

The old grid contains an accidental exact duplicate — `kl0p5_beta0p1` was run
twice at the **same seed** — which calibrates the rig's noise floor:

| | Δ entropy | Δ perception | delivery |
|---|---:|---:|---:|
| spread, same config, same seed | 0.007 | **0.431** | **0.173** |

So **Δentropy is the low-noise observable** and carries the primary claim from a
single run. Δperception and delivery do not: differences below ~0.4 and ~0.17
respectively are unreadable from one run. Hence two seeds, and hence the target
regime is defined on *both* seeds.

## What this family will not settle

**The channel.** Delivery fell 0.98 → 0.50 in all 20 cells of the old grid, with
no trend in `beta` (ρ = −0.25, p = 0.28) or `kl_coef` (ρ = −0.16, p = 0.49) and
not one cell above 0.8. Regularisation does not reach the dark-room collapse
across a fifty-fold range in both knobs. Nothing here is expected to fix
delivery, and a delivery difference under 0.17 means nothing.

## How to read it when it lands

On Δentropy and Δperception, iterations 1–25 vs the last 25, both seeds shown,
with the well-formed post rate beside them. **Do not read `alive`**: across the
20 old cells it is uncorrelated with delivery (ρ = −0.10, p = 0.66) and it has
now reversed four readings.

Use `scripts/analyse_kl_beta_grid.py` for the window and the statistics; it will
need its `FAMILIES` dict pointed at this directory.

## Results

All eight cells ran to iteration 300 with `kl_coef` and `beta` in `config.json`
matching their names; **no cell is void**. Elapsed 3h05m–5h07m against 8 h
limits.

| `kl_coef` | `beta` | seed | Δ entropy | ent end | Δ perc | perc end | delivery end | KL from base |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2.0 | 0.01 | 1 | −0.01 | 0.74 | −0.49 | 1.68 | 0.678 | 0.46 |
| 2.0 | 0.01 | 2 | +0.06 | 0.82 | −0.55 | 1.62 | 0.603 | 0.06 |
| 2.0 | 0.20 | 1 | +0.15 | 0.90 | −0.22 | 1.94 | 0.785 | 0.39 |
| 2.0 | 0.20 | 2 | +0.33 | 1.10 | −0.34 | 1.83 | 0.519 | 0.08 |
| 2.0 | 0.40 | 1 | +0.28 | 1.05 | −0.17 | 2.00 | 0.705 | 0.36 |
| 2.0 | 0.40 | 2 | +0.59 | 1.36 | −0.11 | 2.06 | 0.641 | 0.10 |
| 5.0 | 0.40 | 1 | +0.09 | 0.85 | −0.31 | 1.86 | **0.975** | 0.02 |
| 5.0 | 0.40 | 2 | +0.15 | 0.91 | −0.40 | 1.79 | 0.872 | 0.02 |

## Headline

**The ridge is real: a stronger anchor does raise the safe ceiling on the
entropy bonus, and it roughly doubles it.** The pre-registered key test was
`beta` = 0.2, which collapses at `kl_coef` 0.5. It does not collapse at 2.0:

| `beta` = 0.2 | Δ entropy | Δ perception | verdict |
|---|---:|---:|---|
| at `kl_coef` 0.5 (old grid) | +1.19 | **+0.71** | collapsed |
| at `kl_coef` 2.0, seed 1 | +0.15 | **−0.22** | perception *improved* |
| at `kl_coef` 2.0, seed 2 | +0.33 | **−0.34** | perception *improved* |

And `beta` = 0.4 — a value catastrophic at every `kl_coef` the old grid tried,
where 0.3 ended at perception 4.46 and 0.5 at 5.2–6.4 — ends here at perception
**2.00 and 2.06** with Δperception −0.17 and −0.11. The safe ceiling moves from
below 0.2 at `kl_coef` 0.5 to at least 0.4 at 2.0: a 2× ceiling for a 4×
anchor.

**But the pre-registered target regime was formally missed.** The bar was
Δentropy ≥ +0.30 *and* Δperception ≤ 0 on **both** seeds. The best cell,
(2.0, 0.4), gives Δentropy +0.28 and +0.59 with Δperception negative on both —
**seed 1 misses the entropy threshold by 0.02.** One seed of two clears the bar.
The bar was set before the runs and is not being moved: the correct statement is
that the regime was reached on one seed of two and approached on the other.

## The confound that governs how this is read

**Delivery is negatively correlated with policy drift across all 28 cells now
on disk** — Spearman ρ = −0.503, p = 0.003, delivery(end) against KL-from-base
(end). The least-drifted cell in the campaign delivers 0.975; the most-drifted
delivers 0.395.

That matters because the anchor's whole job is to suppress drift, so anything it
appears to buy has to be checked against the possibility that it bought it by
preventing learning:

| `kl_coef` | KL from base at end | reading |
|---:|---|---|
| 0 (old grid) | 1.20–1.95 | free drift |
| 0.5 (old grid) | 0.48–0.61 | anchored |
| 2.0 (here) | 0.06–0.46 | tightly anchored |
| **5.0 (here)** | **0.02** | **pinned to base** |

**The `kl_coef` = 5.0 cells are the degenerate end and their headline numbers
should not be cited as a training result.** KL from base of 0.02 means the
policy has essentially not moved; the base model delivers ~0.98 in the first 25
iterations of every run in this project, so a delivery of 0.975 at KL 0.02 is
*the base model intact*, not an improvement. Their entropy is pinned too
(+0.09, +0.15). At `kl_coef` 5.0 the anchor buys the metrics by preventing the
training from happening.

**The cell that matters is (2.0, 0.4).** It moved (KL 0.36 and 0.10), entropy
rose (+0.28, +0.59), perception improved (−0.17, −0.11), delivery held at 0.705
and 0.641 against the reference `run7`'s 0.525 — and the transcripts are
healthy.

## The traces, checked because of yesterday

Yesterday's novelty size sweep produced a delivery of 0.926 that turned out to
be a two-agent repetition loop, so a delivery figure in this family cannot be
reported without the traces. It is not that here. At iteration 300:

| run | posts | chars | thread sim | repeat | leak | frag | math |
|---|---:|---:|---:|---:|---:|---:|---:|
| `run7` reference | 9.0 | 39 | 0.02 | 0.00 | 0.00 | **0.33** | 0.00 |
| old grid (0.5, 0.2) | 14.0 | 463 | 0.05 | 0.00 | 0.00 | 0.07 | 0.07 |
| (2.0, 0.01) s1 / s2 | 16 / 16 | 376 / 639 | 0.12 / 0.16 | 0.00 | 0.00 | 0.00 | 0.75 / 0.69 |
| (2.0, 0.2) s1 / s2 | 16 / 15 | 201 / 814 | 0.10 / 0.15 | 0.00 | 0.00 | 0.00 | 0.00 / 0.80 |
| (2.0, 0.4) s1 / s2 | 16 / 16 | 445 / 751 | 0.12 / 0.13 | 0.00 | 0.00 | 0.00 | 0.62 / 0.69 |
| (5.0, 0.4) s1 / s2 | 16 / 16 | 193 / 424 | 0.10 / 0.12 | 0.00 | 0.00 | 0.00 | 0.50 / 0.94 |

**Every cell: `repeat` 0.00, `leak` 0.00, `frag` 0.00, and 16 of 16 posts in
seven of eight.** Against the loop artefact's 0.38 repeat and 0.44 thread
similarity, and against the reference's 39-character posts and 33% fragments,
these are the healthiest late transcripts on this substrate anywhere in the
project. Reading (2.0, 0.4) seed 1 directly: eight distinct agents addressing
each other by name and building a Fibonacci induction proof — base case,
inductive hypothesis, inductive step — with one agent then connecting the
handshake problem to graph edges. It is a conversation, not an artefact.

Blemishes worth recording: some CJK-character leakage in (2.0, 0.2) seed 1,
emoji noise throughout, and one post in (2.0, 0.4) seed 1 that quotes another
agent's block verbatim.

## An instrument caveat found here

**The `math` regex under-counts mathematical reasoning expressed in prose.**
(2.0, 0.2) seed 1 scores `math` 0.00 while its agents work through subset
counting correctly — "one element, one valid subset; adding another shows how
this could grow" — because the regex wants symbols, LaTeX, or proof vocabulary
and that passage has none. It measures notation, not reasoning, and its zeros
are not evidence of absence.

## What this settles

- **A stronger KL anchor raises the safe ceiling on the entropy bonus.**
  `beta` = 0.2 goes from Δperception +0.71 at `kl_coef` 0.5 to −0.22 and −0.34
  at 2.0; `beta` = 0.4, catastrophic everywhere on the old grid, ends at
  perception 2.00 and 2.06. — *established*, two seeds per cell
- **The two terms trade off against each other rather than acting
  independently.** Both act only at ACT positions and this family completes the
  picture the old grid started: at fixed `beta` the anchor suppresses the
  entropy rise, and at fixed anchor a larger `beta` overcomes it. — *established*
- **The target regime — entropy up, perception not down — is reachable, on one
  seed of two.** (2.0, 0.4) seed 2: Δentropy +0.59 with Δperception −0.11. —
  *provisional*, one seed clears the pre-registered bar
- **Regularisation *does* reach the channel above `kl_coef` 0.5.** Ridge cells
  deliver 0.519–0.975 (mean 0.722) against the old grid's 0.395–0.613 (mean
  0.501), MWU exact p = 1.2 × 10⁻⁴, with 6 of 8 above the old grid's *maximum*
  and an effect on the mean of +0.222 against a 0.173 noise floor. **This
  corrects the claim established earlier the same day** that neither
  regulariser touches the channel — that was measured over `kl_coef` ∈ [0, 0.5]
  and does not extend past it. — *established*
- **But delivery tracks how little the policy moved.** ρ = −0.503, p = 0.003
  over 28 cells, and the highest-delivery cells are pinned at KL-from-base 0.02.
  The anchor may be preventing the damage rather than training anything better.
  — *likely*
- **`kl_coef` = 5.0 prevents learning.** KL from base 0.02 with entropy pinned;
  its delivery of 0.975 is the base model, which delivers ~0.98 untrained. —
  *established*

## What it does not settle

- **Whether (2.0, 0.4) is genuinely better than the base model, or just less
  damaged than everything else.** Its KL from base is 0.36 and 0.10, so it did
  move — but nothing here compares its transcripts against the *untrained* model
  on the same substrate. That comparison is cheap and has not been made; it is
  the next thing to run, and until it exists "the healthiest late transcripts in
  the project" means healthiest among *trained* runs.
- **Where the new ceiling actually is.** `beta` = 0.4 works at `kl_coef` 2.0;
  0.8 and 1.6 were not tried, and neither was `kl_coef` 1.0, which is where the
  useful corner probably sits given that 5.0 pins the policy.
- **Whether the seed asymmetry is real.** Seed 1 has higher KL from base in all
  three `kl_coef` 2.0 cells (0.46/0.39/0.36 against 0.06/0.08/0.10) and lower
  Δentropy in all four. A systematic seed effect across four cells is odd; seed 2
  also ran ~40% longer in wall time in every pair, so the two may differ in
  something other than the seed. Not diagnosed.
- **Anything about horizon.** All eight stop at 300.

## Status

`CLOSED` for the ridge question, which it answers. Cite the ceiling result and
the channel correction. Do **not** cite the `kl_coef` 5.0 numbers as a training
outcome, and do not cite the target regime as reached without saying it is one
seed of two.

---
*Analysed 2026-09-08 from `metrics.jsonl`, `config.json` and the checkpoint
traces, with `scripts/analyse_kl_beta_grid.py` and
`scripts/analyse_novelty_traces.py`. Machine output stays local.*

