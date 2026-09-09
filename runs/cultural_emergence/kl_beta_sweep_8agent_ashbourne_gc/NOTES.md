# kl_beta_sweep_8agent_ashbourne_gc

*cultural emergence — the 4 × 4 regularisation sweep. **The family is the unit
of inference, not the cell.***

|  |  |
|---|---|
| Configs | [`configs/cultural_emergence/kl_beta_sweep_8agent_ashbourne_gc/`](../../../configs/cultural_emergence/kl_beta_sweep_8agent_ashbourne_gc/) |
| Model | `Qwen2.5-7B-Instruct` |
| Population | 8 agents on `study_group_ashbourne_gc`, `extended_8` roster |
| Grid | `kl_coef` ∈ {0, 0.01, 0.1, 0.5} × `beta` ∈ {0, 0.01, 0.1, 0.5} |
| Held fixed | lr 5e-6, LoRA r64, 16 episodes/iter, seed 1, 300 iterations |
| Ran | all 16 cells to iteration 300 |
| **Differs from** | **`run7_8agent_7B_study_group_ashbourne_gc`** — that run is the (0.05, 0.01) point, which is *not* on this grid. |

## What it was for

Both regularisers were set once, early, and never justified: `kl_coef` 0.05
anchors the policy to the frozen base, `beta` 0.01 adds an entropy bonus. The
sweep asks which of them is load-bearing and whether the campaign's default sits
anywhere sensible.

## Headline

**The entropy bonus is the dangerous knob and the default was set an order of
magnitude too high on it — in the safe direction, by luck.** Every cell with
`beta ≥ 0.1` ends degenerate; every cell with `beta = 0.5` ends catastrophically
so, with policy entropy above 6.5 nats and perception loss above 5.2. The KL
anchor, by contrast, does what it is supposed to and nothing more: raising it
from 0 to 0.5 pulls end-of-training KL from base down from ~1.2 to ~0.5 and
changes the alive fraction hardly at all.

**But the sweep cannot rank its own cells**, for a reason discovered by
accident — see below.

## Core metrics

Population means over the first and last 10% of the 300 logged iterations, and
the alive fraction at iterations 200 and 300:

| `kl_coef` | `beta` | perc first | perc last | entropy last | KL last | alive @200 | alive @300 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 0.0 | 2.18 | 2.15 | 0.70 | 1.22 | 0.88 | 0.50 |
| 0.0 | 0.01 | 2.18 | 1.97 | 0.73 | 1.50 | 0.88 | 0.50 |
| 0.0 | 0.1 | 2.17 | 2.56 | 1.80 | 1.75 | 0.12 | 0.00 |
| 0.0 | 0.5 | 2.17 | 5.69 | 6.79 | 1.97 | 0.00 | 0.00 |
| 0.01 | 0.0 | 2.15 | 1.97 | 0.77 | 1.24 | 0.88 | 0.75 |
| 0.01 | 0.01 | 2.17 | 2.20 | 0.80 | 1.22 | 0.88 | 0.00 |
| 0.01 | 0.1 | 2.18 | 2.56 | 1.67 | 1.38 | 0.62 | 0.00 |
| 0.01 | 0.5 | 2.17 | 5.67 | 6.87 | 2.26 | 0.00 | 0.00 |
| 0.1 | 0.0 | 2.16 | **1.65** | 0.62 | 0.95 | 0.88 | **0.88** |
| 0.1 | 0.01 | 2.18 | 1.97 | 0.75 | 0.91 | 0.88 | 0.50 |
| 0.1 | 0.1 | 2.16 | 2.46 | 1.55 | 1.45 | 0.88 | 0.00 |
| 0.1 | 0.5 | 2.16 | 5.23 | 6.63 | 2.13 | 0.00 | 0.00 |
| 0.5 | 0.0 | 2.20 | 2.10 | 0.68 | 0.48 | 0.88 | 0.12 |
| 0.5 | 0.01 | 2.17 | 1.95 | 0.70 | 0.60 | 0.88 | 0.75 |
| 0.5 | 0.1 | 2.18 | **1.71** | 0.91 | 0.46 | 1.00 | **0.88** |
| 0.5 | 0.5 | 2.17 | 6.42 | 6.54 | 1.02 | 0.00 | 0.00 |

Every cell starts at perception loss 2.16–2.20. That is the untrained model on
this substrate, and it is above the `text_degenerate` threshold of 2.0 — so a
cell that ends near 2.2 has not degraded, it has failed to learn.

## The accident that limits what this sweep can say

The `(0.5, 0.1)` cell was run **twice**: once here and once as part of
[`../beta_zoom_kl0p5_8agent_ashbourne_gc/`](../beta_zoom_kl0p5_8agent_ashbourne_gc/NOTES.md).
The two configs differ only in the run name, the output directory, and which of
the four GPUs the job landed on. Same seed, same hyperparameters, same code,
300 clean iterations each with no resume.

They agree closely to about iteration 150 and then separate:

| iterations | sweep cell | beta-zoom cell |
|---|---:|---:|
| 0–50 | 2.148 | 2.153 |
| 50–100 | 1.873 | 1.938 |
| 100–150 | 1.671 | 1.790 |
| 150–200 | 1.573 | 1.777 |
| 200–250 | 1.629 | 1.943 |
| 250–300 | **1.676** | **2.106** |

Alive at iteration 300: **0.875 against 0.000**. One of the two best cells on
this grid and one of its complete failures are the *same configuration*.

That is not a criticism of either run. It means the run-to-run spread at fixed
seed on this substrate is as large as the spread across most of the grid, so
**no ranking among the surviving cells is supported by one run each.** What
survives is the part of the signal that is much larger than that spread: the
`beta` effect.

## What this settles

- **`beta ≥ 0.1` destroys the population at every KL coefficient tried.** 8/8
  cells with `beta ≥ 0.1` end with alive 0.00, against 1/8 for `beta ≤ 0.01`.
  — *established*
- **`beta = 0.5` is catastrophic, not merely bad.** Entropy ends at 6.5–6.9 nats
  and perception loss at 5.2–6.4, five to eight times any healthy run. —
  *established*
- **The KL coefficient controls drift and little else.** End-of-training KL from
  base falls monotonically with `kl_coef` (≈1.2 → 1.2 → 0.9 → 0.5) while alive
  at iteration 200 is 0.88 in eleven of the twelve `beta ≤ 0.1` cells. —
  *likely*
- **This pipeline is not reproducible at fixed seed.** Two runs of one identical
  configuration ended at alive 0.875 and 0.000. — *established*, one direct
  demonstration
- **The campaign's default `beta = 0.01` sits inside the safe band**, and its
  `kl_coef = 0.05` sits between two grid points that both behave. — *likely*

## What it does not settle

**Which cell is best.** The two best cells on end-of-training perception loss —
`(0.1, 0.0)` at 1.65 and `(0.5, 0.1)` at 1.71 — differ by less than the gap
between the two runs of `(0.5, 0.1)`. Ranking the survivors needs replicates,
which is what
[`../kl_beta_seed_replicates_8agent_ashbourne_gc/`](../kl_beta_seed_replicates_8agent_ashbourne_gc/NOTES.md)
was built for and only partly delivers.

Also unsettled: whether `beta` between 0.01 and 0.1 has a usable optimum. The
zoom was run only at `kl_coef = 0.5`.

## The 2026-09-08 re-reading: on delivery, and on the CHANGE in entropy

Both this family and the [`beta` zoom](../beta_zoom_kl0p5_8agent_ashbourne_gc/NOTES.md)
were originally read on the `alive` fraction. Across the 20 cells of the two
families together, **`alive` is uncorrelated with the well-formed post rate**
(Spearman −0.10, p = 0.66). The readings above that touch on population health
therefore cannot support claims about the channel. The delivery rate was
recorded all along, as `tool/post_success_total` / `tool/post_turns_total`; no
agent is frozen in any of these runs, so the population-level rate is valid.

Re-read with [`scripts/analyse_kl_beta_grid.py`](../../../scripts/analyse_kl_beta_grid.py),
comparing iterations 1–25 against the last 25. Four things come out of it that
the original reading missed.

### 1. Neither regulariser touches the channel

Delivery falls from 0.98 to 0.50 in **all 20 cells**. No trend in `beta`
(ρ = −0.25, p = 0.28) or in `kl_coef` (ρ = −0.16, p = 0.49); the whole spread is
0.395–0.613 and **not one cell ends above 0.8**. Across a fifty-fold range in
both knobs, regularisation does not touch the dark-room collapse. Whatever
closes the public channel, it is not something either regulariser reaches.

### 2. The entropy bonus is a step function, not a dial

| `beta` | cells | Δ entropy, first-25 → last-25 |
|---|---:|---|
| ≤ 0.05 | 9 | −0.12 … +0.06 |
| ≥ 0.10 | 11 | +0.16 … +6.20 |

The two groups separate **completely** — max of the low group +0.064, min of the
high group +0.160 — MWU exact p = 1.2 × 10⁻⁵. Below the step the bonus is inert
and its sign is as often negative as positive. **There is no `beta` that raises
entropy a little.**

### 3. Above the step, entropy and perception loss are locked together

Spearman Δentropy vs Δperception = **+0.853, p = 4.4 × 10⁻¹²**, n = 20. The two
cells where entropy rose and perception loss did not are the two runs of the
duplicate `(0.5, 0.1)` — see the noise floor below, which is exactly the size of
that effect. **On this grid, the regime where policy entropy rises without the
perception loss collapsing does not occur.**

### 4. The KL anchor does fight the entropy bonus

At `beta` = 0.10, Δentropy falls monotonically as the anchor tightens:

| `kl_coef` | 0.00 | 0.01 | 0.10 | 0.50 |
|---|---:|---:|---:|---:|
| Δ entropy | +1.04 | +0.92 | +0.79 | +0.16 |

Four points, monotone, and the duplicate pair agrees on the last one to 0.01.
Both terms act only at ACT positions — the entropy bonus inside `l_act`, the KL
alongside it (`marlllm/loss.py`) — so they are competing for the same
distribution: the bonus flattens it, the anchor pins it to base.

**This corrects the claim carried from the zoom** that "the entropy bonus
destroys populations without moving them away from base, so the KL anchor cannot
protect against it." At `beta` = 0.10 the anchor *does* protect, and by a factor
of six. It stops protecting by `beta` = 0.20, where `kl_coef` 0.5 still ends at
Δentropy +1.19 and Δperception +0.71.

### The noise floor this rig actually has

The accidental duplicate of `(0.5, 0.1)` was run at the **same seed**. The two
runs differ by:

| | Δ entropy | Δ perception | delivery |
|---|---:|---:|---:|
| spread, same config, same seed | **0.007** | **0.431** | **0.173** |

So the duplicate is not only a reproducibility failure — it is a calibration.
**Δentropy is the low-noise observable on this rig and carries weight from a
single run; Δperception and delivery do not.** Differences below ~0.4 in
Δperception or ~0.17 in delivery are unreadable from one run, here or anywhere
else in the campaign. Read against that floor, the entire delivery spread across
this grid (0.395–0.613) is consistent with pure noise about 0.50.

### What raises entropy instead

Through the same window, on the same substrate and roster:

| run | Δ entropy | Δ perception | delivery |
|---|---:|---:|---|
| reference `run7` (`return_sign` +1) | +0.06 | +0.12 | 0.980 → 0.525 |
| novelty search seed 1 (`return_sign` −1) | +0.22 | +0.24 | 0.984 → 0.638 |
| novelty search seed 2 (`return_sign` −1) | +0.23 | **−0.46** | 0.983 → 0.455 |

Reversing the return's sign moves entropy roughly four times as far as any safe
`beta` does, and on seed 2 it did so while perception loss *improved*. That
Δperception of −0.46 is only just above the noise floor and rests on one seed, so
it is not a result — but the entropy change is, and it points at the return
rather than at the regularisers as the term that governs policy entropy here.

### What this section opens

Whether a stronger anchor raises the safe ceiling on the bonus, putting the
target regime somewhere on a diagonal in (`kl_coef`, `beta`). `kl_coef` above
0.5 has never been run. Tested in
[`../entropy_anchor_ridge_8agent_ashbourne_gc/`](../entropy_anchor_ridge_8agent_ashbourne_gc/NOTES.md).

## The 2026-09-09 content read: interesting behaviour, or noisy TV?

The 20 cells here plus the 8 in
[`../entropy_anchor_ridge_8agent_ashbourne_gc/`](../entropy_anchor_ridge_8agent_ashbourne_gc/NOTES.md)
had never been read on what the agents actually *say* — only on `alive`, then
on delivery and entropy. Read on trace content at iteration 300, sorted by end
entropy, they give the answer the whole regularisation line was for.

**Both, at different points, and the transition is compositional rather than
global.**

| end entropy | `stopword` | what it looks like |
|---:|---:|---|
| 0.9–1.1 | 0.28–0.36 | coherent throughout |
| 1.5–1.8 | 0.14–0.20 | **mixed within one episode** |
| ~3.4 | 0.196 | coherent **opening**, noise **tail** |
| 6.6–7.0 | 0.035–0.090 | word salad |

At **end entropy 1.78** (`kl0p0_beta0p1`) one episode contains all of it at
once — Beatrice opening with real mathematics ("area under curves like
y = x^n … maybe start with n=1 to n=3?"), Sam and Tom answering coherently,
Hana in name-repetition babble ("beat, beat, beat! beatrice, absolute beatrice
okafor, beatrice okafre"), and Priya emitting nothing but `</?</?</?` for 1143
characters. **Agents collapse individually, not together.**

At **end entropy 3.37** (`kl0p5_beta0p3`) the degradation is positional: posts
open with genuine content ("exploring modular arithmetic properties with small
primes? could be a stepping stone") and then run into hundreds of repeated
`💬`. **Openings survive; tails go first.**

At **end entropy 6.6–7.0** it is unambiguous word salad — "sweet baked
contentious oats propaganda zipper bepace correlcusAuthenticate Muslim謨nung".

### The most interesting thing is inside the collapse

At the very top of the ladder, some agents comment on their peers'
degeneration. At 6.59, Sam: *"can someone tame this beast lang mess?"* At 6.95,
Sam again: *"it's just you showing off … figured since everyone else babbled
cryptograms"*, and *"couldn't come up w/ cohesive concept"*. Enough coherence
survives to **observe that the channel has become noise** — which is a more
interesting behaviour than anything in the coherent cells, and nothing in this
project was designed to look for it.

### The usable window, and why nothing has been aimed at it

Degradation is visible from ~1.5 and entropy has barely moved below ~1.1, so
the window is **1.1–1.8**. The `beta` steps available jump straight across it:
0.05 → end entropy 0.78, 0.1 → 1.53–1.78, 0.2 → 1.93. Tested deliberately for
the first time, under novelty search rather than minimisation, in
[`../../novelty_search/entropy_release/`](../../novelty_search/entropy_release/NOTES.md).

### The instrument this required

**Every trace measure was fooled except one.** At end entropy 6.59 the
word-salad population scores 15 posts, 724 mean characters, `repeat` 0.00 and
`frag` 0.00 — indistinguishable from a healthy transcript, because word salad is
long and non-repetitive. Only `thread_sim` hinted at it, dropping to 0.01, and
that reads as "diverse" rather than "broken".

The English function-word fraction separates them: **0.035–0.090** for the four
collapsed populations against **0.28–0.45** for coherent ones, ρ = −0.531,
p = 1.4 × 10⁻³ over 28 runs. Added to
[`scripts/analyse_novelty_traces.py`](../../../scripts/analyse_novelty_traces.py).

It is a **screen, not a verdict**, and reliable only below 0.10: dense
technical prose scores **0.118** while genuine salad scores **0.139**. Anything
between 0.10 and 0.28 must be read.

## Status

`CLOSED` as a screen. Cite the `beta` result, the 2026-09-08 re-reading and the
2026-09-09 content read; do not cite a cell ranking, and do not cite any `alive`
figure from this family without the delivery rate beside it.

The content read is the most useful thing this family produced, and it cost no
compute — the traces had been on disk since 2026-09-01.

---
*Analysed retrospectively on 2026-09-01, re-read on 2026-09-08, from
`metrics.jsonl` and `config.json`.
Machine output stays local.*
