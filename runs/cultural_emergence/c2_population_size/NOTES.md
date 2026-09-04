# c2_population_size

*Tier C item C2 — the five-arm population-size decomposition. **Two arms have
finished; the family's question is not yet answerable.***

|  |  |
|---|---|
| Configs | [`configs/cultural_emergence/c2_population_size/`](../../../configs/cultural_emergence/c2_population_size/README.md) |
| Model | `Qwen2.5-7B-Instruct`, LoRA r64 on q_proj/v_proj |
| Ran to | baseline-large 4/4, rate-matched 5/5, frozen-majority 4/5 at 300; restricted-pairing 2/5 complete, 3 resumed · jobs 6261670–6261683, 6271024–6271029 |
| **Differs from** | **`run8_16agent_7B_study_group_ashbourne_gc`** (N=16 arms) and **C1's N=4 cell** (small arms) — one manipulation each, named per arm. |

## What it was for

N varies three things at once — partner count, per-agent gradient volume, and
context composition — so no existing comparison separates them. Five arms do.
**The claim only becomes causal with all five**, and in particular with
restricted pairing, which is the arm that asks whether outcome tracks partner
count rather than N.

## Headline

**Still incomplete, but the N=16 reference now exists at five seeds and it
changes the frozen-majority reading.** Baseline-large gives `alive` at
iteration 300 of 0.38 / 0.75 / 0.69 / 0.81 across seeds 2–5, with `run8`'s 0.88
as seed 1 — mean 0.70, range 0.38–0.88. The single-seed reference the
frozen-majority arm was read against was the *top* of that range, so the
"learners do better than co-adapting agents" reading is weaker than it looked.
Restricted pairing, the arm that carries the causal claim, is still incomplete.

Separately: **rate-matched N=4 is the healthiest cell measured anywhere in C2 or
C1** — `alive` 1.00 at 300 in four of five seeds. Matching per-agent gradient
volume downward *helps*, which is the direction C10 predicts if the action term
is what does the damage.

## Core metrics, the two finished arms

`alive` at 25 / 100 / 150 / 200 / 300, and the population means:

| run | @25 | @100 | @150 | @200 | @300 | perception loss | KL from base |
|---|---|---|---|---|---|---|---|
| rate-matched s1 | 0.00 | 1.00 | 1.00 | 1.00 | 0.75 | 2.18 → 1.23 | 0.004 → 0.32 |
| rate-matched s2 | 0.00 | 1.00 | 1.00 | 1.00 | **1.00** | 2.16 → 1.54 | 0.004 → 0.22 |
| rate-matched s3 | 1.00 | 1.00 | 1.00 | 1.00 | **1.00** | 2.18 → 1.27 | 0.005 → 0.29 |
| rate-matched s4 | 0.00 | 1.00 | 1.00 | 1.00 | **1.00** | 2.17 → 1.25 | 0.002 → 0.16 |
| rate-matched s5 | 0.00 | 1.00 | 1.00 | 1.00 | **1.00** | 2.23 → 0.99 | 0.002 → 0.14 |
| frozen-majority s1 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 2.36 → 2.13 | 0.018 → 0.32 |
| frozen-majority s2 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 2.37 → 2.19 | 0.012 → 1.32 |
| frozen-majority s3 | 0.00 | 0.00 | 0.00 | 0.50 | 0.00 | 2.34 → 2.09 | 0.064 → 0.24 |
| frozen-majority s4 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 2.34 → 2.05 | 0.028 → 0.77 |
| baseline-large s2 | 0.00 | 1.00 | 1.00 | 0.88 | **0.38** | 2.31 → 2.10 | 0.026 → 1.50 |
| baseline-large s3 | 0.00 | 1.00 | 1.00 | 1.00 | **0.75** | 2.34 → 1.89 | 0.025 → 0.31 |
| baseline-large s4 | 0.00 | 1.00 | 1.00 | 0.94 | **0.69** | 2.32 → 1.82 | 0.023 → 0.64 |
| baseline-large s5 | 0.00 | 1.00 | 1.00 | 1.00 | **0.81** | 2.35 → 1.78 | 0.014 → 0.81 |
| REF `run8` N=16 (seed 1) | 0.00 | 0.88 | 1.00 | 1.00 | 0.88 | 2.35 → 1.71 | 0.026 → 1.59 |
| REF C1 N=4 s1 | 0.00 | 1.00 | 1.00 | 1.00 | 0.75 | 2.18 → 1.51 | 0.009 → 0.33 |

Well-formed post rate at iteration 300, from `tool_counts_per_agent`. For
frozen-majority this **must** be read per learner: fourteen of sixteen members
are frozen and deliver by construction, so the population figure is measuring
the freeze, not the learners.

| run | learners (2 of 16) | whole population |
|---|---|---|
| frozen-majority s1 | 5/8 = **0.62** | 60/64 = 0.94 |
| frozen-majority s2 | 5/8 = **0.62** | 60/64 = 0.94 |
| frozen-majority s3 | 2/8 = **0.25** | 52/64 = 0.81 |
| frozen-majority s4 | 3/8 = **0.38** | 53/64 = 0.83 |
| REF `run8`, all sixteen learning | — | 17/64 = **0.27** |

## What this settles

- **Matching per-agent gradient volume at N=4 does not harm the population and
  may help it.** Four of five rate-matched seeds end `alive` 1.00 with
  perception loss around 1.2; C1's five unmatched N=4 seeds span 0.00–1.00 at
  iteration 300. Less training per iteration, less damage — which is the
  direction C10 predicts, since the action term is what does the damage. —
  *likely*, five seeds against five — this arm and C1
- ~~**A frozen majority's two learners deliver better than `run8`'s sixteen
  co-adapting agents.**~~ **Weakened by baseline-large.** That comparison used
  `run8`'s single seed; the five-seed N=16 reference now spans `alive` 0.38–0.88
  at iteration 300 with `run8` at the top of it. The frozen-majority learners
  are inside that band rather than above it, so the co-adaptation effect this
  arm was reaching for is not demonstrated. — *withdrawn as stated* — this arm
  and baseline-large
- **The N=16 reference is 0.70 mean, 0.38–0.88 range, at five seeds.** Any
  future comparison against N=16 has to clear that spread, which the previous
  single-seed reference concealed. — *established* — baseline-large

## What it does not settle

- **The causal question.** Restricted pairing is the arm that carries it and it
  is still running. Until then "population size" remains withdrawn as a causal
  claim (Tier D) and unsupported as a descriptive one at the small end (C1).
- **Whether frozen-majority's `alive` 0.00 means anything.** The two learners
  sit at perception loss 2.05–2.19, just over the classifier's `perc_loss ≥ 2.0`
  cut, so they label `text_degenerate` while delivering 25–62% well-formed
  posts. The threshold was calibrated on co-adapting populations; against an
  un-adapted, frozen observation stream it is being asked a question it was not
  calibrated for. The same caveat is in
  [`c3_frozen_population/NOTES.md`](../c3_frozen_population/NOTES.md).
- **Seed 5 of frozen-majority**, which died on the GPU-memory preflight guard
  and is requeued as 6279079.
- **Context composition**, which no arm in this design isolates. If restricted
  pairing tracks unrestricted N=16, composition is the remaining candidate and
  needs an arm of its own.
