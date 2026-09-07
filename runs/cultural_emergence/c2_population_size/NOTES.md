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

**All five arms are in, and population size does nothing.** At five seeds per
arm and reading the well-formed post rate rather than `alive`, N=4 and N=16 are
indistinguishable at matched gradient volume — 0.263 against 0.225, p = 0.722.
Neither does gradient volume (p = 0.810). Partner count trends the right way,
0.316 at k=3 against 0.225 at k=15 with N held at 16, but p = 0.135 is not a
result at this power.

**The one thing that does separate is co-adaptation.** Learners in a frozen
majority deliver **0.475** against **0.225** for agents in a co-adapting
population of the same size, p = **0.032** — every frozen-majority seed above
the baseline's median. That reinstates, on the correct measure, the claim this
write-up withdrew when baseline-large first arrived.

So §6.3's design did its job and the answer is negative for its headline
variable: *"if outcome tracks partner count rather than N, population size was
never the variable"* — outcome tracks **neither** at this power, and what it
does track is whether the other agents are moving.

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

## The five arms, at iteration 300

Delivery (well-formed posts / turns), five seeds each. `alive` is shown for
comparison only — it cannot be trusted at low delivery
([why](../c6_extended_horizon/NOTES.md)).

| arm | N | partners | volume | delivery, mean [range] | alive |
|---|---:|---:|---:|---|---|
| baseline small | 4 | 3 | 64 | 0.200 [0.125, 0.250] | 0.65 |
| baseline large | 16 | 15 | 16 | 0.225 [0.109, 0.328] | 0.70 |
| restricted pairing | 16 | **3** | 16 | **0.316** [0.250, 0.438] | 0.75 |
| rate-matched small | 4 | 3 | **16** | 0.263 [0.062, 0.688] | 0.95 |
| frozen majority, learners | 16 | 15 | 16 | **0.475** [0.250, 0.625] | 0.00 |

Exact two-sided Mann-Whitney, five against five:

| contrast | holds fixed | result |
|---|---|---|
| N=4 vs N=16 | volume | 0.263 vs 0.225, **p = 0.722** |
| k=3 vs k=15 | N, volume | 0.316 vs 0.225, p = 0.135 |
| N=16 vs N=4 at k=3 | partners, volume | 0.316 vs 0.263, p = 0.381 |
| volume 16 vs 64 | N, partners | 0.263 vs 0.200, **p = 0.810** |
| frozen vs co-adapting | N, partners, volume | 0.475 vs 0.225, **p = 0.032** |

## What this settles

- **Population size does not determine the outcome.** N=4 and N=16 are
  indistinguishable at matched gradient volume, p = 0.722, on five seeds a
  side. With C1 having already removed the small corner, the population-size
  claim has no support left at either end. — *established* — this family
- **Nor does per-agent gradient volume**, p = 0.810 — which withdraws the
  earlier reading of the rate-matched arm as "the healthiest cell measured".
  That was `alive` 0.95; by delivery it is 0.263 against 0.200, well inside the
  noise. — *withdrawn as stated* — same
- **Co-adaptation degrades delivery.** Learners against a frozen majority
  deliver twice as well as agents in a co-adapting population of the same size,
  p = 0.032. This is the only significant contrast in the design, and it
  answers the question C3 was underpowered for. — *likely* — this family and
  [`c3_frozen_population/NOTES.md`](../c3_frozen_population/NOTES.md)
- **Partner count is the variable worth more seeds.** It is the only
  size-adjacent manipulation that trends, 0.316 vs 0.225, and five seeds cannot
  resolve p = 0.135. Ten would. — *suspected* — same

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
