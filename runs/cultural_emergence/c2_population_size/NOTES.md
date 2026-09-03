# c2_population_size

*Tier C item C2 — the five-arm population-size decomposition. **Two arms have
finished; the family's question is not yet answerable.***

|  |  |
|---|---|
| Configs | [`configs/cultural_emergence/c2_population_size/`](../../../configs/cultural_emergence/c2_population_size/README.md) |
| Model | `Qwen2.5-7B-Instruct`, LoRA r64 on q_proj/v_proj |
| Ran to | rate-matched 5/5 and frozen-majority 4/5 at iteration 300; baseline-large and restricted-pairing still running · jobs 6261670–6261683, 6271024–6271029 |
| **Differs from** | **`run8_16agent_7B_study_group_ashbourne_gc`** (N=16 arms) and **C1's N=4 cell** (small arms) — one manipulation each, named per arm. |

## What it was for

N varies three things at once — partner count, per-agent gradient volume, and
context composition — so no existing comparison separates them. Five arms do.
**The claim only becomes causal with all five**, and in particular with
restricted pairing, which is the arm that asks whether outcome tracks partner
count rather than N.

## Headline

**Nothing yet, by design.** What the two finished arms do show is that the
*rate-matched* manipulation is benign and the *frozen-majority* one is
suggestive, and that is as far as two of five arms can go.

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
| REF `run8` N=16 | 0.00 | 0.88 | 1.00 | 1.00 | 0.88 | 2.35 → 1.71 | 0.026 → 1.59 |
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
- **A frozen majority's two learners deliver better than `run8`'s sixteen
  co-adapting agents**, 0.25–0.62 against 0.27, with all four seeds at or above
  the reference. — *provisional* — the reference is **one** seed, and
  baseline-large exists precisely to replace it with five. Do not cite this
  until it does.

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
