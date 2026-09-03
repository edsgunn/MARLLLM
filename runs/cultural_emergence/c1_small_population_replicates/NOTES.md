# c1_small_population_replicates

*Tier C item C1 — a family of twenty runs whose unit of inference is the family:
five seeds × two substrates × two population sizes.*

|  |  |
|---|---|
| Configs | [`configs/cultural_emergence/c1_small_population_replicates/`](../../../configs/cultural_emergence/c1_small_population_replicates/README.md) |
| Model | `Qwen2.5-7B-Instruct`, LoRA r64 on q_proj/v_proj |
| Populations | 2 and 4 agents on `study_group_ashbourne_gc` and `study_group_strathearn_server` |
| Ran to | iteration 300 of 300, all twenty · jobs 6260359–6260378 |
| **Differs from** | **the four `run10_*` runs of 2026-05-06** — the seed, and nothing else. |

## What it was for

The N=2 and N=4 cells of the population-size figure were each drawn from **one
seed**. The pre-registered test, fixed in writing before these runs: *if the
within-cell spread of `alive` across five seeds is comparable to the
between-cell difference the figure reads as a population-size effect, the N=2
and N=4 points carry no information about population size and the corner comes
out of the figure.*

Seed 1 was re-run rather than reused, because the 2026-05-06 originals executed
against an uncommitted working tree and git cannot establish what code they ran.

## Headline

**The falsification condition fired, and not marginally.** At iteration 200 the
mean within-cell spread of `alive` is **0.44** against N=2-vs-N=4 gaps of 0.20
(Ashbourne) and 0.35 (Strathearn); at iteration 300 the spread is **1.00** — the
entire range — against gaps of 0.05 and 0.15. **No metric separates N=2 from
N=4** on five seeds a side: perception loss, KL from base, policy entropy and
delivery rate all give p ≥ 0.15 by exact Mann-Whitney. The corner carries no
information about population size.

## Core metrics

`alive` fraction at iterations 25 / 100 / 150 / 200 / 300, by seed, with the
single 2026-05-06 seed the figure was drawn from for comparison:

| cell | s1 | s2 | s3 | s4 | s5 | 05-06 | spread @200 / @300 |
|---|---|---|---|---|---|---|---|
| N=2 Ashbourne @200 | 0.50 | 1.00 | 1.00 | 1.00 | 0.50 | 1.00 | **0.50** |
| N=2 Ashbourne @300 | 0.50 | 1.00 | 0.50 | 0.00 | 1.00 | 0.00 | **1.00** |
| N=4 Ashbourne @200 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.75 | **0.00** |
| N=4 Ashbourne @300 | 0.75 | 0.50 | 0.00 | 1.00 | 1.00 | 0.75 | **1.00** |
| N=2 Strathearn @200 | 1.00 | 1.00 | 0.50 | 0.50 | 0.00 | 0.00 | **1.00** |
| N=2 Strathearn @300 | 0.00 | 1.00 | 0.50 | 0.50 | 0.50 | 0.00 | **1.00** |
| N=4 Strathearn @200 | 1.00 | 1.00 | 0.75 | 1.00 | 1.00 | 0.75 | **0.25** |
| N=4 Strathearn @300 | 0.75 | 0.00 | 0.75 | 1.00 | 0.75 | 0.75 | **1.00** |

Continuous metrics, final block, mean [min, max] over five seeds, with the
exact two-sided Mann-Whitney p for N=2 against N=4:

| metric | substrate | N=2 | N=4 | p |
|---|---|---|---|---|
| perception loss | Ashbourne | 1.69 [1.19, 2.11] | 1.50 [1.04, 1.99] | 0.421 |
| perception loss | Strathearn | 1.56 [1.15, 1.87] | 1.47 [0.87, 2.22] | 1.000 |
| KL from base | Ashbourne | 1.51 [0.17, 4.19] | 0.34 [0.20, 0.55] | 0.690 |
| KL from base | Strathearn | 1.13 [0.27, 2.52] | 1.25 [0.20, 4.28] | 0.690 |
| policy entropy | Ashbourne | 0.88 [0.17, 2.11] | 0.71 [0.52, 1.23] | 0.841 |
| policy entropy | Strathearn | 0.67 [0.21, 1.13] | 0.72 [0.39, 1.38] | 1.000 |
| delivery rate | Ashbourne | 0.35 [0.10, 0.73] | 0.22 [0.13, 0.34] | 0.548 |
| delivery rate | Strathearn | 0.32 [0.06, 0.57] | 0.13 [0.05, 0.26] | 0.151 |

## What this settles

- **The N=2 and N=4 cells of the population-size figure are seed noise.** The
  within-cell spread meets or exceeds the between-cell difference at both
  iteration 200 and 300, and no continuous metric separates the sizes at five
  seeds a side. — *established* — this family
- **Two of the four original cells' published readings were seed artefacts.**
  `run10_2agent_…_strathearn_server`'s NOTES headline was *"Dead from iteration
  150 onward"*; the five fresh seeds give 0.50–1.00 alive at 150 and 0.00–1.00
  at 200. `run10_2agent_…_ashbourne_gc` was `text_degenerate` at 100; the fresh
  five are 0.50–1.00. — *established* — same
- **The public channel closes regardless of population size.** Delivery falls
  from 0.93 to 0.25 on average and **fell in 20 of 20 runs**, which replicates
  B4's 63-of-65 on a family designed for a different question. — *established*
  — same
- **`alive` is a poor statistic at small N**, and this is a fact about the
  instrument rather than the substrate: at N=2 it can only take {0, 0.5, 1}, so
  one agent flipping moves it by 0.5. Any figure reading small-N cells off it
  inherits that coarseness. — *established* — same

## What it does not settle

- **Nothing about N=8 or N=16.** This family bounds the small corner only. The
  monotone shape §3.3 already withdrew is untouched; C2 is the arm that asks
  whether size does anything at all, and it needs all five of its arms.
- **Why the fresh seed 1 differs from the 2026-05-06 seed 1** at N=2 Ashbourne
  (0.50/0.50/0.50/0.50 against 0.00/0.00/1.00/1.00/0.00). Code drift and N=2's
  own instability are not separated, and with a within-cell spread of 0.50 at
  that cell they may not be separable. The bridge run did its job — it showed
  the historical run is not safely poolable — without identifying the cause.
- **Whether the decline has a floor.** Every run stops at 300; C6 asks this.
