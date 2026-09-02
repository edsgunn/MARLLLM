# B2 — regime composition as a function of iteration

*Is a regime an outcome, or a snapshot on a trajectory?*

|  |  |
|---|---|
| Script | [`scripts/tier_b/b2_regime_by_iteration.py`](../../../scripts/tier_b/b2_regime_by_iteration.py) |
| Inputs | `metrics.jsonl` for all 108 live runs; balanced panel of 44 cultural-emergence populations |
| Ran | 2026-09-01, no compute |

## What it was for

The paper reports metrics at iteration 200 and calls the result a taxonomy of
*training outcomes*. This recomputes the identical classification on a grid, on
a balanced panel, so "outcome" can be checked.

## Headline

**It is a snapshot, and iteration 200 is on the way down.** Mean alive fraction
across 44 populations observed at every grid point:

| iteration | mean alive | median | runs fully alive | runs fully dead |
|---:|---:|---:|---:|---:|
| 25 | 0.273 | 0.000 | 12 | 32 |
| 100 | **0.858** | 1.000 | 35 | 5 |
| 150 | 0.815 | 1.000 | 25 | 6 |
| 200 | 0.717 | 0.875 | 12 | 6 |
| 300 | 0.464 | 0.562 | **1** | 15 |

**No population improves after iteration 150.** Transitions:

| | down | same | up |
|---|---:|---:|---:|
| 25 → 100 | 0 | 17 | 27 |
| 100 → 150 | 14 | 26 | 4 |
| 150 → 200 | 17 | 27 | **0** |
| 200 → 300 | 31 | 13 | **0** |

39 of 44 end below their own earlier peak. The alive fraction peaks at iteration
100 for 25 of the 44.

## What this settles

- **The state the paper calls an outcome is transient.** — *established*
- **No cultural-emergence population has ever recovered after iteration 150.**
  0/44 in two consecutive transitions. — *established*
- **Every population-level claim in the paper is a claim about iteration 200
  specifically** and must say so. — *established*

## What it does not settle

Why. Nothing here is a mechanism; [B4](../b4_public_channel/NOTES.md) offers a
candidate and cannot establish direction.

The iteration-25 row is also not what it appears: 32 of 44 runs read as 0% alive
there, which [B3](../b3_substrate_relative_regimes/NOTES.md) shows is largely the
absolute perception threshold mislabelling untrained populations on hard
substrates, not a real state.

## Status

`CLOSED`. Supersedes the point estimate at iteration 200 as the way to report
regime composition.

---
*Analysed 2026-09-01. Machine output stays local.*
