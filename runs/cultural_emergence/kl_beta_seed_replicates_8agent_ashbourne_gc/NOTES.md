# kl_beta_seed_replicates_8agent_ashbourne_gc

*cultural emergence — seed replicates of two cells of the regularisation sweep.
**The family is the unit of inference.***

|  |  |
|---|---|
| Configs | [`configs/cultural_emergence/kl_beta_seed_replicates_8agent_ashbourne_gc/`](../../../configs/cultural_emergence/kl_beta_seed_replicates_8agent_ashbourne_gc/) |
| Model | `Qwen2.5-7B-Instruct` |
| Population | 8 agents on `study_group_ashbourne_gc` |
| Cells | `lowreg` = (kl 0.0, β 0.01) and `sweet` = (kl 0.5, β 0.1), seeds 2–4 each |
| Ran | all six to iteration 300 |
| **Differs from** | **[`../kl_beta_sweep_8agent_ashbourne_gc/`](../kl_beta_sweep_8agent_ashbourne_gc/NOTES.md)** — training seed only. Seed 1 of each cell is the sweep run itself. |

## What it was for

The sweep ran one population per cell, which cannot separate a cell from a seed.
These six runs add three more seeds to the two cells the sweep made look most
interesting: the unregularised corner and the apparent sweet spot.

## Headline

**Within-cell seed variation is as large as the between-cell difference the
replicates were meant to confirm.** Across the four seeds of each cell, end-of-
training perception loss spans 1.28–1.68 for `lowreg` and 1.42–2.16 for `sweet`,
and alive at iteration 300 spans 0.50–0.88 and 0.00–0.88 respectively. The two
cells' distributions overlap almost completely.

**The sweet spot does not replicate as a sweet spot.** Its four seeds end at
alive 0.88, 0.00, 1.00, 0.88 — one total failure in four draws.

## Core metrics

| cell | seed | perc first | perc last | entropy last | KL last | alive @200 | alive @300 |
|---|---:|---:|---:|---:|---:|---:|---:|
| lowreg (kl 0.0, β 0.01) | 1 *(sweep)* | 2.18 | 1.97 | 0.73 | 1.50 | 0.88 | 0.50 |
| lowreg | 2 | 2.16 | 1.68 | 1.59 | 1.67 | 1.00 | 0.88 |
| lowreg | 3 | 2.18 | 1.61 | 0.87 | 1.79 | 0.75 | 0.75 |
| lowreg | 4 | 2.19 | 1.28 | 0.44 | 0.97 | 0.88 | 0.88 |
| sweet (kl 0.5, β 0.1) | 1 *(sweep)* | 2.18 | 1.71 | 0.91 | 0.46 | 1.00 | 0.88 |
| sweet | 1 *(beta-zoom duplicate)* | 2.19 | 2.16 | 0.91 | 0.59 | 0.88 | 0.00 |
| sweet | 2 | 2.18 | 1.85 | 0.92 | 1.34 | 1.00 | 0.88 |
| sweet | 3 | 2.18 | 1.45 | 0.77 | 0.23 | 1.00 | 1.00 |
| sweet | 4 | 2.19 | 1.52 | 0.72 | 1.16 | 0.88 | 0.88 |

The `sweet` cell has *five* rows because seed 1 was run twice by accident with
opposite outcomes — see
[`../kl_beta_sweep_8agent_ashbourne_gc/NOTES.md`](../kl_beta_sweep_8agent_ashbourne_gc/NOTES.md).

## What this settles

- **Neither cell can be declared better than the other from these data.** Mean
  alive at iteration 300 is 0.75 (lowreg, n=4) against 0.73 (sweet, n=5), with
  ranges 0.50–0.88 and 0.00–1.00. — *established*
- **A single 300-iteration population on this substrate is a weak observation.**
  The same cell produced both a 1.00-alive and a 0.00-alive outcome. —
  *established*
- **The KL anchor does bind.** End-of-training KL from base is 0.23–1.34 in the
  `sweet` cell (kl_coef 0.5) against 0.97–1.79 in `lowreg` (kl_coef 0.0). —
  *likely*

## What it does not settle

The comparison it was built to make. Three extra seeds is not enough to separate
two cells whose within-cell range is this wide; the design needed either many
more seeds or an evaluation point earlier than 300, where the runs are still
tightly grouped.

## Status

`CLOSED`. The honest reading is a negative one: the sweep's cell ranking does
not survive replication.

---
*Analysed retrospectively on 2026-09-01 from `metrics.jsonl` and `config.json`.
Machine output stays local.*
