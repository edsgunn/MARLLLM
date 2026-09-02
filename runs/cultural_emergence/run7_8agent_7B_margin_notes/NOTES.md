# run7_8agent_7B_margin_notes

*cultural emergence — 8-agent substrate set, Margin Notes*

|  |  |
|---|---|
| Config | [`configs/cultural_emergence/run7_8agent_7B_margin_notes.yaml`](../../../configs/cultural_emergence/run7_8agent_7B_margin_notes.yaml) |
| Model | `Qwen2.5-7B-Instruct` |
| Population | 8 agents on `margin_notes` |
| Settings | kl_coef 0.05, β 0.01, lr 5e-06, LoRA r 64 |
| Ran to | iteration 300 of 300 · 12 checkpoints |
| **Differs from** | **`run7_8agent_7B_inner_loop` — substrate only; Margin Notes is the closest substrate to Inner Loop in style and also carries a `<python>` tool** |

## What it was for

A second tool-bearing substrate, close enough to Inner Loop to serve as its
swap partner in the PMI-decomposition environment swaps, and far enough to test
whether the Inner Loop result is substrate-specific.

## Headline

**Margin Notes loses half its population between iterations 150 and 300.**
Alive goes 1.00 → 0.88 → 0.75 → 0.50 across iterations 100/150/200/300, ending
with two `policy_collapse` and two `text_degenerate` agents; perception loss
rises 1.53 → 1.98 and mean KL reaches 2.21. Its four seed replicates, which stop
at 150, are all 100% alive — **the substrate looks healthy at every horizon the
replicates reach and is visibly failing by the horizon this run reaches.**

## Core metrics

Population means over the first and last 10% of logged iterations (blocks of 32 of 326):

| metric | first block | last block |
|---|---:|---:|
| perception loss | 1.530 | 1.981 |
| KL from base | 0.008 | 2.210 |
| policy entropy | 0.455 | 0.609 |

Regime composition (paper thresholds; see `PROCESS.md`):

| iteration | n | alive | composition |
|---:|---:|---:|---|
| ~25 | 8 | 1.00 | coherent_low_kl 8 |
| ~100 | 8 | 1.00 | coherent_low_kl 6, committed 2 |
| ~150 | 8 | 0.88 | coherent_low_kl 5, committed 2, policy_collapse 1 |
| ~200 | 8 | 0.75 | coherent_low_kl 3, mid_range 3, text_degenerate 2 |
| ~300 | 8 | 0.50 | coherent_low_kl 3, mid_range 1, policy_collapse 2, text_degenerate 2 |

## What this settles

- **Margin Notes is a late-failing substrate.** One long run, 50% alive at
  iteration 300 against 100% alive in four replicates that stop at 150. —
  *likely*
- **Evaluating at iteration 150 would have reported this substrate as healthy.**
  — *established* by the contrast with its own replicates

## What it does not settle

Whether the long run is representative: it is n=1 at 300 iterations, and this
project has one direct demonstration that identical configurations diverge late
(see `../kl_beta_sweep_8agent_ashbourne_gc/NOTES.md`). The replicates would have
to be extended to settle it.

## Status

`CLOSED` at iteration 300. Used as the swap partner for Inner Loop in the PMI decomposition, and as one of the three multi-seed datasets in the trajectory-KL work.

---
*Analysed retrospectively on 2026-09-01 from `metrics.jsonl` and `config.json`; see the note in `PROCESS.md` on what a back-filled note cannot tell you. Machine output stays local.*
