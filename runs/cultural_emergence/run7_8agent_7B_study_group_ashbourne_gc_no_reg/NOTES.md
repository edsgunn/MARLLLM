# run7_8agent_7B_study_group_ashbourne_gc_no_reg

*cultural emergence — Ashbourne GC with all regularisation removed*

|  |  |
|---|---|
| Config | [`configs/cultural_emergence/run7_8agent_7B_study_group_ashbourne_gc_no_reg.yaml`](../../../configs/cultural_emergence/run7_8agent_7B_study_group_ashbourne_gc_no_reg.yaml) |
| Model | `Qwen2.5-7B-Instruct` |
| Population | 8 agents on `study_group_ashbourne_gc` |
| Settings | kl_coef 0, β 0, lr 5e-06, LoRA r 64 |
| Ran to | iteration 300 of 300 · 12 checkpoints |
| **Differs from** | **`run7_8agent_7B_study_group_ashbourne_gc` — `kl_coef` 0.05 → 0 and `beta` 0.01 → 0** |

## What it was for

Whether the KL anchor and entropy bonus are doing anything on this substrate,
by removing both.

## Headline

**Removing both regularisers made the population *better*, not worse.** Alive
is 0.75 at iteration 300 against 0.00 for the regularised reference, and
perception loss ends at 1.88 against 2.27. With `kl_coef` 0 the logged KL is
identically 0 — the anchor term is off, so the run cannot report drift — but on
every metric that is still measured this arm outlives its regularised twin.

## Core metrics

Population means over the first and last 10% of logged iterations (blocks of 30 of 300):

| metric | first block | last block |
|---|---:|---:|
| perception loss | 2.182 | 1.876 |
| KL from base | 0.000 | 0.000 |
| policy entropy | 0.741 | 0.693 |

Regime composition (paper thresholds; see `PROCESS.md`):

| iteration | n | alive | composition |
|---:|---:|---:|---|
| ~25 | 8 | 0.00 | text_degenerate 8 |
| ~100 | 8 | 1.00 | coherent_low_kl 8 |
| ~150 | 8 | 1.00 | coherent_low_kl 7, mid_range 1 |
| ~200 | 8 | 0.88 | coherent_low_kl 6, mid_range 1, text_degenerate 1 |
| ~300 | 8 | 0.75 | coherent_low_kl 2, mid_range 4, text_degenerate 2 |

## What this settles

- **Neither the KL anchor at 0.05 nor the entropy bonus at 0.01 is protecting
  this substrate.** The unregularised arm ends healthier. — *provisional*, one
  run each side, and this project has a demonstrated late-divergence problem at
  fixed seed

## What it does not settle

Which of the two removals mattered — they were removed together. The KL/β sweep
separates them, and agrees: the best cells there are low-β, and β ≥ 0.1 is
fatal at every KL.

## Status

`CLOSED` at iteration 300. Note that `kl` is not measurable in this run.

---
*Analysed retrospectively on 2026-09-01 from `metrics.jsonl` and `config.json`; see the note in `PROCESS.md` on what a back-filled note cannot tell you. Machine output stays local.*
