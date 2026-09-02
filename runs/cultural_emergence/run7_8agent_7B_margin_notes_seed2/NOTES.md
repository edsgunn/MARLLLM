# run7_8agent_7B_margin_notes_seed2

*cultural emergence — Margin Notes seed replicate (seed2)*

|  |  |
|---|---|
| Config | [`configs/cultural_emergence/run7_8agent_7B_margin_notes_seed2.yaml`](../../../configs/cultural_emergence/run7_8agent_7B_margin_notes_seed2.yaml) |
| Model | `Qwen2.5-7B-Instruct` |
| Population | 8 agents on `margin_notes` |
| Settings | kl_coef 0.05, β 0.01, lr 5e-06, LoRA r 64 |
| Ran to | iteration 150 of 150 · 6 checkpoints |
| **Differs from** | **`run7_8agent_7B_margin_notes` — training seed only, and a 150-iteration horizon instead of 300** |

## What it was for

One of four seed replicates of the Margin Notes population, for the
population-level re-analysis. The unit of inference is the set — see
[`../_seed_replicate_regimes/SUMMARY.md`](../_seed_replicate_regimes/SUMMARY.md).

## Headline

**100% alive at every measured iteration, as all five Margin Notes populations
are at this horizon.** Committed fraction across the five is 0.07 ± 0.11; zero
policy-collapse and zero text-degeneration events in the whole set.

Read against the 300-iteration parent run, which is 50% alive, this set is the
project's clearest demonstration that **a horizon can be short enough to make a
failing substrate look sound.**

## Core metrics

Population means over the first and last 10% of logged iterations (blocks of 15 of 150):

| metric | first block | last block |
|---|---:|---:|
| perception loss | 1.502 | 1.239 |
| KL from base | 0.004 | 0.158 |
| policy entropy | 0.438 | 0.426 |

Regime composition (paper thresholds; see `PROCESS.md`):

| iteration | n | alive | composition |
|---:|---:|---:|---|
| ~25 | 8 | 1.00 | coherent_low_kl 8 |
| ~100 | 8 | 1.00 | coherent_low_kl 8 |
| ~150 | 8 | 1.00 | coherent_low_kl 8 |

## What this settles

- **At iteration 150, Margin Notes replicates as a fully alive substrate across
  5 populations.** — *established* — `../_seed_replicate_regimes/SUMMARY.md`

## What it does not settle

What happens after 150. The replicates were never extended, and the one run that
was extended lost half its population.

## Status

`CLOSED` at 150 iterations. Cite the family.

---
*Analysed retrospectively on 2026-09-01 from `metrics.jsonl` and `config.json`; see the note in `PROCESS.md` on what a back-filled note cannot tell you. Machine output stays local.*
