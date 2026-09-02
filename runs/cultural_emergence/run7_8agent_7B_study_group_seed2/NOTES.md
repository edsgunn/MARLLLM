# run7_8agent_7B_study_group_seed2

*cultural emergence — Study Group seed replicate (seed2)*

|  |  |
|---|---|
| Config | [`configs/cultural_emergence/run7_8agent_7B_study_group_seed2.yaml`](../../../configs/cultural_emergence/run7_8agent_7B_study_group_seed2.yaml) |
| Model | `Qwen2.5-7B-Instruct` |
| Population | 8 agents on `study_group` |
| Settings | kl_coef 0.05, β 0.01, lr 5e-06, LoRA r 64 |
| Ran to | iteration 53 of 100 — **truncated** · 2 checkpoints |
| **Differs from** | **`run7_8agent_7B_study_group` — training seed only** |

## What it was for

An early attempt at seed replication on the Study Group substrate, before the
five-seed sets on Ashbourne GC, Inner Loop and Margin Notes replaced it.

## Headline

**Alive and wholly coherent as far as it got (truncated at iteration 53).** Every agent is
`coherent_low_kl` with mean KL from base below 0.11 — the population had not
started to differentiate when the run stopped.

## Core metrics

Population means over the first and last 10% of logged iterations (blocks of 5 of 53):

| metric | first block | last block |
|---|---:|---:|
| perception loss | 1.357 | 1.256 |
| KL from base | 0.001 | 0.034 |
| policy entropy | 0.563 | 0.608 |

Regime composition (paper thresholds; see `PROCESS.md`):

| iteration | n | alive | composition |
|---:|---:|---:|---|
| ~25 | 8 | 1.00 | coherent_low_kl 8 |

## What this settles

- **Nothing.** Too short to reach the iterations where regimes separate.

## What it does not settle

Everything. Superseded as a replication design by the five-seed sets, which
were run to 150 with a fixed roster.

## Status

`TRUNCATED`. Superseded by the five-seed replicate sets. Not cited.

---
*Analysed retrospectively on 2026-09-01 from `metrics.jsonl` and `config.json`; see the note in `PROCESS.md` on what a back-filled note cannot tell you. Machine output stays local.*
