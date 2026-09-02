# run7_8agent_7B_conjecture

*cultural emergence — 8-agent substrate set, Conjecture Inn*

|  |  |
|---|---|
| Config | [`configs/cultural_emergence/run7_8agent_7B_conjecture.yaml`](../../../configs/cultural_emergence/run7_8agent_7B_conjecture.yaml) |
| Model | `Qwen2.5-7B-Instruct` |
| Population | 8 agents on `conjecture` |
| Settings | kl_coef 0.05, β 0.01, lr 5e-06, LoRA r 64 |
| Ran to | iteration 100 of 100 · 4 checkpoints |
| **Differs from** | **`run6_2agent_7B_conjecture` — 8 agents instead of 2** |

## What it was for

Whether Conjecture Inn supports a population rather than a pair. The 2-agent
screen showed the substrate works; this asks whether eight agents sharing one
forum still produce coherent behaviour, and whether any of them commits.

## Headline

**Eight agents on Conjecture Inn stay alive for 100 iterations but perception
loss goes the wrong way.** It rises 1.49 → 1.64 while KL from base reaches
1.70 — the population moves away from the reference policy without getting
better at predicting its world, and three of eight agents end `mid_range`.
The substrate is usable but is not one where surprise minimisation is working.

## Core metrics

Population means over the first and last 10% of logged iterations (blocks of 10 of 100):

| metric | first block | last block |
|---|---:|---:|
| perception loss | 1.490 | 1.640 |
| KL from base | 0.001 | 1.696 |
| policy entropy | 0.672 | 0.767 |

Regime composition (paper thresholds; see `PROCESS.md`):

| iteration | n | alive | composition |
|---:|---:|---:|---|
| ~25 | 8 | 1.00 | coherent_low_kl 8 |
| ~100 | 8 | 1.00 | coherent_low_kl 5, mid_range 3 |

## What this settles

- **Conjecture Inn is not a substrate where perception loss improves.** It rises
  over the run at 8 agents while KL grows. — *provisional*, one run, 100 iterations

## What it does not settle

Whether the rise continues or reverses: the run stops at iteration 100, which
this project's balanced panel shows is roughly where populations peak. Nothing
here speaks to the late decline.

## Status

`CLOSED` at 100 iterations. Not extended; superseded in the campaign's attention by the Study Group family.

---
*Analysed retrospectively on 2026-09-01 from `metrics.jsonl` and `config.json`; see the note in `PROCESS.md` on what a back-filled note cannot tell you. Machine output stays local.*
