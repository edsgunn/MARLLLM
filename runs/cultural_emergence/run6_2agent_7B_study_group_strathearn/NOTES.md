# run6_2agent_7B_study_group_strathearn

*cultural emergence — the 2-agent substrate screen, one arm per environment*

|  |  |
|---|---|
| Config | [`configs/cultural_emergence/run6_2agent_7B_study_group_strathearn.yaml`](../../../configs/cultural_emergence/run6_2agent_7B_study_group_strathearn.yaml) |
| Model | `Qwen2.5-7B-Instruct` |
| Population | 2 agents on `study_group_strathearn` |
| Settings | kl_coef 0.05, β 0.01, lr 5e-06, LoRA r 64 |
| Ran to | iteration 100 of 100 · 4 checkpoints |
| **Differs from** | **the other `run6_2agent` arms — substrate only; model, roster size, KL, β and horizon are held fixed** |

## What it was for

Screening four candidate substrates at the smallest population that still has
an interaction, before committing 8-agent runs to any of them. Study Group, Strathearn roster — the same forum with a different cast.

Two agents is the cheapest configuration that can show whether a substrate
produces conversation at all. It is explicitly *not* a test of whether the
substrate supports a population — that is what run 7 is for.

## Headline

**The substrate screen passed, and it passed too easily to discriminate.** All
four arms are alive at iteration 100 and three of the four are wholly
`coherent_low_kl`; perception loss falls in every arm and KL from base stays
below 0.11 throughout. At 2 agents and 100 iterations nothing has had time to
go wrong, so the screen ranks substrates on *nothing* — its real content is
that none of the four is immediately broken.

## Core metrics

Population means over the first and last 10% of logged iterations (blocks of 10 of 100):

| metric | first block | last block |
|---|---:|---:|
| perception loss | 1.598 | 1.226 |
| KL from base | 0.001 | 0.075 |
| policy entropy | 0.459 | 0.393 |

Regime composition (paper thresholds; see `PROCESS.md`):

| iteration | n | alive | composition |
|---:|---:|---:|---|
| ~25 | 2 | 1.00 | coherent_low_kl 2 |
| ~100 | 2 | 1.00 | coherent_low_kl 2 |

## What this settles

- **All four substrates run without immediate pathology at 2 agents.** Perception
  loss falls in every arm and no agent leaves the alive region. — *established*
- **A 2-agent, 100-iteration screen cannot rank substrates.** The arms are
  indistinguishable on every logged metric, and the differences that later
  mattered (Robotic Athanor's failure, Ashbourne's late collapse) are invisible
  here. — *established*

## What it does not settle

Everything the campaign actually cared about. KL from base never exceeds 0.11,
so no agent has moved far enough for the committed/coherent distinction to
exist; the substrate differences appear at 8 agents and after iteration 150.

## Status

`CLOSED`. Cite as a screen, not as a comparison.

---
*Analysed retrospectively on 2026-09-01 from `metrics.jsonl` and `config.json`; see the note in `PROCESS.md` on what a back-filled note cannot tell you. Machine output stays local.*
