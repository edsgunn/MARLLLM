# run8_16agent_7B_study_group_ashbourne_gc

*cultural emergence — 16-agent Ashbourne GC; the campaign's most-analysed population*

|  |  |
|---|---|
| Config | [`configs/cultural_emergence/run8_16agent_7B_study_group_ashbourne_gc.yaml`](../../../configs/cultural_emergence/run8_16agent_7B_study_group_ashbourne_gc.yaml) |
| Model | `Qwen2.5-7B-Instruct` |
| Population | 16 agents on `study_group_ashbourne_gc` |
| Settings | kl_coef 0.05, β 0.01, lr 5e-06, LoRA r 64 |
| Ran to | iteration 300 of 300 · 12 checkpoints |
| **Differs from** | **`run7_8agent_7B_study_group_ashbourne_gc` — 16 agents instead of 8, on the `extended_16` roster** |

## What it was for

The 16-agent cell of the population-size grid. It carries the population-size
claim, has four seed replicates, and is one of the two forum families in the
sanctioning corpus.

## Headline

**The best-ending population in the campaign, and the population-size story is
weaker than it looks.** At iteration 200 it is 100% alive (12 coherent, 4
committed); at 300 it is 0.88, the highest end-of-run alive fraction of any
forum population. Perception loss falls 2.35 → 1.71 and mean KL reaches 1.59.

But the full grid is not monotone in population size at either evaluation
point, on either substrate — see below. What the grid supports is narrower than
"larger is better".

## Core metrics

Population means over the first and last 10% of logged iterations (blocks of 30 of 300):

| metric | first block | last block |
|---|---:|---:|
| perception loss | 2.353 | 1.707 |
| KL from base | 0.026 | 1.587 |
| policy entropy | 0.870 | 0.601 |

Regime composition (paper thresholds; see `PROCESS.md`):

| iteration | n | alive | composition |
|---:|---:|---:|---|
| ~25 | 16 | 0.00 | text_degenerate 16 |
| ~100 | 16 | 0.88 | mid_range 14, text_degenerate 2 |
| ~150 | 16 | 1.00 | coherent_low_kl 13, committed 3 |
| ~200 | 16 | 1.00 | coherent_low_kl 12, committed 4 |
| ~300 | 16 | 0.88 | coherent_low_kl 6, mid_range 8, policy_collapse 2 |

### The population-size grid it anchors

Alive fraction by substrate and population size, one run per cell:

| substrate | N=2 | N=4 | N=8 | N=16 | N=32 |
|---|---:|---:|---:|---:|---:|
| `ashbourne_gc` @200 | 1.00 | 0.75 | 0.88 | **1.00** | 0.84 |
| `ashbourne_gc` @300 | 0.00 | 0.75 | 0.00 | **0.88** | 0.62 |
| `strathearn_server` @200 | 0.00 | 0.75 | **1.00** | 0.88 | — |
| `strathearn_server` @300 | 0.00 | 0.75 | **0.75** | 0.69 | — |

Every cell is a single run, on a project with a demonstrated late-divergence
problem at fixed seed.

## What this settles

- **Two-agent populations end dead on both substrates.** They are the only
  cells that reach alive 0.00 at iteration 300 and they do so on both. —
  *likely*, one run per cell
- **Above two agents the grid is not monotone in population size**, at either
  evaluation point or on either substrate. The 8-agent Ashbourne cell ends at
  0.00 while the 4-agent cell ends at 0.75, and 32 agents is worse than 16. —
  *established* from the grid above
- **This run is the best-ending forum population measured.** 0.88 alive at
  iteration 300. — *established*
- **Even it declines after iteration 200.** — *established*

## What it does not settle

**Causation, and the shape of the effect.** The two substrates disagree cell by
cell — 2-agent Ashbourne is alive at iteration 200 where 2-agent Strathearn has
been dead since 150, and the 8-agent cells end at 0.00 and 0.75 respectively —
so the evidence supports a substrate × population-size interaction rather than
population size alone. There are no seed replicates at 2 or 4 agents, which is
exactly where the evidence is thinnest. The revised wording — association, not control — is in
[`../_seed_replicate_regimes/REBUTTAL_NOTES.md`](../_seed_replicate_regimes/REBUTTAL_NOTES.md).

## Status

`CLOSED` at iteration 300. Reference population for the trajectory-KL work (Ashbourne-16) and half the sanctioning corpus.

---
*Analysed retrospectively on 2026-09-01 from `metrics.jsonl` and `config.json`; see the note in `PROCESS.md` on what a back-filled note cannot tell you. Machine output stays local.*
