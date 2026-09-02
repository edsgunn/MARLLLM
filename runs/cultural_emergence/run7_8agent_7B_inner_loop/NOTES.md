# run7_8agent_7B_inner_loop

*cultural emergence — 8-agent substrate set, Inner Loop; the reference population for the absorption analyses*

|  |  |
|---|---|
| Config | [`configs/cultural_emergence/run7_8agent_7B_inner_loop.yaml`](../../../configs/cultural_emergence/run7_8agent_7B_inner_loop.yaml) |
| Model | `Qwen2.5-7B-Instruct` |
| Population | 8 agents on `inner_loop` |
| Settings | kl_coef 0.05, β 0.01, lr 5e-06, LoRA r 64 |
| Ran to | iteration 144 of 100 · 5 checkpoints |
| **Differs from** | **`run7_8agent_7B_study_group` — substrate only. Inner Loop is a programming forum where members execute Python in `<python>` tags through a `pybot` tool.** |

## What it was for

Whether a substrate with a real side channel for settling claims — code
execution — behaves differently from a pure conversation forum. This run became
the reference population for every downstream absorption measurement, and its
four seed replicates are the project's cleanest multi-population dataset.

## Headline

**Inner Loop is the substrate where committed agents appear reliably and
nothing collapses.** Three of eight agents end `committed` and five
`coherent_low_kl`, with no agent in a failure regime at any measured iteration.
Mean KL from base reaches 1.90 while perception loss stays flat at ~1.35 —
the population moves a long way without losing its grip on the world, which
is the signature the campaign was looking for.

It is also the substrate that carries the absorption result: the committed
agents here shrink their PMI gap ~6× more than the coherent ones and overshoot
into negative territory ([`_absorption_by_regime/REPORT.md`](../_absorption_by_regime/REPORT.md)).

## Core metrics

Population means over the first and last 10% of logged iterations (blocks of 14 of 144):

| metric | first block | last block |
|---|---:|---:|
| perception loss | 1.317 | 1.353 |
| KL from base | 0.001 | 1.904 |
| policy entropy | 0.588 | 0.755 |

Regime composition (paper thresholds; see `PROCESS.md`):

| iteration | n | alive | composition |
|---:|---:|---:|---|
| ~25 | 8 | 1.00 | coherent_low_kl 8 |
| ~100 | 8 | 1.00 | coherent_low_kl 5, committed 3 |
| ~150 | 8 | 1.00 | coherent_low_kl 5, committed 3 |

## What this settles

- **A code-execution side channel gives a substrate that produces committed
  agents without failure regimes.** 3/8 committed, 0/8 failed, across 144
  iterations. — *likely*, replicated by four seeds at 100 iterations
- **Committed agents can reach KL ≈ 9.6 from base while perception loss stays
  inside the coherent band.** Kai Dempsey is the extreme case. — *established*

## What it does not settle

Whether Inner Loop escapes the late decline. The run stops at iteration 144,
before the window where every long run in this project loses agents. Its
300-iteration sibling `run7_8agent_7B_inner_loop_post_format` does *not* escape
it — 1.00 alive at 200, 0.88 at 300.

## Status

`CLOSED` at iteration 144 (configured for 100 and run past it). The reference run for the absorption, PMI-decomposition and trajectory-KL analyses.

---
*Analysed retrospectively on 2026-09-01 from `metrics.jsonl` and `config.json`; see the note in `PROCESS.md` on what a back-filled note cannot tell you. Machine output stays local.*
