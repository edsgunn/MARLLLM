# run02_8agent_7B_ashbourne_school

*social terminal — memory retune, and the run whose traces found the confabulation bug*

|  |  |
|---|---|
| Config | [`configs/social_terminal/run02_8agent_7B_ashbourne_school.yaml`](../../../configs/social_terminal/run02_8agent_7B_ashbourne_school.yaml) |
| Model | `Qwen2.5-7B-Instruct` |
| Population | 8 agents on `ashbourne_school` |
| Settings | kl_coef 0.05, β 0.01, lr 5e-06, LoRA r 64 |
| Ran to | iteration 72 of 300 — **truncated** · 2 checkpoints |
| **Differs from** | **`run01` — `vllm_gpu_mem_util` 0.25 → 0.18 and 16 episodes per iteration instead of 8** |

## What it was for

Getting further than run 01 by giving the resident vLLM engine less memory and
the trainer more.

## Headline

**It got 72 iterations, and its traces produced the most valuable result on this
substrate: the agents were confabulating.** An agent would emit
`<read>chalkboard</read>` and then continue generating the board's contents in
the same turn — on a read that had actually failed. Nothing in the metrics
showed this; every logged number is healthy (8/8 `coherent_low_kl`, perception
loss 1.53 → 1.32).

This is the case the process note in `PROCESS.md` is about: a well-formed action
that always fails scores as healthy behaviour, and only a human reading traces
finds it.

## Core metrics

Population means over the first and last 10% of logged iterations (blocks of 7 of 72):

| metric | first block | last block |
|---|---:|---:|
| perception loss | 1.496 | 1.307 |
| KL from base | 0.001 | 0.119 |
| policy entropy | 0.512 | 0.574 |

Regime composition (paper thresholds; see `PROCESS.md`):

| iteration | n | alive | composition |
|---:|---:|---:|---|
| ~25 | 8 | 1.00 | coherent_low_kl 8 |

## What this settles

- **Information-returning verbs must end the turn.** An actor that keeps
  generating after committing to a read invents the result. — *established* by
  the traces, and fixed in `run05`/`run06`
- **`vllm_gpu_mem_util` 0.18 is the setting that survives on this rig**; it
  reached iteration 72 where 0.30 OOMs at ~10. — *established*

## What it does not settle

How much of the run's behaviour the confabulation was driving. It was fixed
rather than quantified.

## Status

`SUPERSEDED` by `run06`, which carries the fix. Cite for the confabulation finding.

---
*Analysed retrospectively on 2026-09-01 from `metrics.jsonl` and `config.json`; see the note in `PROCESS.md` on what a back-filled note cannot tell you. Machine output stays local.*
