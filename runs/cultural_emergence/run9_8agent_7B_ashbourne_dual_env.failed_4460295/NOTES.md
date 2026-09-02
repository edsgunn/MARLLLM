# run9_8agent_7B_ashbourne_dual_env.failed_4460295

*cultural emergence — the crashed first attempt at the dual-environment run*

|  |  |
|---|---|
| Config | [`configs/cultural_emergence/run9_8agent_7B_ashbourne_dual_env.yaml`](../../../configs/cultural_emergence/run9_8agent_7B_ashbourne_dual_env.yaml) |
| Model | `Qwen2.5-7B-Instruct` |
| Population | 8 agents on `study_group_ashbourne_gc + margin_notes_ashbourne` |
| Settings | kl_coef 0.05, β 0.01, lr 5e-06, LoRA r 64 |
| Ran to | iteration 0 of 300 — **truncated** · 0 checkpoints |
| **Differs from** | **`run9_8agent_7B_ashbourne_dual_env` — nothing; this is the same config, job 4460295, which died** |

## What it was for

The first submission of the dual-environment run. Kept under its job id so the
failure is findable rather than a silent gap between job numbers.

## Headline

**Two metric records before the job died, and the second one is diagnostic.**
Mean perception loss in record 2 is **30.06** — an order of magnitude above
anything else recorded anywhere in this project. Whatever went wrong went wrong
in the loss, not in the scheduler.

## Core metrics

Population means over the first and last 10% of logged iterations (blocks of 1 of 2):

| metric | first block | last block |
|---|---:|---:|
| perception loss | 2.439 | 30.064 |
| KL from base | 0.000 | 0.103 |
| policy entropy | 0.750 | 0.392 |

## What this settles

- **A dual-environment run can produce a perception loss of 30 nats/token
  within two iterations.** — *established*, but with no diagnosis attached

## What it does not settle

The cause. Nobody traced it at the time, and the re-run
(`run9_8agent_7B_ashbourne_dual_env`) started cleanly at 2.44, so it was not
deterministic.

## Status

`FAILED` — job 4460295. Retained as a record of the failure. Do not cite its numbers as measurements.

---
*Analysed retrospectively on 2026-09-01 from `metrics.jsonl` and `config.json`; see the note in `PROCESS.md` on what a back-filled note cannot tell you. Machine output stays local.*
