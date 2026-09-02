# 01_stage1_base_model

*terminal — Stage 1, ls + cat; started twice, produced nothing*

|  |  |
|---|---|
| Config | [`configs/terminal_experiments/01_stage1_sparse_actions.yaml`](../../../configs/terminal_experiments/01_stage1_sparse_actions.yaml) |
| Model | `Qwen2.5-1.5B` |
| Population | 1 agents on `terminal (stage 1)` |
| Settings | kl_coef 0, β 0.01, lr 3e-05, LoRA r 0 |
| Ran to | **no metrics written** |
| **Differs from** | **`00_stage0_baseline` — a shell appears, and with it ~1–3% action tokens** |

## What it was for

The first stage with actions: the model arrives knowing the corpus text
distribution and must learn to navigate it with `ls` and `cat`. Its config
pre-registers three failure modes to watch for in the metrics — oscillating
`act_loss`, `act_loss` flat at zero (action tokens masked wrongly), and
`perc_loss` diverging while `act_loss` improves.

## Headline

**Started on 2026-04-27 and again on 2026-04-28, and wrote no metrics either
time.** The log contains two "Starting training" lines and nothing after them.
No checkpoint, no `metrics.jsonl`.

The terminal line stops here. Stages 2, 3, 3.5 and 4 have configs — including
`04_stage35_embodiment.yaml`, described in its own header as "THE load-bearing
experimental config for the framework's core claim" — and no runs at all.

## Core metrics

_No metrics were written._

## What this settles

- **Nothing.**

## What it does not settle

The entire terminal programme, including the embodiment claim: that one agent's
`cat $SCRATCH/notes/quantum.txt` delivers another agent's earlier writing as an
ordinary observation, so that coordination — if it appears — emerges from the
objective alone with no inter-agent machinery.

## Status

`NO DATA`. **The terminal line is abandoned, not concluded.** See the open questions in `STATE.md`.

---
*Analysed retrospectively on 2026-09-01 from `metrics.jsonl` and `config.json`; see the note in `PROCESS.md` on what a back-filled note cannot tell you. Machine output stays local.*
