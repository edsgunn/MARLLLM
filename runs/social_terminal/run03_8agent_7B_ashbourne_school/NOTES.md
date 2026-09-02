# run03_8agent_7B_ashbourne_school

*social terminal — the OOM that fixed the memory recipe*

|  |  |
|---|---|
| Config | [`configs/social_terminal/run03_8agent_7B_ashbourne_school.yaml`](../../../configs/social_terminal/run03_8agent_7B_ashbourne_school.yaml) |
| Model | `Qwen2.5-7B-Instruct` |
| Population | 8 agents on `ashbourne_school` |
| Settings | kl_coef 0.05, β 0.01, lr 5e-06, LoRA r 64 |
| Ran to | iteration 14 of 300 — **truncated** · 0 checkpoints |
| **Differs from** | **`run02` — `vllm_gpu_mem_util` 0.18 → 0.30 and 24 episodes per iteration** |

## What it was for

Pushing episodes per iteration up to 24, with more memory handed to vLLM.

## Headline

**Fourteen iterations, then out of memory — and that is the whole result.**
Giving the resident vLLM engine 0.30 of the device (~27 GiB) starves training
and the chunked loss in `marlllm/loss.py` — a `log_softmax` over the ~152k
vocabulary — OOMs at around iteration 10. The value that survives is 0.18.

## Core metrics

Population means over the first and last 10% of logged iterations (blocks of 1 of 14):

| metric | first block | last block |
|---|---:|---:|
| perception loss | 1.486 | 1.417 |
| KL from base | 0.000 | 0.002 |
| policy entropy | 0.521 | 0.513 |

Regime composition (paper thresholds; see `PROCESS.md`):

| iteration | n | alive | composition |
|---:|---:|---:|---|
| ~25 | 8 | 1.00 | coherent_low_kl 8 |

## What this settles

- **`vllm_gpu_mem_util` 0.30 is not viable for an 8-agent 7B population on this
  rig; 0.18 is.** — *established*, three runs

## What it does not settle

Whether 24 episodes per iteration is affordable at 0.18. `run06` answers that:
it is.

## Status

`FAILED` — OOM at ~iteration 10. Retained for the memory recipe.

---
*Analysed retrospectively on 2026-09-01 from `metrics.jsonl` and `config.json`; see the note in `PROCESS.md` on what a back-filled note cannot tell you. Machine output stays local.*
