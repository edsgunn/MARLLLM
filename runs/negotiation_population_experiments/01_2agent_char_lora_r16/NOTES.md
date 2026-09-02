# 01_2agent_char_lora_r16

*negotiation population — two agents, truncated at iteration 389 of 500*

|  |  |
|---|---|
| Config | [`configs/negotiation_population_experiments/01_2agent_char_lora_r16.yaml`](../../../configs/negotiation_population_experiments/01_2agent_char_lora_r16.yaml) |
| Model | `Qwen2.5-1.5B-Instruct` |
| Population | 2 agents on `deal_or_no_deal` |
| Settings | kl_coef 0.1, β 0.01, lr 1e-05, LoRA r 16 |
| Ran to | iteration 389 of 500 — **truncated** · 7 checkpoints |
| **Differs from** | **the other arm of this pair — population size only (2 vs 4 agents)** |

## What it was for

Carrying the population trainer — per-agent LoRA on a shared base, random
pairing — onto the negotiation substrate, as the bridge between the Phase A
two-agent work and the forum populations.

## Headline

**Perception loss collapses by an order of magnitude and the task success rate
never leaves zero.** Perception loss falls 2.18 → 0.21 and entropy
2.07 → 0.34; `success` is logged as 0.000 at every single iteration of
the run.

That combination is the clearest statement anywhere in this project of what the
objective does and does not optimise: **the agents became excellent at
predicting their own conversation and never learned to close a deal.** They are
not the same thing, and nothing in CCSM makes them the same thing.

## Core metrics

Population means over the first and last 10% of logged iterations (blocks of 38 of 389):

| metric | first block | last block |
|---|---:|---:|
| perception loss | 2.176 | 0.214 |
| KL from base | 0.002 | 0.211 |
| policy entropy | 2.065 | 0.344 |

Regime composition (paper thresholds; see `PROCESS.md`):

| iteration | n | alive | composition |
|---:|---:|---:|---|
| ~25 | 2 | 0.00 | text_degenerate 2 |
| ~100 | 2 | 0.00 | policy_collapse 1, text_degenerate 1 |
| ~150 | 2 | 0.00 | policy_collapse 2 |
| ~200 | 2 | 1.00 | coherent_low_kl 2 |
| ~300 | 2 | 0.00 | policy_collapse 2 |

## What this settles

- **Surprise minimisation on this substrate is compatible with a zero external
  success rate for 300+ consecutive iterations.** — *established*, two runs
- **The population trainer runs on `deal_or_no_deal`.** — *established*

## What it does not settle

Whether the pretrained baseline would have done better. It would not have done
much worse: the Phase A baseline evaluation puts untrained agreement at 3–5%.
Nothing here isolates whether training made external performance worse or merely
failed to make it better.

## Status

`TRUNCATED`. Cite the perception-vs-success dissociation.

---
*Analysed retrospectively on 2026-09-01 from `metrics.jsonl` and `config.json`; see the note in `PROCESS.md` on what a back-filled note cannot tell you. Machine output stays local.*
