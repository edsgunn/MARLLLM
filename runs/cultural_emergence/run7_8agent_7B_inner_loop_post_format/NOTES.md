# run7_8agent_7B_inner_loop_post_format

*cultural emergence — Inner Loop with the revised post format, run to 300 iterations*

|  |  |
|---|---|
| Config | [`configs/cultural_emergence/run7_8agent_7B_inner_loop_post_format.yaml`](../../../configs/cultural_emergence/run7_8agent_7B_inner_loop_post_format.yaml) |
| Model | `Qwen2.5-7B-Instruct` |
| Population | 8 agents on `inner_loop` |
| Settings | kl_coef 0.05, β 0.01, lr 5e-06, LoRA r 64 |
| Ran to | iteration 341 of 300 · 13 checkpoints |
| **Differs from** | **`run7_8agent_7B_inner_loop` — revised post formatting, and a full 300-iteration horizon rather than 144** |

## What it was for

Extending the reference Inner Loop population past the horizon the original run
reached, under the revised post format, to see what the substrate does with
three times the training.

## Headline

**The extra 200 iterations bought a worse population.** Alive is 1.00 at
iterations 100, 150 and 200 and 0.88 at 300; perception loss *rises* 1.56 →
1.77 over the run, and the two agents that were `committed` at 200 end
`mid_range`. Inner Loop delays the late decline that the Ashbourne substrates
show but does not avoid it.

## Core metrics

Population means over the first and last 10% of logged iterations (blocks of 34 of 346):

| metric | first block | last block |
|---|---:|---:|
| perception loss | 1.564 | 1.771 |
| KL from base | 0.007 | 1.123 |
| policy entropy | 0.616 | 0.657 |

Regime composition (paper thresholds; see `PROCESS.md`):

| iteration | n | alive | composition |
|---:|---:|---:|---|
| ~25 | 8 | 1.00 | coherent_low_kl 8 |
| ~100 | 8 | 1.00 | coherent_low_kl 6, committed 2 |
| ~150 | 8 | 1.00 | coherent_low_kl 6, committed 2 |
| ~200 | 8 | 1.00 | coherent_low_kl 6, committed 2 |
| ~300 | 8 | 0.88 | coherent_low_kl 5, mid_range 2, policy_collapse 1 |

## What this settles

- **The late decline is not a property of one substrate.** Inner Loop — the
  healthiest substrate in the campaign — also ends below its own peak. —
  *likely*, one long run

## What it does not settle

Whether the post-format change contributed. It is confounded with the horizon:
this is the only Inner Loop run at 300 iterations *and* the only one with the
revised format, so the two cannot be separated here.

## Status

`CLOSED`. Confounded — do not use as a clean format comparison.

---
*Analysed retrospectively on 2026-09-01 from `metrics.jsonl` and `config.json`; see the note in `PROCESS.md` on what a back-filled note cannot tell you. Machine output stays local.*
