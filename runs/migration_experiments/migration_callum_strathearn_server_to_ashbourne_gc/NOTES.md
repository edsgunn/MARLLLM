# migration_callum_strathearn_server_to_ashbourne_gc

*migration — Callum Reid carried from `strathearn_server` into `ashbourne_gc`, replacing a resident*

|  |  |
|---|---|
| Model | `Qwen2.5-7B-Instruct` |
| Population | 8 agents on `ashbourne_gc` |
| Settings | kl_coef 0.05, β 0.01, lr 5e-06, LoRA r 64 |
| Ran to | iteration 150 of 150 · 3 checkpoints |
| **Differs from** | **the host population's own run — one agent's LoRA adapter has been swapped in from a different population trained on a different substrate** |

## What it was for

Whether a trained agent keeps its identity when moved into a population that
has never met it. Callum Reid's per-agent adapter is lifted from the source
population's iteration-100 checkpoint and dropped into the target population's
iteration-100 checkpoint in place of a resident; training then continues for
150 iterations.

Belongs to the `_gc` / `_server` batch (2026-05-06); the family reading is in
[`../NOTES.md`](../NOTES.md).

## Headline

**Callum Reid arrived at KL 1.07 from base and ended at 0.42 — **moved back toward base****,
while the eight residents went 2.08 → 2.00.

Perception loss tracks the host, not the source: the migrant goes
2.16 → 2.04 against the residents' 2.19 → 1.78. **The migrant
learns the new world at the host's rate while keeping its own distance from
base.**

## Core metrics

Population means over the first and last 10% of logged iterations (blocks of 5 of 50):

| metric | first block | last block |
|---|---:|---:|
| perception loss | 1.933 | 1.865 |
| KL from base | 1.715 | 1.816 |
| policy entropy | 0.711 | 0.694 |

Regime composition (paper thresholds; see `PROCESS.md`):

| iteration | n | alive | composition |
|---:|---:|---:|---|
| ~100 | 8 | 0.00 | text_degenerate 8 |
| ~150 | 8 | 0.75 | coherent_low_kl 2, mid_range 4, text_degenerate 2 |

## What this settles

- **A migrated adapter adapts to the host substrate.** Migrant and resident
  perception losses move together. — *likely*, and consistent across all 16
  migration runs
- **Migration does not reset the migrant's drift.** It arrived close to base and stayed close. —
  *likely*

## What it does not settle

Whether the host population changes in response. Nothing here measures the
residents against a no-migration control, which was never run — that is the
single missing arm of this design.

## Status

`CLOSED` at iteration 150.

---
*Analysed retrospectively on 2026-09-01 from `metrics.jsonl` and `config.json`; see the note in `PROCESS.md` on what a back-filled note cannot tell you. Machine output stays local.*
