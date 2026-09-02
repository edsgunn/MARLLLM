# migration_ada_strathearn_server_to_ashbourne_gc

*migration — Ada Whitfield carried from `strathearn_server` into `ashbourne_gc`, replacing a resident*

|  |  |
|---|---|
| Model | `Qwen2.5-7B-Instruct` |
| Population | 8 agents on `ashbourne_gc` |
| Settings | kl_coef 0.05, β 0.01, lr 5e-06, LoRA r 64 |
| Ran to | iteration 150 of 150 · 3 checkpoints |
| **Differs from** | **the host population's own run — one agent's LoRA adapter has been swapped in from a different population trained on a different substrate** |

## What it was for

Whether a trained agent keeps its identity when moved into a population that
has never met it. Ada Whitfield's per-agent adapter is lifted from the source
population's iteration-100 checkpoint and dropped into the target population's
iteration-100 checkpoint in place of a resident; training then continues for
150 iterations.

Belongs to the `_gc` / `_server` batch (2026-05-06); the family reading is in
[`../NOTES.md`](../NOTES.md).

## Headline

**Ada Whitfield arrived at KL 0.27 from base and ended at 0.20 — **stayed where it arrived****,
while the eight residents went 2.10 → 2.10.

Perception loss tracks the host, not the source: the migrant goes
2.22 → 1.71 against the residents' 2.15 → 1.76. **The migrant
learns the new world at the host's rate while keeping its own distance from
base.**

## Core metrics

Population means over the first and last 10% of logged iterations (blocks of 5 of 50):

| metric | first block | last block |
|---|---:|---:|
| perception loss | 1.816 | 1.661 |
| KL from base | 1.706 | 1.780 |
| policy entropy | 0.770 | 0.719 |

Regime composition (paper thresholds; see `PROCESS.md`):

| iteration | n | alive | composition |
|---:|---:|---:|---|
| ~100 | 8 | 0.00 | text_degenerate 8 |
| ~150 | 8 | 0.88 | coherent_low_kl 2, mid_range 5, text_degenerate 1 |

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
