# migration_priya_ashbourne_to_strathearn

*migration — Priya Shah carried from `ashbourne` into `strathearn`, replacing a resident*

|  |  |
|---|---|
| Model | `Qwen2.5-7B-Instruct` |
| Population | 8 agents on `strathearn` |
| Settings | kl_coef 0.05, β 0.01, lr 5e-06, LoRA r 64 |
| Ran to | iteration 150 of 150 · 3 checkpoints |
| **Differs from** | **the host population's own run — one agent's LoRA adapter has been swapped in from a different population trained on a different substrate** |

## What it was for

Whether a trained agent keeps its identity when moved into a population that
has never met it. Priya Shah's per-agent adapter is lifted from the source
population's iteration-100 checkpoint and dropped into the target population's
iteration-100 checkpoint in place of a resident; training then continues for
150 iterations.

Belongs to the `ashbourne` / `strathearn` batch (2026-05-05); the family reading is in
[`../NOTES.md`](../NOTES.md).

## Headline

**Priya Shah arrived at KL 2.85 from base and ended at 3.95 — **kept drifting****,
while the eight residents went 0.47 → 0.65.

Perception loss tracks the host, not the source: the migrant goes
1.44 → 1.45 against the residents' 1.33 → 1.38. **The migrant
learns the new world at the host's rate while keeping its own distance from
base.**

## Core metrics

Population means over the first and last 10% of logged iterations (blocks of 5 of 50):

| metric | first block | last block |
|---|---:|---:|
| perception loss | 1.398 | 1.426 |
| KL from base | 1.056 | 1.105 |
| policy entropy | 0.735 | 0.806 |

Regime composition (paper thresholds; see `PROCESS.md`):

| iteration | n | alive | composition |
|---:|---:|---:|---|
| ~100 | 8 | 1.00 | coherent_low_kl 6, committed 2 |
| ~150 | 8 | 1.00 | coherent_low_kl 6, committed 2 |

## What this settles

- **A migrated adapter adapts to the host substrate.** Migrant and resident
  perception losses move together. — *likely*, and consistent across all 16
  migration runs
- **Migration does not reset the migrant's drift.** It arrived far from base and went further. —
  *likely*

## What it does not settle

Whether the host population changes in response. Nothing here measures the
residents against a no-migration control, which was never run — that is the
single missing arm of this design.

## Status

`CLOSED` at iteration 150.

---
*Analysed retrospectively on 2026-09-01 from `metrics.jsonl` and `config.json`; see the note in `PROCESS.md` on what a back-filled note cannot tell you. Machine output stays local.*
