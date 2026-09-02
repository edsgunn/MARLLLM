# migration_mhairi_strathearn_server_to_ashbourne_gc

*migration — first batch, invalidated: the migrated checkpoint was never loaded*

|  |  |
|---|---|
| Model | `Qwen2.5-7B-Instruct` |
| Population | 8 agents on `—` |
| Settings | kl_coef 0.05, β 0.01, lr 5e-06, LoRA r 64 |
| Ran to | iteration 150 of 150 · 6 checkpoints |
| **Differs from** | **the re-run of the same name one directory up, which starts from the migration checkpoint as intended** |

## What it was for

The first submission of the migration batch. Archived on 2026-05-06 under the
label `kl0_bug`.

## Headline

**These are not migrations.** Every run in this batch that wrote metrics
(7 of 8) opens at KL from base ≈ 0.003 and at the *untrained* perception loss for its substrate (2.14–2.16 on
Ashbourne GC, 1.44–1.50 on Strathearn server). The corrected re-runs open at
KL 0.13–4.66 and perception loss 1.31–2.22 — i.e. from a policy that has
already been trained.

The migration checkpoint was not in effect. Whatever the proximate cause, these
runs trained fresh populations under migration configs, so nothing in them
speaks to migration.

## Core metrics

Population means over the first and last 10% of logged iterations (blocks of 15 of 150):

| metric | first block | last block |
|---|---:|---:|
| perception loss | 2.164 | 1.486 |
| KL from base | 0.003 | 1.197 |
| policy entropy | 0.734 | 0.698 |

Regime composition (paper thresholds; see `PROCESS.md`):

| iteration | n | alive | composition |
|---:|---:|---:|---|
| ~25 | 8 | 0.00 | text_degenerate 8 |
| ~100 | 8 | 1.00 | coherent_low_kl 7, committed 1 |
| ~150 | 8 | 1.00 | coherent_low_kl 7, committed 1 |

## What this settles

- **Nothing about migration.** They are usable, if at all, only as extra fresh
  populations on their substrates.

## What it does not settle

Anything the batch was submitted to answer. Use the re-runs.

## Status

`ARCHIVED` 2026-05-06 — invalid. Do not cite.

---
*Analysed retrospectively on 2026-09-01 from `metrics.jsonl` and `config.json`; see the note in `PROCESS.md` on what a back-filled note cannot tell you. Machine output stays local.*
