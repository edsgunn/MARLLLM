# migration_callum_strathearn_server_to_ashbourne_gc

*migration — first batch, invalidated: the migrated checkpoint was never loaded*

|  |  |
|---|---|
| Model | `—` |
| Ran to | **no metrics written** |
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

_No metrics were written._

## What this settles

- **Nothing about migration.** They are usable, if at all, only as extra fresh
  populations on their substrates.

## What it does not settle

Anything the batch was submitted to answer. Use the re-runs.

## Status

`ARCHIVED` 2026-05-06 — invalid. Do not cite.

---
*Analysed retrospectively on 2026-09-01 from `metrics.jsonl` and `config.json`; see the note in `PROCESS.md` on what a back-filled note cannot tell you. Machine output stays local.*

*This run is the eighth of the batch and wrote no `metrics.jsonl` at all — it
holds one checkpoint directory and nothing else. It is archived with its
siblings because it was submitted with them, not because it was diagnosed.*
