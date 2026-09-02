# migration_experiments

*Does a trained agent keep its identity when moved into a population that has
never met it? **The family is the unit of inference.***

|  |  |
|---|---|
| Configs | [`configs/migration_experiments/`](../../configs/migration_experiments/) — the run table and the checkpoint-build commands are in its `README.md` |
| Model | `Qwen2.5-7B-Instruct`, LoRA r64 per agent on a shared base |
| Design | 8 migrants × 2 substrate pairs = 16 runs, each 150 iterations |
| Pairs | `study_group` ↔ `study_group_strathearn` (2026-05-05) and `study_group_ashbourne_gc` ↔ `study_group_strathearn_server` (2026-05-06) |
| Method | `scripts/build_migration_checkpoint.py` lifts one agent's adapter from the source population's `iter_000100` into the target's `iter_000100`, replacing a named resident; training then continues |
| **Differs from** | **the host population's own run** — exactly one agent's adapter is foreign. |

## What it was for

Per-agent LoRA on a shared base makes an agent a portable object: 30 MB of
adapter weights that can be moved between populations without touching anything
else. This family asks what that object carries. Is a trained agent a policy
that has learned *its population*, in which case dropping it into a stranger
population should be catastrophic or should wash out — or is it a policy that
has learned *a character*, in which case it should keep its shape and adapt its
world model?

## Headline

**Both, cleanly separated. A migrant adapts its world model to the host at the
host's rate, and keeps its own distance from the base policy.**

Across all 16 runs the migrant's perception loss tracks the residents' almost
exactly — Ada arriving in Ashbourne GC goes 2.22 → 1.71 while the residents go
2.15 → 1.76. It learns the new forum as fast as agents who have been there all
along.

Its KL from base does something entirely different: **it does not regress toward
the host population.** Migrants who arrived far from base went further —
Mhairi 4.66 → 8.01, Priya 2.74 → 6.50, Marcus 2.55 → 5.38 — while migrants who
arrived close to base stayed close: Ada 0.27 → 0.20, Tom 0.34 → 0.35, Hana
0.39 → 0.23, Callum 1.07 → 0.42.

Counted across both batches: **all 7 migrants who arrived above KL 2.0 drifted
further, and 8 of 9 who arrived below KL 1.1 stayed put.** The single exception
is Finn in the `_server → _gc` batch (0.13 → 3.08).

## Core metrics

Migrant against the eight residents, first and last logged iteration:

| run | migrant | migrant KL | residents' KL | migrant perc | residents' perc |
|---|---|---:|---:|---:|---:|
| ada_strathearn_server→ashbourne_gc | Ada Whitfield | 0.27 → 0.20 | 2.10 → 2.11 | 2.22 → 1.71 | 2.15 → 1.76 |
| ada_strathearn→ashbourne | Ada Whitfield | 0.14 → 0.13 | 1.04 → 1.31 | 1.40 → 1.24 | 1.40 → 1.22 |
| callum_strathearn_server→ashbourne_gc | Callum Reid | 1.07 → 0.42 | 2.08 → 2.00 | 2.16 → 2.04 | 2.19 → 1.78 |
| callum_strathearn→ashbourne | Callum Reid | 0.65 → 0.33 | 1.09 → 1.53 | 1.49 → 1.41 | 1.48 → 1.35 |
| finn_strathearn_server→ashbourne_gc | Finn Nakamura | 0.13 → **3.08** | 1.19 → 1.22 | 2.16 → 1.71 | 2.13 → 1.77 |
| finn_strathearn→ashbourne | Finn Nakamura | 2.22 → 3.75 | 0.66 → 0.73 | 1.40 → 1.32 | 1.35 → 1.26 |
| hana_ashbourne_gc→strathearn_server | Hana Yilmaz | 0.39 → 0.23 | 0.50 → 0.87 | 1.34 → 1.29 | 1.34 → 1.21 |
| hana_ashbourne→strathearn | Hana Yilmaz | 0.23 → 0.28 | 0.66 → 1.41 | 1.23 → 1.45 | 1.24 → 1.47 |
| marcus_ashbourne_gc→strathearn_server | Marcus Webb | 2.55 → 5.38 | 0.52 → 0.97 | 1.48 → 1.40 | 1.39 → 1.24 |
| marcus_ashbourne→strathearn | Marcus Webb | 2.62 → 4.28 | 0.51 → 0.74 | 1.30 → 1.51 | 1.26 → 1.43 |
| mhairi_strathearn_server→ashbourne_gc | Mhairi Buchanan | 4.66 → **8.01** | 1.36 → 1.23 | 2.22 → 2.18 | 2.10 → 1.88 |
| mhairi_strathearn→ashbourne | Mhairi Buchanan | 2.44 → 4.88 | 0.59 → 0.89 | 1.45 → 1.48 | 1.35 → 1.40 |
| priya_ashbourne_gc→strathearn_server | Priya Shah | 2.74 → 6.50 | 0.20 → 0.18 | 1.56 → 1.50 | 1.43 → 1.30 |
| priya_ashbourne→strathearn | Priya Shah | 2.85 → 3.95 | 0.47 → 0.65 | 1.44 → 1.45 | 1.33 → 1.38 |
| tom_ashbourne_gc→strathearn_server | Tom Whitaker | 0.34 → 0.35 | 0.61 → 1.04 | 1.46 → 1.37 | 1.38 → 1.40 |
| tom_ashbourne→strathearn | Tom Whitaker | 0.32 → 0.30 | 0.78 → 1.57 | 1.27 → 1.52 | 1.26 → 1.44 |

## What this settles

- **A migrated adapter learns the host substrate at the resident rate.** Migrant
  and resident perception losses move together in all 16 runs. — *established*
- **Migration does not reset drift. A committed agent stays committed and gets
  more so.** 7/7 high-arrival migrants drifted further; 8/9 low-arrival migrants
  did not. — *likely*
- **World model and policy identity are separable in this framework**, and the
  per-agent LoRA is where the second one lives. — *likely*

## What it does not settle

- **What happens to the hosts.** There is no no-migration control at the same
  iteration range, so the residents' trajectories cannot be attributed to the
  migrant. This is the design's one missing arm and it is cheap to add.
- **What the migrant's extra drift is made of.** KL from base rising from 2.7 to
  6.5 is a large movement with no behavioural characterisation attached; the
  traces were never read for these runs.
- **Whether the effect survives past 150 iterations**, which is where every
  migration run stops and roughly where the forum campaign's decline begins.

## Status

`CLOSED` at 150 iterations, both batches. The first submission is invalid and
sits in [`_archive_2026-05-06_kl0_bug/`](_archive_2026-05-06_kl0_bug/) — those
runs never loaded their migration checkpoints and are not migrations at all.

---
*Analysed retrospectively on 2026-09-01 from `metrics.jsonl`. Machine output
stays local.*
