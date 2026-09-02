# <run directory name>

*<one line: which family this belongs to and what kind of run it is>*

|  |  |
|---|---|
| Config | `configs/<family>/<name>.yaml` |
| Model | |
| Population | *N* agents on `<env>` |
| Regularisation | `kl_coef` , `beta` , `lr` , LoRA r |
| Ran to | iteration *x* of *y* · *n* checkpoints |
| **Differs from** | **`<the one run this is read against>`** — the single field that changed. |

## What it was for

What question this run exists to answer, and what result would have meant the
idea was wrong. Copy this from the config header, which was written before the
run — do not reconstruct it from the result.

## Headline

The one thing this run establishes, in bold, in a sentence. If it establishes
nothing, say that; a run that failed to separate its arms is a finding about
the design, and recording it stops the design being repeated.

## Core metrics

| metric | first logged iter | last logged iter |
|---|---|---|
| mean perception loss | | |
| mean KL from base | | |
| mean policy entropy | | |

Regime composition at iteration *N*: …

Per-agent table if the population is small enough to read.

## What this settles

- **Claim.** Evidence in a line. — *established | likely | provisional | suspected*

## What it does not settle

What a reader might wrongly take from this run, and what would be needed to
settle it.

## Status

`CLOSED` / `SUPERSEDED by <run>` / `TRUNCATED at iter N` / `NO DATA`, and
whether the run can be cited.

---
*Machine output stays local: `metrics.jsonl`, `traces/`, `checkpoints/`.*
