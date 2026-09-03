# env_likelihood_bias

*Tier C item C9 — arms A and B at seed 1, the environment-likelihood experiment.
**Both runs are void by their own pre-registered criteria.***

|  |  |
|---|---|
| Configs | [`configs/env_likelihood_bias/`](../../configs/env_likelihood_bias/README.md) |
| Model | `Qwen2.5-7B-Instruct`, LoRA r64 on q_proj/v_proj |
| Population | 8 agents on `study_group_ashbourne_gc`, post tags blanked |
| Ran to | iteration 300 of 300, both · jobs 6256587 (A) and 6256588 (B), ~6 h each |
| **Differs from** | **`run7_8agent_7B_study_group_ashbourne_gc`** — post tags blanked so nothing is discarded, plus provenance and the corrected return in B. |

## What it was for

To measure the term the practical loss drops. Arm A logs `ln p_env` without
using it; arm B trains on the corrected return. The decision-relevant outputs
were to be `env_share_of_return` (how large the dropped term is) and
`return_rank_corr` (whether it changes the update direction) — because if the
two returns rank actions alike, the bias does not matter for learning however
large it is.

The configs carried a **void condition**, written before the runs:
`envlp/obs_tokens_measured_frac` must be 1.0, and both
`envlp/misaligned_observations` and `envlp/provenance_misses` must be 0. *"Any
other value means the provenance path is broken and the numbers are void."*

## Headline

**Void, and doubly so.** `envlp/provenance_misses` is not 0: it climbs
monotonically from 3.8 per episode at the start to **497.5** by iteration 300.
And the quantities the experiment exists to produce were **never logged at
all** — `obs_tokens_measured_frac`, `env_share_of_return`, `return_rank_corr`
and `probe_loss` are absent from both runs' `metrics.jsonl`. The experiment
cannot answer its own question from what is on disk, and the pre-registered
condition for trusting any of it fails.

## What was logged, and what it says

| metric | arm A | arm B | void condition |
|---|---|---|---|
| `envlp/misaligned_observations` | 0.0 throughout | 0.0 throughout | **pass** |
| `envlp/provenance_misses` | 3.8 → 497.5 | 4.2 → 487.2 | **fail** (must be 0) |
| `envlp/obs_tokens_measured_frac` | not logged | not logged | **cannot be checked** |
| `envlp/env_share_of_return` | not logged | not logged | — |
| `envlp/return_rank_corr` | not logged | not logged | — |

`provenance_misses` per episode, arm A, growing almost linearly:

| iterations | 1–49 | 50–99 | 100–149 | 150–199 | 200–249 | 250–299 |
|---|---|---|---|---|---|---|
| mean | 40 | 122 | 201 | 286 | 376 | 466 |

## Why the diagnostics are missing

`_env_likelihood_terms` in [`marlllm/loss.py`](../../marlllm/loss.py) returns
early with an empty metrics dict when `env_log_probs is None`. So the entire
diagnostic block — every quantity in the table above except the two trainer-side
counters — is skipped whenever provenance delivered nothing for a batch. The
missing metrics and the rising miss count are the same fault seen from two
sides: **the provenance path did not attach generator log-probs to observation
tokens.**

That `misaligned_observations` stayed at 0 is not reassurance. It counts
tokenisation mismatches among observations that *were* resolved; with resolution
failing there is nothing for it to disagree about.

## What this settles

- **The C9 experiment as submitted does not measure the dropped environment
  term.** Its void condition fails and its outputs are absent. — *established*
  — this run pair
- **Blanking the post tags is not by itself sufficient to make `p_env`
  computable.** The derivation's argument — no private channel, so no latent to
  marginalise — is about the *maths*; the provenance plumbing is a separate
  thing and it is what failed. — *established* — same
- **The failure grows with training**, which is consistent with B5's finding
  that producer-to-receiver alignment degrades with post-format breakdown
  (42–88%). A run whose agents' output degrades will lose provenance as it goes.
  — *suspected*, on the shape of the curve alone — same

## What it does not settle

- **Anything about the size or direction of the dropped term.** That is exactly
  what was being asked and it remains unmeasured. STATE.md's entries on `p_env`
  rest on B5's provenance analysis, not on these runs, and are unaffected.
- **Whether the corrected return trains differently.** Arm B ran to 300 with the
  corrected return in use, but with provenance failing, what it was correcting
  by is unknown. Its policy outcome should not be read as the corrected
  objective's behaviour.
- **What exactly broke.** Whether provenance never attached, attached and was
  dropped in collation, or was lost at the `RolloutBatch` boundary is not
  established here and needs a targeted check before resubmission.

## What to do next

Seeds 2–5 of both arms are written and **remain held** — the gate they were held
on has now failed rather than passed, which is the outcome the hold existed for.
Before resubmitting: instrument `provenance_misses` at the point of failure,
confirm `env_log_probs` is non-None in a short run, and only then release.
