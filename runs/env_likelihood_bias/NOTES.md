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

## Root cause, found 2026-09-03

**One line, and it is not a logging bug.**

`_extract_blocks` in no-tag mode delivered `generation.strip()` as the post,
while `_content_token_ids` required the **unstripped** decode of the action's
tokens to equal the delivered text exactly. Any generation with leading or
trailing whitespace therefore failed the match, and Qwen generations almost
always have some. Reproduced directly:

| generation | provenance, before the fix |
|---|---|
| `hi everyone, shall we start?` | resolved |
| `hi everyone, shall we start?\n` | **miss** |
| ` hi everyone, shall we start?` | **miss** |
| `\n\nhi everyone, shall we start? \n` | **miss** |

A miss sets `content_ids = None`, which is what is handed to every observer, so
the consequence is not a missing log line: **0 of 140 observation tokens carried
`ln p_env`** where a clean generation gave 7 of 140. The term was dropped as
completely as if `provenance` had never been set.

### It also invalidated arm B, silently

`_env_likelihood_terms` returns `(returns, None, {})` when `env_log_probs is
None` — *before* `use_env_likelihood_return` is ever consulted, at
[`marlllm/loss.py`](../../marlllm/loss.py) line 800. So arm B, which declared
`use_env_likelihood_return: True`, **trained on the uncorrected surrogate**.
The two arms are not two arms: they are the same experiment run twice, and
nothing in either job's output said so.

### Why it worsened over training

The strip only bites when the generation has surrounding whitespace, and
degrading policies emit more of it — the same drift that takes per-iteration
cost from 48 s to 157 s by emitting longer, messier generations. Hence 4 misses
per episode at the start and 497 by iteration 300.

### The fix

Three changes, with regression tests in
[`tests/test_env_likelihood.py`](../../tests/test_env_likelihood.py):

1. **No-tag mode no longer strips.** Stripping *is* discarding, and the whole
   premise of no-tag mode is that nothing an agent emits is discarded — that is
   what makes `p_env` the generator's own log-prob over the delivered tokens.
   Emptiness is still judged on the stripped form, so a whitespace-only turn
   still counts as `post_missing` and B4's channel metric is unaffected.
2. **The matcher is strip-aware anyway**, searching a bounded span for an exact
   decode match, so a stripped body elsewhere degrades to a miss rather than
   silently mismatching. Exactness is kept: the ids returned still decode to
   the delivered text and nothing else, because they *are* what gets spliced
   into the observer's context.
3. **The silent fallback is now loud.** A run that declares
   `use_env_likelihood_return` and finds no environment log-probs raises
   instead of quietly training on the surrogate, and a run with
   `env_likelihood_logging` on emits `envlp/env_term_absent` so the absence is
   visible in `metrics.jsonl` rather than being an absence of keys that nothing
   was watching for.

After the fix all five whitespace shapes resolve with zero misses and every
generated token carries `ln p_env`.

**A separate defect, not fixed:** provenance does not resolve in *tagged* mode
either, because there `text` is the content between the tags while the action
tokens include the tags, so no span of the action decodes to the delivered text
by this route. It has presumably never worked. The env-likelihood design uses
no-tag mode so this does not block C9, and B5's 100%-provenance result is a
different measurement (post text appearing as a contiguous span in observers'
contexts) and is unaffected. But `provenance: true` on a tagged substrate is
currently a no-op and should be either fixed or refused.

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
- ~~**What exactly broke.**~~ Found and fixed — see *Root cause* above. What
  remains unestablished is whether anything else in the path is also broken;
  the fix was verified on synthetic generations and in the unit tests, not yet
  on a real run.

## What to do next

Seeds 2–5 of both arms are written and **remain held** — the gate they were held
on has now failed rather than passed, which is the outcome the hold existed for.
**Submitted 2026-09-03.**

- **Arm B re-run as job 6279926**, with the fix in. It is now its own gate: a
  run that cannot compute the corrected return raises rather than falling back,
  so a job that completes has necessarily computed it. The void first attempt is
  kept at `B_corrected_seed1_void_provenance`.
- **The dropped term recovered post hoc from arm A's checkpoints**, job 6280214,
  via [`scripts/tier_b/c9_post_hoc_env_term.py`](../../scripts/tier_b/c9_post_hoc_env_term.py).
  Arm A's adapters and traces are intact, so the measurement the runtime path
  failed to make can be made from them without retraining: score each delivered
  post under its generator's adapter for `E_s`, and each observation block under
  its observer's adapter for `M_s`, then form both returns.

  Two things that reconstruction had to get right, and one it cannot. The
  trainer delivers observations **per turn, not per post** — `env.last()` is
  called only for the acting agent, so an agent receives everything accumulated
  since its previous turn as one block, which is the measurement's real unit.
  And the turn order is imposed from the recorded thread rather than re-drawn,
  which reproduces the realised speaker sequence exactly. What it cannot recover
  is the whitespace the generators actually emitted: the traces hold the
  stripped text, so `E_s` is the adapter's log-prob of the delivered string
  scored fresh, not the fp8 sampling log-prob.

Before releasing seeds 2–5, check the re-run arm B and
confirm two things in its `metrics.jsonl`: `envlp/provenance_misses` at 0, and
`envlp/env_share_of_return` and `envlp/return_rank_corr` actually present. Arm B
is the better smoke than arm A, because it now raises rather than falling back
if the term is missing — a job that completes has necessarily computed the
corrected return.

Seed 1 of both arms should then be **re-run**, not reused: the existing pair is
the same experiment twice.
