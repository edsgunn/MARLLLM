# learning_progress

*Pay the action arm for **improvement** instead of accuracy, so that the
environment's entropy cancels out of the return.*

Separate from the surprise-minimisation suites by design: these runs do not
optimise the surrogate objective, so they are not members of the
`cultural_emergence` family and must not be pooled with it. They sit alongside
[`configs/novelty_search/`](../novelty_search/), which changes the return's
*sign*; this suite changes its *reference*.

## The idea, in one line

The surrogate return sums the agent's own surprise over future observation
tokens, whose expectation is

    E[-ln p_theta] = KL(p_env || p_theta) + H(p_env)

so the practical loss is *the stated objective plus the entropy of the
environment*. That second term is a REINFORCE ascent step on `-H(p_env)`: a
policy gradient whose reward is the negative entropy of the agent's own world.
In a population the world is other agents, and they are plastic — so the term
does not merely *route* the agent through quiet regions, it pays the agent to
*manufacture* them.

`H(p_env)` is common to **any** cross-entropy against `p_env`, so it cancels
identically in a difference of two of them:

    H(p_env, p_theta^-) - H(p_env, p_theta)
        = KL(p_env || p_theta^-) - KL(p_env || p_theta)

It does not matter how bad `p_theta^-` is as a model of the environment. We
never needed `p_env`; we needed a **second reference**. The per-observation
reward becomes

    r_s = ln p_theta(x_s | x_<s) - ln p_theta^-(x_s | x_<s)

and the dark room — where that difference is zero — becomes maximally
unrewarding.

## What is implemented

`learning_progress_return: true` replaces the return's per-token summand with
`M_s - M_s^-` before the backward scan that builds `G_t`. Everything
downstream — the baseline, standardisation, `return_sign` — follows from it.
**The perception arm is untouched**: it still minimises the agent's own
observation surprise, so `perc_loss` remains comparable with every reference
run.

Two references, set by `learning_progress_reference`:

| value | `p_theta^-` | cost | behaviour |
|---|---|---|---|
| `lagged` | this agent's own adapter from earlier in training | one extra frozen adapter + one extra no-grad forward per micro-batch | non-saturating; tracks a moving environment |
| `base` | the frozen pre-trained model the KL anchor already holds | free — the reference forward is already run for KL | saturates once the agent has moved from base; becomes "novelty relative to the prior" |

The lagged reference is **double-buffered**: the reference in force is always
the *second*-most-recent snapshot, never the current weights. At zero lag the
return is identically zero (the plug-in case of
`surprise_minimisation_derivation.md` 5.5.4), so the buffer is what makes the
scheme well posed rather than a refinement of it. The realised lag runs from
`L+1` to `2L` iterations and is logged as `lp/reference_age_iters`.

Both slots start at the initial adapter, so for the first `2L` iterations the
reference *is* the base model and the two variants coincide. They diverge only
after the second rotation. At iteration 1 alone the return is exactly zero,
because no step has been taken and there is genuinely no progress to measure.

## What this does NOT fix

**The length channel.** `KL(p_env || p_theta)` summed over positions is still
extensive in the number of observation tokens, so "post less, elicit less,
pay less" survives the correction untouched. The public channel closes in 20
of 20 runs of `c1_small_population_replicates` and 63 of 65 of B4, and there is
no reason in the derivation for this correction to touch that. **If delivery
still falls, the correction has not failed** — it was never the mechanism the
correction addresses. A fixed observation budget is the separate fix, and it is
not in this suite.

This is pre-registered here rather than in each config so it cannot be
retrofitted as an excuse.

## Read the arms against

- `runs/cultural_emergence/run7_8agent_7B_study_group_ashbourne_gc` — the
  reference at N=8 on the text substrate, seed 1.
- `runs/novelty_search/` — the sign-flip arms at the same cells. Three-way
  comparison at N=8: surrogate, sign reversed, reference changed.
- `runs/env_likelihood_bias/` — the *other* correction of the same dropped
  term, which puts a measured `E_s` back into the return instead of cancelling
  it. The two are mutually exclusive and the trainer refuses to run both.

## A design note the report's "decisive experiment" needs

The source report proposes the N=2 Strathearn cell — "the 100%-policy-collapse
cell" — as the cheap decisive test. **That cell is seed noise.**
`runs/cultural_emergence/c1_small_population_replicates/NOTES.md` re-ran it at
five seeds and found the within-cell spread of `alive` at 1.00, with the
original run's "dead from iteration 150 onward" becoming 0.50–1.00 alive at
150. A single N=2 Strathearn run cannot decide anything, in either direction.

So the suite is built at N=8 on Ashbourne GC instead, where the reference has
company and where the reliable signals live — policy entropy (the one thing
the sign flip moved reliably, at 7 of 7 arms) and the public-channel measures.

## Order of submission

1. `smoke/lp_smoke_2agent.yaml` — CPU, 0.5B, 3 iterations. Verifies the
   pathway before any GPU time is spent. Never cite its numbers.
2. **Tier 1** (4 jobs) — `lp_lagged_*` and `lp_base_*` at 8 agents, 2 seeds
   each. The primary test, and the `lagged`-vs-`base` contrast that separates
   "learning progress" from "any second reference".
3. **Tier 2** (2 jobs) — the lag sweep at seed 1. Only worth spending if
   Tier 1 moves something.
4. **Tier 3** (1 job) — return normalisation off, seed 1.
5. **Tier 4** (2 jobs) — the tool substrate, 2 seeds. The capability readout.

## Cost

Estimated from each reference run's own `metrics.jsonl` p90, per PROCESS.md,
with +9% for the extra no-grad forward the lagged reference costs (it touches
only the ~59% of an iteration that is not rollout).

| tier | jobs | requested | reference p90 |
|---|---:|---:|---|
| 1 — lagged | 2 | 10 h each | 66.3 s/iter (`run7_8agent_7B_study_group_ashbourne_gc`, 315 records) |
| 1 — base | 2 | 9 h each | same; no extra backbone forward |
| 2 — lag sweep | 2 | 10 h each | same |
| 3 — no normalisation | 1 | 10 h | same |
| 4 — tool substrate | 2 | 15 h each | 102.8 s/iter (`run7_8agent_7B_margin_notes`, 326 records) |

Submission, once the smoke passes:

```bash
.venv/bin/python submit_batch.py configs/learning_progress/ --account <project> \
    --hf-home /projects/<proj>/hf_cache
```

Use `.venv/bin/python`, never `uv run` — see PROCESS.md.

## Metrics this suite adds

| key | meaning |
|---|---|
| `<agent>/lp/in_use` | 1.0 when the corrected return is actually training. **A run without this key everywhere is void.** |
| `<agent>/lp/reference_is_base` | which reference is in force |
| `<agent>/lp/ref_surprise_mean` | the reference's own per-token observation surprise |
| `<agent>/lp/progress_per_token` | `ref - own` in nats. Positive = the agent is ahead of its reference. This is the signal the action arm is being paid on, and **its decay to zero is the prediction to watch.** |
| `<agent>/lp/progress_positive_frac` | fraction of observation tokens where the agent beats its reference |
| `<agent>/lp/return_mean` | the corrected return at ACT positions |
| `<agent>/lp/return_surrogate_mean` | the return it replaced, computed on the same batch |
| `<agent>/lp/return_rank_corr` | Spearman between the two. **Near 1.0 means the correction cannot change learning however large it is** — the same discriminator the environment-likelihood study used. |
| `lp/reference_age_iters` | realised lag, population mean |
