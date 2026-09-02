# Environment-likelihood bias: two arms with the private channel removed

Ten configs — arms A and B at five seeds — measuring how large the dropped
environment term is and whether restoring it changes anything. The maths is
[`environment_likelihood_derivation.md`](../../environment_likelihood_derivation.md);
§9a there is this experiment.

## Why the private channel goes

The practical loss drops `Sum_s sigma_s ln p_env(x_s | x_<s)`. In a population
that term is recoverable, because the observation tokens carrying any entropy
are other agents' posts, sampled from adapters we hold. But a generating agent's
private prefix is a latent the observer cannot see, and **contexts persist
across turns**, so the latent accumulates into the whole private history. That
turns the correction into a filtering problem whose weights degenerate over an
episode (§6a). It is not merely large — it stops being estimable.

Blank the post tags and the problem disappears rather than being modelled.
Everything an agent generates is its post, nothing is discarded, the latent is
empty, and

    p_env(y_t | u_t, j) = p_theta_j(y_t | u_t, c_j)   exactly.

Better than exact: **already computed.** The generating agent's sampled
log-probs are stored at rollout and the environment splices its exact token ids
into every observer's context, so each observation token carries its generator's
log-prob with no re-encoding and no alignment step.

Those are the fp8 vLLM sampling log-probs, and that is the point rather than a
compromise: p_env is whatever actually generated the token, and the fp8 engine
is what actually generated it. M_s meanwhile is a bf16 forward, so the two sit
on marginally different footings; the engine gap at identical tokens bounds how
finely an E_s difference can be read and is logged as
`envlp/sampling_vs_training_lp_*`. Nothing to do with staleness — that needs
`async_rollout`, pinned off below.

## Rollout pathway

Pinned to `async_rollout: false`, `ppo_clip: 0.0`, `pack_sequences: false`,
`seq_chunk_size` unset, matching
`runs/cultural_emergence/run7_8agent_7B_study_group_ashbourne_gc`, which predates
all four keys and so ran the synchronous, single-shot, plain-REINFORCE path.
These are also the current defaults, so pinning changes no behaviour — it stops a
later change of default from silently moving this experiment off the pathway it
is meant to be comparable with.

## The two arms

| | return | value head | env term |
|---|---|---|---|
| **A_surrogate** | `-G_t = Sum gamma^(s-t) M_s` — unchanged from every run in this project | fitted to `-G_t` | logged, not used |
| **B_corrected** | `R_t = Sum gamma^(s-t) (E_s - M_s)` | fitted to `R_t` | in the return |

Identical in every other field, including the seed. Both carry a **probe head**
fitted to the environment part of the return alone; its input is detached, like
the main value head's, so it trains only its own parameters and supplies the
history-conditioned mean needed to centre `E_s`.

## What is logged, every iteration

| metric | what it answers |
|---|---|
| `envlp/return_surrogate_mean`, `envlp/return_env_mean`, `envlp/return_corrected_mean` | how big is the dropped term against the return it was dropped from |
| `envlp/env_share_of_return` | its share of the corrected return |
| **`envlp/return_rank_corr`** | **do the two returns rank actions the same way** — the decision-relevant one. If they do, the bias does not matter for learning however large it is |
| `envlp/return_pearson` | the same in levels rather than ranks |
| `envlp/probe_loss`, `envlp/env_resid_mean`, `envlp/env_resid_std` | how predictable the env term is from history — the residual is what the gradient bias is a covariance with |
| `envlp/bias_corr_adv_resid`, `envlp/bias_rank_corr_adv_resid` | sign and strength of that covariance |
| `envlp/obs_tokens_measured_frac` | share of observation tokens with a measured `ln p_env` |
| `envlp/misaligned_observations`, `envlp/provenance_misses` | provenance health |

## Pre-registered

**Void conditions.** `obs_tokens_measured_frac` must be 1.0 and both
`misaligned_observations` and `provenance_misses` must be 0. Anything else means
the provenance path is broken and the run's numbers cannot be used.

**Ceiling.** Other agents' post content is ~32% of an observer's context at 8
agents (73% at 16), and `E_s = 0` on all remaining scaffolding, so
`env_share_of_return` is bounded by roughly a third of the observation stream
here. Read a small share against that ceiling, not against zero.

**The trap in arm B.** The corrected return has expectation
`KL(p_env || p_theta)`, and the perception loss is directly driving that to
zero. So B's action term is *expected* to self-extinguish, and B is expected to
drift toward perception-only behaviour (compare Phase A condition 3: same
perception loss, 4–10× less KL from base, entropy intact). **If B ends at low KL
with entropy intact, that is the flat-optimum prediction of §8 confirmed, not
the objective working.** The discriminating evidence is whether B's advantages
retain variance late in training.

## Running

Seed 1 of each arm first. Release seeds 2–5 only once the void conditions hold
and the pair completes without OOM — this pipeline is not reproducible at fixed
seed (`runs/cultural_emergence/kl_beta_sweep_8agent_ashbourne_gc/NOTES.md`), so
one pair settles nothing on its own and five is the minimum for a comparison.

```bash
.venv/bin/python submit_batch.py \
    configs/env_likelihood_bias/A_surrogate_seed1.yaml \
    configs/env_likelihood_bias/B_corrected_seed1.yaml \
    --account <project> --hf-home /projects/<proj>/hf_cache
```

`smoke/` holds a 2-iteration CPU pair on a 0.5B model that exercises the whole
path end to end. Run it after any change to the provenance code; never cite its
numbers.

## Departures from the reference config

Against `configs/cultural_emergence/run7_8agent_7B_study_group_ashbourne_gc.yaml`:

- `post_open_tag` / `post_close_tag` blanked — no-tag mode
- `provenance: true`
- `post_token_budget` 128 → 384, `total_token_budget` unchanged at 384: the
  private channel's share of the budget has to go somewhere. At 8 agents the
  resulting context (~4.5k tokens) sits below the 16-agent run that completes.
- `thinking_enabled: false`
- `vllm_gpu_mem_util` 0.30 → 0.18, the value that survives on this rig
  (`runs/social_terminal/run03_8agent_7B_ashbourne_school/NOTES.md`)
- the python tool is unavailable in no-tag mode and the env refuses it

**This is a different substrate from the published runs.** The tag-decay and
closing-public-channel failure mode measured in `runs/_tier_b/b4_public_channel/`
cannot occur here. That is deliberate — the experiment is about the bias, not
about reproducing that pathology — but nothing measured here transfers to runs
that have a private channel.
