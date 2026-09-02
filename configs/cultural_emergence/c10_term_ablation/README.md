# C10 — which term does the damage?

Four runs on the reference forum population: perception-only and action-only,
two seeds each. The full loss cell is already on disk as
[`run7_8agent_7B_study_group_ashbourne_gc`](../../../runs/cultural_emergence/run7_8agent_7B_study_group_ashbourne_gc/NOTES.md),
so this is a three-cell dissociation for the cost of four runs.

| arm | `alpha_perc` | `alpha_act` | `alpha_val` |
|---|---:|---:|---:|
| **perception-only** | 1 | 0 | 0 |
| reference (`run7`) | 1 | 1 | 1 |
| **action-only** | 0 | 1 | 1 |

Nothing else moves. Same substrate, roster, hyperparameters, horizon and
rollout pathway.

## Why re-run an ablation the project has already done

The existing perception-only result was measured on the **Phase A negotiation
substrate** in May, on an early and since-rebuilt environment, with two agents
and no population. It says the perception term does the world-modelling and the
action term collapses entropy. Nothing has re-tested that anywhere near where
the paper's claims live.

But staleness is the weaker reason. The sharper one:

**The dark-room incentive reaches behaviour through the action term alone.** The
return enters the loss as the policy gradient's weight; the value head cannot
move behaviour. So `alpha_act = 0` removes the *mechanism*, not just a term.

And everything the decline consists of — the public channel closing in 63 of 65
runs, entropy collapsing, populations dying from iteration ~200 — is currently
pinned on that mechanism by an **association**: the channel closes first, by up
to 200 iterations, in 9 of 12 runs where both are observable. This is the arm
that converts the association into a causal claim or kills it.

## The prediction that would change the paper's shape

Absorption is carried by the **perception** term. `prompt_as_observation` types
character-prompt tokens `OBS`, so they enter `L_perc` and never the policy
gradient — that is the documented mechanism by which an agent learns to predict
its own character prompt.

So if perception-only **keeps absorption and drops the collapse**, the
framework's learning signal and the framework's pathology sit in different
terms. That is a much better paper than *the objective works but kills its
populations*, and it is a concrete stabilisation candidate rather than a
diagnosis.

Action-only is the complement and predicts the mirror image: the incentive runs
undiluted so the channel closes at least as fast as the reference, and
absorption disappears entirely, because with `alpha_perc = 0` the OBS-typed
prompt tokens receive no loss of any kind.

## Two things that do not change, and would be easy to assume otherwise

- **The KL anchor survives at full weight.** `kl_coef * kl` is a separate,
  un-gated term in [marlllm/loss.py](../../../marlllm/loss.py) — the alphas do
  not reach it — so both arms keep `kl_coef: 0.05` exactly as the reference had
  it.
- **The entropy bonus lives inside the action term**
  (`alpha_act * (l_policy - beta * l_ent)`), so `beta` is inert whenever
  `alpha_act` is 0. That is the correct reading of the arm, not an oversight:
  with no policy gradient there is no entropy to bonus.

## The honest failure mode, stated before the data

**Perception-only could be uninformative rather than negative.** If the policy
barely moves — entropy flat, KL from base near zero, PMI gap unchanged — then
"it did not collapse" is trivially true and says nothing at all.

So the arm carries its own positive control, and it is the same measurement
either way: does perception-only still **absorb** (does the own-prompt PMI gap
shrink), and does it still **diverge** across populations? If it shows neither,
this is just *don't train the policy* and there is no result in it. Phase A saw
4–10× less KL from base, not zero, so movement is expected — but that check is
the precondition for reading anything else here.

**Action-only has its own void condition.** With no next-token loss, nothing
anchors the model to fluent text but the KL term, and Tier D already records
that the KL anchor is close to orthogonal to collapse. A run producing only
token soup by iteration 50 answers nothing; if that happens the arm needs a
higher `kl_coef` and a note that it is no longer matched to the reference.

## What this is not

Not the ablation `_tier_b/b4_public_channel/NOTES.md` proposes. That one types
**only public post tokens** as actions and retrains, which is a token-typing
change. This is a term-weight change. Both bear on the same question — does the
closing channel cause the decline or accompany it — from different directions,
and the typing ablation is still unwritten.

## Implementation

`alpha_perc` / `alpha_act` / `alpha_val` were honoured by the loss but
**unreachable from a population config** — `train_population.py` never passed
them, so every forum run in this project has been at 1/1/1 with no way to say
otherwise. Now wired, with tests in
[`tests/test_loss_alphas.py`](../../../tests/test_loss_alphas.py) pinning what
each zero actually removes.

One result worth knowing when reading the metrics: under perception-only the
policy head **still receives gradient**, because next-token prediction over
observation tokens shares the `lm_head`. What is removed is the policy
gradient, not all movement. The test asserts this explicitly so it is not
mistaken for a leak.

## Submitting

```bash
.venv/bin/python submit_batch.py \
    configs/cultural_emergence/c10_term_ablation/ \
    --account brics.a5l \
    --hf-home /lus/lfs1aip2/projects/a5l/egunn/hf_cache \
    --scripts-dir slurm_scripts/
```

The reference run took 5.25 h for 300 iterations; each arm should be comparable
or faster.
