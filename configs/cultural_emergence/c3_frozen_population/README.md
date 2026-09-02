# C3 — one learner in a fixed population

Three runs on the reference substrate: `study_group_ashbourne_gc`, eight
members, **seven of them frozen**. The frozen seven post exactly as they
otherwise would, and the learner reads their posts as observations — they
simply receive no gradient. [REWRITE.md](../../../REWRITE.md) Tier C item **C3**.

## The question

Every failure this project documents was measured in a **co-adapting**
population. That leaves two readings of the decline unseparated:

1. *The agents drove each other into it* — the population co-adapts, the
   observation stream each agent faces gets easier to predict because everyone
   is converging, and the whole thing spirals.
2. *Surprise minimisation does this to any agent against any population* — the
   objective's structure is sufficient, and the multi-agent setting is scenery.

This separates them, because here **the environment cannot move.** The seven
frozen members present a stationary distribution for all 300 iterations, so any
drift the learner shows is its own.

Reading 2 is 5MAX's remaining empirical objection, and there is currently no
run that speaks to it.

## What each outcome would mean

| the lone learner | conclusion |
|---|---|
| declines on the reference schedule | co-adaptation is not doing the work. The decline is single-agent surprise minimisation, the multi-agent framing has no causal role in the failure, and §3.5's mechanism is the whole story. |
| stays healthy where reference agents did not | the failure needs a moving population. That is a genuine result about multi-agent dynamics, and it makes the frozen population a candidate **stabilisation** rather than only a control. |
| declines *faster* | worth taking seriously rather than dismissing: a stationary environment is the easiest possible thing to predict, so the dark-room incentive has the least resistance. |

## Frozen at what

At a **fresh LoRA adapter, which is the identity** — so the seven present the
base model under their character prompts. That is the right null: an
un-adapted crowd.

The same mechanism freezes members at a *trained* culture instead, by adding
`resume_from` pointing at a completed run's checkpoint. That is a different and
also interesting experiment — *can an agent absorb an established culture?* —
and it is the migration experiments' missing control. Not run here, because it
needs a decision about which iteration to freeze at, and the reference
population is already declining by 300.

One caveat if you do it: the optimiser state in a full-population checkpoint
has 912 entries against the 114 a single learner needs, so
`load_population_checkpoint` will warn and start the optimiser fresh. Weights
load correctly; momentum does not carry.

## The three arms

| config | learner | seed |
|---|---|---:|
| `c3_frozen_pop_learner_priya_seed1` | Priya Shah | 1 |
| `c3_frozen_pop_learner_priya_seed2` | Priya Shah | 42 |
| `c3_frozen_pop_learner_marcus_seed1` | Marcus Webb | 1 |

Two axes of variation on purpose. The second seed is because the full loss is
known to be seed-unstable *at the level that decides whether a policy
survives*, so a single run cannot carry a claim about survival. The second
learner is because agents in this project are not interchangeable — the
absorption and commitment results are per-agent — and a result that only holds
for Priya is a result about Priya.

## How to read it

The comparison is
[`run7_8agent_7B_study_group_ashbourne_gc`](../../../runs/cultural_emergence/run7_8agent_7B_study_group_ashbourne_gc/NOTES.md),
which is field-for-field identical apart from the freeze.

**Against its per-agent traces, not its population means.** There is one
learner here; averaging over eight in the reference and over one here would
compare different things. Population means in `metrics.jsonl` average over
*trainable* members only, which is why `pop/n_trainable` and `pop/n_frozen` are
logged every iteration — the denominator is visible rather than inferred.

The reference decline to beat: its public channel falls from 99% to 28% of
turns delivering a well-formed post, and its population declines from iteration
~200.

## Implementation

`frozen_agents` was wired into `PopulationTrainer` for this
([`marlllm/population.py`](../../../marlllm/population.py),
`_setup_frozen_agents`). Frozen members are kept out of the optimiser and their
forward+backward is skipped entirely — which is also why one learner in eight
costs roughly an eighth of the backward.

The freeze is enforced, not assumed: a gradient arriving on a frozen parameter
**raises** at the optimiser step rather than warning, because a leaked freeze
would silently turn this control back into the experiment it controls for.
Parameters shared between a frozen and a trainable member are refused at
construction for the same reason. Tests:
[`tests/test_frozen_population.py`](../../../tests/test_frozen_population.py).

## Known waste, deliberately left alone

`VLLMSamplingEngine.sync_all_adapters` flushes **every** adapter to disk with a
fresh id after each optimiser step, so these runs re-write seven byte-identical
adapters 300 times over. Skipping frozen adapters would be correct — their
weights do not change, so vLLM's cached copy stays valid — but it touches the
rollout pathway these configs deliberately pin, and getting it subtly wrong
would serve stale weights to the *learner*. Not worth that risk before the
first run has produced anything. Revisit if the runs come in slow.

## Submitting

```bash
.venv/bin/python submit_batch.py \
    configs/cultural_emergence/c3_frozen_population/ \
    --account brics.a5l \
    --hf-home /lus/lfs1aip2/projects/a5l/egunn/hf_cache \
    --scripts-dir slurm_scripts/
```

The reference run took 5.25 h for 300 iterations with all eight training; these
should come in well under that.
