# C2 — population size, causally

Fourteen runs, three of the five arms in [REWRITE.md](../../../REWRITE.md) §6.3.
The fourth arm is already on disk; the fifth is blocked on code.

## Why the design exists

N varies **three things at once**: the number of distinct interaction partners,
the per-agent gradient volume, and the context composition. Every size
comparison in this project moves all three together, so "population size
determines the regime" was never separable from two other explanations. Tier D
has already withdrawn the causal claim. This is the design that could earn a
weaker one back.

| arm | N | partners | segments/agent/iter | status |
|---|---:|---:|---:|---|
| baseline small | 4 | 3 | 64 | **on disk** — C1's 4-agent cell, 5 seeds |
| baseline large | 16 | 15 | 16 | **here**, 4 runs (seed 1 is `run8`) |
| large, restricted pairing | 16 | 3 | 16 | **written**, 5 runs — gated on the smoke below |
| small, rate-matched | 4 | 3 | 16 | **here**, 5 runs |
| frozen majority | 16 | 15 | 16, 14 frozen | **here**, 5 runs |

The claim only becomes causal with all five. The restricted-pairing arm is the
one that carries it: **if outcome tracks partner count rather than N,
"population size" was never the variable.**

## What each arm here does

**Baseline large** re-runs N=16 seeds 2–5 at 300 iterations. They ran 150, while
the base ran 300 — so at four of five seeds this arm cannot be read at iteration
200, which is the evaluation point every population-level claim in this project
rests on. Against C1's five N=4 runs at 300 the contrast simply could not be
drawn. Also puts the arm on current code, consistent with C1 and the rest of C2.

**Small, rate-matched** cuts `rollouts` from 16 to 4 at N=4, and nothing else.
With `max_posts` 16 and round-robin, each agent posts `max_posts/N` times per
episode — 4 times at N=4, once at N=16 — so over 16 episodes an N=4 agent gets
**four times** the gradient volume of an N=16 agent. That difference rides along
with every size comparison in the project. Four episodes × 4 posts = 16
segments, matching N=16.

*Why `rollouts` rather than `max_posts`.* Both match volume. `max_posts` 16→4
does it by shortening the thread, which moves context composition too;
`rollouts` holds thread length and composition fixed and moves only the quantity
the arm names. Neither is perfectly surgical — fewer episodes also means noisier
gradients, and that is a real limitation of this arm rather than a footnote.

**Frozen majority** freezes 14 of 16. They post, so the two learners see exactly
the sixteen-member context; they just do not learn. This isolates co-adaptation
at fixed N *and* fixed partner count — the only arm where the population is a
fixed backdrop at full size. Two learners rather than one, so some
agent-to-agent adaptation survives and the arm is not merely
[C3](../c3_frozen_population/README.md) at a larger N.

## The restricted-pairing arm, and why §6.3 mis-scoped it

§6.3 proposed "one new pairing strategy — a fixed partner graph with degree
*k*… a small, well-scoped addition". `pairing_strategy` is a **no-op on every
forum substrate in this project**. Forum envs bind in *named* mode: each role
goes to its same-named member and the whole roster is in every episode, so
`_sample_pairs` is never called. It is a fourth dead config key, alongside
`thinking_enabled`, `frozen_agents` and the `alpha_*` weights.

What the arm actually needed was a **per-episode roster subset**:
`active_agents` in `ForumEnv.reset`, plus `_setup_partner_graph`,
`_episode_roster` and `_bind_episode_roles` in the trainer. With
`partner_graph_degree: 3` the sixteen are partitioned once into four disjoint
groups of four and each episode runs one group — so every agent has exactly
three partners, always the same three, at N=16.

Groups are **cycled, not sampled**, which is what keeps per-agent volume at 16
and the arm comparable: four episodes per group per iteration, four posts each.

Tests: [`tests/test_partner_graph.py`](../../../tests/test_partner_graph.py) —
16 of them, covering the roster arithmetic, the trainer's binding, deep-copy
and reset, the trace, the trajectory filter, and that an unconfigured run is
byte-identical to before.

### The smoke gate

`smoke/c2_restricted_pairing_smoke.yaml` — three iterations at the real N,
model and hardware. The tests cannot reach generation or the loss, and a CPU
smoke is not viable (float32 0.5B generation does not finish an episode in
usable time on the login node). **Release the five seeds only when it passes:**
three iterations complete, every episode trace lists exactly four agents, all
four groups appear, and group membership never changes.

## A wording correction to §6.3

The design table calls the second axis "updates/agent/iter". The trainer takes
**one** optimiser step per agent per iteration either way; what differs is
**action tokens per update**. The confound is real, the label names it
imprecisely, and the rate-matched arm matches the thing that actually varies.

## What is held constant

The rosters nest — `canonical_4` ⊂ `extended_8` ⊂ `extended_16` — so N=4 and
N=16 share characters and the size contrast is not confounded with who is in the
room. Same seed ladder as every other family (1, 42, 625, 8192, 123), same
substrate, same hyperparameters, same 300-iteration horizon, same pinned
synchronous rollout pathway.

## Submitting

```bash
.venv/bin/python submit_batch.py \
    configs/cultural_emergence/c2_population_size/ \
    --account brics.a5l \
    --hf-home /lus/lfs1aip2/projects/a5l/egunn/hf_cache \
    --scripts-dir slurm_scripts/
```

N=16 runs took 7.6 h for 300 iterations; the rate-matched N=4 arm has a quarter
the episodes and should be much faster.
