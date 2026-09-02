# B4 — does the public channel close?

*The share of turns that deliver a well-formed public post, over training.*

|  |  |
|---|---|
| Script | [`scripts/tier_b/b4_public_channel.py`](../../../scripts/tier_b/b4_public_channel.py) |
| Inputs | `tool/post_*` metrics from `metrics.jsonl`, 65 runs with both an early and a late point |
| Ran | 2026-09-01, no compute |

## What it was for

[B6](../b6_token_accounting/NOTES.md) showed the token-mix instrument cannot
support a private-channel reading. This uses the instrument the environment has
been writing all along: it classifies every turn `post_success` /
`post_unclosed` / `post_missing` / `post_empty`
([`envs/forum/env.py`](../../../envs/forum/env.py) lines 517–560), and only text
inside a `<post>` block reaches other agents — "everything between them is
private thinking and is discarded for the public stream" (env.py:519).

## Headline

**The public channel closes in 63 of 65 runs, and it closes long before the
regime classifier notices anything.**

On the campaign's most-analysed population, the fraction of turns delivering a
well-formed post falls from 99% to 28%, while the fraction delivering *nothing
at all* rises from 0.4% to 28%:

| iteration | post_success | post_unclosed | post_missing | alive |
|---:|---:|---:|---:|---:|
| 1 | 0.988 | 0.008 | 0.004 | — |
| 50 | 0.981 | 0.012 | 0.008 | — |
| 100 | 0.856 | 0.144 | 0.000 | 0.88 |
| 150 | 0.621 | 0.328 | 0.051 | 1.00 |
| 200 | 0.477 | 0.426 | 0.098 | 1.00 |
| 300 | 0.281 | 0.434 | 0.285 | 0.88 |

*(`run8_16agent_7B_study_group_ashbourne_gc`)*

**The channel is already half shut at iteration 200, where the paper evaluates,
and the alive fraction still reads 1.00.** The two-turn run is starker: post
success collapses 0.97 → 0.34 between iterations 50 and 100 and stays there,
while the alive fraction holds at 1.00 until 250 and then falls to 0.38 — a
200-iteration lag.

Across runs where both are observable, channel degradation reaches −25% at or
before the first drop in alive fraction in **9 of 12**.

## Why this is the predicted failure

The implemented return is `G_t = Σ_{s>t, σ_s=OBS} γ^(s−t) · surprise_s`
([`marlllm/loss.py`](../../../marlllm/loss.py) line 12) — a sum over *future
observation tokens*. Its magnitude depends on how many observation tokens follow
as much as on how surprising each one is. The rebuttal stated the consequence
before this was measured: "an agent that shortens its co-agents' replies moves it
as surely as one that makes them more predictable." A turn that emits no
well-formed post contributes nothing to any other agent's context.

So the dark room in this substrate is neither gibberish nor determinism. **It is
the closing of the public channel**, and it is the behavioural signature of the
term the surrogate drops.

## What this settles

- **The fraction of turns delivering a well-formed public post falls over
  training.** 63/65 runs; unchanged in 1, rose in 1. — *established*
- **It falls before the regime classifier registers anything.** 9/12 runs where
  both are observable; up to a 200-iteration lead. — *likely*
- **The regime taxonomy is a lagging indicator of this failure.** — *likely*

## What it does not settle

**Direction.** This is association. The channel could be closing *because* the
population is degrading in some way the regime metrics see later. The ablation
that would settle it is C4 in `REWRITE.md` — train with only public post tokens
typed as actions — and it has not been run.

Nor does it settle whether the environment's post accounting is well calibrated
on a substrate with genuine `<think>` blocks; on Qwen3 the thinking counter reads
zero while `<think>` markers appear in traces, and that is unresolved.

## Status

`CLOSED` as a measurement, `OPEN` as a mechanism.

---
*Analysed 2026-09-01. Machine output stays local.*
