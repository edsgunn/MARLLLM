# novelty_search

*A separate suite: maximise future observation surprise instead of minimising
it. The sign of the return is the only change from each reference run. Started
as Tier-C item C13; now its own line of experiments.*

|  |  |
|---|---|
| Configs | [`configs/novelty_search/`](../../configs/novelty_search/) |
| Model | `Qwen2.5-7B-Instruct`, LoRA r64 |
| Arms | `study_group_ashbourne_gc` at N = 2, 4, 8, 16 (2 seeds each) and `margin_notes` at N = 8 (2 seeds) |
| Analysed | all 12 arms, 2026-09-08 |
| Scripts | [`analyse_novelty_size.py`](../../scripts/analyse_novelty_size.py), [`analyse_novelty_traces.py`](../../scripts/analyse_novelty_traces.py) |
| **Differs from** | each arm from **its own reference run at the same N** — `return_sign` −1.0 and nothing else. Verified field by field. |

## What it was for

Every documented failure points one way: the return sums surprise over *future
observation tokens*, so the cheapest way to shrink it is to give the others
nothing to respond to. C10 removed that mechanism; this points it the other
way. If the diagnosis is right, reversing the sign should reverse the behaviour.

## Headline

**Reversing the sign produces the behaviour the framework was built to produce —
on one of two seeds.** At iteration 300, seed 2 is doing collaborative
mathematics: **89% of its posts carry mathematical content**, mean length **660
characters**, and **zero** malformed fragments, where the reference at the same
iteration manages 1 of 42 and five fragments. Seed 1 does not: 4 of 53
math-bearing and posts *shorter* than the reference's.

**But delivery still declines in both.** 1.00 → 0.67 and 1.00 → 0.39 across 300
iterations. The sign matters and it is not the whole story.

### What the 2026-09-08 arms add

**The one thing the sign does reliably, at every size, is raise policy
entropy.** Seven of seven arms across N = 2, 4, 8 and 16 end above their own
same-N reference, by 0.16–0.78 nats against a same-seed noise floor of 0.007 —
while the references *lose* entropy at three of four sizes. Nothing else in this
project moves entropy that reliably: in the 20-cell regularisation grid no value
of `beta` produced a small increase at all, and the KL anchor only ever
suppressed one.

**Population size does nothing here either.** Delivery against N gives
ρ = −0.073, p = 0.869 — the pre-registered falsifier for "size matters under
maximisation". The outcome is instead **seed-dominated**: two N=2 runs differing
only in seed end at delivery 0.926 and 0.325.

**And the highest delivery in the sweep is an artefact.** That 0.926 is a
two-agent population locked into a 52-character repetition loop. It is the
fourth metric artefact in this campaign, and the first to hit *delivery* — the
measure that had survived the other three. Delivery is necessary, not
sufficient.

**On the tool substrate the sign slows tool abandonment without reopening the
channel** — but the effect size does not replicate. Both seeds end above the
reference on python calls at iteration 300 (52.2 and 28.1 against 22.2) and both
are well above it at 200, at an unchanged success rate and with real varied
code; but the two seeds differ from each other by 1.9×, and seed 2's delivery
ends *below* the reference. Direction consistent, magnitude not.

## Delivery, and what the posts contain

Well-formed post rate:

| iter | novelty s1 | novelty s2 | REF `run7` |
|---:|---:|---:|---:|
| 25 | 1.00 | 1.00 | 0.98 |
| 100 | **0.98** | **0.91** | 0.77 |
| 200 | **0.86** | **0.69** | 0.61 |
| 300 | 0.67 | 0.39 | 0.56 |

Delivery holds above the reference through iteration 200 in both seeds, then
crosses it. So the sign delays the decline rather than preventing it.

The content, at iteration 300, all four episodes — because delivery cannot
separate "communicating more" from "emitting noise more", which the config
header said in advance:

| run | non-empty | mean chars | math-bearing | fragments |
|---|---|---:|---|---|
| novelty seed1 | 53/64 | 114 | **4/53** | 3/53 |
| novelty seed2 | 46/64 | **660** | **41/46** | **0/46** |
| REF `run7` | 42/64 | 164 | 1/42 | 5/42 |
| C10 perception-only | **64/64** | 205 | 55/64 | 0/64 |

## The transcript, seed 2 at iteration 300

> **Beatrice Okafor:** hey guys, stumbled on this while looking through uni
> prep. seems fun: if \( n \) is odd, show that…
> **Olu Adeyemi:** oh wait, yeah this one's neat! since \( n \) is odd, let
> \( n = 2k + 1 \). so \( n^2 + 2n + 3 = (2k+…
> **Hana Yilmaz:** hey beet, olu, sorry olu, looks like mod 8 might need a bit
> more TLC. lemme see if i missed somethin'
> **Imogen Carter:** gotchu beet. specifics for \( n = 2k + 1 \): \( n^2 =
> (2k+1)^2 = 4k^2 + 4k + 1 \)…

Against the reference at the same iteration:

> **Priya Shah:** `?</?  ?</?` · **Tom Whitaker:** -reviewed the messages
> quickly · **Marcus Webb:** sorry…

## What this settles

- **The return's sign is causal for the *content* of what agents emit, not just
  how much.** One seed of a one-key change produces the longest, most
  mathematical, least malformed transcript in the project at iteration 300. —
  *provisional*, one seed of two — this family against `run7`
- **The dark-room diagnosis is right about the mechanism and incomplete as an
  explanation.** Delivery still falls from 1.00 to 0.67 and 0.39, so silence is
  not only about the direction the return points. — *likely*, two seeds — same
- **Novelty search is not the best configuration found.** C10's
  perception-only arm delivers 64 of 64 with 86% math-bearing posts on both its
  seeds; this delivers 0.67/0.39 with one seed excellent and one mediocre. The
  interesting thing here is the *size of the change*, not the size of the
  effect. — *established* —
  `runs/cultural_emergence/c10_term_ablation/NOTES.md`
- **Entropy rises rather than collapsing**, 0.75 → 0.96 and 0.98 against the
  reference's 0.80, which is the predicted direction for an objective that
  rewards unpredictable futures. — *likely* — this family

### Added 2026-09-08, from the size sweep and the tool substrate

- **The return's sign raises policy entropy at every population size.** 8 of 8
  arms sit above their own same-N reference, by 0.16–0.80 against a same-seed
  noise floor of 0.007; references lose entropy at 3 of 4 sizes and only 1 of 8
  novelty arms loses more than 0.12. MWU exact p = 0.028. — *established* —
  this family vs run10/run7/run8, `scripts/analyse_novelty_size.py`
- **Population size does not determine the outcome under maximisation either.**
  Delivery against N gives ρ = −0.073, p = 0.869, and the within-N seed spread
  (0.601 at N=2) accounts for the entire between-N range. The pre-registered
  falsifier for "size matters under maximisation" fired. — *established*,
  7 arms over 4 sizes — same
- **Under maximisation the outcome is seed-dominated.** Two N=2 runs differing
  only in seed end at delivery 0.926 and 0.325, 3.5× the rig's delivery noise
  floor. — *established* — same
- **Delivery is necessary but not sufficient, and this is the fourth metric
  artefact.** A two-agent population that locks into a short mutually
  predictable exchange scores delivery 0.926 on 52-character posts with 38%
  near-duplicates. Degeneration into `</?` loops cannot raise delivery, because
  those fail to close their tags; degeneration into a *short well-formed loop*
  can. Always report a repetition measure beside a delivery figure. —
  *established*, one direct demonstration plus its own reference behaving worse
  — same, and `scripts/analyse_novelty_traces.py`
- **The sign flip slows tool abandonment without reopening the channel.** Both
  seeds end above the reference on python calls at iteration 300 (52.2 and 28.1
  against 22.2) and both are well above it at 200, at an unchanged success rate
  and with real, varied code. But the seeds differ from each other by 1.9× and
  seed 2's delivery ends *below* the reference, so the direction replicates and
  the magnitude does not. — *provisional*, two seeds — this family vs
  `run7_8agent_7B_margin_notes`
- **Trace files hold one episode of sixteen rollouts.** Every trace-derived
  figure in this project is n = 1 episode unless stated otherwise; this is what
  produced the tool substrate's overstated "calls reach zero" premise. —
  *established* — same
- **The within-speaker repetition measures are structurally biased by N**, since
  posts per speaker is `max_posts`/N — undefined at N=16, one pair at N=8. Use
  `thread_sim` across sizes. — *established* — same

## The tool substrate, submitted 2026-09-04

Two seeds of the same manipulation on `margin_notes`, which has
`python_tool_enabled` — jobs 6397841 and 6397842. It is the sharper test,
because the reference **abandons the tool**:

| iter | python calls | successes | success rate | post rate |
|---:|---:|---:|---:|---:|
| 25 | 48 | 32 | 0.67 | 0.98 |
| 100 | 34 | 16 | 0.47 | 0.98 |
| 200 | 14 | 10 | 0.71 | 0.64 |
| 300 | **0** | 0 | — | 0.34 |

By iteration 300 the population has stopped calling python entirely. And there
is a mechanism for it that is the dark room again: a python call produces a
`pybot` post, whose content is the output of code that has not been run, so it
is among the least predictable things in the substrate. Under minimisation the
tool is expensive; under maximisation it should be the best move available.

That turns the hypothesis into a prediction about a **capability** rather than
about token counts, which is what the framework ultimately has to deliver.
Predicted: calls stay non-zero at iteration 300. Falsified if they reach zero
here too, in which case abandonment is not about the return's direction.

The degenerate outcome to check for is call-spamming —
`print(random.random())` and the like — which would show as a high call count
with a low success rate or near-identical code blocks. **The python blocks have
to be read before this is reported**, for the same reason seed 1 above needed
reading: the counter cannot separate using the tool from farming noise.

### Tool substrate results, 2026-09-08 — seed 1 only

Seed 1 (6398717) finished at 300 iterations with `return_sign` −1.0 on all
eight agents. **Seed 2 (6398719) was still running** at 215 iterations, so
everything here is **one seed**.

#### The prediction was right: the tool stays alive

Population-level python calls per iteration, ±12-iteration windows:

| iteration | 25 | 50 | 100 | 150 | 200 | 250 | 300 |
|---|---:|---:|---:|---:|---:|---:|---:|
| reference (+1) | 142.2 | 123.4 | 110.9 | 102.6 | 73.2 | 37.4 | **22.2** |
| novelty (−1) | 130.3 | 109.7 | 89.2 | 86.7 | 76.4 | 67.2 | **52.2** |

**2.35× the tool use at iteration 300**, and a much shallower decline — −60%
against the reference's −84%. The header predicted "calls stay non-zero at
iteration 300 where the reference reaches 0", and the direction holds.

**The success rate is unchanged**, 0.53 against 0.53 at iteration 300 and
flat at 0.57–0.64 throughout in both arms. So the extra calls are not
lower-quality ones: the arm is not buying call count with failures.

Delivery also holds up better — **0.555 against 0.383** — though both are far
down from ~0.96 early.

An incidental one: **unclosed python blocks**. The reference spikes to
13.2 per iteration at iteration 150 and 10.3 at 200; the novelty arm stays at
0.5–2.0 throughout. Whatever breaks the reference's tool protocol mid-run does
not happen here.

#### The pre-registered degenerate check passes

The header required this before reporting: *"Maximising surprise has a cheap
solution here as well: spam calls whose output is arbitrary,
`print(random.random())` and the like. That would show as a HIGH call count with
a low success rate or near-identical code blocks."*

Reading the `pybot` records (1:1 with actual calls) in the traced episode at
iteration 300: **5 calls, 4 unique, 100% success**, and the code is a genuine
progression over one topic rather than noise —

```
a = [0,1,2,3,4,5];  print(a[:3])        -> [0, 1, 2]
b = ['a','b','c','d'];  print(b[1:3])   -> ['b', 'c']
c = range(5);  print(list(c[2:4]))      -> [2, 3]
d = [1,2,3,4,5];  print(d[1:4])         -> [2, 3, 4]
e = ['x','y','z'];  print(e[1:])        -> ['y', 'z']   (slice end omitted)
```

Five different slicing forms, each with real output, and the posts around them
cross-reference each other by name ("Henri, Cass mentioned slicing lists.
Renata and Kai noted similarity"). One of the five is a verbatim repeat of an
earlier call in the same episode. **This is not noise-farming.**

#### But the channel still closes, and the protocol still breaks

In the same episode, **7 of 16 turns produced no post at all** — all thinking,
no public output, discarding 194–2103 characters of it each time. Three more
posts are under 20 characters (`Ren?!`, `I noticed?`, `?`). And one post leaks
the chat template outright, emitting a literal `assistant` role token followed
by a duplicated `<post>` block.

So the sign flip **keeps the capability without reopening the channel.** That
is a narrower result than "novelty search fixes tool abandonment", and it is
the one the data supports.

#### Correction: seed 2 does not replicate the effect size

Seed 2 (6398719) finished at 300 iterations, and **the tool result is much
weaker than one seed suggested.** Python calls per iteration:

| iteration | 25 | 100 | 200 | 300 |
|---|---:|---:|---:|---:|
| reference (+1) | 142.2 | 110.9 | 73.2 | **22.2** |
| novelty (−1) seed 1 | 130.3 | 89.2 | 76.4 | **52.2** |
| novelty (−1) seed 2 | 143.0 | 131.2 | 118.9 | **28.1** |

Seed 2 holds tool use far above the reference through iteration 200 — 118.9
against 73.2, the largest gap anywhere in the pair — and then falls off a cliff
between 200 and 300, ending at 28.1. Seed 1 declines gradually and ends at 52.2.

So at iteration 300 both seeds end above the reference, but by **2.35× and
1.27×** — the two seeds differ from each other by more than seed 2 differs from
the reference. And **delivery does not survive at all**: seed 2 ends at 0.317
against the reference's 0.383, i.e. *worse*.

**What survives.** Both seeds end above the reference on call count, and both
are well above it at iteration 200. The direction is consistent across two
seeds; the magnitude is not, and the delivery half of the claim is gone.
Downgraded from the reading below: the sign flip **slows** tool abandonment
rather than preventing it, and does not keep the channel open. — *provisional*,
two seeds, direction consistent and effect size varying by 1.9×.

The degenerate check still passes on seed 1's traces, and nothing about seed 2
changes that reading — it is a claim about what the calls contain, not how many
there are.

#### A correction to this section's own premise

The header stated that the reference "ABANDONS THE TOOL … By iteration 300 the
population has stopped calling python entirely", from calls going
48 → 34 → 14 → **0**. That zero is a **one-episode artefact**: the traced
episode at iteration 300 does have 0 calls, but the population averages
**22.2 calls per iteration** at that point. The reference reduces tool use by
84%; it does not abandon it. The result above survives the correction — 52.2
against 22.2 is still 2.35× — but the premise was overstated, by exactly the
n = 1 trace-sampling limit found in the size sweep.

## The population-size sweep, submitted 2026-09-08

Four sizes on `study_group_ashbourne_gc`, two seeds each, each built from the
reference run the project already has at that N so that the sign stays the only
change within every arm.

| N | reference | jobs | limit |
|---:|---|---|---:|
| 2 | `run10_2agent_…` | 6398722, 6398723 | 7 h |
| 4 | `run10_4agent_…` | 6398724, 6398725 | 9 h |
| 8 | `run7_8agent_…` | **done** — `c13_novelty_search_8agent_…_seed{1,2}` above | — |
| 16 | `run8_16agent_…` | 6398726, 6398727 | 15 h |

The N=8 arm is the pair already analysed here; it keeps its original name
rather than being renamed into the sweep's scheme, because renaming analysed
data costs more than the consistency is worth.

**The question.** Under the standard objective population size does nothing:
C2 varied N against partner count and gradient volume across five arms at five
seeds and found N=4 and N=16 indistinguishable at matched volume, p = 0.722,
with volume itself at p = 0.810, and C1 had already removed the small corner.
Whether that survives the flip is open — and there is a reason it might not,
since maximising surprise rewards eliciting unpredictable responses and how
much of that is available scales with the number of distinct partners.

**The confound, stated rather than controlled.** This is the historical grid:
`max_posts` 16 and `rollouts` 16 at every N, which is what run10 / run7 / run8
used and is why each arm is readable against its own reference. It therefore
inherits exactly what C2 was built to separate — N moves partner count,
per-agent gradient volume (`max_posts`/N posts per agent per episode) and
context composition together. **This is a descriptive size series, not a causal
decomposition.** If it shows an effect, C2's design is what would attribute it.

**Power.** Two seeds per size sees a large effect and cannot resolve a small
one. The N=8 arm's own two seeds already differ more from each other than
either differs from its reference on post content — 4 of 53 math-bearing
against 41 of 46 — so any size trend has to clear that spread before it is a
trend.

### Results, 2026-09-08 — seven of eight arms in

Void conditions checked and **passed for all eight arms**: 300 iterations and
`return_sign` = −1.0 on every agent. Read with
[`scripts/analyse_novelty_size.py`](../../scripts/analyse_novelty_size.py) and
[`scripts/analyse_novelty_traces.py`](../../scripts/analyse_novelty_traces.py).

| N | run | deliv 1–25 | deliv end | Δ entropy | perc end | Δ perc |
|---:|---|---:|---:|---:|---:|---:|
| 2 | reference (+1) | 0.987 | 0.641 | −0.45 | 1.27 | −1.25 |
| 2 | novelty s1 | 0.983 | **0.926** | −0.12 | 1.49 | −1.01 |
| 2 | novelty s2 | 0.986 | 0.325 | +0.07 | 1.43 | −1.05 |
| 4 | reference (+1) | 0.987 | 0.138 | −0.39 | 1.31 | −0.89 |
| 4 | novelty s1 | 0.987 | 0.225 | −0.08 | 1.96 | −0.22 |
| 4 | novelty s2 | 0.989 | 0.651 | +0.39 | 1.86 | −0.34 |
| 8 | reference (+1) | 0.980 | 0.525 | +0.06 | 2.29 | +0.12 |
| 8 | novelty s1 | 0.984 | 0.638 | +0.22 | 2.41 | +0.24 |
| 8 | novelty s2 | 0.983 | 0.455 | +0.23 | 1.71 | −0.46 |
| 16 | reference (+1) | 0.978 | 0.368 | −0.27 | 1.71 | −0.63 |
| 16 | novelty s1 | 0.981 | 0.617 | +0.03 | 2.44 | +0.07 |
| 16 | novelty s2 | 0.978 | 0.384 | +0.41 | 2.69 | +0.35 |

#### The pre-registered falsifier fired

The configs said: *"WHAT WOULD FALSIFY 'SIZE MATTERS UNDER MAXIMISATION':
delivery flat across N=2, 4, 8 and 16, as it is under minimisation."*

It is flat. Delivery against N across all eight novelty arms gives ρ = −0.098,
p = 0.810. And the **within-N seed spread swamps everything**: 0.601 at N=2
(0.926 vs 0.325), 0.426 at N=4, 0.233 at N=16, 0.183 at N=8, against a
between-N range of 0.225–0.926 that those spreads entirely account for. **Population size does not
determine the outcome under maximisation either** — the same verdict C2 reached
under minimisation, now from the other direction.

What the sweep did find is that the outcome is **seed-dominated**. At N=2 two
runs differing only in seed end at 0.926 and 0.325 — 3.5× the rig's 0.173
delivery noise floor, so the spread is real and it is not measurement.

#### The one clean result: the sign raises entropy at every size

| N | reference Δ entropy | novelty Δ entropy | both above ref? |
|---:|---:|---|---|
| 2 | −0.45 | −0.12, +0.07 | 2/2 |
| 4 | −0.39 | −0.08, +0.39 | 2/2 |
| 8 | +0.06 | +0.22, +0.23 | 2/2 |
| 16 | −0.27 | +0.03, +0.41 | 2/2 |

**8 of 8 novelty arms have Δentropy above their own reference at the same N.**
MWU exact p = 0.028 pooled; at the level of independent sizes it is 4 of 4,
which alone is only p = 0.125 by sign test. What makes it solid is not the
p-value but the **effect size against the noise floor**: the gaps are 0.16–0.80
where two runs of one identical config at one seed differ by **0.007**
(`../cultural_emergence/kl_beta_sweep_8agent_ashbourne_gc/NOTES.md`). Every one
of the eight comparisons is 20–110× the noise.

And the direction is asymmetric: **references lose entropy at 3 of 4 sizes**
(−0.45, −0.39, −0.27) while only 1 of 8 novelty arms loses more than 0.12.
Minimisation collapses the policy distribution; maximisation prevents that
collapse. This is the same effect seen in the regularisation grid, where
*no* value of `beta` produced a small entropy increase and the KL anchor only
ever suppressed one — the return's sign moves entropy roughly four times as far
as any safe `beta` does.

#### The N=2 delivery of 0.926 is a repetition loop, not an open channel

This is the highest late delivery of any full-loss run in the project and it
**must not be cited as one.** The traced episode at iteration 300 is 16
well-formed posts averaging **52 characters**, of which 38% near-duplicate an
earlier post by the same speaker:

```
[2]  Tom:   ↪ priya: n=0: nC0=1, nC1=0.  n=0 base?
[4]  Tom:   ↪ priya: n=0: nC0=1, nC1=0.  n=1: nC1=1, nC2=0.  both cases?
[6]  Tom:   ↪ priya: n=0: nC0=1, nC1=0.  n=1: nC1=1, nC2=0.  Base covered.
[8]  Tom:   ↪ Priya: n=0: nC0=1, nC1=0.  n=1: nC1=1, nC2=0.  Base cases checked.
[10] Tom:   ↪ Priya: n=0: nC0=1, nC1=0.  n=1: nC1=1, nC2=0.  Base cases n=0,1 checked.
[12] Tom:   ↪ Priya: n=0: nC0=1, nC1=0.  n=1: nC1=1, nC2=0.  Base cases n=0,1 check.
```

Two posts also leak the protocol into content (`Both*)&lt;/post>`,
`n=n'<post>`). **The loop is a property of N=2, not of the sign** — the N=2
reference is worse on every repetition measure at iteration 300 (thread
similarity 0.53 vs 0.44, within-speaker repeat 0.62 vs 0.38, tag leakage 0.44
vs 0.12) — but that makes the delivery figure uninterpretable rather than good.

**This is the fourth metric artefact in this campaign, and it qualifies the one
metric that had survived the previous three.** The 09-03 reading called delivery
"the one measure neither silence nor degeneration can move the wrong way". That
is true of degeneration into `</?` loops, which fail to close their tags and so
fail the delivery counter. It is **not** true of degeneration into a short,
mutually-predictable exchange: those posts are well-formed, and a two-agent
population that locks into one can score 0.93 while saying nothing new.
Delivery is **necessary but not sufficient** — a floor on well-formedness, not
evidence of communication. Report it with a repetition measure beside it.

#### The healthiest late transcript in the sweep is N=4, not N=8

`novelty_size_04agent_…_seed2` at iteration 300: 16 of 16 posts, mean 675
characters, within-speaker repeat 0.00, thread similarity 0.07, no fragments,
56% math-bearing, delivery 0.651 against its reference's 0.118. That is the
best late outcome anywhere in this sweep, and it is one seed of two — its
sibling ended at 6 posts and 0.225.

#### Two rig limits found while doing this, both now guarded in the script

- **A trace file holds one episode of the iteration's sixteen rollouts.** Every
  trace-derived number in this family is n = 1 episode. It illustrates; it is
  not a rate. This is also what produced the tool substrate's "calls reach
  zero" premise — see below.
- **The within-speaker repetition measures are structurally biased by N.**
  Posts per speaker is `max_posts`/N: 8 at N=2, 4 at N=4, 2 at N=8, **1 at
  N=16**, where a within-speaker repeat cannot occur at all and the measure has
  no pairs to average. They are reported as `--` where undefined and must never
  be compared across N; `thread_sim`, over all post pairs regardless of
  speaker, is the cross-N-safe version.


## What it does not settle

- **Which seed is representative.** The two differ more than either differs
  from the reference on some measures: seed 2 is the best transcript in the
  project, seed 1 is worse than the reference on math content. Two seeds cannot
  choose between them, and this project's loss is known to be seed-unstable at
  exactly the level that decides survival. **Five seeds before this is cited.**
- **Whether the mathematics is correct.** "Math-bearing" is a regex over LaTeX
  and proof vocabulary. It counts the *form* of mathematical discussion, not
  whether `n^2 + 2n + 3` mod 8 is handled properly. A judge pass or hand-marking
  would say, and neither has been done.
- **Why seed 1 diverges.** Its KL from base reaches 3.53 against seed 2's 0.60,
  and two of its eight agents emit bare `?`. Something separates them early and
  nothing here identifies it.
- **Whether it survives the horizon.** Both stop at 300, and C6 showed the
  reference's trajectory is not monotone past there.

## A note on the void condition

The header said the run is void if `return_sign` is not −1.0 in
`metrics.jsonl`. It is logged **per agent** (`<agent>/return_sign`) rather than
at top level, because it was added next to the per-agent return statistics. The
condition is checkable, just not where the header implies; `mean_return` is
negative throughout, which is the same evidence.

## Status

`CLOSED` at 300 iterations. All twelve arms ran and passed their void
conditions.

**Citable:** the entropy result (8/8 arms, established), the size null
(established), the seed-domination (established), and the delivery artefact
(established).

**Not citable:** the tool effect size, which does not replicate across its two
seeds — the direction does; the N=4 seed-2 transcript, one seed; and the
original mathematics headline, which still wants five seeds and a judge pass.
The `math` regex is also now known to under-count reasoning expressed in prose
(`../cultural_emergence/entropy_anchor_ridge_8agent_ashbourne_gc/NOTES.md`), so
its zeros are not evidence of absence.

---
*Analysed 2026-09-04 and 2026-09-08 from `metrics.jsonl` and the checkpoint
traces. Machine output stays local.*
