# c13_novelty_search

*Maximise future observation surprise instead of minimising it. The sign of the
return is the only change. Two seeds, jobs 6383538 and 6383539.*

|  |  |
|---|---|
| Configs | [`configs/cultural_emergence/c13_novelty_search/`](../../../configs/cultural_emergence/c13_novelty_search/) |
| Model | `Qwen2.5-7B-Instruct`, LoRA r64, 8 agents on `study_group_ashbourne_gc` |
| Ran to | iteration 300 of 300, both · 2h58m and 4h13m |
| **Differs from** | **`run7_8agent_7B_study_group_ashbourne_gc`** — `return_sign` −1.0. Verified field by field as the only difference. |

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
