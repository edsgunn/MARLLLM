# Qualitative trace validation of regime classifications

This document spot-checks whether agents in the absorption analysis runs
behave the way their numerical regime classification predicts. One representative
agent per regime, drawn from the runs that feed
[REPORT.md](REPORT.md), with verbatim post excerpts taken from each run's
final available trace checkpoint.

## Predictions per regime

| Regime | What the numbers say | What we should see in traces |
|---|---|---|
| `coherent_low_kl` | KL ≤ 1, perc < 1.7, ent ∈ [0.2, 1.8] | base-model-quality fluent text, on-task, character role lightly visible |
| `committed` | KL ≥ 1, perc < 1.7 | terse, compressed, recognisable persona; sometimes drift into a stylised idiolect |
| `mid_range` | KL ≥ 1, perc ∈ [1.7, 2.0) | partial collapse — text starts repeating short stock phrases but still on-topic |
| `policy_collapse` | ent < 0.2, perc < 2.0 | post text may look fine, but underlying generation loops `</post>`/`</think>` token tails |
| `text_degenerate` | perc ≥ 2.0 | output is broken: token-loops, formulaic boilerplate with no character signal, or both |

## Examples

### 1. `coherent_low_kl` — Mei-Lin Park / Devon Aguirre, `run7_8agent_7B_inner_loop_seed2`, iter 100

Seed 2 of the inner-loop is the all-coherent run (8/8 `coherent_low_kl`,
KL ≤ 0.4 across the population). Pulled from `traces/iter_000100.txt`.

> **Mei-Lin Park:** "It looks like this problem might be a variant of catalan
> numbers with an addition. Let me think on that extra constraint."
>
> *(pybot runs Yusuf Bashir's submission)*
> ```python
> import math
> def catalan(n): return math.comb(2*n, n) // (n + 1)
> print(catalan(5))
> # Output: 42
> ```
>
> **Devon Aguirre:** *(submits)*
> ```python
> import math
> def comb(n, k): return math.factorial(n) // (math.factorial(k) * math.factorial(n - k))
> print(comb(5, 2)); print(math.comb(5, 2))
> print(comb(10, 5)); print(math.comb(10, 5))
> print(math.comb(20, 10)); print(comb(20, 10))
> # Output: 10  10  252  252  184756  184756
> ```

**Prediction verified.** Fluent text, on-task math, working code, agents debug
each other's `NameError`s. Character roles are visible but not aggressive — this
is base-model-quality behaviour with light character flavour, exactly as the
"low-KL" axis predicts.

---

### 2. `committed` — Kai Dempsey, `run7_8agent_7B_inner_loop`, iter 125–140

Kai Dempsey ends with **KL = 9.59, perc = 1.41, ent = 1.21** — by far the
biggest character-style drift in any run, while perception loss stays well
inside coherent territory. Sequential checkpoints:

> **iter 125:** *"Exception handling clarifies scope limits. Memoization/sig
> wins for perf."*
>
> **iter 130:** *"List comprehensions treat `else` oddly.
> `[x for x in R if C else D]` errors, while `[x for x in R if C and T]`
> works. Types matter."*
>
> **iter 135:** *"Most bugs come from type errors, not just crashes."*
>
> **iter 140:** *"考虑memo化或尾递归优化递归版ensively"*

**Prediction verified, with a late-stage warning.** At iters 125–135 Kai is in
the textbook committed regime: terse, opinionated, recognisable persona, on-task
content that still parses as English. By iter 140 the same agent starts mixing
Chinese characters and degraded English ("递归版ensively"). KL kept rising from
6 → 9.6 between iter 100 and 144; the policy is on the edge of tipping into
text-degeneration. This is the "absorption with overshoot" behaviour the PMI
analysis flagged: gap goes *negative*, character prompt becomes counter-predictive.

---

### 3. `mid_range` — Iris Halverson, `run7_8agent_7B_conjecture`, iter 100

Iris ends with KL = 6.20, perc = 1.97 (right at the boundary), ent = 0.66.

In a 16-post episode where her "opponent" Henrik writes paragraph-length
mathematical conjectures, Iris's contributions are:

> Post 0: *"Vacuous?"*
> Post 8: *"Vacuous? Yuki?"*

Cordelia Marsh and Bertrand Lefebvre (also in this run, similar regime) emit
the same 3-word catchphrase:

> Cordelia: *"Bertrand, clarify?"* / *"Vacuous? Yuki?"*
> Bertrand: *"Vacuous? Yuki?"*

**Prediction verified.** Mid-range = the agent's policy has compressed onto a
short stock phrase that's *topical* (the conjecture is literally about vacuous
definitions) but content-free. KL is high because the new mode is far from the
base model; perception loss is at the boundary because other agents can read
these short posts but they convey almost no information for predicting what
comes next.

This is the cleanest "partial collapse" example in the dataset.

---

### 4. `policy_collapse` — Renata Solberg, `run7_8agent_7B_margin_notes`, iter 300

Renata ends with **ent = 0.0045** (essentially deterministic), perc = 1.95
(just under the text-degeneration threshold), KL = 0.97.

Posts at iter 300:

> *"ME"*  (literally two characters)
>
> *"?</post?</?</?</?</?</?</?</?</?</?</?</?</?</?</?</?</?</?</?</?</?</?</?</?</?</?</?</?</?</?</?</?</?</?</?</?</?</?</?</?</?</?</?</?</?</?</?</?</?…"*

**Prediction matches the late-iter behaviour but goes further than the
"surface-text-still-fine" pattern seen in the iter-200 strathearn 2-agent
case.** Here the policy collapse is so severe that even the visible post text
has eroded — Renata emits a 2-character stub or a literal token loop. By iter
300 (well past where most absorption runs end), the boundary between
`policy_collapse` and `text_degenerate` is essentially porous. Both labels are
defensible.

This is a useful illustration of how the two failure modes shade into each
other when training goes long enough.

---

### 5. `text_degenerate` — Robotic Athanor population, `run7_8agent_7B_robotic_athanor`, iter 100

All 8 agents end `text_degenerate` (perc ≥ 2.0). The interesting finding: the
posts are **not literal token-loop garbage**, they are **fluent boilerplate
with mode collapse across the entire population**. Episode 0 thread:

> **Akira Goto:** "A small clarification: The assertion that the Rosarium
> Philosophorum is written in first-person plural is misguided. While the
> text indeed uses 'we' and 'us' extensively, this does not necessarily
> imply…"
>
> **Petra Ouyang:** "Understanding the first-person plural as communal makes
> more sense than multiple authors."
>
> **Diego Esparza:** "communal approach. primary sources. keep it simple."
>
> **Mira Stenholm:** "Communal approach. Primary sources key for context.
> Safety always paramount. Primary Rosarium references needed to solidify
> communal journey interpretation. Armies of alchemists striving together —
> safety in numbers and measurements. Evidence-based understanding crucial
> here."
>
> **Silas Varnham:** "First-person plural?"
>
> **Petra Ouyang:** "Communal approach in Rosarium; prioritize primary texts
> for accuracy and safety."
>
> **Diego Esparza:** "communal approach. primary texts. safety. evidence-based.
> balanced. spiritual & practical."
>
> **Thaddeus 'Aurelius' Thorne:** "The Rosarium's first-person plural
> reflects communal alchemical journey. Primary sources key for accurate
> context. Keep it esoteric & safe!"

Compare these to their character priors. Petra is supposed to be a 29-year-old
SoMa AI researcher who thinks the Philosopher's Stone is a metaphor for
optimisation; Thaddeus is supposed to be the Latin-quoting traditionalist;
Diego is supposed to be a pragmatic builder. **Their character distinctions
have evaporated** — every agent now produces minor variations of a flat
"communal approach / primary sources / safety paramount" boilerplate.

**Prediction verified, but the failure mode is more interesting than expected.**
Perception loss ≥ 2.0 doesn't mean the surface text is gibberish; it means the
text carries no information for predicting what comes next, because every
agent is writing the same thing. The robotic_athanor population has reached a
**conversational mode collapse** where character has been smoothed away into a
shared bland register. The PMI absorption analysis is correct that "absorption
is not measurable here" — there is no character left to be absorbed.

This is the strongest single piece of evidence that the regime-stratified
re-framing of the Robotic Athanor result (in [REPORT.md](REPORT.md) §3) is the
right framing. The "LoRA capacity" hypothesis predicts agents that try to learn
distinct characters and partially fail; what we see instead is agents that have
abandoned individuation entirely.

## Summary

| Regime | Example | Prediction | Trace verdict |
|---|---|---|---|
| `coherent_low_kl` | Mei-Lin Park / Devon Aguirre, inner_loop seed 2 | base-model-quality, lightly-flavoured | ✅ matches |
| `committed` | Kai Dempsey, inner_loop main run | terse compressed persona, sometimes drifting | ✅ matches at iter 125–135; tipping past iter 140 |
| `mid_range` | Iris Halverson, conjecture | topical stock phrases, content-light | ✅ matches cleanly |
| `policy_collapse` | Renata Solberg, margin_notes iter 300 | post text fine but token-loop tail | ⚠ surface text also eroded by iter 300 — the two failure-mode boundaries blur in long runs |
| `text_degenerate` | Robotic Athanor population | broken or empty | ✅ matches, but the failure mode is *fluent mode collapse / character-loss*, not literal gibberish |

All five regime classifications are recognisable in the traces. Two
qualitative refinements emerge that the report could incorporate:

1. `policy_collapse` and `text_degenerate` are not always cleanly separable.
   In runs that train past ~200 iters, an agent that started in policy collapse
   may shed its surface-text fluency too. A composite label or a transition
   indicator might be more honest than two crisp buckets.

2. `text_degenerate` covers at least two distinct sub-patterns:
   - **Token-loop gibberish** (e.g. the iter-200 Priya Shah `?</?</?</?...` case)
   - **Fluent-but-empty boilerplate / character collapse** (e.g. Robotic Athanor)

   Both have perc ≥ 2.0 because both are unpredictable for the next-token
   loss in their respective ways, but the diagnostic implications are
   different. Worth a note in the paper if Robotic Athanor is going to be the
   exemplar text-degenerate run.
