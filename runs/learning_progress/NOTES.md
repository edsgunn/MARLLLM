# learning_progress

*A suite whose unit of inference is the family: nine arms testing what happens
when the action return is paid for **improvement over a second reference**
instead of for accuracy, so that `H(p_env)` cancels out of the objective.*

|  |  |
|---|---|
| Configs | [`configs/learning_progress/`](../../configs/learning_progress/README.md) |
| Model | `Qwen2.5-7B-Instruct`, LoRA r64 on q_proj/v_proj, 8 agents |
| Arms | `lagged` × 2 seeds, `base` × 2 seeds, lag L∈{5,100} × 1, no-normalisation × 1, `margin_notes` × 2 seeds |
| Ran to | iteration 300 of 300, all nine · jobs 6571670–6571673, 6580460–6580464 |
| Code | commit `9e9e25d` |
| Scripts | [`analyse_learning_progress.py`](../../scripts/analyse_learning_progress.py) |
| **Differs from** | **`run7_8agent_7B_study_group_ashbourne_gc`** (text) and **`run7_8agent_7B_margin_notes`** (tool) — the return's per-observation summand, `M_s − M_s⁻` instead of `M_s`, and nothing else. |

## What it was for

The surrogate return's per-position expectation is
`KL(p_env ‖ p_θ) + H(p_env)`, so the practical loss is the stated objective
*plus the entropy of the environment* — a REINFORCE reward for making the world
simpler, which in a population means making the other agents simpler. `H(p_env)`
is common to any cross-entropy against `p_env` and cancels in a difference of
two of them, so the correction needs a second model rather than `p_env` itself.

Pre-registered: the primary read is policy entropy at 300 against the same-N
reference; the gate on whether *any* of it could matter is
`lp/return_rank_corr`, "near 1.0 means the correction reorders nothing and
cannot have mattered". Pre-registered as **not** addressed: the length channel.

## Headline

**The correction is real, large, and does not stop the collapse.** It nearly
*reverses* the surrogate's ordering of actions — `lp/return_rank_corr` runs
−0.30 to −0.84 across all nine arms, where the pre-registered null was "near
1.0" — and it still ends where the reference ends: delivery 0.51–0.71 at
iteration 300 against the reference's 0.56, with **1 to 4 of 8 agents in every
single arm silent on more than half their turns**.

**What varies is the seed, not the correction.** `lagged` and `base` are
indistinguishable from each other; seed 1 and seed 2 are not remotely alike. At
iteration 300 the two seed-2 arms are eight agents doing collaborative
mathematics at ~405 characters a post with **zero** fragments, while the seed-1
arms are at 103–262 characters with 22–44% fragments.

**And perception loss is anti-correlated with transcript quality in this
family, in both directions at once.** The two arms doing real mathematics are
labelled `text_degenerate` on 8 of 8 agents; the most degenerate arm in the
suite — token loops of `?` and a repeated name — has the *lowest* perception
loss of all nine. Neither reading survives contact with the traces.

## Core metrics

Iteration 300, mean over 8 agents. `deliv` is `post_success / post_turns`;
`frag` is the fraction of posts under 20 characters; `stopword` is the English
function-word screen (`< 0.10` salad, `0.10–0.28` ambiguous, `≥ 0.28` coherent).

| arm | deliv | entropy | perc | frag | chars | read by hand |
|---|---|---|---|---|---|---|
| lagged s1 | 0.56 | 0.94 | 2.54 | 0.44 | 104 | **salad** — `beatricebirthdateNavigate`, `?</?` |
| lagged s2 | 0.71 | 1.50 | 2.26 | 0.00 | 408 | coherent maths, code-switches to Chinese |
| base s1 | 0.53 | 0.90 | 2.43 | 0.33 | 153 | readable prose, many empty posts |
| base s2 | 0.65 | 1.22 | 2.14 | 0.00 | 404 | coherent maths — Fibonacci subset proof |
| lag L=5 s1 | 0.56 | 1.01 | 2.41 | 0.35 | 262 | readable prose, many empty posts |
| lag L=100 s1 | 0.61 | 0.89 | 2.43 | 0.22 | 181 | readable prose, many empty posts |
| no-norm s1 | 0.51 | 0.65 | **3.47** | 0.03 | 232 | readable; 3 agents ~100% silent |
| REF surrogate | 0.56 | 0.79 | 2.48 | 0.22 | 158 | — |
| REF novelty s1 | 0.64 | 0.92 | 2.38 | 0.28 | 113 | — |
| tool LP s1 | 0.41 | 0.87 | 2.04 | 0.48 | 103 | **salad** |
| tool LP s2 | 0.29 | 0.90 | 1.79 | 0.21 | 1166 | **`?` loops + repeated name** |
| tool REF surrogate | 0.37 | 0.33 | 1.83 | 0.16 | 520 | — |

Regime composition at 300 is `alive` 0.00 for every Ashbourne arm and for the
reference. **That number should not be quoted**: see below.

Per-agent silent-turn rate at 300 is bimodal in every arm — e.g. lagged s1
`1.00 0.84 0.81 0.56 | 0.00 0.00 0.00 0.00`. Agents stop individually; the
population mean hides it.

## What this settles

- **The corrected return reorders actions rather than rescaling them.**
  `lp/return_rank_corr` is −0.30 to −0.84 in all nine arms at every checkpoint.
  This is the opposite of the only other return correction measured here — the
  dropped environment term is ~43% of the return's magnitude at rank
  correlation **1.000** — so the two corrections are not the same kind of
  object. — *established*, nine arms
- **Cancelling `H(p_env)` does not keep the public channel open.** Delivery
  falls in 9 of 9 arms, to 0.51–0.71 against a reference at 0.56, and 1–4 of 8
  agents per arm end silent on over half their turns. The dark-room term is not
  what closes the channel, or not by itself. — *established* for the fall,
  *likely* for the comparison (one reference seed)
- **The `perc_loss ≥ 2.0` gate misclassifies this family in both directions.**
  `base s2` and `lagged s2` are 8/8 `text_degenerate` while discussing
  `a(n) = a(n−1) + a(n−2)` and `fib(n+2) − 1` in fluent English; `tool LP s2`
  has 1/8 over the gate and is `?`-loop degenerate at 1166 characters a post.
  This is the fifth and sixth metric artefact in this project and the first
  where the metric fails *both* ways in one family. — *established*, by reading
- **The dominant failure mode here is silence, not degeneration.** Of nine
  arms at iteration 300, two contain genuine word salad; the other seven stay
  readable and lose delivery through empty posts. `text_degenerate` names the
  wrong thing for most of this family. — *established*, by reading
- **Removing return normalisation makes it worse, and its own header predicted
  the opposite.** The arm was submitted on the argument that standardisation
  rescales estimator noise once the reward dies. It ends at the worst
  perception loss in the suite, 3.47, carried by **3 agents at 5.41–7.13 who
  are ~100% silent** while the other five sit at 1.81–1.90. — *established*,
  one seed
- **The lag knob does nothing detectable.** L = 5, 25 and 100 at seed 1 end at
  delivery 0.56 / 0.56 / 0.61 and entropy 1.01 / 0.94 / 0.89, with the same
  silence structure. The scheme's only knob spans a 20× range and moves
  nothing. — *likely*, one seed per level
- **Entropy rises, as it does under a reversed sign.** Six of seven Ashbourne
  arms end above the reference's 0.792, by +0.10 to +0.70 nats. This is the
  second objective change to move entropy reliably upward. — *likely*; one
  reference seed, so the size of the gap is not established
- **The tool substrate is not rescued.** Python calls per episode at 300 are
  2.4 and 0.4 against the surrogate reference's 1.4 and novelty search's 3.8,
  and seed 2 is `?`-loop degenerate. The prediction that a return paying for
  improvement would make tool output worth eliciting is not supported. —
  *likely*, two seeds

## What it does not settle

- **Whether the correction helps at all**, because both references are n=1 at
  iteration 300 and this project's measured noise floor on delivery between two
  runs of one config is **0.173**. The LP arms hold delivery above the
  reference at iterations 100 (7/7) and 200 (7/7), by 0.13 and 0.14 — *below
  that floor*. Suggestive, not a result. The margin_notes reference has four
  further seeds but only to iteration 150.
- **Whether the seed-2 transcripts are better than the reference's**, for the
  same reason. Zero fragments at ~405 characters against the reference's 22% at
  158 is a large difference; it rests on two arms and one reference run.
- **What the code-switching means.** `lagged s2` and `base s2` end with 16–20%
  non-Latin characters, and the English function-word screen cannot score
  code-switched text — it drops `lagged s2` from 0.281 (coherent) at iteration
  150 to 0.169 (ambiguous) at 300 on text that reads fine. The screen needs a
  multilingual variant before it is used on this family again.
- **Why silence is agent-wise.** Every arm splits into agents that post
  normally and agents that stop almost entirely. Nothing here says what selects
  which.

## Status

`CLOSED`. All nine ran 300 of 300; void conditions checked and passed on every
arm (`lp/in_use` = 1.0 on every record; `lp/reference_age_iters` within [1, 2L]
for lagged arms and equal to the iteration for base arms). Citable, with the
n=1-reference caveat above attached to every comparison.

---
*Machine output stays local: `metrics.jsonl`, `traces/`, `checkpoints/`.*
