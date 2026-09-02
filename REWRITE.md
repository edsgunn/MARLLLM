# REWRITE.md — plan for the resubmission

What we want to claim, what we can already claim, what has to be answered to
close the gap, and what changes in the paper.

Written 2026-09-01 against [`STATE.md`](STATE.md) and the per-run notes under
`runs/`. Reviews and rebuttal are the NeurIPS 2026 thread for submission 34736.

**Read this first:** five things the retrospective documentation turned up are
not in the paper, not in the rebuttal, and not in the extended analyses. Two of
them break parts of the current experimental design and one of them is probably
the best result available to us. They are in §3.

**Updated 2026-09-01, second pass.** All eight Tier-B analyses have now been run
([`runs/_tier_b/`](runs/_tier_b/README.md)). Results are folded in below. **Two
things in the first version of this document were wrong and are retracted in
place:** §3.5's mechanism (it is not private deliberation) and §5.1's cost claim
(restoring the environment term is not nearly free). Both were caught by the
falsification conditions written into the analysis scripts before they ran.

---

## 1. Where we stand

**5MAX** did the review the paper deserved and, after the rebuttal, moved to a
position we can work with: the reframe from *demonstrating* cultural emergence
to *establishing measurable preconditions* is "substantially more defensible";
the gradient-bias characterisation "directly addresses my principal theoretical
objection"; the 15 replicate populations "materially strengthen the empirical
section". Three things remain outstanding in their words: the zero-bias
condition **has not been established in the actual substrates**; the swap
control is a likelihood comparison that does not show behavioural persistence
in rollouts, and **does not isolate co-adaptation** from ordinary continued
training; and the reframe must be carried consistently through title, abstract,
contributions, experimental questions and conclusions.

**pd7C** rejected the premise rather than the execution: no clear research
question, an instruction-tuned base contradicts the self-supervision story, and
persona specification is prompt engineering orthogonal to the objective. The
rebuttal answered the third with the swap control and conceded the second
correctly (the prior is to be specified, not eliminated). The first is a
framing failure and is fixed by the reframe.

**ZLns** asked four questions. Three were answered. **Q4 was not: how much do
private thinking tokens contribute to the training signal, and would results
change if only public post tokens were treated as actions?** That question turns
out to be the most important one anybody asked — see §3.5. ZLns also wants the
MindForge citation and flags the bibliography style.

---

## 2. The reframe

The rebuttal's three preconditions are the right spine, and they should be
stated as the paper's object rather than as a defence:

1. **Reachability** — agents can move substantively away from base behaviour
   without collapsing or becoming mutually unintelligible.
2. **Internalisation** — what they acquire is held in weights rather than
   re-read from the prompt.
3. **Population specificity** — what they acquire is specific to their
   population rather than a drift all populations share.

Each necessary, none sufficient. We have evidence bearing on all three.

**Add a fourth, which we have already measured and never used:**

4. **Portability** — what an agent acquires survives transplant into a
   population that has never met it, and is not overwritten by the host.

The migration experiments show this cleanly and are not in the paper at all
([`runs/migration_experiments/NOTES.md`](runs/migration_experiments/NOTES.md)).
Portability is closer to a transmission channel than anything else in the
corpus: a precondition for culture is that acquired structure can *move*, and
sixteen runs say it moves and keeps its shape. This is the cheapest available
upgrade to the paper's contribution list — the runs exist, the analysis is done.

**And one honest negative, which a preconditions paper should want:**

5. **We looked for sanctioning and did not find it.** Rung 0 over both five-seed
   forum families: approval median 0.656 reactions per post, disapproval median
   0.062, no trend across 300 iterations, and the asymmetry is already at its
   final value at iteration 0
   ([`runs/sanctioning/NOTES.md`](runs/sanctioning/NOTES.md)). A norm needs a
   sanctioning response that can go either way; in these populations one
   direction is at the floor and training does not move it. Reporting this
   converts "we did not demonstrate conventions" from an absence into a
   measurement with an instrument attached.

---

## 3. Five things the retrospective found that the paper does not know

### 3.1 The state described by Precondition 1 is transient, and iteration 200 is inside the window

Across a balanced panel of 44 cultural-emergence populations measured at
iterations 25, 100, 150, 200 and 300, the mean alive fraction runs

> 0.27 → **0.86** → 0.82 → 0.72 → **0.46**

**No forum population improved after iteration 150.** Of the 44 measured at both
150 and 200, none rose; of the 44 measured at both 200 and 300, none rose and 31
fell. 39 of 44 end below their own earlier peak.

The clearest single case is the 8-agent Ashbourne GC run, which is 88% alive at
iteration 200 and **0% alive at 300**
([`runs/cultural_emergence/run7_8agent_7B_study_group_ashbourne_gc/NOTES.md`](runs/cultural_emergence/run7_8agent_7B_study_group_ashbourne_gc/NOTES.md)).
This also reconciles two of our own tables that look contradictory: the iter-200
classification lists that run as mostly alive and the absorption report lists it
as 8/8 text-degenerate. Both are right; they read different iterations.

Margin Notes makes the same point from the other side: 100% alive across four
seed replicates that stop at iteration 150, and 50% alive in the one run that
continued to 300. **A short horizon can make a failing substrate look sound.**

*What this does to the paper.* Figure 1 reports end-of-training metrics at
iteration 200 and the text calls them training outcomes. They are not outcomes;
they are a snapshot one step before a decline that every long run shows. This is
not fatal — but it must be the paper's own finding rather than a referee's.
Restated, it is a first-order result for a preconditions paper: **the enabling
state is reachable and is not an attractor.** Culture requires persistence, so
the horizon over which the state persists is precisely a precondition
measurement. Report the alive fraction as a function of iteration, not at a
point.

### 3.2 The pipeline is not reproducible at fixed seed

One configuration was run twice by accident — the `(kl_coef 0.5, β 0.1)` cell,
once in the sweep and once in the beta zoom. The two configs differ in run name,
output directory, and which of four GPUs the job landed on. Same seed, same
hyperparameters, same code, 300 clean iterations each, no resume.

They track each other to about iteration 150 and then separate. Alive fraction
at iteration 300: **0.875 against 0.000**
([`runs/cultural_emergence/kl_beta_sweep_8agent_ashbourne_gc/NOTES.md`](runs/cultural_emergence/kl_beta_sweep_8agent_ashbourne_gc/NOTES.md)).

*What this does to the paper.* Figure 1 is one run per cell. The run-to-run
spread at fixed seed is as large as the spread across most of the design. This
is 5MAX's Q3 objection, and it is worse than they knew — the problem is not only
that agents within a population are dependent, it is that **the population is
not a reproducible unit either.** The 15 replicates address it where they exist
(N=16 Ashbourne, N=8 Inner Loop, N=8 Margin Notes) and nowhere else.

Disclose it, and make it useful: it tells the field what n is needed for this
class of experiment, which nobody currently reports.

### 3.3 The population-size grid is not monotone above N=2

Alive fraction, one run per cell:

| substrate | N=2 | N=4 | N=8 | N=16 | N=32 |
|---|---:|---:|---:|---:|---:|
| `ashbourne_gc` @200 | 1.00 | 0.75 | 0.88 | 1.00 | 0.84 |
| `ashbourne_gc` @300 | 0.00 | 0.75 | 0.00 | 0.88 | 0.62 |
| `strathearn_server` @200 | 0.00 | 0.75 | 1.00 | 0.88 | — |
| `strathearn_server` @300 | 0.00 | 0.75 | 0.75 | 0.69 | — |

What survives: **two-agent populations end dead on both substrates**, and
**N=32 is worse than N=16 at both evaluation points**. What does not survive:
any ordering in between, on either substrate, at either iteration. The 8-agent
Ashbourne cell ends at 0.00 while the 4-agent cell ends at 0.75.

The rebuttal already withdrew "population size controls regime accessibility"
and the N=16 commitment claim. This says the retreat has to go further than
"association": the *shape* asserted in Appendix G — "the failure mode is
progressively eliminated by population scale" — is not in the data either.

### 3.4 The perception-loss threshold is not substrate-neutral

Perception loss at iteration 1, before any training: `study_group_ashbourne_gc`
2.16–2.50, `robotic_athanor` 2.17, `inner_loop` 1.23, `ashbourne_school` 1.60.
The text-degenerate cut is 2.0.

So on two of our substrates **the untrained model is already classified as
text-degenerate**, and whole populations are labelled degenerate at iteration 25
before training could have broken anything. `text_degenerate` is currently a
mixture of *has broken* and *never learned*, and which one it means depends on
the substrate.

This is exactly what happened on Robotic Athanor. Perception loss there is
2.18 → 2.12 across 100 iterations — flat at its starting value — and entropy is
1.07 → 1.02, neither low nor falling
([`runs/cultural_emergence/run7_8agent_7B_robotic_athanor/NOTES.md`](runs/cultural_emergence/run7_8agent_7B_robotic_athanor/NOTES.md)).
The paper says these agents' "action distributions have degraded" and that this
"is what one would expect when action distributions have degraded". They have
not degraded. They never moved. The paper's *conclusion* survives — the LoRA
capacity explanation is unnecessary — but its stated reason is wrong, and the
correct reason is stronger: **absorption is undefined for a population that
never learned to predict its substrate at all.**

*Fix.* Define the perception threshold **relative to the substrate's own
iteration-0 baseline**, not absolutely. Then "degraded" means "worse than the
untrained model", which is what every reader already assumes it means. This
changes Figure 1 and it removes an inconsistency a referee will otherwise find
in the released traces.

### 3.5 The public channel closes — and it closes before anything else shows

**Correction, twice.** The first version read a rising `thinking_frac` as agents
moving generation into a private channel. The second version retracted that on
the grounds that there are no `<think>` tags anywhere. **The second version was
wrong.** In this framework `<think>` is not what makes a token private — the
environment routes by tag, and "everything else" outside `<post>`/`<python>` is
private by construction ([`envs/forum/env.py`](envs/forum/env.py):290, 518).
`thinking_tokens = n_act − post_tokens` implements that definition correctly.

What [B6](runs/_tier_b/b6_token_accounting/NOTES.md) does establish is subtler
and worse for the metric: `thinking_frac` includes `<python>`, which is
**public** — 68–80% of early non-post tokens on the two tool substrates — and it
cannot tell deliberation from a post the agent wrote and failed to tag. Reading a
fair sample of late `post_missing` turns finds all three: untagged posts (one
agent writing in Strathearn's own pinned format, another giving a careful
in-persona argument about list-versus-tuple types), genuine deliberation, and
token loops (8–11%). Tags and intent have come apart in both directions — *"Note
to self: …"* was published inside a `<post>` at margin_notes iteration 300.

**So do not build the section on `thinking_frac`.** Build it on the
environment's own post accounting, which measures the effect directly and is
agnostic about intent.

**The corrected finding is stronger, and the environment was already measuring
it.** `envs/forum/env.py` classifies every turn `post_success` / `post_unclosed`
/ `post_missing` / `post_empty`, and only text inside a `<post>` block reaches
other agents. [B4](runs/_tier_b/b4_public_channel/NOTES.md):

| iteration | post_success | post_unclosed | post_missing | alive |
|---:|---:|---:|---:|---:|
| 1 | 0.988 | 0.008 | 0.004 | — |
| 50 | 0.981 | 0.012 | 0.008 | — |
| 100 | 0.856 | 0.144 | 0.000 | 0.88 |
| 150 | 0.621 | 0.328 | 0.051 | 1.00 |
| 200 | **0.477** | 0.426 | 0.098 | **1.00** |
| 300 | 0.281 | 0.434 | 0.285 | 0.88 |

*(`run8_16agent_7B_study_group_ashbourne_gc`; post_success fell in **63 of 65
runs** measured.)*

**The channel is already half shut at iteration 200, where the paper evaluates,
and the alive fraction still reads 1.00.** The two-turn run is starker: post
success collapses 0.97 → 0.34 between iterations 50 and 100 and stays there,
while the alive fraction holds at 1.00 until 250 and then falls to 0.38. Across
runs where both are observable, the channel degrades 25% at or before the first
drop in alive fraction in **9 of 12**.

**What this does and does not license.** "Agents learn to go quiet" is **not**
supported — a large share of these turns are agents trying to speak into a
tagging convention that has decayed, and nothing in the objective rewards a
well-formed tag. What is supported is the *consequence*, and it is identical
either way: under a return that sums surprise over future observation tokens, an
intended-but-untagged post and a deliberate silence do the same thing to
everyone else's `G_t`. The observation stream shrinks regardless.

*Why this matters more than the numbers.* This is the rebuttal's own theoretical
prediction, measured. The implemented return is
`G_t = Σ_{s>t, σ_s=OBS} γ^(s−t) · surprise_s`
([`marlllm/loss.py`](marlllm/loss.py) line 12) — a sum over *future observation
tokens*, whose magnitude depends on how many follow as much as on how surprising
each is. The rebuttal: "an agent that shortens its co-agents' replies moves it as
surely as one that makes them more predictable."

So the dark room here is neither gibberish nor determinism. **It is the closing
of the public channel**, it is the behavioural signature of the term the
surrogate drops, and **the regime taxonomy is a lagging indicator of it.**

What this is not: causal. The ablation is C4.

## 4. The claims ledger

### Tier A — supported by evidence in hand

| # | Claim | Confidence | Evidence |
|---|---|---|---|
| A1 | The objective is a single self-supervised loss recovering pre-training as an exact limiting case, with an exactly characterised gradient bias | established (derivation) | §3.1 + corrected Appendix A |
| A2 | Training outcomes occupy a continuous, bimodally-occupied space that a threshold partition describes but does not create | established | `_regime_threshold_sweep_3d/` — contrast positive at 100% of 2,850 grid points; k-means ARI 1.00 on seeds; GMM modes 0.13 / 2.84 nats |
| A3 | Fixed prompt-borne conditioning is absorbed into weights, character-specifically, in the low-drift stratum | established | `_absorption_swap_by_regime/` — own +0.073, swap −0.002, differential +0.076, n=32 |
| A4 | Environment description is absorbed as a distinct component, ~half the magnitude of character | likely | `_pmi_decomposition/` |
| A5 | In the high-drift stratum, absorption is ~21% character-specific and the rest is generic drift | likely | same swap control |
| A6 | Parallel populations' trajectory distributions diverge monotonically from a ≈0 iteration-25 baseline, three substrates, no plateau | established | `_trajectory_kl_combined/` |
| A7 | Per-character drift-from-base and inter-population divergence are different quantities; the inverse correlation was an entropy artefact | established | `_diagnostics_d2d3/` |
| A8 | A trained adapter transplanted into a naive population acquires the host's world model at the resident rate and keeps its own distance from base | likely | migration, 16 runs; 7/7 high-arrival drifted further, 8/9 low-arrival did not |
| A9 | Perception improvement and task success are dissociable | established | negotiation population: perception loss ÷10, success 0.000 every iteration |
| A10 | The perception term produces the world-model improvement; the action term moves the policy and narrows entropy | established | Phase A condition 3 vs 2 |
| A11 | β ≥ 0.1 is fatal at every KL coefficient; the KL anchor controls drift and not collapse | established | KL/β sweep, 16 cells + 4 zoom cells |
| A12 | The enabling state is transient: no forum population improves after iteration 150 | established | §3.1, 44-population balanced panel |
| A13 | The public channel closes over training in every long forum run | established | §3.5, five runs |
| A14 | Sanctioning is detectable and asymmetric, and training does not move it | likely | `runs/sanctioning/` |
| A15 | The pipeline is not reproducible at fixed seed | established | §3.2, one direct demonstration |
| A16 | The public channel closes over training in 63/65 runs, and leads the regime decline by up to 200 iterations | established (fall) / likely (lead) | [B4](runs/_tier_b/b4_public_channel/NOTES.md) |
| A17 | Cross-population divergence is not an artefact of collapsed adapters; pooled magnitudes are inflated where failure regimes are present (Margin Notes −45%) | likely | [B1](runs/_tier_b/b1_policy_distance_by_regime/NOTES.md) |
| A18 | The absorption contrast is horizon-clean on the Inner Loop seed replication (+0.160) and largely horizon-driven on the cross-environment pool (+0.199 → +0.024) | likely | [B7](runs/_tier_b/b7_absorption_horizon/NOTES.md) |
| A19 | Movement from base is strongly base-model dependent: no Qwen3 agent reaches KL ≥ 1 where 27/182 Qwen2.5-7B agents do at the same iteration | likely | [B8](runs/_tier_b/b8_model_generation/NOTES.md) |
| A20 | Observation-token provenance resolves at 100%, so the gradient bias is measurable without new instrumentation | established | [B5](runs/_tier_b/b5_env_likelihood/NOTES.md) |

### Tier B — done (2026-09-01), no new training

All eight ran. Scripts in [`scripts/tier_b/`](scripts/tier_b/), findings in
[`runs/_tier_b/`](runs/_tier_b/README.md). Each script's docstring carries its
pre-registration; two of the eight overturned a claim already written here.

| # | Question | Outcome |
|---|---|---|
| B1 | Policy distance restricted to alive characters | **Survives on 2 of 3.** Inner Loop ×1.00, Ashbourne ×0.95, Margin Notes ×0.55. Growth from baseline ×27–×112 either way. Report the restricted version; state the Margin Notes sensitivity. |
| B2 | Regime composition by iteration | **A snapshot, not an outcome.** Balanced panel of 44: alive 0.27 → 0.86 → 0.82 → 0.72 → 0.46. **0/44 improve after iteration 150.** 39/44 end below their own peak. |
| B3 | Substrate-relative perception thresholds | **Two substrates start above the degenerate cut** (Robotic Athanor 2.44, Ashbourne GC 2.20, vs Inner Loop 1.35). **50% of labels move** under a substrate-anchored convention; 40/79 runs read 0% alive at iteration 25 absolutely against 17/79 relatively. |
| B4 | Does the public channel close? | **Yes — 63/65 runs**, and it leads the regime decline by up to 200 iterations. See §3.5. |
| B5 | Is `p_env` recoverable? | **Provenance yes (100%), stored log-probs no.** The bias measurement is well posed; restoring the term costs a forward pass. §5 corrected. **B5b (the covariance) needs GPU and moves to Tier C.** |
| B6 | What does `thinking_frac` contain? | **Not what the name says.** It includes `<python>` (public; 68–80% of early non-post tokens on tool substrates) and cannot tell deliberation from an untagged post. §3.5 re-grounded on the environment's post accounting instead. |
| B7 | Is absorption measured while populations are healthy? | **Seeds yes, cross-env no.** Contrast +0.160 unchanged on the seed replication (0% of its measurements are post-decline); +0.199 → +0.024 on the cross-environment pool (31% are). |
| B8 | Does this depend on the base model? | **Strongly.** No Qwen3 agent ever reaches KL ≥ 1; mean drift 6–14× lower at matched iterations; and the Qwen3 populations are *healthier*. |

### Tier C — needs new runs

| # | Experiment | Scale | Answers |
|---|---|---|---|
| C1 | N=2 and N=4 replicates, 5 seeds × 2 substrates × 2 sizes, to iteration 300 with checkpoints at 100/150/200/300 | 20 runs, small | Task 2; the unreplicated corner of Figure 1 |
| C2 | Population-size decomposition (§6.3) | 15–20 runs | Task 3 — the causal question |
| C3 | Frozen-population, solo-agent, static-transcript controls | 6–9 runs | Task 7; 5MAX's remaining empirical objection |
| C4 | Public-tokens-only arm (`thinking_enabled: false`, or thinking typed as neither ACT nor OBS) | 2–4 runs | §3.5; ZLns Q4 |
| C5 | Restored-`p_env` arm (§5) | 2–4 runs | Removes the dark-room incentive from the objective, if §5 holds |
| C6 | One existing dataset extended past 300 iterations | 1–2 runs | Does divergence plateau; does the decline bottom out |
| C7 | **B5b — the action↔future-environment-entropy covariance**, moved here from Tier B: no logged artefact carries per-token log-probs, so it needs a scoring pass over checkpoints | GPU, no new training | 5MAX's remaining theoretical objection; §5.1 |
| C8 | **The private-prefix correction `I(z; y \| u)`**, pre-registered with falsification conditions in `scripts/tier_b/c7_env_likelihood_bias.py` (dry-run works; scoring pass unimplemented) | GPU, ~2k passes per run | Whether the naive cross-agent substitution is usable as-is |
| **C9** | **The environment-likelihood experiment.** Two arms on `study_group_ashbourne_gc` with the post tags blanked, so nothing an agent emits is discarded and `p_env` is exact rather than estimated. A logs the term without using it; B trains on the corrected return. Implemented and configured: [`configs/env_likelihood_bias/`](configs/env_likelihood_bias/README.md) | 8-agent runs, 10 configs, seed 1 of each first | How large the dropped term is, whether it changes the update direction, and what the corrected objective does |

### Tier D — dropped, and say so

- "Conventions not present in the substrate specification emerge, persist,
  transform, and spread between agents" — withdrawn in rebuttal. Title and
  abstract must follow.
- "Population size determines / controls which regime is accessible" — withdrawn;
  and §3.3 says even the monotone *shape* is unsupported.
- "Commitment is reliably accessed only at N=16" — withdrawn in rebuttal.
- "Committed agents absorb roughly six times more character information" as the
  headline — reallocate: the clean character-specific signal is in the low-drift
  stratum (A3/A5).
- "Negative PMI gaps consistent with full absorption with overshoot" for
  policy-collapse agents — withdrawn; both gaps go negative together.
- "The dark-room failure is progressively eliminated by population scale" —
  not supported (§3.3), and §3.5 offers a different mechanism.
- "Robotic Athanor agents' action distributions have degraded" — false as
  stated (§3.4); keep the conclusion, replace the reason.
- The upper-bound argument and the descent-direction conjecture — sign error,
  withdrawn in rebuttal, replaced by the exact bias.
- "The KL term prevents policy collapse" (§3.1 of the paper) — the sweep says β
  is the controlling knob and the KL anchor is close to orthogonal to collapse.

---

## 5. The objective: the bias, and the term we may not have to drop

The rebuttal's exact result is right and should go in the main body, not an
appendix:

> ∇L − ∇L_approx = E[ Σ_t (1−σ_t) ∇ln p_θ(x_t | x_<t) · Ẽ_t ]

with `Ẽ_t` the history-centred future environmental log-likelihood. The
surrogate is biased exactly to the extent that an agent's action is coupled to
the environmental surprise that follows it; the sufficient zero-bias condition is
mean-independence of that quantity from the action.

5MAX's remaining objection is that this **characterises** the problem without
establishing anything about our substrates. Two responses, in increasing order
of ambition.

### 5.1 Measure the coupling (Task 1) — tractable, and cheaper than it looks

The reviewer proposed branch resampling: fix a prefix, resample the focal
agent's turn K times, roll each branch forward, compare the environmental
uncertainty that follows. That works, but it needs `p_env` — which the paper
says cannot be evaluated.

**In this setting `p_env` is not a black box.** The environment of agent *i* is:
the persona prefix (deterministic, `ln p_env = 0`), structural markers
(deterministic, 0), tool output (deterministic given the code, 0), and **other
agents' posts — which are other LoRA adapters over the same frozen base, whose
per-token log-probabilities we already compute and store.**

**Correction (2026-09-01).** [B5](runs/_tier_b/b5_env_likelihood/NOTES.md) ran
this check and the answer is half yes. The pieces below are present, and
observation-token provenance resolves at **100%** — every post appears as a
contiguous interior token span in every other agent's context, at both early and
late checkpoints. **But the producer's own tokenisation is not the receiver's**
(42–88%, degrading with post-format breakdown), because the text on the thread is
the *extracted* post, re-encoded. So `act_log_probs_old` cannot simply be looked
up. The claim below that the term is "recoverable at close to zero marginal cost"
is **withdrawn**: restoring it means re-scoring the producing agent's adapter on
the receiver's tokenisation — one extra forward pass per agent per episode.
Feasible, since the adapters share the frozen base and are swapped in anyway; not
free. The *measurement* (B5b) is unaffected and well posed.

The pieces are verified present:

- `RolloutBatch.agent_id_mask` — "same shape — agent index; −1 for PROMPT/pad"
  ([`marlllm/types.py`](marlllm/types.py) line 102). **Every token carries its
  producer's identity**, observation tokens included.
- `RolloutBatch.act_log_probs_old` — per-token log-probs at ACT positions
  ([`marlllm/types.py`](marlllm/types.py) line 103).
- `_build_agent_trajectories` builds per-agent trajectories from one shared
  episode ([`marlllm/population.py`](marlllm/population.py) line 1550), so
  agent *j*'s ACT tokens are agent *i*'s OBS tokens.

So for the OBS tokens in *i*'s trajectory that carry `agent_id_mask == j`, the
generating conditional is *j*'s policy. **The dropped term is therefore
measurable** — B5 confirmed the provenance resolves — **but at the cost of
re-scoring, not a lookup** (see the correction above).

*One honest subtlety, confirmed by B5.* The formalism writes
`p_env(x_s | x_<s)` conditioned on *i*'s context; the actual generating
distribution was *j*'s policy conditioned on *j*'s context. These are the same
event with different conditioning sets — and B5 shows they are not even the same
token sequence. For measuring the bias this is fine; the paper should state which
object it is reporting.

**Now formalised.** [`environment_likelihood_derivation.md`](environment_likelihood_derivation.md)
works out the conditioning problem properly and the answer is better than
expected: **the adjustment is generator-specific, not observer-specific.** Under
full observability every observer of a post shares one `p_env`, because an
agent's own persona and private state are conditionally independent of what
others emit given the public thread. The single correction is to marginalise the
generator's private prefix, it equals `I(z; y | u)` exactly, it is non-negative,
it is **zero when the generator emits no private tokens** — which B6 measures as
the case at every early checkpoint — and it is bounded above by the term being
dropped, so substituting is never worse than dropping. Fog-of-war substrates
(`social_terminal`) are the exception and need separate treatment.

### 5.2 Restore the term (the ambitious version)

If 5.1 holds, the return can be computed as

> R_t = Σ_{s>t} σ_s ( ln p_env(x_s) − ln p_θ(x_s) )

with `ln p_env` taken from the producing agent's stored log-probs and zero for
deterministic environment tokens. This is **not** the plug-in estimate the
appendix rules out: that substitution uses *i*'s own model and cancels to zero.
Using *j*'s actual policy does not cancel — it yields a cross-agent
log-likelihood ratio, the excess surprise of *i*'s world model relative to the
true generator.

The consequences, if it works, are large:

- The gradient bias is zero by construction over agent-produced tokens, which is
  most of the entropy in the observation stream.
- **The dark-room incentive disappears from the objective.** The omitted term
  *is* the environmental entropy; restoring it makes the agent indifferent to how
  much its co-agents say and sensitive only to how well it predicts them. §3.5
  says the closing of the public channel is that incentive expressing itself, so
  this arm has a specific, falsifiable prediction, now with a measured target:
  **under the restored return, `post_success` should not fall from 0.99 to
  0.28.**
- It converts the theory section from "here is a bias we characterise" to "here
  is a bias we characterise, and here is the substrate class in which it can be
  removed" — which is a contribution rather than a caveat.

*Risks, stated up front.* It is not free — one extra forward pass per agent per
episode under the producing agent's adapter (B5). The return becomes a difference
of two large noisy quantities, so variance may rise sharply and the value
baseline may need rework.
Frozen or API agents have no stored log-probs and would need a fallback. And it
changes the objective, so nothing measured under the old return transfers — this
is a new experiment, not a re-analysis. **Do not make the paper depend on it.**
Run 5.1 regardless; treat 5.2 as a section we add if it works and drop if it
does not.

---

## 6. Task-by-task

### 6.1 Characterise the bias and measure the covariance
§5. The join is verified — provenance resolves at 100%
([B5](runs/_tier_b/b5_env_likelihood/NOTES.md)) — so what remains is a GPU
scoring pass computing the covariance per substrate and per population size.
This is C7, not a Tier-B analysis. **Prediction worth pre-registering:** the
coupling should be larger in small populations, since one agent's marginal
influence on aggregate future entropy falls as N grows — which is the rebuttal's
own hypothesis and would tie the bias directly to the population-size result.

### 6.2 Replicate N=2 and N=4
5 seeds × {2, 4} × {ashbourne_gc, strathearn_server} = 20 runs, to iteration 300,
checkpoints at 100/150/200/300 so the horizon is measurable rather than assumed.
These are the cheapest runs in the project. **Do not stop them at 150** — the
existing replicates do, and that is why they cannot speak to §3.1.

### 6.3 Population size, causally
Manipulation is not the problem; we already vary N. The problem is that N varies
three things at once: the number of *distinct interaction partners*, the number
of *gradient updates per agent per iteration*, and the *context composition*.
The 32-agent run scaled episode length to hold per-post budget fixed, which moved
the second of these — so it is not a clean size contrast either.

A design that separates them:

| arm | N | partners per agent | updates/agent/iter | isolates |
|---|---:|---:|---:|---|
| baseline small | 4 | 3 | high | — |
| baseline large | 16 | 15 | low | — |
| **large, restricted pairing** | 16 | 3 | low | partner count, holding N |
| **small, rate-matched** | 4 | 3 | low | update rate, holding N |
| **frozen majority** | 16 | 15 (14 frozen) | low | co-adaptation, holding N and partners |

`pairing_strategy` currently offers `random_no_self`, `random_with_self` and
`round_robin` ([`marlllm/population.py`](marlllm/population.py) lines 26–28).
The restricted-pairing arm needs one new strategy — a fixed partner graph with
degree *k*. That is a small, well-scoped addition and it is the arm that makes
the claim causal: **if outcome tracks partner count rather than N, "population
size" was never the variable.**

5 seeds per arm. This is the single most valuable block of compute on the list.

### 6.4 Clean up code for release
The release is already promised in the checklist and its absence was noted by
5MAX. Two things the retrospective adds: **`runs/**` is now out of git with
`runs/**/*.md` back in**, so the write-ups are releasable without the 28 TB; and
the config `STATUS:` lines mean a reader can tell which configs describe runs
that happened. One live defect to fix before release: one Phase A config was
edited after its run and no longer names the model that ran
([`runs/phase_a_substrate_validation/02_ccsm_focal_competitive/NOTES.md`](runs/phase_a_substrate_validation/02_ccsm_focal_competitive/NOTES.md)).

### 6.5 Rename the regimes
Current names assert mechanism ("committed", "collapse") and one of them is
substrate-dependent (§3.4). Name the *measurement*, on two axes plus a flag:

| current | proposed | why |
|---|---|---|
| coherent low-KL | **near-reference** | describes position, not virtue |
| committed coherent | **displaced** (predictive) | "committed" asserts a mechanism the dose-response result withdrew |
| mid-range | **marginal** | unchanged in substance |
| policy collapse | **narrowed** | entropy < 0.2; drop the high-entropy branch from the definition and state it never fires (max observed 1.59) |
| text degenerate | **unmodelled** | and define it *relative to the substrate's iteration-0 perception loss* (§3.4), so it means "worse than untrained" |

Add the new one from §3.5: **withdrawn** — the agent's turns have stopped
delivering well-formed public posts, regardless of the other three axes. B4 shows
the instrument already exists (`tool/post_success_per_ep`) and fires 100–200
iterations before any current regime does. Do **not** define it from
`thinking_frac`, which B6 shows is a different quantity.

### 6.6 What we can claim about enabling cultural emergence
§2 and §4. In one sentence for the abstract: *we establish that the enabling
state is reachable but transient, that what is acquired is weight-held and
character-specific, that it is portable between populations, that parallel
populations' policies diverge from a clean baseline — and that the objective as
implemented contains an incentive that closes the public channel, which we
identify, measure, and can remove.*

### 6.7 Isolate co-adaptation
`frozen_agents` already exists and is honoured by the trainer
([`marlllm/config.py`](marlllm/config.py) line 31,
[`marlllm/trainer.py`](marlllm/trainer.py) line 70) — Phase A used it for the
focal-vs-frozen design. **The frozen-population control is a config change, not
a code change.** Three arms:

- **Frozen majority** — one focal agent trains, N−1 frozen at base. Social
  diversity without co-adaptation.
- **Solo** — N=1 against a replayed transcript. No other agents at all.
- **Static transcript** — train on logged episodes from a finished run. No
  interaction, but the same data distribution.

Run all three on one substrate with the absorption and policy-distance
measurements attached. Together they answer: does absorption need other agents,
does it need *adapting* other agents, does it need interaction at all. This is
5MAX's largest remaining empirical objection and it is cheap.

### 6.8 Articulate the absorption controls properly
Restate the swap table with the stratification carrying its own caveats:

- **near-reference (n=32)** — own +0.073, swap −0.002, differential +0.076. The
  clean result. Swap pinned at zero throughout is the signature nothing else
  predicts. **This is the headline, not the committed stratum.**
- **displaced (n=12)** — own +0.245, swap +0.193, differential +0.051. Real but
  ~21% character-specific.
- **narrowed (n=8)** — both negative, differential ≈ 0. Withdraw "overshoot".
- **unmodelled (n=23)** — differential +0.381, and **not interpretable**: this
  stratum pools Robotic Athanor, which never learned (§3.4), with Ashbourne
  agents that were alive early and collapsed late. Two populations, one label.
  Split it or drop it from the table.
- **environment slot** — a distinct component at ~half the magnitude. New, clean,
  and it means the submitted PMI gap was bundling two processes.

Add the honest scope line 5MAX asked for: this is a likelihood comparison, not an
identification decision, and it does not show that an unprompted agent behaves in
character during interactive rollouts. If we want that, it is a prompt-ablated
rollout scored by a held-out classifier — cheap, and it would close ZLns's Q3.

---

## 7. What a referee will find if we do not say it first

We are committing to release code, traces and the analysis pipeline. Everything
in §3 is then findable in an afternoon. In rough order of how badly it lands if
someone else finds it:

1. **The iteration-200 evaluation point sits before a universal decline** (§3.1).
2. **One configuration, two runs, opposite outcomes** (§3.2) — against a figure
   with one run per cell.
3. **The untrained model is above the degenerate threshold on two substrates**
   (§3.4), so "8/8 text-degenerate at every measured iteration" partly means
   "never started".
4. ~~The policy-distance headline is not restricted to coherent agents~~ —
   **closed by [B1](runs/_tier_b/b1_policy_distance_by_regime/NOTES.md).** Report
   the restricted version and the Margin Notes sensitivity and this is a
   strength rather than an exposure.
5. **The public channel closing** (§3.5), which is in the `tool/post_*` metrics
   of every run we release, and is already half shut at the iteration we
   evaluate.
6. **The absorption contrast is horizon-driven on the cross-environment pool**
   ([B7](runs/_tier_b/b7_absorption_horizon/NOTES.md)) — +0.199 falls to +0.024
   on healthy-window measurements. Lead with the seed replication.
7. **No Qwen3 agent ever reaches KL ≥ 1**
   ([B8](runs/_tier_b/b8_model_generation/NOTES.md)), so the stratum the
   absorption result lives in may be a property of one base model.

Each of these is a better paper if we report it and a worse one if a reviewer
does. Four of the five are results.

---

## 8. How far to trust the extended analyses

You said you do not fully trust them. Calibration, given what the run data shows:

| experiment | verdict |
|---|---|
| **1 — swapped-character control** | **Trust.** Clean construction, real n, and the interpretation is appropriately self-critical. One repair: the `text_degenerate` row pools two different populations (§3.4) and should not be reported as one stratum. |
| **2 — threshold sensitivity** | **Trust.** Mechanical, exhaustive, and the conclusion is stated against interest (the taxonomy does labelling, not contrast work). |
| **3 — PMI decomposition** | **Trust the character/environment half; the history half correctly reports its own failure.** The iter-25 baseline of +1.10 nats is a real disconfirmation and the report says so. Cite the env-specific result; do not cite the history swap for anything. |
| **4 — trajectory KL** | **Trust the construction and the headline.** Baselines pass (diagonal 0.000000, iter-25 ≈ 0.001–0.008). Note in its favour: the measurement window (25–150) sits *inside* the healthy window, so it is not contaminated by §3.1. |
| **5 — diagnostics** | **Trust the retraction; trust the replacement headline with one correction.** The retraction of "committed agents converge" is well made. [B1](runs/_tier_b/b1_policy_distance_by_regime/NOTES.md) has now supplied the missing control: restricting to characters alive in both populations changes the end value by 0–5% on Inner Loop and Ashbourne and by **−45% on Margin Notes**, whose 1.359 is the least trustworthy number in the D3 table. Report `both_alive`. |

The general pattern: the analyses are honest about their own failures, which is
why they are usable. Their weakness is not fabrication, it is that they were all
computed at a fixed horizon and stratified by labels assigned at that horizon,
in a project where the horizon turns out to matter more than anything else.

[B7](runs/_tier_b/b7_absorption_horizon/NOTES.md) has now put a number on that
weakness where it bites hardest: on the cross-environment pool, 31% of the PMI
measurements are taken while the population is already below its own best, and
dropping them takes the committed-vs-coherent contrast from +0.199 to +0.024. The
Inner Loop seed replication has none — its window ends before the decline — and
its contrast is unchanged. **That is the dataset to build the paper on.**

---

## 9. Section-by-section rewrite

**Title.** Drop "Cultural Emergence" as the object. Something like *Preconditions
for Cultural Emergence in Populations of Surprise-Minimising Language Models*,
or a title that leads on the objective and its measurable bias.

**Abstract.** Delete "develop emergent behavioural conventions through
co-adaptation alone" and "Conventions not present in the substrate specification
emerge, persist, transform, and spread between agents". Replace with the four
preconditions, the transience result, and the bias. State the negative
(sanctioning) in the abstract — it is what makes the framing credible.

**§1 Introduction.** Keep the pre-training analogy; it is the paper's best idea
and no reviewer attacked it. Replace the two "enabling conditions" with the four
preconditions and add: we also identify an incentive the implemented objective
introduces, and show its behavioural signature.

**§3.1 Objective.** Promote the exact gradient bias from appendix to main body.
Remove the upper-bound claim and the descent-direction conjecture. State plainly
that the implemented objective is a **biased surrogate whose faithfulness depends
on a measurable coupling**, and report the measurement (B5). Correct "the KL term
prevents policy collapse" — the sweep says β does.

**§4.1 Experiment 1.** Recompute with substrate-relative thresholds (§3.4) and
report regime composition **as a function of iteration**, not at 200. Add the
transience result (§3.1) as a finding. Remove all causal language about
population size; report the grid as it is (§3.3), with the two claims that
survive. Rename regimes (§6.5). Add the reproducibility measurement (§3.2) as a
methods-level result.

**§4.2 Experiment 2.** Lead with the swap control and the near-reference
stratum. Demote the 6× ratio. Fold in the environment decomposition. Add the
scope sentence about likelihood-vs-identification.

**New §4.3 — population specificity.** Trajectory KL and policy distance with
the `both_alive` restriction reported as primary (B1 done). State the Margin
Notes sensitivity in the text, not a footnote: −45% is large enough that a
referee who finds it unannounced will not believe the rest.

**New §4.4 — portability.** The migration experiments (§2, item 4). This is new
material at zero compute cost and it is the most culture-shaped result we have.

**New §4.5 — the public channel.** §3.5, on the environment's own post
accounting (**not** on `thinking_frac`, which B6 shows measures nothing of the
kind). Present as the third failure mode, tied to the derivation, and make the
lead time the point: it fires 100–200 iterations before any current regime label
moves. With the ablation (C4) if it runs in time.

**§5 Discussion.** Replace "population size and regime accessibility are the same
finding from two angles" — they are not (§3.3). Keep the policy/world-model
unification argument; it is good and unchallenged. Replace the Robotic Athanor
paragraph's reasoning (§3.4).

**Limitations.** Add, in this order: the enabling state is transient and our
evaluation point is inside the window; the public channel is already half shut at
that point; the pipeline is not reproducible at fixed seed and the
population-level n required is larger than we used; **the central stratum does
not appear at all on Qwen3, so the commitment result may be a property of one
base model** (B8); the cross-environment absorption contrast is horizon-driven
and only the seed replication is clean (B7); no data above 7B; the co-adaptation
controls (if C3 has not run); and that we have not demonstrated conventions, only
looked for them with a named instrument and not found them.

**Citations.** Add MindForge (ZLns). Fix the bibliography style.

**Checklist.** Item 7 currently answers "No" to error bars on the grounds that
small n makes parametric tests inappropriate. With C1 and C2 that answer changes
to Yes, at the population level. Item 5 changes to Yes on release.

---

## 10. Ordering

Nothing below the line depends on new compute except where marked.

**Done, no compute:** all eight Tier-B analyses
([`runs/_tier_b/`](runs/_tier_b/README.md)). What remains at zero compute is
drafting the reframe (§9) against Tier A, which is now 20 claims.

**Next, small compute:** C1 N=2/N=4 replicates (20 small runs) · C3
co-adaptation controls (6–9 runs, config-only — `frozen_agents` already exists) ·
C4 public-tokens-only ablation (2–4 runs, and B4 has given it a measured target
to move) · C7 the covariance measurement (GPU scoring, no training).

**Then, the block that decides the paper:** C2 population-size decomposition
(15–20 runs, needs one new pairing strategy).

**Superseded by C9.** C8's estimator turns out to be ill-posed on a substrate
with private thinking: contexts persist, so the latent is the accumulated
private history and marginalising it is a filtering problem that degenerates
over an episode (`environment_likelihood_derivation.md` §6a). C9 removes the
private channel instead, which makes the correction exactly zero and `p_env`
computable rather than estimable. Run C9; keep C8 only if a *thinking* substrate
ever needs the number.

**Optional, high upside:** C5 the restored-`p_env` arm — now the B arm of C9. §5.1 has verified; the open risk is now §8 of the derivation — the corrected
return is a proper divergence whose optimum is flat across worlds, so the action
gradient may self-extinguish where the surrogate's does not. Never on the
critical path.

**Sequencing note.** C1 and C3 are independent of everything else and are the
cheapest runs in the project; start them while the Tier B analyses run. C2 is
the long pole. If compute is short, C2 beats C5: without it the population-size
claim stays associational, which is survivable — but it is the claim the paper
has been asked twice to justify.
