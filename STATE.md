# State of the project

A living summary: what we believe, how strongly, what we are running, and what
we still do not know. **No numbers or derivations here** — every claim points at
the document that carries the evidence.

How this document is kept true is [`PROCESS.md`](PROCESS.md). What the framework
is and how to run it is [`README.md`](README.md). The chronology is
[`runs/TIMELINE.md`](runs/TIMELINE.md). What we intend to claim in the
resubmission, and what has to be answered first, is [`REWRITE.md`](REWRITE.md).

Last updated: 2026-09-04 (seventh update: the novelty-search ablation).

---

## Confidence key

| | meaning |
|---|---|
| **established** | multiple runs, or one run with a controlled contrast and adequate sample. Would be surprising to overturn. |
| **likely** | one clean measurement, or several suggestive ones. Believe it, but it could move. |
| **provisional** | one run, confounded, or small sample. Treat as a working assumption. |
| **suspected** | a reading of the data we have not tested directly. |

**Novelty markers.** 🆕 marks what was added or materially changed in the most
recent update, 🔶 the update before it, 🔸 the one before that, and unmarked is
settled. Markers are demoted on every update — see `PROCESS.md`.

The 2026-09-01 back-fill has now aged out of the markers. This document was written
retrospectively that day, so those entries mean specifically: **derived from the
runs' own metrics during the back-fill, and not stated in any report written at
the time.** Everything unmarked comes from the analysis reports under
`runs/cultural_emergence/_*`.

---

## What we have learned

Organised by subject, not by when we found it. **A new finding goes into the
category it belongs to. If none fits, the categories are wrong and get revised —
never appended to.**

### What the objective actually optimises

*The framework's central question. CCSM has no reward function, so what it
maximises and what we want are related only by argument.*

- **Surprise minimisation and task success are dissociable, and on the one
  substrate where both were measured they came apart completely.** Perception
  loss fell more than tenfold while the logged task success rate stayed at
  exactly 0.000 for every iteration of both runs. — *established* —
  `runs/negotiation_population_experiments/*/NOTES.md`
- **The perception term does the world-model learning; the action term moves
  the policy and collapses its entropy.** Perception-only reaches the same
  perception loss as the full loss with four to ten times less KL from base and
  with entropy intact. — *established* —
  `runs/phase_a_substrate_validation/03_perception_only_*/NOTES.md`
- **The full loss is seed-unstable at the level that decides whether a policy
  survives.** The headline Phase A condition ends with entropy ≤ 0.003 at seed
  42 under both character prompts, and at 0.58–0.75 at seed 137. — *established*
  — `runs/phase_a_substrate_validation/NOTES.md`
- **A character prompt's predictive value over an agent's own actions shrinks
  over training, and for committed agents it goes negative** — the prompt
  becomes worse than uninformative. — *established* —
  `runs/cultural_emergence/_absorption_by_regime/REPORT.md`
- **That shrinkage is absorption of *this* character, not generic drift from
  base — in the coherent stratum.** Own-vs-none shrinks while swap-vs-none stays
  at zero throughout. — *established* —
  `runs/cultural_emergence/_absorption_swap_by_regime/REPORT.md`
- **In the committed stratum it is mostly generic drift**, with a real but
  smaller character-specific component on top. — *likely* — same
- **Environment description is absorbed as well as character description, by a
  comparable amount.** The published PMI gap bundles both; each carries about
  half the load. — *likely* —
  `runs/cultural_emergence/_pmi_decomposition/REPORT.md`
- **The implemented return rewards shortening the observation
  stream, and its behavioural signature is now measured.** `G_t` sums surprise
  over *future observation tokens*, so its magnitude depends on how many follow
  as much as on how surprising each is. The fraction of turns delivering a
  well-formed public post falls in **63 of 65 runs** — on the reference
  population from 99% to 28%, with 28% of turns delivering nothing at all. —
  *established* — `runs/_tier_b/b4_public_channel/NOTES.md`
- **`p_env` is not a black box in this substrate class, but
  restoring it is not free.** Observation-token provenance resolves at 100% —
  every post appears as a contiguous token span in every other agent's context —
  so the dropped term is measurable. The producing agent's own tokenisation is
  *not* the receiver's, so its stored log-probs do not align: restoring the term
  costs a forward pass per agent per episode, not a lookup. — *established* —
  `runs/_tier_b/b5_env_likelihood/NOTES.md`
- **The experiment built to measure the dropped environment
  term measured nothing, because no-tag mode stripped the generation while the
  provenance matcher required the unstripped decode.** Any surrounding
  whitespace missed, so no observation token carried `ln p_env`; and because
  the corrected return is selected only after that check, **arm B silently
  trained on the surrogate** and is a duplicate of arm A. Fixed, with
  regression tests; both arms need re-running. — *established* —
  `runs/env_likelihood_bias/NOTES.md`
- **A run can declare an objective it is not training on.**
  Nothing in six hours of either arm's output said the corrected return was
  unavailable — the signal was an *absence* of `envlp/*` keys. The path now
  raises instead. Worth assuming other declared-but-unused settings exist,
  alongside the four keys the code never read. — *established* — same
- **The environment conditional is observer-independent under
  full observability.** An agent's own persona and private deliberation are
  conditionally independent of what other agents emit, given the public thread,
  so every observer of a post shares one `p_env`. The correction is a property of
  the *generator's* turn, computed once per post rather than once per
  (post, observer) pair. — *established* (derivation) —
  `environment_likelihood_derivation.md`
- **The single adjustment needed is the mutual information
  between the generator's private prefix and its public post.** Exact pointwise
  identity; non-negative; zero when the generator emits no private tokens; and
  bounded above by the term currently being dropped, so substituting the
  generator's log-probs is never worse than dropping and is *exact* early in
  training. — *established* (derivation), unmeasured — same
- **The committed-vs-coherent absorption contrast is
  horizon-sensitive.** Intact at +0.160 on the Inner Loop seed replication, whose
  window ends before any decline; falls from +0.199 to +0.024 on the
  cross-environment pool when measurements taken during decline are dropped. The
  seed replication is the citable one. — *likely* —
  `runs/_tier_b/b7_absorption_horizon/NOTES.md`
- **The perception term does the learning and the action term does the
  damage, and the separation is total on the reference substrate.**
  Perception-only ends 300 iterations with `alive` 1.00 on both seeds,
  perception loss 2.13 → 0.70, entropy intact and 61 of 64 turns delivering a
  well-formed post; the full loss manages 36 of 64 with all eight agents
  degenerate. — *established* — `runs/cultural_emergence/c10_term_ablation/NOTES.md`
- **The dark-room incentive is causal, not merely associated with the
  decline.** It reaches behaviour through the action term alone, and removing
  that term removes both the channel closure and the population failure.
  Isolating it instead — action-only — misses 24 to 30 of 64 turns entirely and
  drives perception loss to 4.2, above base. — *established* — same
- **Perception-only is a candidate stabilisation, not yet a result.** It
  moves the policy about six times less than the full loss (KL 0.22 against
  1.32), matching Phase A, so it is not vacuous — but whether it still *absorbs*
  needs the PMI scoring pass, which has not run. Until then it could be
  "don't train the policy" wearing a better outcome. — *provisional* — same
- **Absorption scales continuously with movement from base**, not in a step at
  any threshold: on the cleanest subset a linear model in KL beats a step model
  on R², AIC and leave-one-out error. — *established* —
  `runs/cultural_emergence/_regime_threshold_sweep_3d/REBUTTAL_NOTES.md`

- **Removing the normalisation destroys the run; removing the discount does
  nothing.** γ=1 with normalisation kept delivers 0.50 against the reference's
  0.56 — the earlier "helps survival" reading rested on `alive` and is
  withdrawn. γ=1 without it reaches perception loss
  **37**, entropy 0.00, and 62 of 64 turns delivering nothing — the pure
  dark-room outcome, worse than C10's action-only arm. — *provisional*, one
  seed each — `runs/cultural_emergence/c11_discount_normalisation/NOTES.md`
- **The un-normalised arms test the arm mis-scaling, not the derived
  objective.** At γ=1 the raw return is ~2300 against a perception loss of
  ~1.4, so the action arm enters the summed loss ~1600× larger and the run is
  action-only in all but name. This was predicted in the config headers before
  submission. Whether the derived objective trains with a *working* baseline is
  open until the PopArt arm reports. — *established* — same

- **A working baseline does not make the derived objective trainable, and
  the scale mismatch is an arm-weighting problem rather than a baseline one.**
  PopArt takes the value head from diverging at 1.45e8 to a stable O(10) loss —
  a seven-million-fold improvement, value kept in the return's units — and the
  run still ends at perception loss 39.6, entropy 0.00 and 7 well-formed posts
  of 64. A fitted baseline centres the advantage but cannot shrink it: its scale
  is the return's spread, which PopArt measures at σ = 1552. — *established* —
  `runs/cultural_emergence/c11_discount_normalisation/NOTES.md`
- **The return's scale is genuinely non-stationary**, so the per-batch
  standardisation was papering over something real: PopArt's running moments
  move μ 697 → 861 and σ 344 → 1552 across 300 iterations. — *established* —
  same

- **Reversing the return's sign produces the behaviour the framework was
  built to produce, on one of two seeds.** Maximising future observation
  surprise instead of minimising it — a one-key change — gives a population
  doing collaborative mathematics at iteration 300: 89% of posts math-bearing,
  mean 660 characters, zero malformed fragments, against the reference's 1 of 42
  and five fragments. The other seed does not, at 4 of 53. — *provisional*, one
  seed of two — `runs/novelty_search/NOTES.md`
- **The dark-room diagnosis is right about the mechanism and incomplete as
  an explanation.** Delivery holds above the reference through iteration 200
  under a reversed sign, then falls anyway — 1.00 to 0.67 and 0.39. Silence is
  not only about the direction the return points. — *likely*, two seeds — same
- **Entropy rises under a reversed sign** rather than collapsing, 0.75 →
  0.96 and 0.98 against the reference's 0.80 — the predicted direction for an
  objective that rewards unpredictable futures. — *likely* — same

- 🆕 (2026-09-15) **The surrogate return is the stated objective plus the
  entropy of the environment.** Per observation position
  `E[-ln p_theta] = KL(p_env ‖ p_theta) + H(p_env)`, so dropping `E_s` does not
  add noise — it adds a REINFORCE ascent step on `-H(p_env)`, a policy gradient
  whose reward is the negative entropy of the agent's own world. In a
  population that world is other agents, and they are plastic, so the term pays
  each agent to *manufacture* a quiet environment rather than merely to find
  one. — *established* (derivation), unmeasured in our substrates —
  `environment_likelihood_derivation.md`, `marlllm/config.py`
- 🆕 (2026-09-15) **`H(p_env)` cancels in a difference of any two
  cross-entropies, so the fix needs a second model and not `p_env`.**
  `H(p_env, p_theta^-) − H(p_env, p_theta) = KL(p_env ‖ p_theta^-) − KL(p_env ‖ p_theta)`
  for *any* reference, however poor a model of the environment it is. That
  makes the correction cheap where measuring `p_env` is not — B5 priced that at
  a forward pass per agent per episode — and it is why the frozen base the KL
  anchor already holds is a valid second reference. — *established*
  (derivation) — `marlllm/config.py`, `configs/learning_progress/README.md`
- 🆕 (2026-09-15) **Correcting the entropy term does not touch the length
  channel.** `KL(p_env ‖ p_theta)` summed over positions is still extensive in
  the number of observation tokens, so "post less, elicit less, pay less"
  survives any second-reference correction untouched. The two mechanisms need
  separate fixes, and a falling delivery rate is not evidence against the
  entropy one. — *established* (derivation) —
  `configs/learning_progress/README.md`

### The substrate — what a world must be for anything to run at all

*Not a research question. These are preconditions: get one wrong and every
measurement downstream is void.*

- **Chat-format context management is the single largest lever in the project's
  history.** Without it, observation and action tokens reach the model as one
  undelimited stream with no signal for which it generated. 67 runs are
  invalidated by its absence and another 79 by getting its token positions
  wrong. — *established* — `runs/no_context_management/README.md`,
  `runs/old_delimiters/README.md`
- **The character prompt has to be inside the perception loss.** Typed `PAD` it
  is excluded from every loss term, and the thing absorption measures is then
  not in the objective. — *established* — `runs/prompts_as_padding/README.md`
- **Information-returning verbs must end the turn, or the agent invents what
  it has not been told.** An actor that keeps generating after committing to a
  read confabulates the result of a read that failed. Nothing in the metrics
  showed this; it was found by reading traces. — *established* —
  `runs/social_terminal/run02_8agent_7B_ashbourne_school/NOTES.md`
- **Framework-inserted scaffolding must be typed OBS, not ACT.** Only
  model-generated tokens are actions; inserted turn-close tokens in the PPO
  ratio are not. — *established* — `runs/social_terminal/run06_8agent_7B_ashbourne_school/NOTES.md`
- **Robotic Athanor is unlearnable under this objective, and not because it
  degrades.** Perception loss is flat at its initial value across 100 iterations
  at both 2 and 8 agents, with entropy neither low nor falling. This corrects
  the "action distributions degrade into low-entropy gibberish" reading in the
  absorption report, and supports that report's conclusion by a different route.
  — *established* — `runs/cultural_emergence/run7_8agent_7B_robotic_athanor/NOTES.md`
- **One population cannot hold two environments at this budget.** The
  dual-environment arm is dead from iteration 25 and never recovers above 25%
  alive, where the single-environment run on the same roster is 88% alive at
  iteration 200. — *likely* —
  `runs/cultural_emergence/run9_8agent_7B_ashbourne_dual_env/NOTES.md`
- **Substrates differ in starting perception loss by nearly a nat**, so
  absolute perception loss is not comparable across substrates without the
  baseline. — *established* — `runs/cultural_emergence/run8_16agent_7B_study_group_strathearn_server/NOTES.md`
- **`deal_or_no_deal` at 1.5B is a floor-effect substrate.** The pretrained
  model reaches agreement on 3.1–4.7% of 64 episodes, so an agreement-rate
  criterion cannot show an improvement of any size. — *established* —
  `runs/phase_a_substrate_validation/01_pretrained_baseline_*/NOTES.md`

### What populations become

- **Populations end in one of a small number of regimes**, described by KL from
  base, perception loss and policy entropy. The taxonomy is a *descriptive
  partition of a continuous, bimodally-occupied space* — "committed" names the
  upper mode of an outcome distribution, not a mechanistically distinct phase. —
  *established* — `runs/cultural_emergence/_regime_threshold_sweep_3d/REBUTTAL_NOTES.md`
- **The labels are non-arbitrary because the covariate is bimodal**, with the
  threshold sitting in a sparse valley: a mixture on log KL selects two
  components at 0.13 and 2.84 nats. — *likely*; the bimodality evidence rests on
  the inner-loop seed replicates — same
- **Failure comes in two kinds that must not be merged.** `policy_collapse`
  agents emit readable text over a policy narrowed to near-determinism;
  `text_degenerate` agents emit broken text. — *established* —
  `runs/cultural_emergence/_population_figures/trace_verification_iter200.md`
- **The high-entropy branch of `policy_collapse` has never fired.** All 7
  policy-collapse agents in 220 labelled agents classify via `entropy < 0.2`;
  the maximum entropy observed anywhere is 1.59. — *established* —
  `runs/cultural_emergence/_regime_threshold_sweep_3d/REPORT.md`
- **Population size and substrate are jointly associated with regime outcome.**
  Large-population non-collapse replicates across 15 independent populations;
  small-population outcomes vary by substrate and are single observations. The
  causal reading is not supported. — *established* —
  `runs/cultural_emergence/_seed_replicate_regimes/REBUTTAL_NOTES.md`
- **Above two agents, the population-size grid is not monotone at all.** On
  Ashbourne GC the alive fraction at iteration 300 runs 0.00, 0.75, 0.00, 0.88,
  0.62 for 2, 4, 8, 16 and 32 agents; Strathearn disagrees cell by cell. What
  the grid supports is that **two-agent populations end dead on both
  substrates** and that 32 agents is worse than 16 — not "larger is better". —
  *established* from the grid; every cell is one run —
  `runs/cultural_emergence/run8_16agent_7B_study_group_ashbourne_gc/NOTES.md`
- **Which individual agent commits is seed-random, not a property of the
  character sheet.** With rosters held fixed across seeds, Fleiss κ ≈ 0. The
  reproducible object is the population-level composition. — *established* —
  `runs/cultural_emergence/_seed_replicate_regimes/SUMMARY.md`
- **What a population becomes depends strongly on the base
  model.** At matched iterations on one substrate with identical regularisation,
  **no Qwen3-4B or Qwen3-8B agent ever reaches KL ≥ 1 from base**, where 27 of
  182 Qwen2.5-7B agents do; mean drift is 6–14× lower and the populations are
  *healthier*. "Commitment" — the stratum the absorption result lives in — is
  not something the objective reliably produces. — *likely* —
  `runs/_tier_b/b8_model_generation/NOTES.md`
- **More action tokens per episode accelerates policy collapse.** Both
  two-turn arms end far below their single-turn counterparts, and the failure
  mode is entropy collapse rather than text degeneration. — *likely* —
  `runs/cultural_emergence/run11_16agent_7B_study_group_ashbourne_gc_2turns/NOTES.md`

- **The N=2 and N=4 cells of the population-size figure are seed noise.**
  Five seeds a side: within-cell spread of `alive` is 0.44 against between-cell
  gaps of 0.20 and 0.35 at iteration 200, and 1.00 against 0.05 and 0.15 at 300.
  No continuous metric separates the sizes (all p ≥ 0.15, exact
  Mann-Whitney). The pre-registered falsification condition fired. —
  *established* —
  `runs/cultural_emergence/c1_small_population_replicates/NOTES.md`
- **Two published cell readings were seed artefacts.** "Dead from iteration
  150 onward" for 2-agent Strathearn becomes 0.50–1.00 alive at 150 across five
  seeds. — *established* — same
- **A single agent minimising surprise against a frozen population can
  still collapse completely**, so co-adaptation is not necessary for the
  failure. Whether it is *sufficient* is unresolved: three learners span 0.00 to
  1.00 well-formed delivery against a co-adapting reference of 0.56. —
  *likely* for the necessity half, *inconclusive* for the rest —
  `runs/cultural_emergence/c3_frozen_population/NOTES.md`

- **The N=16 reference spans `alive` 0.38–0.88 at iteration 300 across
  five seeds**, mean 0.70, where it was previously a single run at 0.88 — the
  top of the range. Comparisons drawn against that single seed were comparing
  against the best case. — *established* —
  `runs/cultural_emergence/c2_population_size/NOTES.md`
- ~~**Matching per-agent gradient volume downward gives the healthiest cell.**~~
  **Withdrawn.** It rested on `alive` 0.95; by delivery it is 0.263 against
  0.200, p = 0.810. — *withdrawn* —
  `runs/cultural_emergence/c2_population_size/NOTES.md`

- **Population size does not determine the outcome, at either end.** All
  five arms of the decomposition are in: N=4 against N=16 at matched gradient
  volume gives 0.263 vs 0.225 delivery, p = 0.722 on five seeds a side, and
  volume itself gives p = 0.810. With C1 having removed the small corner, the
  population-size claim has no support left. — *established* —
  `runs/cultural_emergence/c2_population_size/NOTES.md`
- **Co-adaptation degrades delivery, and it is the only variable in that
  design that separates.** Learners against a frozen majority deliver 0.475
  against 0.225 for agents in a co-adapting population of the same size,
  p = 0.032, every seed above the baseline's median. — *likely* — same
- **Partner count is the one size-adjacent variable worth more seeds**:
  0.316 at three partners against 0.225 at fifteen with N held at 16, but
  p = 0.135 is not a result at five seeds. — *suspected* — same

- 🔸 **The return's sign raises policy entropy at every population size.**
  **Eight of eight** novelty-search arms across N = 2, 4, 8 and 16 end above
  their own same-N reference, by 0.16–0.80 nats against a same-seed noise floor
  of 0.007; the references *lose* entropy at three of four sizes and only one of
  eight arms loses more than 0.12. MWU exact p = 0.028. Nothing else in the
  project moves entropy this reliably. — *established*, promoted from seven arms
  — `runs/novelty_search/NOTES.md`
- **Population size does not determine the outcome under maximisation
  either.** Delivery against N gives ρ = −0.098, p = 0.810, firing the
  pre-registered falsifier; the within-N seed spread accounts for the whole
  between-N range. Same verdict as C2 reached under minimisation, from the
  other direction. — *established*, 8 arms over 4 sizes — same
- **Under maximisation the outcome is seed-dominated.** Two N=2 runs
  differing only in seed end at delivery 0.926 and 0.325 — 3.5× the rig's
  delivery noise floor. — *established* — same
- 🔸 **The sign flip slows tool abandonment without reopening the channel,
  and its effect size does not replicate.** Both seeds end above the reference
  on python calls at iteration 300 (52.2 and 28.1 against 22.2) and both are
  well above it at 200, at an unchanged success rate and with real varied code —
  but the two seeds differ from each other by 1.9×, and seed 2's delivery ends
  *below* the reference (0.317 vs 0.383). Direction replicates, magnitude does
  not. **Corrects** the one-seed reading of "keeps the tool alive". —
  *provisional*, two seeds — same

- 🔶 **Released entropy gives interesting behaviour *and* noisy TV, at
  different points on a ladder, and the transition is compositional rather than
  global.** Read from the trace content of the 28 regularisation runs, which had
  never been read on content before. End entropy ~0.9–1.1: coherent. ~1.5–1.8:
  **mixed within one episode** — some agents holding coherent mathematics while
  others emit `beat, beat, beat! beatrice, absolute beatrice` or a bare
  `</?</?</?` loop. ~3.4: coherent **opening**, noise **tail** — real content
  followed by hundreds of repeated emoji. ~6.6–7.0: word salad. Openings survive
  longest, tails go first, and agents collapse individually rather than
  together. — *established* —
  `runs/cultural_emergence/kl_beta_sweep_8agent_ashbourne_gc/NOTES.md`
- 🔶 **The usable window is around end entropy 1.1–1.8 and no run has ever
  been aimed at it.** Above 1.8 degradation is visible; below 1.1 entropy has
  barely moved. — *likely* — same
- 🔶 **Agents comment on their peers' degeneration from inside the
  collapse.** At end entropy 6.6–7.0, amid word salad: "can someone tame this
  beast lang mess?", "figured since everyone else babbled cryptograms",
  "couldn't come up w/ cohesive concept". Enough coherence survives to observe
  that the channel has become noise. — *provisional*, two runs — same

### Training horizon — what happens if you keep going

*This whole section is new. It was not visible in any analysis written at the
time, because every published analysis evaluates at a fixed iteration.*

- **Forum populations peak around iteration 100 and decline from there.**
  Across a balanced panel of 44 cultural-emergence populations measured at
  iterations 25, 100, 150, 200 and 300, the mean alive fraction runs
  0.27 → 0.86 → 0.82 → 0.72 → 0.46. — *established* —
  `runs/_tier_b/b2_regime_by_iteration/NOTES.md`
- **No forum population has ever improved after iteration 150.** Of the 44
  measured at both 150 and 200, none rose; of the 44 measured at both 200 and
  300, none rose and 31 fell. 39 of 44 end below their own earlier peak. —
  *established* — same
- **The campaign's evaluation point sits one step before the decline.** The
  clearest single case is the 8-agent Ashbourne GC run: 88% alive at iteration
  200 and **0% at 300**. This reconciles two published tables that appear to
  contradict each other — they are reading different iterations of one run. —
  *established* —
  `runs/cultural_emergence/run7_8agent_7B_study_group_ashbourne_gc/NOTES.md`
- **A short horizon can make a failing substrate look sound.** Margin Notes is
  100% alive across four seed replicates that stop at iteration 150, and 50%
  alive in the one run that continued to 300. — *established* —
  `runs/cultural_emergence/run7_8agent_7B_margin_notes/NOTES.md`
- **The public channel closes long before the regime classifier
  registers anything** — by up to 200 iterations. On the reference population the
  channel is already half shut at iteration 200 while the alive fraction still
  reads 1.00. The taxonomy is a lagging indicator of this failure. — *likely* —
  `runs/_tier_b/b4_public_channel/NOTES.md`
- **The decline is not inherent to the objective.** Both social-terminal runs
  that reached iteration 300 ended alive, and one improved between 200 and 300 —
  the only multi-agent run in the project that does. — *likely*, two runs —
  `runs/social_terminal/run06_8agent_7B_ashbourne_school/NOTES.md`

- **`alive` is invalid wherever delivery is low: its dominant input moves
  with the failure it is meant to detect.** Perception loss is computed over
  observation tokens, so as agents stop posting the stream becomes
  deterministic scaffolding and `perc_loss` falls through the classifier's
  `< 1.7` gate. Within the 8-agent extension, delivery and perception correlate
  at r = +0.85 while `alive` moves the other way; across ten (run, iteration)
  points delivery and `alive` correlate at only r = +0.23, and the two
  references invert — `run7` delivers 56% of turns and scores 0.00, `run8`
  delivers 27% and scores 0.88. — *established* —
  `runs/cultural_emergence/c6_extended_horizon/NOTES.md`
- **The extended populations do not recover; they fall silent.** At
  iteration 900 the 8-agent reference delivers 5 well-formed posts in 64 turns
  while scoring `alive` 0.62, with four of five "alive" agents posting nothing.
  Whether the decline bottoms out is unanswered and not answerable with this
  classifier. — *established* — same

### Whether populations become *different from each other*

- **Parallel populations trained from different seeds on the same substrate
  produce trajectory distributions that diverge monotonically over training**, by
  two orders of magnitude between iterations 25 and 100, with a clean ≈0
  baseline at iteration 25. Replicated on three substrates. — *established* —
  `runs/cultural_emergence/_trajectory_kl_combined/REPORT.md`
- **The divergence holds at the policy level, not merely the realisation
  level.** Symmetric KL between per-character adapters at matched contexts grows
  ×60–×200 from iteration 25 to the end. This is the cleanest form of the claim
  and should be cited as primary. — *established* —
  `runs/cultural_emergence/_diagnostics_d2d3/REPORT.md`
- **Divergence does not plateau at any horizon we can measure.** — *likely* —
  `runs/cultural_emergence/_trajectory_kl_combined/REPORT.md`
- **RETRACTED: "committed agents converge across seeds, coherent agents
  diverge."** The strong inverse correlation (r ≈ −0.77) between per-character
  KL-from-base and cross-population contribution is a realisation-level
  artefact. At the policy level it collapses to near zero or slightly positive,
  and per-character contribution correlates +0.55 with the agent's own entropy
  deficit. The claim was replicated on three datasets and was still wrong. —
  *established* — `runs/cultural_emergence/_diagnostics_d2d3/REPORT.md`
- **The divergence is not an artefact of collapsed adapters, but
  the pooled magnitudes are inflated where failure regimes are present.**
  Restricting to characters alive in both populations changes the end-of-training
  policy distance by 0–5% on Inner Loop and Ashbourne and by **−45% on Margin
  Notes**. Growth from the iteration-25 baseline is ×27–×112 either way. Report
  the restricted version. — *likely* —
  `runs/_tier_b/b1_policy_distance_by_regime/NOTES.md`
- **Failure regimes must be carved out of any divergence claim.** Restricting to
  source-coherent characters raises the signal 20–40%. — *established* — same

### What an agent carries between populations

- **A migrated adapter learns the host substrate at the resident rate.**
  Migrant and resident perception losses move together in all 16 migration runs.
  — *established* — `runs/migration_experiments/NOTES.md`
- **Migration does not reset drift: a committed agent stays committed and gets
  more so.** All 7 migrants arriving above KL 2.0 drifted further; 8 of 9
  arriving below KL 1.1 did not. — *likely* — same
- **World model and policy identity are separable, and per-agent LoRA is where
  the second one lives.** — *likely* — same

### Norms and sanctioning

- **Sanctioning is detectable at high volume in these transcripts**, and it is
  overwhelmingly approval: disapproval runs at a median of 0.062 reactions per
  post against approval's 0.656. — *established* — `runs/sanctioning/NOTES.md`
- **The asymmetry is the base model's, not something the populations learned.**
  It is at its iteration-0 value before any training and shows no trend across
  300 iterations. — *likely* — same
- **A norm needs a sanctioning response that can go either way, and
  disapproval is at the floor.** — *suspected*; this is a reading of the Rung-0
  table, not a measured result — same

### Regularisation

All five entries below rest on the 20 cells of
`runs/cultural_emergence/kl_beta_sweep_8agent_ashbourne_gc/` and
`.../beta_zoom_kl0p5_8agent_ashbourne_gc/`, re-read on 2026-09-08 on the
delivery rate and on the *change* in entropy
(`scripts/analyse_kl_beta_grid.py`; the re-reading is written up in the sweep's
NOTES).

- **Neither regulariser touches the public channel *within* `kl_coef` ≤ 0.5.**
  Delivery falls from 0.98 to 0.50 in all 20 cells of the old grid, with no
  trend in `beta` (ρ = −0.25, p = 0.28) or `kl_coef` (ρ = −0.16, p = 0.49), and
  not one cell above 0.8. — *established*, over `kl_coef` ∈ [0, 0.5] only —
  sweep NOTES, the 2026-09-08 re-reading
- 🔸 **But it does above that, and this corrects the entry above, which was
  stated without its range earlier the same day.** At `kl_coef` 2.0–5.0 delivery
  is 0.519–0.975 (mean 0.722) against the old grid's 0.395–0.613 (mean 0.501),
  MWU exact p = 1.2 × 10⁻⁴. **But most of it is the pinned cells:** excluding
  `kl_coef` 5.0 the effect falls to +0.155, below the 0.173 noise floor, and at
  the only matched-ratio pairing it is +0.147, also below it. A group
  difference (p = 1.5 × 10⁻³), not a per-cell one. — *likely*, downgraded from
  *established* —
  `runs/cultural_emergence/entropy_anchor_ridge_8agent_ashbourne_gc/NOTES.md`
- 🔶 **RETRACTED: "a stronger KL anchor raises the safe ceiling on the
  entropy bonus".** `beta` and `kl_coef` act through roughly their **ratio** —
  Spearman Δentropy vs `beta`/`kl_coef` = +0.831, p = 7.8 × 10⁻¹¹, n = 21,
  better than `beta` alone (+0.755). `beta` 0.4 at `kl_coef` 2.0 is the *same*
  operating point as `beta` 0.1 at `kl_coef` 0.5. The ceiling did not rise;
  both axes were rescaled. The key comparison was never at matched operating
  point: at `beta` 0.2 the entropy change differs 4–8× between `kl_coef` 0.5
  and 2.0, and no cell in that family ever reached the entropy level at which
  damage appears (max +0.59 against the +1.19 it was meant to rescue). The
  intervention became smaller, not safer. — *established* —
  `runs/cultural_emergence/entropy_anchor_ridge_8agent_ashbourne_gc/NOTES.md`
- 🔶 **`beta` and `kl_coef` are one knob, not two, over most of their range.**
  Because they act through their ratio, raising both together cannot reach a new
  regime by construction — which is why the ridge design could not answer the
  question it was built for. — *established* — same
- 🔶 **WITHDRAWN: "the target regime is reachable" at (2.0, 0.4).** That
  cell ends at entropy 1.36, below the 1.5–1.8 band where degradation begins,
  so it never tested the trade-off. It is a low-entropy cell, not a solved
  high-entropy one. — *withdrawn* — same
- 🔸 **Delivery tracks how little the policy moved from base.** Spearman
  ρ = −0.503, p = 0.003 over all 28 regularisation cells. The least-drifted cell
  delivers 0.975; the most-drifted 0.395. So the anchor may buy the channel by
  suppressing learning rather than by training better. — *likely* — same
- 🔸 **`kl_coef` = 5.0 prevents learning outright.** KL from base 0.02 with
  entropy pinned (+0.09, +0.15); its delivery of 0.975 is the base model, which
  delivers ~0.98 untrained. Its numbers must not be cited as a training outcome.
  — *established* — same
- **The entropy bonus is a step function, not a dial.** For `beta` ≤ 0.05
  (9 cells) |Δentropy| ≤ 0.12 with the sign as often negative as positive; for
  `beta` ≥ 0.10 (11 cells) Δentropy ≥ +0.16. The groups separate completely,
  MWU exact p = 1.2 × 10⁻⁵. **There is no `beta` that raises entropy a little.**
  — *established* — same
- **Above the step, entropy and perception loss are locked together.** Spearman
  Δentropy vs Δperception = +0.853, p = 4.4 × 10⁻¹², n = 20. **The regime where
  policy entropy rises without the perception loss collapsing does not occur
  anywhere on this grid.** — *established* — same
- **`beta` ≥ 0.1 destroys the population, and `beta` = 0.5 catastrophically**
  (entropy above 6.5 nats, perception above 5.2). The cliff is below 0.2. —
  *established* — same
- **The KL anchor does fight the entropy bonus.** At `beta` = 0.10, Δentropy
  falls monotonically +1.04 → +0.92 → +0.79 → +0.16 as `kl_coef` goes
  0 → 0.01 → 0.10 → 0.50; the duplicate pair agrees on the last point to 0.01.
  Both terms act only at ACT positions, so they compete for the same
  distribution. The anchor stops protecting by `beta` = 0.20. — *established* —
  same
  - **CORRECTS** the previous entry "the entropy bonus destroys populations
    without moving them from base, so the KL anchor cannot protect against it",
    which was drawn from four cells at one `kl_coef`. **Withdrawn.**
- **The KL coefficient controls drift, and the entropy bonus.** Raising it from
  0 to 0.5 halves end-of-training KL from base; it also suppresses the entropy
  rise sixfold, per the entry above. It does not change delivery. — *likely* —
  same
- **`alive` does not track the channel.** Across the 20 cells it is uncorrelated
  with the well-formed post rate, ρ = −0.10, p = 0.66. Fourth independent
  confirmation. Do not cite an `alive` figure from these families without the
  delivery rate beside it. — *established* — same
- **No cell ranking among the surviving settings is supported.** The
  differences between the good cells are smaller than the spread between two
  runs of one identical configuration. The zoom's "best cell in the campaign"
  (`kl` 0.5, `beta` 0.05, alive 0.75, perception 1.75) is **withdrawn**: it ends
  at delivery 0.474, like everything else. — *established* — same
- **Removing both regularisers made one population *better*, not worse.** —
  *provisional*, one run each side —
  `runs/cultural_emergence/run7_8agent_7B_study_group_ashbourne_gc_no_reg/NOTES.md`
- **What raises policy entropy is the return's sign, not the regularisers.**
  Through the same window on the same substrate, `return_sign` = −1 gives
  Δentropy +0.22/+0.23 against the reference's +0.06 — roughly four times any
  safe `beta` — and on seed 2 perception loss *fell* 0.46 while entropy rose.
  The Δperception rests on one seed and only just clears the noise floor; the
  Δentropy does not. — *likely* for the entropy, *provisional* for the
  perception — `runs/novelty_search/NOTES.md`
- **Open: whether a stronger anchor licenses a larger bonus.** `kl_coef` above
  0.5 had never been run. Being tested at (2.0, 0.01/0.2/0.4) and (5.0, 0.4),
  two seeds each — `runs/cultural_emergence/entropy_anchor_ridge_8agent_ashbourne_gc/NOTES.md`,
  jobs 6404722–6404729, submitted 2026-09-08.

**The noise floor of this rig, which every entry above is read against.** The
old grid contains an accidental duplicate run at the **same seed**; the two runs
differ by 0.007 in Δentropy, **0.431 in Δperception** and **0.173 in delivery**.
So Δentropy carries weight from a single run and Δperception and delivery do
not — differences below ~0.4 and ~0.17 respectively are unreadable from one run,
anywhere in the campaign. — *established* — sweep NOTES

### Engineering and reproducibility

- **This pipeline is not reproducible at fixed seed.** One configuration was
  run twice by accident, differing only in run name, output directory and which
  GPU it landed on. The two runs track each other to about iteration 150 and
  then separate, ending at alive 0.875 and 0.000. — *established*, one direct
  demonstration — `runs/cultural_emergence/kl_beta_sweep_8agent_ashbourne_gc/NOTES.md`
- **Async rollout with `max_staleness` 1 under a PPO clip reproduces
  synchronous training.** Perception loss, entropy and KL all match the
  synchronous siblings. — *likely*, two matched pairs —
  `runs/cultural_emergence_qwen3_async/*/NOTES.md`
- **Sequence packing has never produced a training iteration.** Four
  submissions, zero metric records. — *established* —
  `runs/cultural_emergence_qwen3_packed/README.md`
- **`vllm_gpu_mem_util` 0.30 starves the trainer and OOMs the chunked loss;
  0.18 survives.** — *established*, three runs —
  `runs/social_terminal/run03_8agent_7B_ashbourne_school/NOTES.md`
- **`uv run` breaks the hand-built vLLM venv.** Use `.venv/bin/python`. —
  *established* — `README.md`
- 🆕 (2026-09-15) **PEFT selects an adapter's weights by plain substring, so
  any auxiliary adapter whose name contains a live adapter's name is written
  into that adapter's save file.** `get_peft_model_state_dict` filters on
  `adapter_name in key`, and `save_pretrained(selected_adapters=[...])` sits on
  the vLLM sync path after every optimiser step — so the rollout engine would
  have been handed a file holding two policies' weights. Caught before it ran;
  auxiliary adapters are now named from a hash of the live name, with a
  regression test. Any future second adapter must do the same. — *established*
  — `marlllm/agent.py`, `tests/test_learning_progress.py`
- 🆕 (2026-09-15) **`compute_loss_chunked` silently ignored `return_sign`.**
  The single-shot path has applied it since the novelty-search ablation landed;
  the chunked path never did, so a chunked novelty run would have trained the
  unflipped objective. **No run was affected** — every novelty config leaves
  `seq_chunk_size` unset — but it is the same failure class as the six dead
  keys and as arm B training on the surrogate: a setting the code accepts and
  one code path does not honour. Fixed. — *established* — `marlllm/loss.py`
- **A config in git can stop describing the run that used it.** One Phase A
  arm ran on the base model while its tracked config was later edited to name
  the Instruct model, confounding the prompt-swap diagnostic the design turned
  on. This is why every config needs a `STATUS:` line. — *established* —
  `runs/phase_a_substrate_validation/02_ccsm_focal_competitive/NOTES.md`
- **Asking for the 24 h maximum was costing more queue time
  than it saved.** Forty-six jobs at the `workq` cap reserved 1,080 node-hours
  for about 250 hours of work, making all of them unbackfillable; limits
  estimated from each reference run's own p90 per-iteration cost cut the
  reservation 60% with a 1.5× margin still in hand. Per-iteration cost is
  stable enough to plan from — 47 s at N=2 to 111 s at N=16 on 4 × GH200. —
  *established* — `PROCESS.md`, "Ask for the walltime the run needs"
- **Counting non-empty thread text overstates the public channel by up to
  a factor of two.** An unclosed `<post>` block still has its text extracted
  onto the thread, so only `tool_counts_per_agent`'s `post_success` measures
  what B4's claim is about — on `run8` at iteration 300 the two read 0.27 and
  0.80. Using the proxy hid the frozen-majority effect entirely. — *established*
  — `runs/cultural_emergence/c2_population_size/NOTES.md`
- **A population-mean metric is meaningless when most of the population is
  frozen.** With 7 of 8 or 14 of 16 members frozen at base, the population
  post-success rate is the freeze fraction — 0.88, 0.94 — and says nothing about
  the learners, whose own rates are 0.00–1.00 and 0.25–0.62. `pop/n_frozen` is
  logged every iteration so the denominator is visible. — *established* —
  `runs/cultural_emergence/c3_frozen_population/NOTES.md`
- **Six config keys the code does not read**, and the last two
  are worse than the rest. `thinking_enabled`; `frozen_agents`, in the
  negotiation trainer only; the three `alpha_*` weights, so every forum run
  trained at 1/1/1; `pairing_strategy`, inert on forum substrates; and now
  **`gamma` and `normalise_returns`**, passed by none of the five trainers. The
  first four made an experiment *inexpressible*; these two silently fixed the
  objective's shape for every run in the project's history. All six were found
  by trying to write an experiment rather than by reading code. — *established*
  — `marlllm/config.py`,
  `configs/cultural_emergence/c2_population_size/README.md`
- **The dropped environment term is large but does not change
  the update ordering.** Recovered post hoc from arm A's checkpoints: ~43% of
  the return's magnitude at four checkpoints, rank correlation 1.000 throughout,
  because a post is 6–36% less surprising per token to its author than to its
  readers so the correction is same-signed and smaller. That margin narrows
  toward parity over training. — *likely* —
  `runs/_tier_b/c9_post_hoc_env_term/NOTES.md`
- **The discount is not in the derived objective, and it
  front-weights credit inside every post by ~48×.** γ=0.99 is applied
  *positionally per token*, a 100-token horizon against a 128–384-token post.
  On a real 386-token post the future-observation mass at an action token runs
  2.0 at the first token to 97.1 at the last; at γ=1 it is 2057, flat.
  `surprise_minimisation_derivation.md` §6.2 introduces γ only as variance
  reduction "at the cost of bias", so γ=1 is the derived return. — *established*
  — `marlllm/loss.py` `_compute_returns`
- **Return normalisation is not the free choice it is in RL,
  because the action term is summed with a supervised one.** In RL only the
  ordering of advantages matters and scale is absorbed by the learning rate;
  here `α_perc·L_perc` is in nats while the normalised advantage is in standard
  deviations, so `α_perc = α_act = 1` sets an exchange rate nobody chose. The
  gain is 1/std, which *rises* as returns flatten, so the action term holds full
  weight against perception as its own signal vanishes. — *likely*, from the
  code and the scale mismatch; `return_norm_gain` is now logged to test it —
  `marlllm/loss.py`

### Measurement — what our instruments do and do not show

*Read the first entry before citing any `alive` number.*

- **Measuring perception loss on post content only does not
  repair the classifier — predictability is simply the wrong target.** Content
  is *more* predictable than scaffolding at 12 of 13 checkpoints, and
  content-only loss is lowest where delivery is lowest (0.78 nats/token at
  delivery 0.234), because a degenerate `</?` loop is content and is maximally
  predictable — at iteration 900 one such post is 82% of everything delivered.
  Any loss over a stream the agents control can be reduced by emitting less or
  by emitting something trivially repetitive. — *established* —
  `runs/_tier_b/content_perception/NOTES.md`
- **The environment's framing is not deterministic**, because
  it carries the speaker's name — which of eight agents posts next is the least
  predictable thing in the stream. Scaffolding NLL runs 1.8–2.7, above content
  almost everywhere. Any future token split has to account for that rather than
  treat framing as free. — *established* — same
- **The regime classifier cannot be trusted at low delivery,
  and this is the second time its inputs have been fooled by the failure they
  measure.** The first was a policy looping `</post>` while still scoring
  healthy, which is why `degenerate` was split in two. This one is deeper: the
  perception input falls *because* agents stop posting. Any population-size or
  horizon reading drawn off `alive` needs re-checking against the well-formed
  post rate. C10's and C11's conclusions survive that check; C6's did not. —
  *established* — `runs/cultural_emergence/c6_extended_horizon/NOTES.md`
- **Agents within a population are not independent samples.** Perception-loss
  ICC across seeds is 0.56–0.88, a design effect of 4.9–13.7: a 16-agent
  population is worth about 6 independent agents. The population is the unit of
  analysis. — *established* —
  `runs/cultural_emergence/_seed_replicate_regimes/REBUTTAL_NOTES.md`
- **Regime membership is unstable within a run.** 38% of cross-environment
  agents passed through two or more regimes during the absorption window, so
  every regime label must say whether it is contemporaneous or end-of-training.
  — *established* — `runs/cultural_emergence/_absorption_by_regime/REPORT.md`
- **`text_degenerate` conflates "has not learned yet" with "has broken".** The
  untrained model's perception loss is 2.44 on Robotic Athanor and 2.20 on
  Ashbourne GC, both above the 2.0 cut, against 1.35 on Inner Loop — a spread
  wider than the gap between the two thresholds. — *established* —
  `runs/_tier_b/b3_substrate_relative_regimes/NOTES.md`
- **Half of all regime labels move if the perception cut is
  anchored to the substrate instead of fixed.** 40 of 79 runs read as 0% alive at
  iteration 25 under the absolute cuts against 17 of 79 under relative ones. Which
  convention is right is open; that the choice is load-bearing is not. —
  *established* — same
- **Falling perception loss does not mean a healthy population.** Ten runs
  decline strictly across every 50-iteration block; one of them
  (`run10_2agent_7B_study_group_ashbourne_gc`, 2.47 → 1.33) ends with both
  agents in `policy_collapse`. A population can get better at predicting itself
  by becoming deterministic. — *established* —
  `runs/cultural_emergence/run10_2agent_7B_study_group_ashbourne_gc/NOTES.md`
- **`thinking_enabled` is a dead config key.** No code reads it,
  in the env or the trainer, and 86 configs set it — including the run billed as
  "Qwen3-8B with native thinking". The forum env always extracts `<post>`, so
  the private channel exists by construction whatever the key says. Native
  thinking, where it happens, comes from the chat template via
  `marlllm/context_formatter.py`, not from this. — *established* —
  `scripts/build_migration_configs.py` line 41 records the same fact
- **`thinking_frac` measures the environment's private channel
  correctly, and its name is wrong twice over.** It includes `<python>`, which is
  *public* — 68–80% of early non-post tokens on the two tool substrates — and it
  cannot distinguish deliberation from a post the agent wrote and failed to tag.
  — *established* — `runs/_tier_b/b6_token_accounting/NOTES.md`
- **Structural tag compliance decays over training, and the
  environment discards what falls outside a tag.** `<post>` opens per close go
  1.7 → 87; late `post_missing` turns include agents writing in the substrate's
  own pinned format without wrapping it. Nothing in the objective rewards a
  well-formed tag. — *established* — same
- **Single-iteration endpoints are noise.** Per-iteration standard deviation
  of population-mean perception loss is comparable to the change a whole run
  produces; quote block means. — *established* —
  `runs/terminal_experiments/00_stage0_baseline/NOTES.md`
- **A swap or counterfactual construction needs an iteration-25 baseline.** The
  cross-run history swap failed this test at +1.10 nats/token before training
  had done anything, and the cultural-learning measurement built on it does not
  stand. — *established* —
  `runs/cultural_emergence/_pmi_decomposition/REPORT.md`
- **The committed-vs-coherent contrast survives its own thresholds**, positive at
  100% of 2,850 grid points in both datasets, and an unsupervised alternative
  recovers the same split. — *established* —
  `runs/cultural_emergence/_regime_threshold_sweep_3d/REPORT.md`
- **Metrics chosen in advance cannot find a failure mode nobody thought of.**
  Both of the worst bugs found in this project — the `</post>` token loop behind
  readable text, and the confabulated read results — were invisible in every
  logged number and obvious in the traces. — *established* —
  `runs/cultural_emergence/_population_figures/trace_verification_iter200.md`,
  `runs/social_terminal/run02_8agent_7B_ashbourne_school/NOTES.md`

---

## What we are running

| job | config | what it is for |
|---|---|---|
| 6571670, 6571671 | [`learning_progress/lp_lagged_*`](configs/learning_progress/README.md) | Tier 1 of the learning-progress suite, two seeds. The return's per-observation summand becomes `M_s - M_s^-` against this agent's own adapter from 26–50 iterations ago, which cancels `H(p_env)` exactly. Read against `run7_8agent_7B_study_group_ashbourne_gc` on policy entropy first — the one quantity the sign flip moved at 8 of 8 arms — and on `lp/return_rank_corr`, which decides whether the correction could have mattered at all. |
| 6571672, 6571673 | [`learning_progress/lp_base_*`](configs/learning_progress/README.md) | The same correction against the frozen pre-trained model instead. The cancellation is exact for any second reference, so this arm separates "the cancellation is what matters" from "the reference has to track". Nearly free — the reference forward already runs for the KL term. |
| 6580460, 6580461 | [`learning_progress/lp_lag{005,100}_*`](configs/learning_progress/README.md) | The lag knob at L=5 and L=100, bracketing Tier 1's L=25. Lag is the scheme's only knob and it interpolates the whole family: 0 makes the return identically zero, infinite makes it novelty relative to base. |
| 6580462 | [`learning_progress/lp_lagged_nonorm_*`](configs/learning_progress/README.md) | The same correction with `normalise_returns` off. Motivated *by* Tier 1: lagged seed 1 has `lp/progress_per_token` at ~0.000 while KL climbs past 3.2, which is a policy moving hard on a reward that has gone to zero — the signature of standardisation rescaling estimator noise back to unit variance. |
| 6580463, 6580464 | [`learning_progress/lp_lagged_margin_notes_*`](configs/learning_progress/README.md) | The tool substrate, two seeds. The reference abandons python entirely by iteration 300; a return that pays for improvement should make tool output — structured, learnable, and not suppressible by the population — worth eliciting. Read on the python counters and the traces, not on `alive`. |

Held against these before reading them. `lp/progress_per_token` is *expected*
to decay, because the perception arm drives the improvement the action arm is
paid for — an arm ending at low KL with entropy intact is the action term
self-extinguishing, not the objective working, and has to be read against C10's
perception-only arm before it counts as a success.

**And the pre-registration's other prediction is already in doubt.** It says
this correction cannot touch the length channel, because `KL(p_env ‖ p_theta)`
stays extensive in observation tokens. At iteration 120–175 the Tier 1 posts
are getting *longer* — mean post tokens per turn 49–58 at iteration 25 against
76–149 now — with silent turn rate at 0.000–0.035, where the public channel
closes in 20 of 20 runs of C1 and 63 of 65 of B4. **This is not a result yet**:
the runs are unfinished and unanalysed, `mean_post_tokens` and
`silent_turn_rate` postdate the reference run so there is no matched
comparison, and the trace-based well-formed post rate — the measure PROCESS.md
requires beside any of this — has not been computed. Recorded here so that if
it holds up it is not mistaken for something that was expected.

One config is **held** rather than queued: [`env_likelihood_bias/B_corrected_seed1`](configs/env_likelihood_bias/README.md)
has failed three times and must not be resubmitted before
`observation_segments` is instrumented.

The novelty-search jobs previously listed here (6398717–6398727) have finished
and are analysed in `runs/novelty_search/NOTES.md`. The entropy-release family
(6434469–6434474) is no longer in the queue either, but its `NOTES.md` still
reads **RUNNING** — that file needs closing out against what is on disk.

---

## What we want to learn

- **Delivery is necessary but not sufficient, and this is the fourth
  metric artefact.** A two-agent population locked into a short mutually
  predictable exchange scores delivery 0.926 on 52-character posts of which 38%
  near-duplicate an earlier post by the same speaker. Degeneration into `</?`
  loops *cannot* raise delivery — those fail to close their tags — which is why
  the 09-03 reading called it the one measure degeneration cannot move the
  wrong way; degeneration into a short **well-formed** loop can. Always report a
  repetition measure beside a delivery figure. — *established*, one direct
  demonstration — `runs/novelty_search/NOTES.md`,
  `scripts/analyse_novelty_traces.py`
- **A trace file holds one episode of the iteration's sixteen rollouts.**
  Every trace-derived figure in this project is n = 1 episode unless stated
  otherwise. This is what produced the tool substrate's overstated "python
  calls reach zero" premise — the traced episode had 0 calls while the
  population averaged 22.2 per iteration. — *established* — same
- **Within-speaker repetition measures are structurally biased by
  population size.** Posts per speaker is `max_posts`/N — 8 at N=2, 1 at N=16,
  where a within-speaker repeat cannot occur at all. Never compare them across
  N; use pairwise similarity over all post pairs instead. — *established* —
  same

- 🔸 **The `math` regex under-counts mathematical reasoning expressed in
  prose.** A population working correctly through subset counting — "one
  element, one valid subset; adding another shows how this could grow" — scores
  0.00, because the regex wants symbols, LaTeX or proof vocabulary. It measures
  notation, not reasoning; its zeros are not evidence of absence. — *established*
  — `runs/cultural_emergence/entropy_anchor_ridge_8agent_ashbourne_gc/NOTES.md`

- 🔶 **Every trace-content measure except one is fooled by
  high-entropy collapse.** At end entropy 6.59 a word-salad population scored
  15 posts, 724 mean characters, `repeat` 0.00 and `frag` 0.00 —
  indistinguishable from a healthy transcript, because word salad is long and
  non-repetitive. The English function-word fraction (`stopword`) separates
  them: 0.035–0.090 for the four collapsed populations against 0.28–0.45 for
  coherent ones, ρ = −0.531, p = 1.4 × 10⁻³ over 28 runs. — *established* —
  `scripts/analyse_novelty_traces.py`
- 🔶 **But `stopword` is a screen, not a verdict, and is reliable only below
  0.10.** Between 0.10 and 0.28 it cannot tell dense technical prose from word
  salad: a genuine discussion of induction base cases scores 0.118 while real
  salad scores 0.139. Anything in that band must be read. **This also withdraws
  "the healthiest late transcript in the sweep is N=4 seed 2" being read off
  metrics** — it scores 0.118 and had to be confirmed by reading, which it
  passed. — *established* — same

### What we cannot currently see

- **Anything above 7B**, and — since the Qwen3 comparison — anything about
  whether the central stratum generalises at all. Every claim about commitment
  is a claim about Qwen2.5-7B-Instruct; the 14B and 32B attempts produced no
  data and were never diagnosed. — `runs/_tier_b/b8_model_generation/NOTES.md`,
  `runs/cultural_emergence_14B/README.md`
- **What a committed agent's extra drift is made of.** KL from base rising from
  2.7 to 6.5 during a migration is a large movement with no behavioural
  characterisation attached; those traces were never read.
- **Whether cultural divergence keeps growing.** It has not plateaued at any
  measured horizon, and no run goes far enough to find out.
- **Whether sanctioning is targeted.** Rung 1 is gated on a judge validation
  that has not been run.

### What we have not run

- **The Phase A evaluation.** Six trained checkpoints sit on disk;
  `evaluate_negotiation.py` and `analyze_prompt_conditional.py` were never run
  over any of them, so criteria (a) and (b) of the framework's founding claim are
  untested. Conditions 4 (action-only) and 5 (symmetric CCSM) have configs and no
  runs — a different thing from the forum term ablation now running, which is
  the same contrast on the substrate the paper's claims actually live on. This
  is cheap and it is the largest single gap in the project.
- **The terminal programme past Stage 0.** Stage 1 started twice and wrote
  nothing; Stages 2, 3, 3.5 and 4 have configs and no runs — including the
  embodiment config its own header calls "THE load-bearing experimental config
  for the framework's core claim".
- **A no-migration control** at the same iteration range as the migration runs.
  Without it the residents' trajectories cannot be attributed to the migrant.
- **The dual-environment control.** The long-episode variant that would say
  whether the failure is the environment split or the halved token budget got
  six iterations.
- **A frozen-population, solo-agent or static-transcript control** — beyond the
  frozen-population arm now running, the other two still have neither code nor
  configs.

### Open questions

- **What fraction of undelivered turns are untagged posts rather than
  deliberation?** Reading a sample finds both, plus token loops. The number
  decides whether the closing public channel is a policy pathology or a
  substrate one, and it needs a classifier or a hand-coded sample. —
  `runs/_tier_b/b6_token_accounting/NOTES.md`
- **Answered for the term, not for the typing.** Removing the action term
  removes the channel closure and the population failure, so the direction is
  causal (C10, above). What remains open is the *token-typing* version — typing
  only public post tokens as actions and retraining — which would say whether
  the effect runs through what is scored as an action rather than through the
  policy gradient as a whole. Still unwritten. —
  `runs/cultural_emergence/c10_term_ablation/NOTES.md`,
  `runs/_tier_b/b4_public_channel/NOTES.md`
- **Why does the social-terminal substrate not decline?** Its two full-horizon
  runs end alive where every forum population does not. Whether that is the
  substrate, the model, or the shorter public channel is unseparated.
- **Is the decline the same event as the entropy collapse?** `policy_collapse`
  is the dominant late failure mode in the forum runs, and the entropy bonus is
  the one hyperparameter known to control entropy catastrophically. Nobody has
  checked whether the safe-side default is merely slower rather than safe.
- **What is the right evaluation horizon?** Iteration 200 was chosen before the
  decline was visible, and the runs say it is not a converged state. C1 now adds
  that the choice is load-bearing in a second way: within-cell seed spread of
  `alive` is 0.44 at iteration 200 and 1.00 at 300, so *which* iteration a
  figure is drawn at changes what it shows as much as the condition does. C6
  will say whether there is a floor. —
  `runs/cultural_emergence/c1_small_population_replicates/NOTES.md`
- **How large is the gradient bias in our substrates?** Still unmeasured, and
  the first attempt to measure it failed on plumbing rather than on the idea:
  B5 established the measurement is well posed at 100% provenance resolution,
  but the C9 runs built to do it lost provenance progressively and logged none
  of the diagnostics. Needs the path fixed, then a GPU scoring pass. —
  `runs/env_likelihood_bias/NOTES.md`, `runs/_tier_b/b5_env_likelihood/NOTES.md`
- **How large is the private-prefix correction?** Predicted near zero early and
  growing with the pre-post region; pre-registered with its falsification
  conditions in `scripts/tier_b/c7_env_likelihood_bias.py`, not yet run.
- 🆕 (2026-09-15) **Does the corrected return still produce policy movement?**
  Restoring `p_env` makes the return a proper divergence whose optimum is flat
  across worlds, so advantages may vanish at convergence where the surrogate's
  do not. That may be why the surrogate moves policies at all. **Now being
  asked directly**: jobs 6571670–6571673 train on a difference of two
  cross-entropies, whose reward goes to zero exactly where there is nothing
  left to learn. `lp/progress_per_token` decaying to zero with entropy intact
  and low KL would be the action arm self-extinguishing — which has to be read
  against C10's perception-only arm, not as the objective working. —
  `environment_likelihood_derivation.md`,
  `configs/learning_progress/README.md`
- 🆕 (2026-09-15) **Does the correction reorder actions at all?** The one prior
  measurement of a return correction in this project found it changed magnitude
  and not ordering — the dropped environment term is ~43% of the return with
  rank correlation 1.000 — and a correction that does not reorder cannot change
  learning however large it is. The learning-progress return is a different
  object (a difference of two surprises, not a same-signed additive term) so
  the prior does not transfer, but it is the pre-registered null.
  `lp/return_rank_corr` is logged per batch to settle it, and it is the first
  thing to read at iteration 25. — `runs/_tier_b/c9_post_hoc_env_term/NOTES.md`,
  `configs/learning_progress/README.md`
- **Does absorption survive a horizon-clean window on more than one dataset?**
  It does on the Inner Loop seed replication and largely does not on the
  cross-environment pool, where restricting to healthy-window measurements also
  shortens the window. The two readings are not yet separated. —
  `runs/_tier_b/b7_absorption_horizon/NOTES.md`
- **Why did four separate configurations die immediately after their
  iteration-0 trace dump?** Qwen3.5 (12 submissions), Qwen2.5-14B (2), sequence
  packing (4) and the async first attempt (2) all share the signature. One cause
  is more likely than four, and none was traced.
- **Can cultural learning be measured at all?** The cross-run history swap did
  not separate it from baseline transcript surprise. The proposed fix —
  permuted-own-run history, rotating who said what while keeping the content —
  has not been implemented. — `runs/cultural_emergence/_pmi_decomposition/REPORT.md`
- **Does CCSM ever improve an external metric?** It has never been shown to. The
  one substrate where success was logged returned exactly zero throughout, and
  the substrate designed to test it had no headroom.

---

## Reading order

1. This document.
2. [`PROCESS.md`](PROCESS.md) — how it is kept true.
3. [`runs/TIMELINE.md`](runs/TIMELINE.md) — what happened when.
4. [`runs/README.md`](runs/README.md) — where everything is.
5. [`runs/_tier_b/README.md`](runs/_tier_b/README.md) — the eight analyses that
   spend no compute, each with its pre-registration in the script docstring.
6. Whichever `NOTES.md` or `REPORT.md` a claim above cites.

For the cultural-emergence results specifically, read
[`_trajectory_kl_combined/REPORT.md`](runs/cultural_emergence/_trajectory_kl_combined/REPORT.md)
and [`_diagnostics_d2d3/REPORT.md`](runs/cultural_emergence/_diagnostics_d2d3/REPORT.md)
together, in that order — the second retracts a reading of the first.

---

## Standing caveats

- **Almost every run in this project is truncated.** State the iteration with
  every number.
- **Rooms, runs and substrates desynchronise.** Never compare raw iteration
  counts across substrates without the baseline.
- **A single run cannot rank two schemes here.** Demonstrated directly: one
  configuration, two runs, opposite outcomes.
- **The regime classifier will label anything**, including a two-iteration smoke
  test. A regime is only meaningful for a run that trained.
- **The retrospective notes record what the data shows, not what was expected.**
  A back-filled note cannot tell you whether a result surprised its author,
  which is the information a pre-registered header exists to preserve. See the
  last section of `PROCESS.md`.
