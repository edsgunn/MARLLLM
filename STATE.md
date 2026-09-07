# State of the project

A living summary: what we believe, how strongly, what we are running, and what
we still do not know. **No numbers or derivations here** — every claim points at
the document that carries the evidence.

How this document is kept true is [`PROCESS.md`](PROCESS.md). What the framework
is and how to run it is [`README.md`](README.md). The chronology is
[`runs/TIMELINE.md`](runs/TIMELINE.md). What we intend to claim in the
resubmission, and what has to be answered first, is [`REWRITE.md`](REWRITE.md).

Last updated: 2026-09-04 (sixth update: PopArt, and C2 complete).

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
- 🔸 **The experiment built to measure the dropped environment
  term measured nothing, because no-tag mode stripped the generation while the
  provenance matcher required the unstripped decode.** Any surrounding
  whitespace missed, so no observation token carried `ln p_env`; and because
  the corrected return is selected only after that check, **arm B silently
  trained on the surrogate** and is a duplicate of arm A. Fixed, with
  regression tests; both arms need re-running. — *established* —
  `runs/env_likelihood_bias/NOTES.md`
- 🔸 **A run can declare an objective it is not training on.**
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
- 🔸 **The perception term does the learning and the action term does the
  damage, and the separation is total on the reference substrate.**
  Perception-only ends 300 iterations with `alive` 1.00 on both seeds,
  perception loss 2.13 → 0.70, entropy intact and 61 of 64 turns delivering a
  well-formed post; the full loss manages 36 of 64 with all eight agents
  degenerate. — *established* — `runs/cultural_emergence/c10_term_ablation/NOTES.md`
- 🔸 **The dark-room incentive is causal, not merely associated with the
  decline.** It reaches behaviour through the action term alone, and removing
  that term removes both the channel closure and the population failure.
  Isolating it instead — action-only — misses 24 to 30 of 64 turns entirely and
  drives perception loss to 4.2, above base. — *established* — same
- 🔸 **Perception-only is a candidate stabilisation, not yet a result.** It
  moves the policy about six times less than the full loss (KL 0.22 against
  1.32), matching Phase A, so it is not vacuous — but whether it still *absorbs*
  needs the PMI scoring pass, which has not run. Until then it could be
  "don't train the policy" wearing a better outcome. — *provisional* — same
- **Absorption scales continuously with movement from base**, not in a step at
  any threshold: on the cleanest subset a linear model in KL beats a step model
  on R², AIC and leave-one-out error. — *established* —
  `runs/cultural_emergence/_regime_threshold_sweep_3d/REBUTTAL_NOTES.md`

- 🔶 **Removing the normalisation destroys the run; removing the discount does
  nothing.** γ=1 with normalisation kept delivers 0.50 against the reference's
  0.56 — the earlier "helps survival" reading rested on `alive` and is
  withdrawn. γ=1 without it reaches perception loss
  **37**, entropy 0.00, and 62 of 64 turns delivering nothing — the pure
  dark-room outcome, worse than C10's action-only arm. — *provisional*, one
  seed each — `runs/cultural_emergence/c11_discount_normalisation/NOTES.md`
- 🔶 **The un-normalised arms test the arm mis-scaling, not the derived
  objective.** At γ=1 the raw return is ~2300 against a perception loss of
  ~1.4, so the action arm enters the summed loss ~1600× larger and the run is
  action-only in all but name. This was predicted in the config headers before
  submission. Whether the derived objective trains with a *working* baseline is
  open until the PopArt arm reports. — *established* — same

- 🆕 (2026-09-04) **A working baseline does not make the derived objective trainable, and
  the scale mismatch is an arm-weighting problem rather than a baseline one.**
  PopArt takes the value head from diverging at 1.45e8 to a stable O(10) loss —
  a seven-million-fold improvement, value kept in the return's units — and the
  run still ends at perception loss 39.6, entropy 0.00 and 7 well-formed posts
  of 64. A fitted baseline centres the advantage but cannot shrink it: its scale
  is the return's spread, which PopArt measures at σ = 1552. — *established* —
  `runs/cultural_emergence/c11_discount_normalisation/NOTES.md`
- 🆕 (2026-09-04) **The return's scale is genuinely non-stationary**, so the per-batch
  standardisation was papering over something real: PopArt's running moments
  move μ 697 → 861 and σ 344 → 1552 across 300 iterations. — *established* —
  same

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

- 🔸 **The N=2 and N=4 cells of the population-size figure are seed noise.**
  Five seeds a side: within-cell spread of `alive` is 0.44 against between-cell
  gaps of 0.20 and 0.35 at iteration 200, and 1.00 against 0.05 and 0.15 at 300.
  No continuous metric separates the sizes (all p ≥ 0.15, exact
  Mann-Whitney). The pre-registered falsification condition fired. —
  *established* —
  `runs/cultural_emergence/c1_small_population_replicates/NOTES.md`
- 🔸 **Two published cell readings were seed artefacts.** "Dead from iteration
  150 onward" for 2-agent Strathearn becomes 0.50–1.00 alive at 150 across five
  seeds. — *established* — same
- 🔸 **A single agent minimising surprise against a frozen population can
  still collapse completely**, so co-adaptation is not necessary for the
  failure. Whether it is *sufficient* is unresolved: three learners span 0.00 to
  1.00 well-formed delivery against a co-adapting reference of 0.56. —
  *likely* for the necessity half, *inconclusive* for the rest —
  `runs/cultural_emergence/c3_frozen_population/NOTES.md`

- 🔶 **The N=16 reference spans `alive` 0.38–0.88 at iteration 300 across
  five seeds**, mean 0.70, where it was previously a single run at 0.88 — the
  top of the range. Comparisons drawn against that single seed were comparing
  against the best case. — *established* —
  `runs/cultural_emergence/c2_population_size/NOTES.md`
- 🔶 ~~**Matching per-agent gradient volume downward gives the healthiest cell.**~~
  **Withdrawn.** It rested on `alive` 0.95; by delivery it is 0.263 against
  0.200, p = 0.810. — *withdrawn* —
  `runs/cultural_emergence/c2_population_size/NOTES.md`

- 🆕 (2026-09-04) **Population size does not determine the outcome, at either end.** All
  five arms of the decomposition are in: N=4 against N=16 at matched gradient
  volume gives 0.263 vs 0.225 delivery, p = 0.722 on five seeds a side, and
  volume itself gives p = 0.810. With C1 having removed the small corner, the
  population-size claim has no support left. — *established* —
  `runs/cultural_emergence/c2_population_size/NOTES.md`
- 🆕 (2026-09-04) **Co-adaptation degrades delivery, and it is the only variable in that
  design that separates.** Learners against a frozen majority deliver 0.475
  against 0.225 for agents in a co-adapting population of the same size,
  p = 0.032, every seed above the baseline's median. — *likely* — same
- 🆕 (2026-09-04) **Partner count is the one size-adjacent variable worth more seeds**:
  0.316 at three partners against 0.225 at fifteen with N held at 16, but
  p = 0.135 is not a result at five seeds. — *suspected* — same

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

- 🔶 **`alive` is invalid wherever delivery is low: its dominant input moves
  with the failure it is meant to detect.** Perception loss is computed over
  observation tokens, so as agents stop posting the stream becomes
  deterministic scaffolding and `perc_loss` falls through the classifier's
  `< 1.7` gate. Within the 8-agent extension, delivery and perception correlate
  at r = +0.85 while `alive` moves the other way; across ten (run, iteration)
  points delivery and `alive` correlate at only r = +0.23, and the two
  references invert — `run7` delivers 56% of turns and scores 0.00, `run8`
  delivers 27% and scores 0.88. — *established* —
  `runs/cultural_emergence/c6_extended_horizon/NOTES.md`
- 🔶 **The extended populations do not recover; they fall silent.** At
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

- **The entropy bonus is the dangerous knob.** `beta ≥ 0.1` destroys the
  population at every KL coefficient tried; `beta = 0.5` is catastrophic, ending
  with entropy above 6.5 nats. The cliff is below 0.2. — *established* —
  `runs/cultural_emergence/kl_beta_sweep_8agent_ashbourne_gc/NOTES.md`,
  `runs/cultural_emergence/beta_zoom_kl0p5_8agent_ashbourne_gc/NOTES.md`
- **The entropy bonus destroys populations without moving them from base**, so
  the KL anchor cannot protect against it. — *established* — same
- **The KL coefficient controls drift and little else.** Raising it from 0 to
  0.5 halves end-of-training KL from base and barely changes the alive fraction.
  — *likely* — same
- **No cell ranking among the surviving settings is supported.** The
  differences between the good cells are smaller than the spread between two
  runs of one identical configuration. — *established* — same
- **Removing both regularisers made one population *better*, not worse.** —
  *provisional*, one run each side —
  `runs/cultural_emergence/run7_8agent_7B_study_group_ashbourne_gc_no_reg/NOTES.md`

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
- 🔸 **Counting non-empty thread text overstates the public channel by up to
  a factor of two.** An unclosed `<post>` block still has its text extracted
  onto the thread, so only `tool_counts_per_agent`'s `post_success` measures
  what B4's claim is about — on `run8` at iteration 300 the two read 0.27 and
  0.80. Using the proxy hid the frozen-majority effect entirely. — *established*
  — `runs/cultural_emergence/c2_population_size/NOTES.md`
- 🔸 **A population-mean metric is meaningless when most of the population is
  frozen.** With 7 of 8 or 14 of 16 members frozen at base, the population
  post-success rate is the freeze fraction — 0.88, 0.94 — and says nothing about
  the learners, whose own rates are 0.00–1.00 and 0.25–0.62. `pop/n_frozen` is
  logged every iteration so the denominator is visible. — *established* —
  `runs/cultural_emergence/c3_frozen_population/NOTES.md`
- 🔸 **Six config keys the code does not read**, and the last two
  are worse than the rest. `thinking_enabled`; `frozen_agents`, in the
  negotiation trainer only; the three `alpha_*` weights, so every forum run
  trained at 1/1/1; `pairing_strategy`, inert on forum substrates; and now
  **`gamma` and `normalise_returns`**, passed by none of the five trainers. The
  first four made an experiment *inexpressible*; these two silently fixed the
  objective's shape for every run in the project's history. All six were found
  by trying to write an experiment rather than by reading code. — *established*
  — `marlllm/config.py`,
  `configs/cultural_emergence/c2_population_size/README.md`
- 🔸 **The dropped environment term is large but does not change
  the update ordering.** Recovered post hoc from arm A's checkpoints: ~43% of
  the return's magnitude at four checkpoints, rank correlation 1.000 throughout,
  because a post is 6–36% less surprising per token to its author than to its
  readers so the correction is same-signed and smaller. That margin narrows
  toward parity over training. — *likely* —
  `runs/_tier_b/c9_post_hoc_env_term/NOTES.md`
- 🔸 **The discount is not in the derived objective, and it
  front-weights credit inside every post by ~48×.** γ=0.99 is applied
  *positionally per token*, a 100-token horizon against a 128–384-token post.
  On a real 386-token post the future-observation mass at an action token runs
  2.0 at the first token to 97.1 at the last; at γ=1 it is 2057, flat.
  `surprise_minimisation_derivation.md` §6.2 introduces γ only as variance
  reduction "at the cost of bias", so γ=1 is the derived return. — *established*
  — `marlllm/loss.py` `_compute_returns`
- 🔸 **Return normalisation is not the free choice it is in RL,
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

- 🔶 **Measuring perception loss on post content only does not
  repair the classifier — predictability is simply the wrong target.** Content
  is *more* predictable than scaffolding at 12 of 13 checkpoints, and
  content-only loss is lowest where delivery is lowest (0.78 nats/token at
  delivery 0.234), because a degenerate `</?` loop is content and is maximally
  predictable — at iteration 900 one such post is 82% of everything delivered.
  Any loss over a stream the agents control can be reduced by emitting less or
  by emitting something trivially repetitive. — *established* —
  `runs/_tier_b/content_perception/NOTES.md`
- 🔶 **The environment's framing is not deterministic**, because
  it carries the speaker's name — which of eight agents posts next is the least
  predictable thing in the stream. Scaffolding NLL runs 1.8–2.7, above content
  almost everywhere. Any future token split has to account for that rather than
  treat framing as free. — *established* — same
- 🔶 **The regime classifier cannot be trusted at low delivery,
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

**Nothing.** The 2026-09-02/04 batch is complete: 60 of 66 jobs reached their
configured iteration count, and every family has a write-up.

One config is **held** rather than queued:
[`env_likelihood_bias/B_corrected_seed1`](configs/env_likelihood_bias/README.md)
has failed three times and must not be resubmitted before
`observation_segments` is instrumented — the guard has localised the fault to it
returning empty for some agents' observation steps, and three-minute crashes
have stopped being informative.

---

## What we want to learn

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
- **Does the corrected return still produce policy movement?** Restoring `p_env`
  makes the return a proper divergence whose optimum is flat across worlds, so
  advantages may vanish at convergence where the surrogate's do not. That may be
  why the surrogate moves policies at all. — `environment_likelihood_derivation.md`
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
