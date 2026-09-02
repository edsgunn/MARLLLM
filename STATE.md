# State of the project

A living summary: what we believe, how strongly, what we are running, and what
we still do not know. **No numbers or derivations here** — every claim points at
the document that carries the evidence.

How this document is kept true is [`PROCESS.md`](PROCESS.md). What the framework
is and how to run it is [`README.md`](README.md). The chronology is
[`runs/TIMELINE.md`](runs/TIMELINE.md). What we intend to claim in the
resubmission, and what has to be answered first, is [`REWRITE.md`](REWRITE.md).

Last updated: 2026-09-01 (second update: the Tier-B analyses).

---

## Confidence key

| | meaning |
|---|---|
| **established** | multiple runs, or one run with a controlled contrast and adequate sample. Would be surprising to overturn. |
| **likely** | one clean measurement, or several suggestive ones. Believe it, but it could move. |
| **provisional** | one run, confounded, or small sample. Treat as a working assumption. |
| **suspected** | a reading of the data we have not tested directly. |

**Novelty markers.** 🔶 marks what was added or materially changed in the most
recent update. This document was written retrospectively on 2026-09-01, so 🆕
here means specifically: **derived from the runs' own metrics during the
back-fill, and not stated in any report written at the time.** Everything
unmarked comes from the analysis reports under `runs/cultural_emergence/_*`.
Markers are demoted on every subsequent update — see `PROCESS.md`.

---

## What we have learned

Organised by subject, not by when we found it. **A new finding goes into the
category it belongs to. If none fits, the categories are wrong and get revised —
never appended to.**

### What the objective actually optimises

*The framework's central question. CCSM has no reward function, so what it
maximises and what we want are related only by argument.*

- 🔶 **Surprise minimisation and task success are dissociable, and on the one
  substrate where both were measured they came apart completely.** Perception
  loss fell more than tenfold while the logged task success rate stayed at
  exactly 0.000 for every iteration of both runs. — *established* —
  `runs/negotiation_population_experiments/*/NOTES.md`
- 🔶 **The perception term does the world-model learning; the action term moves
  the policy and collapses its entropy.** Perception-only reaches the same
  perception loss as the full loss with four to ten times less KL from base and
  with entropy intact. — *established* —
  `runs/phase_a_substrate_validation/03_perception_only_*/NOTES.md`
- 🔶 **The full loss is seed-unstable at the level that decides whether a policy
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
- 🆕 (2026-09-01) **The implemented return rewards shortening the observation
  stream, and its behavioural signature is now measured.** `G_t` sums surprise
  over *future observation tokens*, so its magnitude depends on how many follow
  as much as on how surprising each is. The fraction of turns delivering a
  well-formed public post falls in **63 of 65 runs** — on the reference
  population from 99% to 28%, with 28% of turns delivering nothing at all. —
  *established* — `runs/_tier_b/b4_public_channel/NOTES.md`
- 🆕 (2026-09-01) **`p_env` is not a black box in this substrate class, but
  restoring it is not free.** Observation-token provenance resolves at 100% —
  every post appears as a contiguous token span in every other agent's context —
  so the dropped term is measurable. The producing agent's own tokenisation is
  *not* the receiver's, so its stored log-probs do not align: restoring the term
  costs a forward pass per agent per episode, not a lookup. — *established* —
  `runs/_tier_b/b5_env_likelihood/NOTES.md`
- 🆕 (2026-09-01) **The environment conditional is observer-independent under
  full observability.** An agent's own persona and private deliberation are
  conditionally independent of what other agents emit, given the public thread,
  so every observer of a post shares one `p_env`. The correction is a property of
  the *generator's* turn, computed once per post rather than once per
  (post, observer) pair. — *established* (derivation) —
  `environment_likelihood_derivation.md`
- 🆕 (2026-09-01) **The single adjustment needed is the mutual information
  between the generator's private prefix and its public post.** Exact pointwise
  identity; non-negative; zero when the generator emits no private tokens; and
  bounded above by the term currently being dropped, so substituting the
  generator's log-probs is never worse than dropping and is *exact* early in
  training. — *established* (derivation), unmeasured — same
- 🆕 (2026-09-01) **The committed-vs-coherent absorption contrast is
  horizon-sensitive.** Intact at +0.160 on the Inner Loop seed replication, whose
  window ends before any decline; falls from +0.199 to +0.024 on the
  cross-environment pool when measurements taken during decline are dropped. The
  seed replication is the citable one. — *likely* —
  `runs/_tier_b/b7_absorption_horizon/NOTES.md`
- **Absorption scales continuously with movement from base**, not in a step at
  any threshold: on the cleanest subset a linear model in KL beats a step model
  on R², AIC and leave-one-out error. — *established* —
  `runs/cultural_emergence/_regime_threshold_sweep_3d/REBUTTAL_NOTES.md`

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
- 🔶 **Information-returning verbs must end the turn, or the agent invents what
  it has not been told.** An actor that keeps generating after committing to a
  read confabulates the result of a read that failed. Nothing in the metrics
  showed this; it was found by reading traces. — *established* —
  `runs/social_terminal/run02_8agent_7B_ashbourne_school/NOTES.md`
- 🔶 **Framework-inserted scaffolding must be typed OBS, not ACT.** Only
  model-generated tokens are actions; inserted turn-close tokens in the PPO
  ratio are not. — *established* — `runs/social_terminal/run06_8agent_7B_ashbourne_school/NOTES.md`
- 🔶 **Robotic Athanor is unlearnable under this objective, and not because it
  degrades.** Perception loss is flat at its initial value across 100 iterations
  at both 2 and 8 agents, with entropy neither low nor falling. This corrects
  the "action distributions degrade into low-entropy gibberish" reading in the
  absorption report, and supports that report's conclusion by a different route.
  — *established* — `runs/cultural_emergence/run7_8agent_7B_robotic_athanor/NOTES.md`
- 🔶 **One population cannot hold two environments at this budget.** The
  dual-environment arm is dead from iteration 25 and never recovers above 25%
  alive, where the single-environment run on the same roster is 88% alive at
  iteration 200. — *likely* —
  `runs/cultural_emergence/run9_8agent_7B_ashbourne_dual_env/NOTES.md`
- 🔶 **Substrates differ in starting perception loss by nearly a nat**, so
  absolute perception loss is not comparable across substrates without the
  baseline. — *established* — `runs/cultural_emergence/run8_16agent_7B_study_group_strathearn_server/NOTES.md`
- 🔶 **`deal_or_no_deal` at 1.5B is a floor-effect substrate.** The pretrained
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
- 🔶 **Above two agents, the population-size grid is not monotone at all.** On
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
- 🆕 (2026-09-01) **What a population becomes depends strongly on the base
  model.** At matched iterations on one substrate with identical regularisation,
  **no Qwen3-4B or Qwen3-8B agent ever reaches KL ≥ 1 from base**, where 27 of
  182 Qwen2.5-7B agents do; mean drift is 6–14× lower and the populations are
  *healthier*. "Commitment" — the stratum the absorption result lives in — is
  not something the objective reliably produces. — *likely* —
  `runs/_tier_b/b8_model_generation/NOTES.md`
- 🔶 **More action tokens per episode accelerates policy collapse.** Both
  two-turn arms end far below their single-turn counterparts, and the failure
  mode is entropy collapse rather than text degeneration. — *likely* —
  `runs/cultural_emergence/run11_16agent_7B_study_group_ashbourne_gc_2turns/NOTES.md`

### Training horizon — what happens if you keep going

*🔶 This whole section is new. It was not visible in any analysis written at the
time, because every published analysis evaluates at a fixed iteration.*

- 🔶 **Forum populations peak around iteration 100 and decline from there.**
  Across a balanced panel of 44 cultural-emergence populations measured at
  iterations 25, 100, 150, 200 and 300, the mean alive fraction runs
  0.27 → 0.86 → 0.82 → 0.72 → 0.46. — *established* —
  `runs/_tier_b/b2_regime_by_iteration/NOTES.md`
- 🔶 **No forum population has ever improved after iteration 150.** Of the 44
  measured at both 150 and 200, none rose; of the 44 measured at both 200 and
  300, none rose and 31 fell. 39 of 44 end below their own earlier peak. —
  *established* — same
- 🔶 **The campaign's evaluation point sits one step before the decline.** The
  clearest single case is the 8-agent Ashbourne GC run: 88% alive at iteration
  200 and **0% at 300**. This reconciles two published tables that appear to
  contradict each other — they are reading different iterations of one run. —
  *established* —
  `runs/cultural_emergence/run7_8agent_7B_study_group_ashbourne_gc/NOTES.md`
- 🔶 **A short horizon can make a failing substrate look sound.** Margin Notes is
  100% alive across four seed replicates that stop at iteration 150, and 50%
  alive in the one run that continued to 300. — *established* —
  `runs/cultural_emergence/run7_8agent_7B_margin_notes/NOTES.md`
- 🆕 (2026-09-01) **The public channel closes long before the regime classifier
  registers anything** — by up to 200 iterations. On the reference population the
  channel is already half shut at iteration 200 while the alive fraction still
  reads 1.00. The taxonomy is a lagging indicator of this failure. — *likely* —
  `runs/_tier_b/b4_public_channel/NOTES.md`
- 🔶 **The decline is not inherent to the objective.** Both social-terminal runs
  that reached iteration 300 ended alive, and one improved between 200 and 300 —
  the only multi-agent run in the project that does. — *likely*, two runs —
  `runs/social_terminal/run06_8agent_7B_ashbourne_school/NOTES.md`

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
- 🆕 (2026-09-01) **The divergence is not an artefact of collapsed adapters, but
  the pooled magnitudes are inflated where failure regimes are present.**
  Restricting to characters alive in both populations changes the end-of-training
  policy distance by 0–5% on Inner Loop and Ashbourne and by **−45% on Margin
  Notes**. Growth from the iteration-25 baseline is ×27–×112 either way. Report
  the restricted version. — *likely* —
  `runs/_tier_b/b1_policy_distance_by_regime/NOTES.md`
- **Failure regimes must be carved out of any divergence claim.** Restricting to
  source-coherent characters raises the signal 20–40%. — *established* — same

### What an agent carries between populations

- 🔶 **A migrated adapter learns the host substrate at the resident rate.**
  Migrant and resident perception losses move together in all 16 migration runs.
  — *established* — `runs/migration_experiments/NOTES.md`
- 🔶 **Migration does not reset drift: a committed agent stays committed and gets
  more so.** All 7 migrants arriving above KL 2.0 drifted further; 8 of 9
  arriving below KL 1.1 did not. — *likely* — same
- 🔶 **World model and policy identity are separable, and per-agent LoRA is where
  the second one lives.** — *likely* — same

### Norms and sanctioning

- 🔶 **Sanctioning is detectable at high volume in these transcripts**, and it is
  overwhelmingly approval: disapproval runs at a median of 0.062 reactions per
  post against approval's 0.656. — *established* — `runs/sanctioning/NOTES.md`
- 🔶 **The asymmetry is the base model's, not something the populations learned.**
  It is at its iteration-0 value before any training and shows no trend across
  300 iterations. — *likely* — same
- 🔶 **A norm needs a sanctioning response that can go either way, and
  disapproval is at the floor.** — *suspected*; this is a reading of the Rung-0
  table, not a measured result — same

### Regularisation

- 🔶 **The entropy bonus is the dangerous knob.** `beta ≥ 0.1` destroys the
  population at every KL coefficient tried; `beta = 0.5` is catastrophic, ending
  with entropy above 6.5 nats. The cliff is below 0.2. — *established* —
  `runs/cultural_emergence/kl_beta_sweep_8agent_ashbourne_gc/NOTES.md`,
  `runs/cultural_emergence/beta_zoom_kl0p5_8agent_ashbourne_gc/NOTES.md`
- 🔶 **The entropy bonus destroys populations without moving them from base**, so
  the KL anchor cannot protect against it. — *established* — same
- 🔶 **The KL coefficient controls drift and little else.** Raising it from 0 to
  0.5 halves end-of-training KL from base and barely changes the alive fraction.
  — *likely* — same
- 🔶 **No cell ranking among the surviving settings is supported.** The
  differences between the good cells are smaller than the spread between two
  runs of one identical configuration. — *established* — same
- 🔶 **Removing both regularisers made one population *better*, not worse.** —
  *provisional*, one run each side —
  `runs/cultural_emergence/run7_8agent_7B_study_group_ashbourne_gc_no_reg/NOTES.md`

### Engineering and reproducibility

- 🔶 **This pipeline is not reproducible at fixed seed.** One configuration was
  run twice by accident, differing only in run name, output directory and which
  GPU it landed on. The two runs track each other to about iteration 150 and
  then separate, ending at alive 0.875 and 0.000. — *established*, one direct
  demonstration — `runs/cultural_emergence/kl_beta_sweep_8agent_ashbourne_gc/NOTES.md`
- 🔶 **Async rollout with `max_staleness` 1 under a PPO clip reproduces
  synchronous training.** Perception loss, entropy and KL all match the
  synchronous siblings. — *likely*, two matched pairs —
  `runs/cultural_emergence_qwen3_async/*/NOTES.md`
- 🔶 **Sequence packing has never produced a training iteration.** Four
  submissions, zero metric records. — *established* —
  `runs/cultural_emergence_qwen3_packed/README.md`
- 🔶 **`vllm_gpu_mem_util` 0.30 starves the trainer and OOMs the chunked loss;
  0.18 survives.** — *established*, three runs —
  `runs/social_terminal/run03_8agent_7B_ashbourne_school/NOTES.md`
- **`uv run` breaks the hand-built vLLM venv.** Use `.venv/bin/python`. —
  *established* — `README.md`
- 🔶 **A config in git can stop describing the run that used it.** One Phase A
  arm ran on the base model while its tracked config was later edited to name
  the Instruct model, confounding the prompt-swap diagnostic the design turned
  on. This is why every config needs a `STATUS:` line. — *established* —
  `runs/phase_a_substrate_validation/02_ccsm_focal_competitive/NOTES.md`

### Measurement — what our instruments do and do not show

- **Agents within a population are not independent samples.** Perception-loss
  ICC across seeds is 0.56–0.88, a design effect of 4.9–13.7: a 16-agent
  population is worth about 6 independent agents. The population is the unit of
  analysis. — *established* —
  `runs/cultural_emergence/_seed_replicate_regimes/REBUTTAL_NOTES.md`
- **Regime membership is unstable within a run.** 38% of cross-environment
  agents passed through two or more regimes during the absorption window, so
  every regime label must say whether it is contemporaneous or end-of-training.
  — *established* — `runs/cultural_emergence/_absorption_by_regime/REPORT.md`
- 🔶 **`text_degenerate` conflates "has not learned yet" with "has broken".** The
  untrained model's perception loss is 2.44 on Robotic Athanor and 2.20 on
  Ashbourne GC, both above the 2.0 cut, against 1.35 on Inner Loop — a spread
  wider than the gap between the two thresholds. — *established* —
  `runs/_tier_b/b3_substrate_relative_regimes/NOTES.md`
- 🆕 (2026-09-01) **Half of all regime labels move if the perception cut is
  anchored to the substrate instead of fixed.** 40 of 79 runs read as 0% alive at
  iteration 25 under the absolute cuts against 17 of 79 under relative ones. Which
  convention is right is open; that the choice is load-bearing is not. —
  *established* — same
- 🔶 **Falling perception loss does not mean a healthy population.** Ten runs
  decline strictly across every 50-iteration block; one of them
  (`run10_2agent_7B_study_group_ashbourne_gc`, 2.47 → 1.33) ends with both
  agents in `policy_collapse`. A population can get better at predicting itself
  by becoming deterministic. — *established* —
  `runs/cultural_emergence/run10_2agent_7B_study_group_ashbourne_gc/NOTES.md`
- 🆕 (2026-09-01) **`thinking_frac` measures the environment's private channel
  correctly, and its name is wrong twice over.** It includes `<python>`, which is
  *public* — 68–80% of early non-post tokens on the two tool substrates — and it
  cannot distinguish deliberation from a post the agent wrote and failed to tag.
  — *established* — `runs/_tier_b/b6_token_accounting/NOTES.md`
- 🆕 (2026-09-01) **Structural tag compliance decays over training, and the
  environment discards what falls outside a tag.** `<post>` opens per close go
  1.7 → 87; late `post_missing` turns include agents writing in the substrate's
  own pinned format without wrapping it. Nothing in the objective rewards a
  well-formed tag. — *established* — same
- 🔶 **Single-iteration endpoints are noise.** Per-iteration standard deviation
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

**Nothing.** No job has been submitted since 2026-06-23.

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
  runs. This is cheap and it is the largest single gap in the project.
- **The terminal programme past Stage 0.** Stage 1 started twice and wrote
  nothing; Stages 2, 3, 3.5 and 4 have configs and no runs — including the
  embodiment config its own header calls "THE load-bearing experimental config
  for the framework's core claim".
- **A no-migration control** at the same iteration range as the migration runs.
  Without it the residents' trajectories cannot be attributed to the migrant.
- **The dual-environment control.** The long-episode variant that would say
  whether the failure is the environment split or the halved token budget got
  six iterations.
- **Seed replicates at 2 and 4 agents**, which is exactly where the
  population-size evidence is weakest.
- **Any forum run past iteration 300.** The decline has no measured end.

### Open questions

- **What fraction of undelivered turns are untagged posts rather than
  deliberation?** Reading a sample finds both, plus token loops. The number
  decides whether the closing public channel is a policy pathology or a
  substrate one, and it needs a classifier or a hand-coded sample. —
  `runs/_tier_b/b6_token_accounting/NOTES.md`
- **Does the closing public channel cause the decline, or accompany it?** The
  channel closes first, by up to 200 iterations, in 9 of 12 runs where both are
  observable — but that is association. The ablation that would settle direction
  is to type only public post tokens as actions and retrain; it has not been
  run. — `runs/_tier_b/b4_public_channel/NOTES.md`
- **Why does the social-terminal substrate not decline?** Its two full-horizon
  runs end alive where every forum population does not. Whether that is the
  substrate, the model, or the shorter public channel is unseparated.
- **Is the decline the same event as the entropy collapse?** `policy_collapse`
  is the dominant late failure mode in the forum runs, and the entropy bonus is
  the one hyperparameter known to control entropy catastrophically. Nobody has
  checked whether the safe-side default is merely slower rather than safe.
- **What is the right evaluation horizon?** Iteration 200 was chosen before the
  decline was visible. Every population-level claim in this project is a claim
  about that iteration, and the runs themselves now say it is not a converged
  state.
- **How large is the gradient bias in our substrates?** The exact expression is
  a conditional covariance between an agent's action score and the environmental
  surprise that follows it. The measurement is now known to be well posed —
  provenance resolves at 100% — and needs a GPU scoring pass that has not been
  run. — `runs/_tier_b/b5_env_likelihood/NOTES.md`
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
