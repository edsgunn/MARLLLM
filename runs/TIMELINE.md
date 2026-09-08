# Timeline

Every campaign in the order it ran, one line of outcome each. **This document
holds no beliefs** — it is the chronology, and `STATE.md` is the synthesis.

Dates are the run directories' own modification times, so they are when a
campaign finished writing rather than when it was conceived. Counts are run
directories, not jobs — several directories were submitted more than once.

---

## April 2026 — the framework, and two invalidating bugs

| dates | campaign | n | outcome |
|---|---|---:|---|
| 04-17 | `old_trainer/negotiation` | 1 | The first run in the project. |
| 04-17 → 04-23 | `old_trainer/negotiation_experiments` | 19 | Model, size and learning-rate sweep on deal-or-no-deal. |
| 04-20 → 04-21 | `old_trainer/concordia_experiments` | 13 | The Concordia scenario set. **The only place these ever ran.** |
| 04-21 → 04-23 | `old_trainer/experiments` | 4 | Scratch. |
| 04-24 | `old_trainer/negotiation_{prompt,lora_shared,role}_experiments` | 36 | Prompt variation, shared-base LoRA, role shuffling. |
| 04-24 | — | — | **Population-based training replaces the per-experiment trainers.** Everything above is now `old_trainer/`. |
| 04-24 → 04-27 | `no_context_management/*` | 67 | The negotiation families re-run under the population trainer. |
| 04-27 | — | — | **Context formatter added** (`e760344`). Everything above read its context as an undelimited token stream; all 67 runs are invalidated. |
| 04-27 → 04-28 | `old_delimiters/*` | 67 | The same families re-run again. |
| 04-28 | `terminal_experiments` | 2 | Stage 0 passes its σ-routing gate. Stage 1 starts twice and writes nothing. **The terminal line stops here.** |
| 04-30 | `old_delimiters/negotiation_entropy_kl_ablation`, `.../phase_a` | 12 | Entropy/KL ablation and the first Phase A attempt. |
| 04-30 | — | — | **Special-token positions fixed** (`5fa0b3f`). All 79 `old_delimiters/` runs are invalidated. The negotiation families have now been run three times. |

## May 2026 — Phase A, then the forum campaign

| dates | campaign | n | outcome |
|---|---|---:|---|
| 05-01 | `phase_a_substrate_validation` | 6 | Training completes; **the evaluation that defines the claim is never run.** Perception loss halves; the headline condition is seed-unstable. |
| 05-02 → 05-04 | `negotiation_population_experiments` | 2 | Perception loss falls tenfold; task success rate stays at exactly 0.000 throughout. |
| 05-02 → 05-03 | `prompts_as_padding/cultural_emergence` | 6 | The first cultural-emergence runs. Invalidated: the character prompt was typed `PAD`, so it was outside every loss. |
| 05-03 | — | — | **`prompt_as_observation` defaults to true.** The prompt enters the perception loss, which is the mechanism absorption is measured through. |
| 05-04 | `smoke_test` | 1 | vLLM rollout path confirmed end to end. |
| 05-04 → 05-22 | **`cultural_emergence`** | 78 | The main campaign. Substrate screen, substrate × population-size grid, five-seed replicate sets, the 4×4 regularisation sweep, the two-turn and dual-environment arms. |
| 05-05 → 05-06 | `migration_experiments` | 24 | 16 migrations across two substrate pairs; the first batch of 8 is invalidated — the migration checkpoints never loaded. |
| 05-11 | `cultural_emergence_14B`, `_32B` | 3 | **No data.** The scaling question is untouched above 7B. |
| 05-14 | `cultural_emergence_qwen3p5_attic` | 12 | **No data.** Twelve submissions, three per configuration, none trained. Never diagnosed. |
| 05-14 → 05-15 | `cultural_emergence_qwen3_packed*` | 4 | **No data.** Sequence packing has never produced a training iteration. |
| 05-15 | `cultural_emergence_qwen3` | 4 | Qwen3-4B/8B port cleanly and drift far less than Qwen2.5-7B. All truncated below iteration 105. |
| 05-15 → 05-16 | `cultural_emergence_qwen3_async*` | 4 | Async rollout under a PPO clip matches synchronous training on every metric. |

## June 2026 — the social-terminal substrate

| dates | campaign | n | outcome |
|---|---|---:|---|
| 06-03 → 06-05 | `social_terminal` | 6 | A spatial fog-of-war substrate. Three runs lost to memory tuning; the traces of one found agents confabulating the results of failed reads, which produced the turn-ending upgrade. **`run04` and `run06` are the only runs in the project that reached their configured horizon in a healthy state.** |
| 06-23 | `sanctioning` | — | Rung 0 over the two five-seed forum families: constant approval, near-absent disapproval, no trend across training. Rung 1 is gated on a judge validation that has not been run. |

## July–August 2026 — analysis, no new training

| dates | work | outcome |
|---|---|---|
| 07-31 | `_seed_replicate_regimes`, `_regime_threshold_sweep_3d` | The population-size claim is re-stated at the population level; causal language is dropped. The regime taxonomy is re-framed as a descriptive partition of a continuous, bimodally-occupied space. |
| — | `_absorption_by_regime`, `_absorption_swap_by_regime`, `_absorption_threshold_sensitivity` | Absorption survives a swapped-character control in the coherent stratum; the "LoRA capacity" reading of Robotic Athanor is dropped. |
| — | `_pmi_decomposition` | Character and environment absorption separate cleanly. The cultural-learning measurement fails its own iter-25 baseline and does not stand. |
| — | `_trajectory_kl*`, `_diagnostics_d2d3`, `_policy_distance` | Cross-population divergence replicates on three substrates and holds at the policy level. **The regime reinterpretation built on it is retracted** — it was an entropy-overconfidence artefact. |

## September 2026

| date | work | outcome |
|---|---|---|
| 09-01 | Retrospective documentation | `STATE.md`, `PROCESS.md`, this file, and a `NOTES.md` for every live run. Two readings in the existing reports were corrected against the runs' own metrics; see `STATE.md`. |
| 09-01 | Tier-B analyses ([`_tier_b/`](_tier_b/README.md)) | Eight pre-registered re-analyses, no compute. The public channel closes in 63/65 runs and leads the regime decline by up to 200 iterations; the policy-distance headline survives a collapse control on 2 of 3 datasets; no Qwen3 agent ever reaches KL ≥ 1. Two claims written down the same morning were overturned: `thinking_frac` does not measure private deliberation, and restoring the dropped environment term is not free. |
| 09-02 | Environment-likelihood experiment ([`configs/env_likelihood_bias/`](../configs/env_likelihood_bias/README.md)) | Implemented and submitted: jobs 6256587 (arm A, surrogate return with the environment term logged) and 6256588 (arm B, corrected return). Post tags blanked so nothing an agent emits is discarded and `p_env` is computed rather than estimated. Seeds 2-5 written and held. |
| 09-02 | Tier C item C1 ([`configs/cultural_emergence/c1_small_population_replicates/`](../configs/cultural_emergence/c1_small_population_replicates/README.md)) | Implemented and submitted as jobs 6260359-6260378: 5 seeds × 2 substrates × 2 population sizes, replicating the N=2/N=4 corner of the population-size figure where every cell is n=1. Seed 1 re-run rather than reused — the 05-06 originals ran against an uncommitted tree. Verified in the process that `thinking_enabled` has been an inert key since 05-05; REWRITE.md C4 corrected accordingly. |
| 09-02 | Tier C items C3, C6 and C10 ([`c3_frozen_population/`](../configs/cultural_emergence/c3_frozen_population/README.md), [`c6_extended_horizon/`](../configs/cultural_emergence/c6_extended_horizon/README.md), [`c10_term_ablation/`](../configs/cultural_emergence/c10_term_ablation/README.md)) | Nine jobs submitted, 6260985-6260989 and 6261085-6261088. Two config keys that the loss and trainer honoured but no population config could reach were wired: `frozen_agents` (implemented in `PopulationTrainer`) and the three `alpha_*` term weights — so every forum run in this project until now was at 1/1/1 with no way to say otherwise. C3 trains one learner against seven frozen members; C6 resumes the 8- and 16-agent references past 300; C10 is the perception-only/action-only dissociation on the reference population. |
| 09-02 | Tier C item C2 ([`c2_population_size/`](../configs/cultural_emergence/c2_population_size/README.md)) | Fourteen jobs submitted, 6261670-6261683, plus a smoke at 6262498. Implemented the per-episode roster subset the restricted-pairing arm needs: `pairing_strategy` turned out to be a fourth dead key, a no-op on every forum substrate, so REWRITE.md 6.3's "one new pairing strategy" was the wrong mechanism. `partner_graph_degree` partitions the population into fixed groups instead. That arm's five seeds are written and held on the smoke. |

| date | work | outcome |
|---|---|---|
| 09-03 | C2's partner-graph smoke (6262498) | **Passed** in 5m39s. Every episode roster was size 4 rather than 16, all four groups appeared four times each, and membership never changed. The restricted-pairing arm's five seeds were released on it. |
| 09-03 | Walltime audit | Every job had inherited the 24 h `workq` cap: 1,080 node-hours reserved for ~250 hours of work, none of it backfillable. Limits re-estimated from each reference run's p90 per-iteration cost and lowered in place with `scontrol`; reservation down 60% with a 1.5x margin. The method is now in `PROCESS.md`. |
| 09-03 | First Tier-C results land | 36 of 50 jobs finished at their full iteration count, none truncated. One failure: `c2_frozen_majority_seed5` died in 63 s on `submit_batch`'s GPU-memory preflight guard — 549 MiB of a previous tenant's memory against its 500 MiB threshold, a node-cleanup race — resubmitted as 6279079. **No run has been analysed yet.** |
| 09-03 | The walltime estimate that was wrong | C2's restricted-pairing arm was sized from its 3-iteration smoke (54 s/iter) and given 10 h; steady state is ~150 s/iter, so all five project 9.4–11.1 h against it. Per-iteration cost drifts 3× over a run and the growth is in generation, not the update (`time/rollout_frac` 61% → 77%). A limit cannot be raised once set, so the recovery is `--resume`. Configs re-timed to 19 h; `PROCESS.md` now says never to size from a smoke. |
| 09-03 | Tier-C analysis ([C1](cultural_emergence/c1_small_population_replicates/NOTES.md), [C3](cultural_emergence/c3_frozen_population/NOTES.md), [C10](cultural_emergence/c10_term_ablation/NOTES.md), [C2 partial](cultural_emergence/c2_population_size/NOTES.md), [C9](env_likelihood_bias/NOTES.md)) | **C10 is the result of the batch:** perception-only ends 300 iterations alive on both seeds with perception loss 2.13 → 0.70 and 61/64 well-formed posts, against 36/64 and eight degenerate agents for the full loss; action-only misses 24–30 of 64 turns. **C1 fired its falsification condition** — the N=2/N=4 corner is seed noise, no metric separates the sizes. **C3 is inconclusive** at three runs. **C9 is void** — both arms breach their provenance condition and logged none of their diagnostics. |
| 09-03 | The objective's own settings ([C11](../configs/cultural_emergence/c11_discount_normalisation/), [c9 post hoc](_tier_b/c9_post_hoc_env_term/NOTES.md)) | The dropped environment term recovered from arm A's checkpoints: ~43% of the return's magnitude, rank correlation 1.000, so it does not reorder updates. Then two defects in the objective itself. γ=0.99 is applied positionally per token — a 100-token horizon against a 384-token post, giving a 48× within-post credit gradient where the derived γ=1 is flat — and it is variance reduction from §6.2, not part of the derivation. Return normalisation is not free here because the action term is summed with a supervised one, and its 1/std gain rises as returns flatten. Neither key was reachable from any config: dead keys five and six. Both wired; C11 submitted as 6280551/6280553. |
| 09-03 | C11, C6 and the N=16 reference ([C11](cultural_emergence/c11_discount_normalisation/NOTES.md), [C6](cultural_emergence/c6_extended_horizon/NOTES.md), [C2](cultural_emergence/c2_population_size/NOTES.md)) | **The apparent recovery is an artefact, and it invalidates the metric rather than the runs.** The 8-agent reference scores `alive` 0.62 at iteration 900 while delivering 5 well-formed posts in 64 turns, four of five "alive" agents posting nothing. Perception loss is computed over observation tokens, so silence makes the stream predictable and pushes `perc_loss` through the classifier's gate — delivery and perception correlate at r = +0.85 within the run while `alive` moves the other way. Whether the decline bottoms out is unanswered and not answerable with this classifier. Caught by reading the traces, which is the check PROCESS.md says metrics cannot do. **C11**: γ=1 alone ends at 0.50 against the reference's 0.00, but removing normalisation reaches perception loss 37 with 62 of 64 turns delivering nothing — the arm mis-scaling, exactly as its header predicted, not a test of the derived objective. **The N=16 reference** is now five seeds spanning 0.38–0.88, and the frozen-majority claim read against its single best seed is withdrawn. |
| 09-03 | Content vs scaffolding perception loss ([content_perception](_tier_b/content_perception/NOTES.md)) | **The proposed repair fails, and its own pre-registered check caught it.** Measuring perception loss on post content only was meant to make the metric immune to silence; content turns out *more* predictable than scaffolding at 12 of 13 checkpoints, and content-only loss is lowest where delivery is lowest — because a degenerate `</?` loop is content and is maximally predictable, 82% of everything delivered at iteration 900. Predictability is the wrong target when the agents control the stream. Two incidental findings: the framing is not deterministic either, since it carries the speaker's name; and delivery rate is monotone 0.98 → 0.08 across iterations 25–900 with no recovery, which is the one measure neither silence nor degeneration can move the wrong way. |
| 09-04 | PopArt, and C2 complete ([C11](cultural_emergence/c11_discount_normalisation/NOTES.md), [C2](cultural_emergence/c2_population_size/NOTES.md)) | **PopArt works and rescues nothing.** The value head goes from diverging at 1.45e8 to a stable O(10) loss, value kept in the return's units — and the run still ends at perception 39.6, entropy 0.00, 7 posts of 64. So the scale mismatch is an arm-weighting problem, not a baseline one: a fitted baseline centres the advantage but its scale is the return's spread, σ = 1552. **C2's five arms are in and population size does nothing** — N=4 vs N=16 p = 0.722, volume p = 0.810. The only variable that separates is co-adaptation: frozen-majority learners deliver 0.475 against 0.225, p = 0.032. Two of my own claims withdrawn, both of which rested on `alive`. |
| 09-04 | Novelty search ([C13](novelty_search/NOTES.md)) | **Reversing the return's sign — one key — produces the behaviour the framework was built for, on one of two seeds.** Seed 2 at iteration 300 is doing collaborative mathematics: 89% of posts math-bearing, mean 660 characters, zero fragments, against the reference's 1 of 42 and five fragments. Seed 1 is not, at 4 of 53. Delivery holds above the reference through iteration 200 in both and falls anyway by 300, so the dark-room diagnosis is right about the mechanism and incomplete as an explanation. Entropy rises rather than collapsing. Not the best configuration found — C10's perception-only still delivers 64 of 64 on both seeds — but by far the smallest change to produce a transcript like it. |
| 09-08 | Novelty search becomes its own suite ([`novelty_search/`](novelty_search/NOTES.md)) | Moved out of `cultural_emergence/c13_*` — reversing the return's sign is an objective change, not a cultural-emergence cell. Two `margin_notes` seeds submitted (6398717, 6398719) to ask whether tool use improves under a reversed sign, and a six-cell population-size sweep at N = 2/4/16 × 2 seeds (6398722–6398727). `submit_batch.py` gained a guard that skips configs whose STATUS says CLOSED/VOID/ARCHIVED/SUPERSEDED/HELD, after a directory submit nearly resubmitted two completed and analysed runs over their own data. |
| 09-08 | The regularisation grid, re-read ([sweep](cultural_emergence/kl_beta_sweep_8agent_ashbourne_gc/NOTES.md), [zoom](cultural_emergence/beta_zoom_kl0p5_8agent_ashbourne_gc/NOTES.md)) | **Asked whether policy entropy can be made to rise without the perception loss collapsing; the 20 cells already on disk answer no, and needed no compute to say so.** They had only ever been read on `alive`, which across them is uncorrelated with the well-formed post rate (ρ = −0.10, p = 0.66) — the delivery rate was in `metrics.jsonl` all along. Re-read on delivery and on the *change* in entropy: **neither knob touches the channel** (delivery 0.98 → 0.50 in all 20, no trend in either, not one cell above 0.8, across a 50× range in both); **the entropy bonus is a step function, not a dial** (|Δentropy| ≤ 0.12 for `beta` ≤ 0.05, ≥ +0.16 for `beta` ≥ 0.10, complete separation, p = 1.2e-5); and **above the step, entropy and perception loss are locked** (ρ = +0.853, p = 4.4e-12). One claim overturned: the zoom's "the KL anchor cannot protect against the entropy bonus" is wrong with `kl_coef` varied — at `beta` = 0.10 it suppresses the entropy rise sixfold, monotone over four points. The zoom's best-cell claim is withdrawn; it ends at delivery 0.474. **And the accidental duplicate turns out to be a calibration, not just a reproducibility failure:** same config, same seed, the two runs differ by 0.007 in Δentropy but **0.431 in Δperception and 0.173 in delivery**, which is the noise floor every reading in the campaign is now held against. |
| 09-08 | The entropy-anchor ridge ([`entropy_anchor_ridge/`](cultural_emergence/entropy_anchor_ridge_8agent_ashbourne_gc/NOTES.md)) | Eight jobs, 6404722–6404729. The one direction the old grid left untested: if the anchor is what stops a large bonus wrecking the text, a stronger anchor should raise the safe `beta`. `kl_coef` had never been run above 0.5. (2.0, 0.01) as control, (2.0, 0.2) as the key test — that `beta` fails at 0.5 — plus (2.0, 0.4) and (5.0, 0.4), two seeds each because Δperception and delivery are unreadable from one run at the noise floor measured this morning. Refutation is pre-registered as a real outcome: it would mean the uniform entropy bonus is the wrong instrument, and close the question. |

---

## What is currently running

**The entropy-anchor ridge**, 8 jobs (6404722–6404729), submitted 2026-09-08 at
15:11 — 4 cells × 2 seeds, 300 iterations each, 08:00:00 limits.

**Novelty search**, 3 of 8 jobs still going: `margin_notes` seeds 1 and 2
(6398717, 6398719) and the N=16 size cell (6398727). The other five size cells
have finished and are **not yet analysed**.

`env_likelihood_bias/B_corrected_seed1` is **held**: instrument
`observation_segments` before a fourth attempt.

`provenance: true` is a **silent no-op on tagged substrates** and has now
blocked two analyses. Fix it or make it refuse.
