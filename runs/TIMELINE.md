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

---

## What is currently running

| job | started | what |
|---|---|---|
| 6256587 | 2026-09-02 | `A_surrogate_seed1` — environment-likelihood experiment, arm A |
| 6256588 | 2026-09-02 | `B_corrected_seed1` — arm B, corrected return |

The first job submitted since 2026-06-23. Seeds 2-5 of each arm are written and
held until the provenance void conditions are confirmed —
[`configs/env_likelihood_bias/README.md`](../configs/env_likelihood_bias/README.md).
