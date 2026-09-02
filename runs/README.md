# Runs — index

A **map** of what exists and where. It holds no beliefs: what a run *showed* is
in its `NOTES.md`, and what the project believes is in `STATE.md`.

Run data is not in git. `.gitignore` keeps `runs/**` out and lets `runs/**/*.md`
back in, so a clone has every write-up and none of the ~28 TB of checkpoints,
traces, metrics and figures they describe.

---

## Live campaigns

| directory | n | substrate / task | what it is | write-up |
|---|---:|---|---|---|
| [`cultural_emergence/`](cultural_emergence/) | 78 | forum substrates, Qwen2.5-7B | The main campaign: substrate screen, substrate × population-size grid, five-seed replicate sets, the 4×4 KL/β sweep, two-turn and dual-environment arms | per-run `NOTES.md`; sweeps have a family `NOTES.md` |
| [`social_terminal/`](social_terminal/) | 6 | `ashbourne_school`, three rooms with fog of war | A spatial substrate; the observation-principle upgrade | per-run `NOTES.md` |
| [`migration_experiments/`](migration_experiments/) | 16 | forum substrates | One agent's adapter moved into a population that has never met it | [`NOTES.md`](migration_experiments/NOTES.md) + per-run |
| [`phase_a_substrate_validation/`](phase_a_substrate_validation/) | 6 | `deal_or_no_deal` | The framework's first pre-registered experiment | [`NOTES.md`](phase_a_substrate_validation/NOTES.md) + per-run |
| [`sanctioning/`](sanctioning/) | — | analysis over the forum corpus | Rungs 0–1 of the sanctioning measurement framework | [`NOTES.md`](sanctioning/NOTES.md) |
| [`negotiation_population_experiments/`](negotiation_population_experiments/) | 2 | `deal_or_no_deal` | The population trainer on the negotiation substrate | per-run `NOTES.md` |
| [`cultural_emergence_qwen3/`](cultural_emergence_qwen3/) | 4 | `study_group_ashbourne_gc` | Qwen3-4B/8B portability | per-run `NOTES.md` |
| [`cultural_emergence_qwen3_async/`](cultural_emergence_qwen3_async/) | 2 | `study_group_ashbourne_gc` | Async rollout + PPO clip | per-run `NOTES.md` |
| [`terminal_experiments/`](terminal_experiments/) | 2 | filesystem shell | Stage 0 gate; Stage 1 never ran | per-run `NOTES.md` |
| [`smoke_test/`](smoke_test/) | 1 | — | vLLM pipeline check, not an experiment | [`NOTES.md`](smoke_test/vllm_smoke/NOTES.md) |

## Cross-run analyses

Directories under `cultural_emergence/` beginning with `_` hold analyses that
span runs. Each carries its own `REPORT.md`; the `.csv` tables and figures they
link to stay local.

| directory | question |
|---|---|
| [`_population_figures/`](cultural_emergence/_population_figures/) | the regime classification at iteration 200, plus [`trace_verification_iter200.md`](cultural_emergence/_population_figures/trace_verification_iter200.md) |
| [`_population_figures_annotated/`](cultural_emergence/_population_figures_annotated/), [`_qualitative_traces/`](cultural_emergence/_qualitative_traces/) | which dot is which agent, and what that agent actually wrote |
| [`_seed_replicate_regimes/`](cultural_emergence/_seed_replicate_regimes/) | does regime composition replicate across seeds |
| [`_regime_threshold_sweep_3d/`](cultural_emergence/_regime_threshold_sweep_3d/) | does the taxonomy survive its own thresholds |
| [`_absorption_summary/`](cultural_emergence/_absorption_summary/), [`_absorption_by_regime/`](cultural_emergence/_absorption_by_regime/) | per-token PMI absorption, stratified by regime |
| [`_absorption_swap_by_regime/`](cultural_emergence/_absorption_swap_by_regime/) | is it absorption of *this* character, or generic drift |
| [`_absorption_threshold_sensitivity/`](cultural_emergence/_absorption_threshold_sensitivity/) | the same result without any thresholds |
| [`_pmi_decomposition/`](cultural_emergence/_pmi_decomposition/) | character vs environment vs history components |
| [`_trajectory_kl*/`](cultural_emergence/_trajectory_kl_combined/) | do parallel populations diverge |
| [`_policy_distance/`](cultural_emergence/_policy_distance/), [`_diagnostics_d1_coherent_only/`](cultural_emergence/_diagnostics_d1_coherent_only/), [`_diagnostics_d2d3/`](cultural_emergence/_diagnostics_d2d3/) | is that divergence a policy fact or a sampling artefact |
| [`_seed_aggregate_inner_loop/`](cultural_emergence/_seed_aggregate_inner_loop/) | pooled inner-loop seed data |

## Tier-B analyses

[`_tier_b/`](_tier_b/README.md) — eight questions answered from data already on
disk, no new training. Each carries its pre-registration in the docstring of its
script in [`../scripts/tier_b/`](../scripts/tier_b/) and its findings in a
`NOTES.md`. Two of them overturned a claim that had already been written down.

## Attics — runs that cannot be cited

Never deleted. Each directory has a `README.md` giving the reason.

| directory | n | why |
|---|---:|---|
| [`old_trainer/`](old_trainer/) | 75 | predate population-based training (2026-04-24) |
| [`no_context_management/`](no_context_management/) | 67 | context was an undelimited token stream (2026-04-27) |
| [`old_delimiters/`](old_delimiters/) | 79 | special tokens in the wrong positions (2026-04-30) |
| [`prompts_as_padding/`](prompts_as_padding/) | 6 | the character prompt was typed `PAD`, outside every loss (2026-05-03) |
| [`cultural_emergence/unshuffled/`](cultural_emergence/unshuffled/) | 4 | fixed agent-to-role assignment |
| [`cultural_emergence/archive/`](cultural_emergence/archive/) | 2 | superseded first attempts |
| [`migration_experiments/_archive_2026-05-06_kl0_bug/`](migration_experiments/_archive_2026-05-06_kl0_bug/) | 8 | the migration checkpoints never loaded |
| [`cultural_emergence_qwen3p5_attic/`](cultural_emergence_qwen3p5_attic/) | 12 | Qwen3.5 never trained; never diagnosed |
| [`cultural_emergence_qwen3_packed_attic/`](cultural_emergence_qwen3_packed_attic/) | 2 | sequence packing never trained |
| [`cultural_emergence_qwen3_async_attic/`](cultural_emergence_qwen3_async_attic/) | 2 | async workers started, trainer never consumed; fixed on resubmission |

## Jobs that produced no data

Findable on purpose, so the failure is not a silent gap between job ids.

| directory | what was attempted |
|---|---|
| [`cultural_emergence_14B/`](cultural_emergence_14B/) | Qwen2.5-14B, 8 and 16 agents |
| [`cultural_emergence_32B/`](cultural_emergence_32B/) | Qwen2.5-32B — did not reach a `config.json` |
| [`cultural_emergence_qwen3_packed/`](cultural_emergence_qwen3_packed/) | sequence packing |
| [`cultural_emergence/run12_64agent_7B_study_group_ashbourne_gc/`](cultural_emergence/run12_64agent_7B_study_group_ashbourne_gc/) | the top of the population-size ladder |
| [`cultural_emergence/run11_16agent_7B_margin_notes_2turns/`](cultural_emergence/run11_16agent_7B_margin_notes_2turns/) | the third arm of the two-turn experiment |
| [`cultural_emergence/run9_8agent_7B_ashbourne_dual_env.failed_4460295/`](cultural_emergence/run9_8agent_7B_ashbourne_dual_env.failed_4460295/) | job 4460295; two records, the second with a perception loss of 30 nats |
| [`terminal_experiments/01_stage1_base_model/`](terminal_experiments/01_stage1_base_model/) | Stage 1 of the terminal programme |

## Reading order

New to the project: [`../STATE.md`](../STATE.md), then
[`TIMELINE.md`](TIMELINE.md), then whichever `NOTES.md` a claim cites.

New to the cultural-emergence results specifically:
[`cultural_emergence/run8_16agent_7B_study_group_ashbourne_gc/NOTES.md`](cultural_emergence/run8_16agent_7B_study_group_ashbourne_gc/NOTES.md)
(the most-analysed population), then
[`cultural_emergence/_trajectory_kl_combined/REPORT.md`](cultural_emergence/_trajectory_kl_combined/REPORT.md)
and [`cultural_emergence/_diagnostics_d2d3/REPORT.md`](cultural_emergence/_diagnostics_d2d3/REPORT.md)
together — the second retracts a reading of the first.

Template for a new run's write-up: [`_TEMPLATE_NOTES.md`](_TEMPLATE_NOTES.md).
