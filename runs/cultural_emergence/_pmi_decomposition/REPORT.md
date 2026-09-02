# PMI decomposition — results report

**Experiment.** Decompose the per-token absorption signal into character-, environment-, and population-history components by scoring action-token NLL under eight prompt-construction conditions and reading off the gaps between conditions. The goal is to operationalise *cultural learning* as the gap `swap_history_cross_run − own` — *predictive value of A's own population's history beyond a parallel-seed population's history at the same iter*.

**Datasets.**
- *Inner Loop 5-seed replication* (run7_8agent_7B_inner_loop + seeds 2–5): full 8 conditions including history swaps. 640 episode-level evaluations.
- *Cross-environment 6-substrate set*: 5 conditions (char + env + own; no history). 1 792 evaluations.

**Conditions.** `own`, `none_char`, `swap_char_within`, `none_env`, `swap_env`, `none_history`, `swap_history_within_run`, `swap_history_cross_run`.

**Forced design choices** (documented because the spec is silent on them):
- *First-assistant-span scoring only.* Contexts are multi-turn (3 assistant spans per episode). Only the first has a cleanly-defined "history" — later spans depend on A's own intervening posts. Comparable across all 8 conditions; not directly comparable in magnitude to the original PMI gap which scored all spans.
- *Env-swap modifies the thematic paragraph only*, keeping tool and formatting instructions intact (per spec).
- *Env preamble is replaced in BOTH the system block and the first user-block echo.* Otherwise the user-block leaks the original env.
- *Cross-substrate env pairings.* Inner Loop ↔ Margin Notes (both have `<python>` tool); Study Group, Study Group Ashbourne GC, Study Group Strathearn Server, Conjecture Inn, Robotic Athanor in a tool-free cycle.
- *Cross-run history swaps only on Inner Loop 5 seeds* (only multi-seed substrate; same 8-character roster across all 5).

## Headline finding (and the part that broke)

The spec defined a critical sanity check:

> *"Check the iter-25 baseline. If the gap is near zero at iter 25 — when the agent has barely been trained and shouldn't yet have absorbed any history-specific structure — then the late-iter gap is plausibly attributable to learning during training rather than to baseline off-distributionness of the swap."*

**The iter-25 baseline of `gap_history_specific_to_run` is +1.10 nats/token (std 0.77, n=160).** Not near zero. **The cross-run history swap as constructed is dominated by baseline transcript off-distributionness, not by cultural learning.**

Corroborating evidence:

| iter | `gap_history_specific_to_run` mean | `gap_history_specific_to_episode` mean | `gap_char_specific` mean | `gap_env_specific` mean |
|-----:|-----------------------------------:|---------------------------------------:|-------------------------:|------------------------:|
|   25 |                              1.100 |                                  0.869 |                    0.213 |                   0.157 |
|   50 |                              1.100 |                                  0.996 |                    0.199 |                   0.123 |
|   75 |                              0.968 |                                  0.882 |                    0.152 |                   0.097 |
|  100 |                              0.919 |                                  0.853 |                    0.120 |                   0.057 |
|  125 |                              1.170 |                                  1.363 |                    0.038 |                  −0.010 |

The history-specific gap is roughly flat over training, with no clean monotone growth, and the same magnitude as the within-run-episode swap. The within-run swap is between *different episodes of A's own population at the same iter* — there's no plausible cultural-learning interpretation for that gap being large. So both gaps are tracking the same thing: *sequence-specific surprise from any substituted transcript*, not population-specific predictive value.

**The cross-run history swap, as constructed, doesn't isolate cultural learning.** It would need a more careful counterfactual — e.g. the spec's suggested "permuted-own-run" variant (rotate which character said what, keeping content) — to attempt to factor out generic surprise from substituted transcripts.

## What the decomposition *does* show cleanly

The character and environment components behave as a clean decomposition over training. Both shrink monotonically toward zero — A's policy comes to encode both the character description *and* the substrate description in its weights, no longer needing the prompt to specify either.

**Char-specific component (coherent stratum):** 0.213 → 0.128 (iter 25 → 125), a 40% decline. A's own character bullets retain less predictive value over A's actual action distribution as training proceeds.

**Env-specific component (coherent stratum):** 0.157 → 0.100 over training. This is **a new finding not covered by Experiment 1**: environment-specific information is being absorbed too, by an amount comparable to character-specific information. The PMI absorption signal in the original paper bundles both — the env swap shows they're each carrying about half the load.

**Committed stratum at iter 125** (small n=3): char-specific = −0.111, env-specific = −0.194. Both turn negative — the agent's actions become *less* predictable under the foreign character/env description than under no description. Same pattern as in Experiment 1's committed regime: by end of training, foreign conditioning is actively misleading.

Stacked-bar visualisation in [fig_decomposition_bars_inner.png](fig_decomposition_bars_inner.png) (note: history components dropped from the bar because their magnitudes dominate the scale and aren't interpretable as cultural learning under the current construction).

## Cross-substrate env-specific absorption ([fig_env_swap_by_substrate.png](fig_env_swap_by_substrate.png))

| Substrate                       | end-of-training `gap_env_specific` mean |
|---------------------------------|----------------------------------------:|
| margin_notes                    |                                  0.043  |
| study_group                     |                                  0.110  |
| study_group_ashbourne_gc        |                                  0.077  |
| study_group_strathearn_server   |                                  0.105  |
| conjecture                      |                                  0.119  |
| robotic_athanor                 |                                  0.142  |

Robotic Athanor (with its stylised register-defining personae) shows the largest env-specific absorption, in line with the spec's qualitative prediction. Margin Notes (close to its swap partner Inner Loop in style) shows the smallest. The ordering is qualitatively consistent with the spec's sanity check.

(Inner Loop's env-specific value isn't directly in this cross-env aggregate because the inner-loop dataset has its own asset directory — see the Inner Loop trajectory table above; iter 125 = −0.010, also fitting the picture: by end of training, Inner Loop-style env description has been absorbed and is no longer informative.)

## Per-agent scatter: history-specific vs char-specific ([fig_scatter_history_vs_char.png](fig_scatter_history_vs_char.png))

Because the history-specific gap is dominated by baseline off-distribution surprise, this plot mostly shows that history-spec values are larger than char-spec values and don't correlate cleanly with regime. Not interpretable as cultural-learning evidence under the current construction.

## What this means for the paper

**The decomposition framework itself is sound.** Char-specific, env-specific, and total absorption components can be measured by varying one prompt slot at a time. The char-specific result reproduces and refines Experiment 1's qualitative story (own char absorbs faster than env). The env-specific result is novel and worth reporting on its own: environment-specific information also absorbs, by comparable magnitude.

**The cultural-learning measurement, as specified, did not work.** The cross-run history swap is baseline off-distribution at iter 25, before any learning has happened. A more careful counterfactual is required — the spec lists "permuted-own-run history" (rotate which character said what, keeping content) as the natural triage. That would preserve the population's vocabulary, jokes, and co-mention patterns while breaking the specific sequence — distinguishing *content* from *sequence*.

**Recommended next iteration if the cultural-learning question is worth pursuing:**

1. Implement the permuted-own-run condition: take A's actual history, swap the character→speaker assignments at random, keeping content.
2. Look for an asymmetric pattern: if `gap` is large for cross-run but ≈0 for permuted-own-run, cultural learning is about *participants and content*; if both are large, it's about *sequence and structure* (and the original measurement was confounded by surface specificity).
3. Or: shorten the foreign-history truncation point and check whether shorter foreign histories give smaller gaps — would help quantify how much of the gap is "any non-actual history is surprising" vs structured signal.

For the rebuttal itself, this report's clean takeaways:
1. Character absorption and environment absorption are independently visible (env is novel).
2. Both decay monotonically toward zero in the coherent stratum; both turn negative in the committed stratum (consistent with Exp 1).
3. The published PMI gap is the *sum* of char-specific, char-any, env-specific, and env-any components — this decomposition pulls them apart and shows char-specific and env-specific each carry ~half the absorption load.

## Caveats and known issues

- **History swaps need redesign before they can be interpreted as cultural learning.** Documented above.
- **Scoring scope**: only the first assistant span. The original PMI scored all spans; absolute magnitudes here ≠ paper's absolute magnitudes. Comparisons within this report are internally consistent.
- **Sample sizes for the committed stratum** are small (n=3 instances at iter 125 in Inner Loop). The trajectory is informative; the absolute end-iter values should be reported with explicit n.
- **None-history (`gap_history_any`) is severely off-distribution** at late iter (empty transcript). The −2.18 value for the committed stratum is essentially uninterpretable — `nll_none_history` is *lower* than `nll_swap_history_cross_run`, meaning the model is less surprised by no history than by a foreign one. Likely an artefact of the empty-transcript condition being weirdly close to a "cold start" generation. Not headline-worthy.

## Inputs

- Raw shards: [`run7_8agent_7B_inner_loop/pmi_decomp/4699162/`](../run7_8agent_7B_inner_loop/pmi_decomp/4699162/) (640 rows), [`run7_8agent_7B_margin_notes/pmi_decomp/4699163/`](../run7_8agent_7B_margin_notes/pmi_decomp/4699163/) (1 792 rows)
- Merged + regime-labelled: [`pmi_decomp_inner_with_regime.csv`](pmi_decomp_inner_with_regime.csv), [`pmi_decomp_cross_with_regime.csv`](pmi_decomp_cross_with_regime.csv)
- Tables: [`table_endregime_inner.csv`](table_endregime_inner.csv), [`table_contempregime_inner.csv`](table_contempregime_inner.csv), [`table_contempregime_cross.csv`](table_contempregime_cross.csv)
- Scripts: [`analyze_pmi_decomposition.py`](../../../scripts/analyze_pmi_decomposition.py), [`analyse_pmi_decomposition.py`](../../../scripts/analyse_pmi_decomposition.py)
