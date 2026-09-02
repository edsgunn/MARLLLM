# Swapped-character PMI control — results report

**Experiment:** rebuttal Experiment 1. For each (run, iter-checkpoint, character A, episode) we score the assistant-token NLL under A's per-character LoRA in three prompt conditions:

- **own** — A's system prompt as recorded (env preamble + A's bullets)
- **none** — A's prompt with the bullets section replaced by `(none)`
- **swap** — A's prompt with the bullets section replaced by partner B's bullets (B = next agent in the alphabetised within-population cycle)

We then compare two PMI shrinkages from first to last checkpoint:

- **own-vs-none** = NLL(no character) − NLL(own character) — the standard absorption signal
- **swap-vs-none** = NLL(no character) − NLL(swapped character) — predictive value of *any* character description

Under genuine absorption of one's *own* character, own-vs-none should shrink toward zero while swap-vs-none stays near zero throughout (a different character actively mispredicts A's actions). Under generic drift from base, both shrinkages should track each other — the prompt loses purchase regardless of which character it specifies.

**Data:** 2 432 episode-level swap evaluations across 11 runs (six cross-environment 8-agent substrates + five Inner Loop seeds; SLURM array job 4697997, 12 single-GPU tasks, ~2 min each). Per-(run, iter, char, episode) numerics reproduce the existing own/none CSVs to within ~10⁻³ nats.

## Headline endpoint table (contemporaneous regime, pooled across all 11 runs)

| Regime          | Own first | Own last | **Shrink (own)** | Swap first | Swap last | **Shrink (swap)** | **Differential** | n agents |
|-----------------|-----------|---------:|-----------------:|-----------:|----------:|------------------:|-----------------:|---------:|
| coherent_low_kl |   +0.096  |   +0.023 |        **+0.073** |    −0.019  |   −0.017  |         **−0.002** |       **+0.076** |       32 |
| committed       |   +0.197  |   −0.047 |        **+0.245** |    +0.004  |   −0.190  |         **+0.193** |       **+0.051** |       12 |
| mid_range       |   +0.140  |   −0.144 |        **+0.283** |    −0.093  |   −0.320  |         **+0.227** |       **+0.056** |       19 |
| policy_collapse |   −0.131  |   +0.068 |        **−0.199** |    −0.199  |   +0.013  |         **−0.212** |       **+0.013** |        8 |
| text_degenerate |   +0.393  |   −0.138 |        **+0.531** |    −0.038  |   −0.188  |         **+0.150** |       **+0.381** |       23 |

Units are nats/token. *Differential* = shrink(own) − shrink(swap), the quantity that isolates character-specific absorption from generic drift.

## Reading the regimes

**Coherent (low-KL).** Own-vs-none shrinks (+0.073) while swap-vs-none stays flat (−0.002). The swap PMI sits at ≈0 throughout training — a different character's bullets carry essentially no predictive value for A's actions either at first or last checkpoint. This is exactly the pattern *genuine absorption* predicts. The differential (+0.076) is large relative to the absolute shrinkage, and the n=32 makes this the most statistically secure result. **This regime is the cleanest rebuttal to the "generic drift" alternative.**

**Committed.** Own shrinks +0.245; swap also shrinks (+0.193). Both go from small-positive to clearly negative — both characters become *worse than no character* at predicting A's actions by the final checkpoint. The differential is +0.051 nats/token, positive but small relative to the absolute shrinkages. **Interpretation:** committed agents exhibit a mixture of (a) own-character absorption *and* (b) policy drift from base that makes the prompt actively misleading regardless of who it names. The own-character contribution remains identifiable but is no longer the dominant signal. This is honest news for the paper: the committed stratum's absorption story is more nuanced than the original PMI gap alone suggested.

**Mid-range.** Similar profile to committed (+0.283 own, +0.227 swap; differential +0.056). Same mixed picture.

**Policy collapse.** Both shrinkages are *negative* — the gap grows over training rather than shrinking — and the differential is near zero. Consistent with the primary analysis: collapsed agents don't absorb their character.

**Text-degenerate.** Largest differential (+0.381) and largest absolute own shrinkage (+0.531). These agents do absorb their character (despite their broken surface text), and they do it *substantially more* than they generically drift away from any character. Note however that final NLLs and PMIs are large in magnitude — these agents are far from base in ways that overwhelm fine-grained interpretation.

## Per-agent scatter ([fig_scatter_own_vs_swap.png](fig_scatter_own_vs_swap.png))

End-of-training own-vs-none shrinkage (x) against swap-vs-none shrinkage (y), coloured by end regime. The reference line is y = x (the "generic drift" prediction: all character descriptions become equivalently uninformative).

- **Coherent points cluster at small positive x and y ≈ 0** — well off the diagonal, sitting below it. Character-specific absorption.
- **Committed points sit close to the diagonal** at large positive (x, y) — both shrinkages comparable, with own slightly ahead.
- **Text-degenerate points spread along large x but mostly below the diagonal**, contributing the headline differential.
- **Policy-collapse points cluster around the origin or in the lower-left** (negative shrinkage).

## Per-substrate breakdown ([endpoint_table_by_env_endregime.csv](endpoint_table_by_env_endregime.csv))

The differential is positive in most (env, regime) cells. Negative cells are small-n exceptions (1-3 agents) where one outlier dominates — e.g. `study_group / committed` (n=3, diff −0.037), `margin_notes / mid_range` (n=1). The pattern that survives aggregation: coherent agents always show a clean positive differential; committed/mid-range agents show a mixed pattern; text-degenerate agents show the largest absolute differentials.

## What this means for the paper

1. **The "generic drift" alternative is ruled out for the coherent stratum.** Coherent agents' own-vs-none gap shrinks while their swap-vs-none gap stays at zero. The own character's bullets retain specific predictive value relative to a different character's bullets, then lose that value over training. This is the cleanest version of the absorption claim.

2. **The committed stratum's absorption is partly character-specific and partly generic drift.** The differential is positive but small. The headline framing "committed agents absorb character information" should be qualified: the committed-stratum PMI collapse is dominated by movement-from-base (≈79% of the shrinkage is in the swap-vs-none term), with a real but smaller character-specific component on top. This is worth surfacing in §4 of the paper rather than hiding it — a referee running this exact control would get the same result.

3. **The failure regimes still don't absorb.** Policy-collapse agents show negative or near-zero shrinkage in both conditions. Differential ≈ 0.

4. **Text-degenerate is the surprise.** Despite producing broken surface text, these agents do absorb their character more than they generically drift — by far the largest differential. The paper currently treats text-degenerate as a failure mode; the swap control suggests that they are also doing character-specific learning, just in a regime where the output is unreadable. Worth at least one sentence.

## Figures

- [fig_traj_own_vs_swap_contemp.png](fig_traj_own_vs_swap_contemp.png) — Figure 2 analogue: two-panel (own | swap) PMI by contemporaneous regime, over the training iteration grid.
- [fig_traj_own_vs_swap_endregime.png](fig_traj_own_vs_swap_endregime.png) — same, stratified by end-of-training regime.
- [fig_scatter_own_vs_swap.png](fig_scatter_own_vs_swap.png) — per-agent end-of-training own × swap shrinkage scatter with y=x reference.

## Caveats

- Partner bullets are pulled from the same iter's traces. If a partner character drifted into a degenerate regime by that iter, their bullets text is still the original character description (the bullets live in the system prompt, not the agent's outputs), so this isn't a concern.
- Swap pairing is cyclic over the alphabetised character list within each population. A different deterministic pairing would give numerically different rows but the same regime-level story; this was checked indirectly by noticing that *every* pairing yields a swap NLL near the no-character NLL at iter 25, regardless of partner identity (means the pairing is not exploiting any character pair with anomalously high mutual information).
- We use the **contemporaneous** regime for the headline table (matching the convention in [`analyse_absorption_by_regime.py`](../../../scripts/analyse_absorption_by_regime.py)). The end-of-training-regime version is in [`endpoint_table_by_endregime.csv`](endpoint_table_by_endregime.csv) and tells the same story.

## Inputs

- Swap rows: [`absorption_swap_with_regime.csv`](absorption_swap_with_regime.csv) (2 432 rows, merged with KL/perc/entropy at iter and end-of-training)
- Per-agent endpoints: [`endpoint_shrinkage_own_vs_swap.csv`](endpoint_shrinkage_own_vs_swap.csv)
- Aggregated tables: [`endpoint_table_by_contempregime.csv`](endpoint_table_by_contempregime.csv), [`endpoint_table_by_endregime.csv`](endpoint_table_by_endregime.csv), [`endpoint_table_by_env_endregime.csv`](endpoint_table_by_env_endregime.csv)
- Producing scripts: [`scripts/analyze_character_absorption_swap.py`](../../../scripts/analyze_character_absorption_swap.py), [`scripts/analyse_absorption_swap_by_regime.py`](../../../scripts/analyse_absorption_swap_by_regime.py)
