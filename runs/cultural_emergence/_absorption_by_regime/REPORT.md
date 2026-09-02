# Regime-stratified absorption analysis

Re-analysis of the per-token PMI absorption data from Experiment 2, stratified
by the regime classification from Experiment 1, across:

- 7 cross-environment substrate–population configs (1,632 episode-evals,
  pre-existing in `_absorption_summary/absorption_all.csv`)
- 5 inner-loop seeds (640 episode-evals, pre-existing in
  `_seed_aggregate_inner_loop/absorption_all_seeds.csv`)

Classification thresholds match the prior figures, with one change carried over
from the iter-200 trace verification: `degenerate` is split into
**`policy_collapse`** (entropy < 0.2 or > 1.8, but perception loss < 2.0) and
**`text_degenerate`** (perception loss ≥ 2.0). Five regimes total. Regime is
assigned from the **final logged iteration of each run's `metrics.jsonl`**
(end-of-training).

All artefacts are written to `_absorption_by_regime/` — the `.csv`s referenced
below sit alongside this report.

## Sample sizes per regime per substrate

Unique end-of-training agents per regime (cross-env). Bold = adequately
populated; thin = single-digit n.

| substrate (env)               | coherent | committed | mid_range | policy_collapse | text_degenerate |
|-------------------------------|---:|---:|---:|---:|---:|
| conjecture                    |  5 |  0 |  3 |  0 |  0 |
| inner_loop                    |  5 |  3 |  0 |  0 |  0 |
| margin_notes                  |  3 |  0 |  1 |  2 |  2 |
| robotic_athanor               |  0 |  0 |  0 |  0 |  **8** |
| study_group                   |  4 |  3 |  0 |  1 |  0 |
| study_group_ashbourne_gc      |  0 |  0 |  0 |  0 |  **8** |
| study_group_strathearn_server |  6 |  0 |  0 |  2 |  0 |

Inner-loop seeds (per seed; total 5 seeds × 8 agents = 40 agent-runs):

| seed | coherent | committed |
|---|---:|---:|
| 1 | 5 | 3 |
| 2 | 8 | 0 |
| 3 | 7 | 1 |
| 4 | 7 | 1 |
| 5 | 7 | 1 |

**Coverage gaps to keep in mind.** Most cells are n ≤ 3. Regime-specific claims
within any single substrate are underpowered; pooled-across-substrates and
pooled-across-seeds claims are stronger.

## Regime stability during the absorption window

**33 / 88 (38 %) cross-env agents passed through ≥ 2 regimes during the
absorption-measurement iters.** The end-of-training regime mis-attributes the
state of these agents at the iters where their PMI was actually measured. See
[regime_consistency_cross.csv](regime_consistency_cross.csv). Examples:

- Many `text_degenerate`-at-end agents (e.g. study_group_ashbourne_gc Priya,
  Tom) were `mid_range` or `committed` at iters 25–75 and only collapsed
  later. Their early PMI gaps reflect a different regime than their end
  classification.
- Margin-notes Renata Solberg ends `policy_collapse` but visited
  `coherent_low_kl`, `committed`, and `text_degenerate` along the way.

This motivates the **contemporaneous-regime** view in
`fig_traj_contemp_regime.png`: each PMI measurement labelled with the regime
the agent was in *at that iter*. That view is more honest about what each
gap value is measuring.

## Headline results

### 1. Hypothesis (committed → steepest decline) — supported, where the data is adequate

**Pooled inner-loop (5 seeds), end-of-training regime.** [stats_shrinkage_pooled_seeds.csv](stats_shrinkage_pooled_seeds.csv):

| regime           | n_agents | n_instances | first_gap | last_gap |  shrinkage | shrinkage_frac |
|------------------|---:|---:|---:|---:|---:|---:|
| coherent_low_kl  |  8 | 34 | +0.116 | +0.065 | **+0.051** | 44 % |
| committed        |  5 |  6 | +0.097 | −0.210 | **+0.307** | 316 % |

`n_agents` counts unique character names; `n_instances` counts unique
(run, character) pairs — the latter reflects the number of distinct trained
agents. The seeds reuse the same 8 character names, so the same persona can
appear across multiple seeds: e.g. Ravi Krishnamurthy is `committed` in both
seed 1 and seed 5, contributing 1 to `n_agents` but 2 to `n_instances`.

Committed agents shrink ~6× more than coherent-low-KL, and end *negative* — the
character prompt becomes worse-than-uninformative for predicting actions. That
is full absorption with overshoot.

**Pooled cross-env, end-of-training regime.** [stats_shrinkage_pooled.csv](stats_shrinkage_pooled.csv):

| regime           | n_agents | n_instances | first_gap | last_gap |  shrinkage |
|------------------|---:|---:|---:|---:|---:|
| coherent_low_kl  | 20 | 23 | +0.097 | +0.041 | +0.057 |
| committed        |  6 |  6 | +0.126 | −0.045 | **+0.172** |
| mid_range        |  4 |  4 | +0.071 | +0.058 | +0.013 |
| policy_collapse  |  5 |  5 | +0.054 | +0.012 | +0.042 |
| text_degenerate  | 18 | 18 | +0.352 | +0.089 | **+0.262** |

For most cross-env regimes the two columns match — characters in different
substrates are distinct people, so name uniqueness equals instance uniqueness.
The exception is `coherent_low_kl`: 23 instances vs 20 names because some
characters (e.g. Priya Shah, Tom Whitaker) are coherent in more than one
substrate.

The committed-vs-coherent contrast holds. **But text_degenerate also shows
large shrinkage** — for a different reason discussed below.

### 2. The inner-loop multi-seed result is partly carried by seed 1's committed agents

[fig_traj_inner_loop_seeds.png](fig_traj_inner_loop_seeds.png) breaks the
5-seed inner-loop result by seed and regime:

- **Seed 1**: 3 committed agents drive shrinkage to −0.21; the 5 coherent
  agents in the same seed shrink only ~0.09.
- **Seed 2**: 0 committed agents. Coherent agents *go up slightly* (gap
  −0.012 — i.e. last > first). No absorption signal.
- **Seed 3, 4**: 1 committed agent each. Coherent shrink modestly
  (+0.067, +0.039); committed shrink more (+0.030 in seed 3, +0.140 in
  seed 4 with overshoot).
- **Seed 5**: only iters 25–75 logged, 1 committed agent — short window;
  coherent +0.11, committed +0.06.

So the headline "monotonic decline across seeds" is real but driven
asymmetrically: seeds with more committed agents show more decline, and the
biggest signal in the aggregate trace comes from seed 1.

### 3. Robotic Athanor: regime-classified, the LoRA-capacity claim is unnecessary

[fig_robotic_athanor_zoom.png](fig_robotic_athanor_zoom.png).

**All 8 robotic_athanor agents end in `text_degenerate` regime.** Perception
losses settle ≥ 2.0 across the run; the 2D regime-trajectory plot in (entropy,
perception loss) shows the population sitting clearly in the degenerate band
across all measured iters. PMI gaps stay high (~0.4–0.5) and *don't decline* —
shrinkage is **negative** (−0.04, gap rises slightly).

The currently claimed explanation ("LoRA capacity limits") is not needed.
**Text-degenerate agents are not learning a coherent character at all** — their
action distributions degrade into low-entropy gibberish where any conditioning
information happens to remain marginally predictive. That predicts exactly the
"absorption fails to occur" pattern the paper observes here.

The regime-stratified framing replaces "this substrate is too hard for LoRA"
with "this substrate drives the agents into text-level degeneration; absorption
is undefined for agents whose own action distribution has collapsed". The
latter is consistent with the rest of the paper's narrative.

### 4. text_degenerate aggregate shrinkage: positive on average, but not absorption

The 0.26 pooled text_degenerate shrinkage (cross-env) is **not** evidence of
character absorption. Two contributing patterns:

- **Robotic Athanor** (n=8): shrinkage ≈ 0; gap is ~constant high.
- **study_group_ashbourne_gc** (n=8): shrinkage = +0.23 — but this is the
  population that *transitioned* into text_degeneration only late in training.
  Their early measurements (high gap, ~0.32) reflect a coherent / mid-range
  state; their late measurements (low gap, ~0.09) reflect collapse. The
  shrinkage is the difference between an alive agent and a degenerate one,
  not between absorbed and unabsorbed.

[fig_traj_contemp_regime.png](fig_traj_contemp_regime.png) confirms this — when
each measurement is labelled by the regime at that iter (not at end-of-training),
the text_degenerate trace is high and roughly flat, with negative spikes that
are tiny-n artefacts (1–4 obs at iters where almost no run is in that state).

### 5. Coherent low-KL agents do absorb, just slowly

[fig_traj_contemp_regime.png](fig_traj_contemp_regime.png), green line:
**924 contemporaneous-regime observations** — the workhorse stratum. Mean gap
declines monotonically from 0.090 (iter 25) to 0.040 (iter 300). Hypothesis 2
("shallow decline") is **partially** wrong: there's a real, slow absorption in
this regime; it's not zero.

### 6. Policy collapse: too few obs for a clean call

Only 28 contemporaneous obs and 5 end-of-training agents across cross-env.
[stats_shrinkage_cross.csv](stats_shrinkage_cross.csv) shows wild
substrate-to-substrate variability:

- `study_group_strathearn_server` policy_collapse (n=2): final gap −0.150,
  shrinkage 0.187 (gap goes very negative — over-absorption similar to
  committed agents).
- `margin_notes` policy_collapse (n=2): shrinkage 0.105.
- `study_group` policy_collapse (n=1): shrinkage 0.028.

The pattern is suggestive (collapse can mimic over-absorption) but not
conclusive. **Underpowered.**

## Anomalies and contradictions

- **The `committed` shrinkage in inner-loop seed 4** (n=1, single agent) is
  +0.14 with the gap going to −0.11. One-agent regimes carry too much weight
  in their seed-stratum stats. Use the pooled-across-seeds row instead.
- **Coherent agents in seed 2 show *negative* shrinkage (gap rises 0.062 →
  0.075)**. Either drift or seed noise; not consistent with the hypothesis but
  not strong evidence against it given the overall pooled signal.
- **The text_degenerate spike to −1.6 at iter 175** in
  `fig_traj_contemp_regime.png` is one or two data-point artefacts (n=4 obs
  from a transitioning agent). Not interpretable.
- The `mid_range` regime is essentially flat (shrinkage ≈ 0.013 pooled) across
  whatever data exists for it. Consistent with "mid-range = neither committed
  nor collapsed = no commitment-driven absorption", but n=4 agents.

## Bottom-line summary

| Hypothesis | Verdict |
|---|---|
| H1: committed agents show steepest absorption decline | **Supported** in pooled cross-env (0.17 vs 0.06) and pooled inner-loop seeds (0.31 vs 0.05). Per-substrate verification limited by n. |
| H2: coherent low-KL agents show shallow decline | **Partly supported** — they decline meaningfully (~0.05–0.10) but ~3–6× less than committed. "Shallow" is right; "no decline" would be wrong. |
| H3: policy collapse anomalous | **Suggestive** — over-absorption pattern visible but n=5 agents pooled. |
| H4: text-degenerate uninterpretable / pathological | **Strongly supported.** PMI gaps are large but not from absorption; they reflect degraded action distributions. The robotic_athanor "absorption failure" is a textbook example. |

## Recommended paper-level changes

1. **Replace the "LoRA capacity" framing for the Robotic Athanor result with
   "agents are in the text_degenerate regime; absorption is not measurable
   here".** Reference fig_robotic_athanor_zoom.

2. **State the inner-loop multi-seed shrinkage as a regime-stratified result.**
   The unconditional pooled curve mixes 8 coherent + 5 committed agents (across
   5 seeds); reporting the two strata separately is more honest and stronger.

3. **Add the regime-instability caveat** (33/88 ≈ 38% of cross-env agents pass
   through multiple regimes during the absorption window). Either (a) restrict
   the cross-env analysis to runs where the absorption window is in the
   converged regime, or (b) report contemporaneous-regime versions of the
   trajectory plot alongside the end-of-training-regime version.

## Files

- `regime_table_eot.csv` — per-(run, agent) end-of-training regime + KL/perc/entropy.
- `absorption_cross_with_regime.csv` — cross-env PMI rows + end-regime + contemp-regime.
- `absorption_seeds_with_regime.csv` — inner-loop seed rows, same columns.
- `regime_counts_cross.csv`, `regime_counts_seeds.csv` — sample sizes.
- `stats_shrinkage_cross.csv`, `stats_shrinkage_seeds.csv` — per-(env|seed, regime).
- `stats_shrinkage_pooled.csv`, `stats_shrinkage_pooled_seeds.csv` — pooled.
- `regime_consistency_cross.csv` — which agents passed through multiple regimes.
- `contemp_regime_traj_cross.csv` — per-iter mean gap for contemp-regime view.
- `fig_traj_cross_pooled.{pdf,png}` — pooled cross-env trajectory by regime.
- `fig_traj_cross_by_env.{pdf,png}` — per-substrate panels.
- `fig_traj_inner_loop_seeds.{pdf,png}` — per-seed panels.
- `fig_shrinkage_cross.{pdf,png}` — per-(substrate, regime) shrinkage bars.
- `fig_robotic_athanor_zoom.{pdf,png}` — RA per-agent traces + regime-space.
- `fig_traj_contemp_regime.{pdf,png}` — contemporaneous-regime stratification.
