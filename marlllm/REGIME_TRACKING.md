# Live regime tracking — implementation notes

The post-hoc classifier in `scripts/plot_population_regime_figures.py` is now
applied at training-time. Every record in `metrics.jsonl` gets stamped with a
regime label per agent + population aggregates + slope-based early-warning
signals, and live PNG plots are rendered into `<output_dir>/regime/`.

## Files

- [marlllm/regime_tracking.py](regime_tracking.py) — `RegimeThresholds`,
  `RegimeTracker`, viz, end-of-run summary, pre-collapse checkpoint copier.
  Pure-Python, no GPU work.
- [marlllm/regime_token_signals.py](regime_token_signals.py) — Tier-3
  token-level signals from rollouts (tail-loop rate, non-ASCII rate, top-token
  fraction). Optional; off by default.
- [marlllm/config.py](config.py) — config knobs under the
  `regime_*` prefix. Defaults match the post-hoc calibration.
- [marlllm/trainer.py](trainer.py) — single integration point around the
  metrics-write block; tracker is constructed in `__init__` and invoked once
  per iter before `_write_metrics_jsonl`.
- [tests/test_regime_tracking.py](../tests/test_regime_tracking.py) — pytest
  suite covering the classifier, slope helpers, tracker state machine, alerts,
  pre-collapse trigger, and viz.

## What it adds to metrics.jsonl per iter

Per agent (one set per agent_id):

- `<agent>/regime` — one of `coherent_low_kl | committed | mid_range | policy_collapse | text_degenerate`
- `<agent>/regime_streak` — consecutive iters in current regime
- `<agent>/kl_slope_5` — KL slope over the last 5 iters (nats / iter)
- `<agent>/entropy_slope_5` — entropy slope, same window
- (with `regime_token_signals = True`): `<agent>/tail_loop_rate`,
  `<agent>/non_ascii_rate`, `<agent>/top_token_frac`

Population-level:

- `regime/n_<category>` and `regime/frac_<category>` for each of the 5 buckets
- `regime/n_transitions_last_iter`
- `regime/divergence` — Shannon entropy of the population fraction distribution
- `regime/iters_since_first_collapse` (−1 if no collapse yet)
- `regime/pre_collapse_trigger` (1 when at least one agent newly entered
  `text_degenerate` this iter, else 0)
- `regime/pre_collapse_agents` — comma-separated list when trigger fires

## Console output (added to `train.log`)

One extra line per `log_every` iters:

```
2026-05-07 11:23:45 INFO     regime iter   125 | coh 5/8 | comm 2/8 | pcol 1/8 | trans 1
```

Plus per-event WARNINGs:

```
WARNING  regime: Kai Dempsey: KL ramping (4.96, slope +0.180/iter) — commitment overshoot risk
WARNING  regime: Mhairi Buchanan: entropy declining (0.31, slope -0.0420/iter) — approaching policy collapse
WARNING  regime: first collapse at iter 75: 1 policy_collapse, 0 text_degenerate
WARNING  regime: NEW text_degenerate at iter 198: Priya Shah — consider rolling back.
WARNING  regime: saved pre-collapse snapshot to checkpoints/pre_collapse_iter_000198
```

## Live PNG outputs (in `<output_dir>/regime/`)

Updated every `regime_viz_every` iters (default 25) and once more at end of run:

- `regime_trajectory.png` — stacked area of population composition over iters
- `agent_regime_strip.png` — Gantt-style strip plot, one row per agent
- `regime_summary.md` — end-of-run markdown with final regime per agent + final composition

## Pre-collapse checkpoint

When any agent first crosses into `text_degenerate`, the trainer copies the
most recent regular checkpoint (`checkpoints/iter_NNNNNN/`) into
`checkpoints/pre_collapse_iter_<current_iter>/`. This guarantees you have a
roll-back point at most `checkpoint_every` iters before the collapse, without
needing to set `checkpoint_every=1`. Disable by setting
`regime_pre_collapse_checkpoint = False` in the config.

## Configuration

All defaults match the post-hoc calibration (7B-LoRA study-group runs):

```python
regime_tracking_enabled: bool = True
regime_kl_threshold: float = 1.0
regime_perc_coherent: float = 1.7
regime_perc_degen: float = 2.0
regime_ent_low: float = 0.2
regime_ent_high: float = 1.8
regime_slope_window: int = 5
regime_viz_every: int = 25
regime_token_signals: bool = False
regime_pre_collapse_checkpoint: bool = True
```

If you train on a substrate / model size where the baseline `perc_loss` differs
substantially from study-group's ~1.2–1.5, **re-calibrate** the
`regime_perc_coherent` / `regime_perc_degen` thresholds — otherwise the live
classifier will mis-tag whole populations.

## Cost

- **Compute**: negligible. Two `_slope()` linear regressions per agent per
  iter, both O(window). The pop aggregation is O(n_agents). With
  `regime_viz_every = 25`, plot rendering is ~50 ms every 25 iters on the main
  rank. Tier-3 token signals (off by default) add a per-rollout pass over
  generated tokens — measurable but small (<0.5% of rollout time empirically).
- **Disk**: an extra ~20 fields per iter in `metrics.jsonl`. Two ~30 kB PNGs
  rewritten each viz interval.
- **Memory**: per-agent ring buffer of last 64 iters' (kl, perc, ent) + a
  history list of (iter, regime) capped at `history_len = 1000`. <100 kB
  total for typical populations.

## Running the tests

```bash
.venv/bin/python -m pytest tests/test_regime_tracking.py -v
```

The suite covers 28 checks across classifier boundaries, slope helpers,
streak/transition counting, alert firing, pre-collapse trigger semantics, end-of-run
summary, and the pre-collapse checkpoint copier. It does *not* exercise the
trainer integration directly — that requires a full training run.

## What is *not* implemented (and why)

- **Vocabulary cosine similarity to base model** — listed in the original
  proposal as a smoother alternative to KL, but it requires holding the
  per-agent first-iter token histogram in memory and comparing per iter.
  `top_token_frac` is the lighter proxy that exists in the Tier-3 module.
- **Live dashboard / wandb integration** — the file-based outputs are
  dashboard-agnostic; if you add wandb later, wrap `tracker.update()`'s
  returned `additions` dict and forward.
- **Threshold sensitivity check** — there's no built-in "try N nearby
  thresholds, report classification stability". Add post-hoc if needed.
