"""Tests for the live regime tracker.

Covers: classification thresholds, slope computation, streak counters,
transitions, pre-collapse trigger, population aggregates, and the optional
viz rendering.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from marlllm.regime_tracking import (
    REGIMES,
    RegimeThresholds,
    RegimeTracker,
    _parse_agent_metrics,
    _shannon,
    _slope,
    copy_pre_collapse_checkpoint,
)


# ----------------- classifier ------------------------------------------------

def test_classifier_each_regime():
    t = RegimeThresholds()
    # coherent: low KL, low perc, mid entropy
    assert t.classify(kl=0.2, perc=1.3, entropy=0.7) == "coherent_low_kl"
    # committed: high KL, low perc, mid entropy
    assert t.classify(kl=4.0, perc=1.4, entropy=0.6) == "committed"
    # mid-range: perc in [1.7, 2.0)
    assert t.classify(kl=0.5, perc=1.85, entropy=0.6) == "mid_range"
    # policy collapse: entropy below threshold, perc still alive
    assert t.classify(kl=0.3, perc=1.5, entropy=0.05) == "policy_collapse"
    # policy collapse: entropy above upper threshold
    assert t.classify(kl=0.3, perc=1.5, entropy=1.95) == "policy_collapse"
    # text degenerate: perc >= 2.0 wins regardless of entropy
    assert t.classify(kl=0.3, perc=2.7, entropy=0.05) == "text_degenerate"
    assert t.classify(kl=0.3, perc=2.7, entropy=1.0) == "text_degenerate"


def test_classifier_boundary_conditions():
    t = RegimeThresholds()
    # Exactly at KL=1.0 → committed (>= boundary).
    assert t.classify(kl=1.0, perc=1.4, entropy=0.6) == "committed"
    # Exactly at perc=1.7 → mid-range (>= boundary, < degen).
    assert t.classify(kl=0.5, perc=1.7, entropy=0.6) == "mid_range"
    # Exactly at perc=2.0 → text_degenerate.
    assert t.classify(kl=0.5, perc=2.0, entropy=0.6) == "text_degenerate"


def test_classifier_unknown_on_missing():
    t = RegimeThresholds()
    assert t.classify(kl=None, perc=1.3, entropy=0.5) == "unknown"


# ----------------- helpers ---------------------------------------------------

def test_slope_basic():
    # Perfect linear increase: slope = 1
    assert _slope([0, 1, 2, 3], [0, 1, 2, 3]) == pytest.approx(1.0)
    # Perfect linear decrease.
    assert _slope([3, 2, 1, 0], [0, 1, 2, 3]) == pytest.approx(-1.0)
    # Flat — slope 0.
    assert _slope([1, 1, 1, 1], [0, 1, 2, 3]) == 0.0


def test_slope_edge_cases():
    assert _slope([], []) == 0.0
    assert _slope([5.0], [10]) == 0.0
    # Zero variance in x.
    assert _slope([1, 2], [3, 3]) == 0.0


def test_shannon_entropy():
    # Uniform over 5 bins.
    p = [0.2] * 5
    import math
    assert _shannon(p) == pytest.approx(math.log(5))
    # Degenerate — all mass on one bin.
    assert _shannon([1.0, 0.0, 0.0]) == pytest.approx(0.0)


# ----------------- _parse_agent_metrics --------------------------------------

def test_parse_agent_metrics_extracts_canonical_fields():
    metrics = {
        "agent_0/kl": 0.5, "agent_0/perc_loss": 1.4, "agent_0/entropy": 0.8,
        "agent_0/extra": 99.0,
        "Priya Shah/kl": 4.0, "Priya Shah/perc_loss": 1.5,
        "Priya Shah/entropy": 0.5,
        "total_loss": 1.23,  # not per-agent
    }
    out = _parse_agent_metrics(metrics)
    assert set(out.keys()) == {"agent_0", "Priya Shah"}
    assert out["agent_0"]["kl"] == 0.5
    assert out["Priya Shah"]["entropy"] == 0.5
    # Agents with incomplete data are dropped.
    assert _parse_agent_metrics({"foo/kl": 0.1}) == {}


# ----------------- tracker behaviour -----------------------------------------

def make_metrics(values: dict[str, tuple[float, float, float]]) -> dict:
    """Build a flat metrics dict from {agent_id: (kl, perc, ent)}."""
    out = {}
    for agent, (kl, perc, ent) in values.items():
        out[f"{agent}/kl"] = kl
        out[f"{agent}/perc_loss"] = perc
        out[f"{agent}/entropy"] = ent
    return out


def test_tracker_basic_population_aggregates():
    tr = RegimeTracker()
    metrics = make_metrics({
        "a": (0.2, 1.3, 0.7),     # coherent
        "b": (4.0, 1.4, 0.6),     # committed
        "c": (0.3, 1.5, 0.05),    # policy collapse
        "d": (0.3, 2.7, 0.05),    # text degenerate
    })
    add, alerts = tr.update(iteration=10, metrics=metrics)
    assert add["a/regime"] == "coherent_low_kl"
    assert add["b/regime"] == "committed"
    assert add["c/regime"] == "policy_collapse"
    assert add["d/regime"] == "text_degenerate"
    # Population aggregates.
    assert add["regime/n_coherent_low_kl"] == 1
    assert add["regime/n_committed"] == 1
    assert add["regime/n_policy_collapse"] == 1
    assert add["regime/n_text_degenerate"] == 1
    assert add["regime/n_mid_range"] == 0
    for r in REGIMES:
        if add[f"regime/n_{r}"] > 0:
            assert add[f"regime/frac_{r}"] == pytest.approx(0.25)
    # First-collapse marker.
    assert add["regime/iters_since_first_collapse"] == 0
    # Pre-collapse trigger fires on the new text_degenerate.
    assert add["regime/pre_collapse_trigger"] == 1
    assert "d" in add["regime/pre_collapse_agents"]


def test_tracker_transitions_and_streaks():
    tr = RegimeTracker()
    # Iter 1: agent a is coherent.
    tr.update(1, make_metrics({"a": (0.2, 1.3, 0.7)}))
    # Iter 2: still coherent → streak 2, no transition.
    add2, _ = tr.update(2, make_metrics({"a": (0.2, 1.3, 0.7)}))
    assert add2["a/regime_streak"] == 2
    assert add2["regime/n_transitions_last_iter"] == 0
    # Iter 3: enters committed → streak resets, transition counted.
    add3, _ = tr.update(3, make_metrics({"a": (3.0, 1.4, 0.6)}))
    assert add3["a/regime"] == "committed"
    assert add3["a/regime_streak"] == 1
    assert add3["regime/n_transitions_last_iter"] == 1


def test_tracker_slope_computation():
    tr = RegimeTracker(window=4)
    # Linearly declining entropy over 4 iters → negative slope.
    seq = [(1, 0.9), (2, 0.8), (3, 0.7), (4, 0.6)]
    for i, ent in seq:
        add, alerts = tr.update(i, make_metrics({"a": (0.3, 1.3, ent)}))
    # Final slope should be ~ -0.1 per iter.
    assert add["a/entropy_slope_4"] == pytest.approx(-0.1, abs=1e-6)


def test_tracker_pre_collapse_trigger_only_on_entry():
    tr = RegimeTracker()
    # Already text_degenerate at iter 1 — first entry, trigger fires.
    add1, _ = tr.update(1, make_metrics({"a": (0.3, 2.5, 0.05)}))
    assert add1["regime/pre_collapse_trigger"] == 1
    # Iter 2: still text_degenerate — streak > 1, trigger does NOT fire again.
    add2, _ = tr.update(2, make_metrics({"a": (0.3, 2.6, 0.04)}))
    assert add2["regime/pre_collapse_trigger"] == 0


def test_tracker_alerts_entropy_decline():
    tr = RegimeTracker(window=5)
    # Walk entropy down while keeping it in the "still alive" range,
    # crossing into the alert window (ent < 0.5 and slope < -0.02).
    entropies = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
    last_alerts: list[str] = []
    for i, ent in enumerate(entropies, start=1):
        _, last_alerts = tr.update(i, make_metrics({"a": (0.3, 1.4, ent)}))
    assert any("approaching policy collapse" in a for a in last_alerts), last_alerts


def test_tracker_summary_line_and_md(tmp_path: Path):
    tr = RegimeTracker()
    for it in (1, 2, 3):
        add, _ = tr.update(it, make_metrics({
            "a": (0.2, 1.3, 0.7),
            "b": (3.0, 1.4, 0.6),
            "c": (0.3, 1.5, 0.05),
        }))
    line = tr.summary_line(it, {**add})
    assert "regime iter" in line
    assert "coh" in line and "pcol" in line
    md = tmp_path / "summary.md"
    tr.write_summary_md(md)
    text = md.read_text()
    assert "agents tracked: 3" in text
    assert "first collapse: iter 1" in text


def test_tracker_render_does_not_crash(tmp_path: Path):
    tr = RegimeTracker()
    for i in range(1, 8):
        tr.update(i, make_metrics({
            "a": (0.2, 1.3, 0.7),
            "b": (3.0 + 0.1 * i, 1.4, 0.6),
        }))
    # render_* are no-ops without matplotlib; either way they must not raise.
    tr.render_population_plot(tmp_path / "pop.png")
    tr.render_agent_strip(tmp_path / "strip.png")


# ----------------- pre-collapse checkpoint copy -------------------------------

def test_copy_pre_collapse_checkpoint(tmp_path: Path):
    ck = tmp_path / "checkpoints"
    (ck / "iter_000050").mkdir(parents=True)
    (ck / "iter_000050" / "meta.pt").write_text("blob")
    (ck / "iter_000075").mkdir()
    (ck / "iter_000075" / "meta.pt").write_text("blob")
    # Trigger at iter 80 should copy iter_000075 (most recent < 80).
    dst = copy_pre_collapse_checkpoint(ck, iteration=80, prev_iteration=79)
    assert dst is not None
    assert dst.name == "pre_collapse_iter_000080"
    assert (dst / "meta.pt").read_text() == "blob"


def test_copy_pre_collapse_checkpoint_none_if_empty(tmp_path: Path):
    ck = tmp_path / "checkpoints"
    ck.mkdir()
    assert copy_pre_collapse_checkpoint(ck, iteration=10, prev_iteration=9) is None
