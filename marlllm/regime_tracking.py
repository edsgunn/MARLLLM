"""Live regime classification and population tracking during training.

The post-hoc analysis in `scripts/plot_population_regime_figures.py` and
`scripts/analyse_absorption_by_regime.py` showed that the (KL, perception
loss, entropy) triple cleanly separates training agents into five regimes:

- coherent_low_kl  — KL < kl_thr, perc < perc_coherent, ent ∈ [ent_low, ent_high]
- committed        — KL ≥ kl_thr, perc < perc_coherent, ent ∈ [ent_low, ent_high]
- mid_range        — perc ∈ [perc_coherent, perc_degen)
- policy_collapse  — perc < perc_degen, entropy outside [ent_low, ent_high]
- text_degenerate  — perc ≥ perc_degen

This module reuses those thresholds at training-time so we can:
- stamp `<agent>/regime` on every record in metrics.jsonl
- aggregate population composition per iter (`regime/n_<cat>`, fractions)
- compute slope-based early-warning signals (entropy_slope_5, kl_slope_5)
- detect imminent collapse and trigger a pre-collapse checkpoint copy
- render live regime-composition and per-agent strip plots
- emit a one-line console summary that fits in train.log

Everything is pure-function except the `RegimeTracker` ring-buffer, which is
the only stateful piece. Thresholds are configurable via `RegimeThresholds`
so substrates/model sizes with different baselines can be re-calibrated.
"""

from __future__ import annotations

import json
import math
import shutil
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

# Optional plotting deps — guarded so the tracker still works on a headless
# minimal install. If matplotlib is missing we just skip the viz step.
try:  # pragma: no cover - import-only
    import matplotlib  # noqa: F401
    matplotlib.use("Agg")
    import matplotlib.pyplot as _plt
    _MPL_OK = True
except Exception:  # pragma: no cover
    _plt = None
    _MPL_OK = False

# ---------------------------------------------------------------- thresholds

REGIMES = ("coherent_low_kl", "committed", "mid_range",
           "policy_collapse", "text_degenerate")
REGIME_COLORS = {
    "coherent_low_kl": "#117733",
    "committed":       "#332288",
    "mid_range":       "#DDCC77",
    "policy_collapse": "#88CCEE",
    "text_degenerate": "#CC6677",
}
REGIME_LABELS = {
    "coherent_low_kl": "coherent (low KL)",
    "committed":       "committed",
    "mid_range":       "mid-range",
    "policy_collapse": "policy collapse",
    "text_degenerate": "text degenerate",
}


@dataclass
class RegimeThresholds:
    """Boundaries for the five regimes. Defaults match the post-hoc analysis
    calibrated against 7B-LoRA study-group runs. Override via training-config
    when running on a different substrate or model size."""

    kl_threshold: float = 1.0
    perc_coherent: float = 1.7
    perc_degen: float = 2.0
    ent_low: float = 0.2
    ent_high: float = 1.8

    def classify(self, kl: float, perc: float, entropy: float) -> str:
        if perc is None or kl is None or entropy is None:
            return "unknown"
        if perc >= self.perc_degen:
            return "text_degenerate"
        if entropy < self.ent_low or entropy > self.ent_high:
            return "policy_collapse"
        if perc < self.perc_coherent and kl < self.kl_threshold:
            return "coherent_low_kl"
        if perc < self.perc_coherent and kl >= self.kl_threshold:
            return "committed"
        return "mid_range"


# ---------------------------------------------------------------- helpers

def _slope(values: Iterable[float], iters: Iterable[int]) -> float:
    """Simple linear-regression slope (Δvalue / Δiter) over a short window.
    Returns 0.0 if fewer than 2 points or zero variance in iters."""
    vs = list(values)
    its = [float(i) for i in iters]
    n = len(vs)
    if n < 2:
        return 0.0
    mean_x = sum(its) / n
    mean_y = sum(vs) / n
    num = sum((its[i] - mean_x) * (vs[i] - mean_y) for i in range(n))
    den = sum((its[i] - mean_x) ** 2 for i in range(n))
    if den == 0:
        return 0.0
    return num / den


def _shannon(probs: Iterable[float]) -> float:
    h = 0.0
    for p in probs:
        if p > 0:
            h -= p * math.log(p)
    return h


_AGENT_FIELDS = ("kl", "perc_loss", "entropy")


def _parse_agent_metrics(metrics: dict) -> dict[str, dict[str, float]]:
    """Extract per-agent kl/perc_loss/entropy from a flat metrics dict.

    The trainer flattens these as `<agent_id>/<field>`; this is the inverse
    of `marlllm/trainer.py:227-229` aggregation.
    """
    out: dict[str, dict[str, float]] = defaultdict(dict)
    for k, v in metrics.items():
        if "/" not in k:
            continue
        head, tail = k.rsplit("/", 1)
        if tail in _AGENT_FIELDS:
            out[head][tail] = float(v)
    return {agent: m for agent, m in out.items()
            if all(f in m for f in _AGENT_FIELDS)}


# ---------------------------------------------------------------- tracker

@dataclass
class RegimeTracker:
    """Stateful per-agent buffer + per-iter aggregation.

    Holds a small ring buffer of recent metrics per agent for slope/streak
    computation. Memory footprint: O(n_agents × window).
    """

    thresholds: RegimeThresholds = field(default_factory=RegimeThresholds)
    window: int = 5
    history_len: int = 1000  # cap for in-memory viz buffer per agent

    # Per-agent ring buffer for slopes/streaks.
    _recent: dict[str, deque] = field(init=False, default_factory=lambda: defaultdict(lambda: deque(maxlen=64)))
    # Long-running per-iter history for live viz: list of (iter, regime) per agent.
    _agent_history: dict[str, list[tuple[int, str]]] = field(init=False, default_factory=lambda: defaultdict(list))
    # Population-level fraction history: list of (iter, {regime: frac}).
    _pop_history: list[tuple[int, dict[str, float]]] = field(init=False, default_factory=list)
    # Last regime seen per agent for transition counting.
    _last_regime: dict[str, str] = field(init=False, default_factory=dict)
    # Streak counters: iters in current regime per agent.
    _streak: dict[str, int] = field(init=False, default_factory=lambda: defaultdict(int))
    # Set to the iter when any agent first hits a collapse regime.
    _first_collapse_iter: int | None = field(init=False, default=None)

    # ------- main entry point ------------------------------------------------
    def update(self, iteration: int, metrics: dict) -> tuple[dict, list[str]]:
        """Classify each agent at this iter, compute derived/aggregate fields,
        and return (additions_to_metrics, list_of_alert_strings).

        `additions_to_metrics` is merged into the trainer's metrics dict
        before it is written to metrics.jsonl. `alerts` is a list of
        human-readable warning strings to print to console / train.log.
        """
        agent_data = _parse_agent_metrics(metrics)
        additions: dict = {}
        alerts: list[str] = []

        regime_counts: dict[str, int] = {r: 0 for r in REGIMES}
        transitions = 0
        agents_this_iter = []

        for agent, m in agent_data.items():
            kl, perc, ent = m["kl"], m["perc_loss"], m["entropy"]
            regime = self.thresholds.classify(kl, perc, ent)
            agents_this_iter.append((agent, regime, kl, perc, ent))

            # Per-agent ring buffer for slope computation.
            buf = self._recent[agent]
            buf.append((iteration, kl, perc, ent))
            # Slope window: take last `window` entries.
            window_buf = list(buf)[-self.window:]
            iters_w = [e[0] for e in window_buf]
            kl_slope = _slope([e[1] for e in window_buf], iters_w)
            ent_slope = _slope([e[3] for e in window_buf], iters_w)

            # Streak: how many consecutive iters in this regime.
            prev = self._last_regime.get(agent)
            if prev == regime:
                self._streak[agent] += 1
            else:
                if prev is not None:
                    transitions += 1
                self._streak[agent] = 1
            self._last_regime[agent] = regime

            # Stamp per-agent derived fields.
            additions[f"{agent}/regime"] = regime
            additions[f"{agent}/regime_streak"] = self._streak[agent]
            additions[f"{agent}/kl_slope_{self.window}"] = kl_slope
            additions[f"{agent}/entropy_slope_{self.window}"] = ent_slope

            regime_counts[regime] = regime_counts.get(regime, 0) + 1

            # Append to long-running per-agent history (for viz).
            hist = self._agent_history[agent]
            hist.append((iteration, regime))
            if len(hist) > self.history_len:
                del hist[0]

            # Per-agent early-warning alerts.
            # Sustained entropy decline while still alive — collapse approaching.
            if (regime in ("coherent_low_kl", "committed", "mid_range")
                    and ent_slope < -0.02 and ent > self.thresholds.ent_low
                    and ent < 0.5):
                alerts.append(
                    f"{agent}: entropy declining ({ent:.3f}, slope "
                    f"{ent_slope:+.4f}/iter) — approaching policy collapse"
                )
            # Runaway KL — heading for overshoot/text degeneration.
            if (regime == "committed" and kl_slope > 0.1
                    and kl > 2.0 * self.thresholds.kl_threshold):
                alerts.append(
                    f"{agent}: KL ramping ({kl:.2f}, slope {kl_slope:+.3f}/iter) "
                    f"— commitment overshoot risk"
                )

        # Population aggregates.
        n_agents = sum(regime_counts.values()) or 1
        for r in REGIMES:
            additions[f"regime/n_{r}"] = regime_counts.get(r, 0)
            additions[f"regime/frac_{r}"] = regime_counts.get(r, 0) / n_agents
        additions["regime/n_transitions_last_iter"] = transitions
        additions["regime/divergence"] = _shannon(
            regime_counts.get(r, 0) / n_agents for r in REGIMES
        )

        # First-collapse marker.
        n_collapse = regime_counts.get("policy_collapse", 0) + regime_counts.get("text_degenerate", 0)
        if n_collapse > 0 and self._first_collapse_iter is None:
            self._first_collapse_iter = iteration
            alerts.append(
                f"first collapse at iter {iteration}: "
                f"{regime_counts.get('policy_collapse', 0)} policy_collapse, "
                f"{regime_counts.get('text_degenerate', 0)} text_degenerate"
            )
        if self._first_collapse_iter is not None:
            additions["regime/iters_since_first_collapse"] = (
                iteration - self._first_collapse_iter
            )
        else:
            additions["regime/iters_since_first_collapse"] = -1

        # Save fraction snapshot for live viz.
        frac_snapshot = {r: regime_counts.get(r, 0) / n_agents for r in REGIMES}
        self._pop_history.append((iteration, frac_snapshot))

        # Pre-collapse trigger: any agent crossing INTO text_degenerate this iter.
        pre_collapse_agents = [
            a for a, regime, *_ in agents_this_iter
            if regime == "text_degenerate"
            and self._streak.get(a, 0) == 1  # streak == 1 means newly entered
        ]
        if pre_collapse_agents:
            additions["regime/pre_collapse_trigger"] = 1
            additions["regime/pre_collapse_agents"] = ",".join(sorted(pre_collapse_agents))
            alerts.append(
                f"NEW text_degenerate at iter {iteration}: "
                f"{', '.join(pre_collapse_agents)} — consider rolling back."
            )
        else:
            additions["regime/pre_collapse_trigger"] = 0

        return additions, alerts

    # ------- console summary -------------------------------------------------
    def summary_line(self, iteration: int, metrics: dict) -> str:
        """Compact one-line regime composition for the console / train.log."""
        parts = [f"regime iter {iteration:5d}"]
        n_total = sum(int(metrics.get(f"regime/n_{r}", 0)) for r in REGIMES)
        for r in REGIMES:
            count = int(metrics.get(f"regime/n_{r}", 0))
            if count == 0:
                continue
            short = {"coherent_low_kl": "coh", "committed": "comm",
                     "mid_range": "mid", "policy_collapse": "pcol",
                     "text_degenerate": "tdeg"}[r]
            parts.append(f"{short} {count}/{n_total}")
        trans = int(metrics.get("regime/n_transitions_last_iter", 0))
        if trans:
            parts.append(f"trans {trans}")
        return " | ".join(parts)

    # ------- visualisations --------------------------------------------------
    def render_population_plot(self, out_path: Path) -> None:
        """Stacked-area plot of regime composition over training iters."""
        if not _MPL_OK or not self._pop_history:
            return
        iters = [it for it, _ in self._pop_history]
        stacks = {r: [snap.get(r, 0.0) for _, snap in self._pop_history]
                  for r in REGIMES}
        fig, ax = _plt.subplots(figsize=(6.5, 3.0), constrained_layout=True)
        ax.stackplot(iters,
                     [stacks[r] for r in REGIMES],
                     colors=[REGIME_COLORS[r] for r in REGIMES],
                     labels=[REGIME_LABELS[r] for r in REGIMES],
                     alpha=0.92)
        ax.set_xlim(min(iters), max(iters))
        ax.set_ylim(0, 1.001)
        ax.set_xlabel("training iteration")
        ax.set_ylabel("fraction of agents")
        ax.set_title("Live regime composition", fontsize=10)
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5),
                  frameon=False, fontsize=7)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, format="png", dpi=160,
                    bbox_inches="tight", pad_inches=0.04)
        _plt.close(fig)

    def render_agent_strip(self, out_path: Path) -> None:
        """Per-agent regime strip plot — Gantt-like, one row per agent."""
        if not _MPL_OK or not self._agent_history:
            return
        agents = sorted(self._agent_history.keys())
        all_iters: set[int] = set()
        for h in self._agent_history.values():
            all_iters.update(it for it, _ in h)
        if not all_iters:
            return
        min_it, max_it = min(all_iters), max(all_iters)
        fig, ax = _plt.subplots(
            figsize=(6.5, max(2.0, 0.30 * len(agents) + 0.5)),
            constrained_layout=True,
        )
        for row, agent in enumerate(agents):
            hist = self._agent_history[agent]
            # Build contiguous runs of same regime.
            runs: list[tuple[int, int, str]] = []
            if not hist:
                continue
            start_it, cur_reg = hist[0]
            prev_it = start_it
            for it, reg in hist[1:]:
                if reg != cur_reg:
                    runs.append((start_it, prev_it, cur_reg))
                    start_it, cur_reg = it, reg
                prev_it = it
            runs.append((start_it, prev_it, cur_reg))
            for s, e, reg in runs:
                width = max(e - s, 1)
                ax.barh(row, width, left=s, height=0.85,
                        color=REGIME_COLORS.get(reg, "0.5"),
                        edgecolor="none")
        ax.set_yticks(range(len(agents)))
        ax.set_yticklabels(agents, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlim(min_it - 0.5, max_it + 0.5)
        ax.set_xlabel("training iteration")
        ax.set_title("Per-agent regime trajectory", fontsize=10)
        handles = [
            _plt.Line2D([], [], color=REGIME_COLORS[r], lw=6,
                        label=REGIME_LABELS[r])
            for r in REGIMES
            if any(any(rg == r for _, rg in self._agent_history[a]) for a in agents)
        ]
        ax.legend(handles=handles, loc="center left",
                  bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=7)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, format="png", dpi=160,
                    bbox_inches="tight", pad_inches=0.04)
        _plt.close(fig)

    # ------- end-of-run summary ---------------------------------------------
    def write_summary_md(self, out_path: Path) -> None:
        """End-of-run markdown summary."""
        lines = ["# Regime tracking summary", ""]
        lines.append(f"- thresholds: {self.thresholds}")
        lines.append(f"- agents tracked: {len(self._agent_history)}")
        lines.append(f"- iters logged: {len(self._pop_history)}")
        if self._first_collapse_iter is not None:
            lines.append(f"- first collapse: iter {self._first_collapse_iter}")
        else:
            lines.append("- first collapse: never (no agent reached policy_collapse / text_degenerate)")
        lines.append("")
        lines.append("## Final regime per agent")
        lines.append("")
        lines.append("| agent | final regime | streak |")
        lines.append("|---|---|---:|")
        for agent in sorted(self._agent_history.keys()):
            reg = self._last_regime.get(agent, "?")
            streak = self._streak.get(agent, 0)
            lines.append(f"| {agent} | {reg} | {streak} |")
        if self._pop_history:
            final_iter, final_frac = self._pop_history[-1]
            lines.append("")
            lines.append(f"## Final population composition (iter {final_iter})")
            lines.append("")
            lines.append("| regime | fraction |")
            lines.append("|---|---:|")
            for r in REGIMES:
                if final_frac.get(r, 0) > 0:
                    lines.append(f"| {REGIME_LABELS[r]} | {final_frac[r]:.1%} |")
        out_path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------- pre-collapse ckpt

def copy_pre_collapse_checkpoint(
    checkpoints_dir: Path,
    iteration: int,
    prev_iteration: int | None,
) -> Path | None:
    """If the previous iter's checkpoint directory still exists on disk, copy
    it into `checkpoints/pre_collapse_iter_<iteration>/`. Returns the path of
    the copy, or None if no prior checkpoint was found.

    The trainer normally writes a checkpoint every `checkpoint_every` iters,
    so the "previous iter" may not be the iter immediately before this one —
    we find the most recent `iter_NNNNNN/` directory that is strictly less
    than `iteration` and copy that.
    """
    checkpoints_dir = Path(checkpoints_dir)
    if not checkpoints_dir.exists():
        return None
    candidates = sorted(
        (p for p in checkpoints_dir.iterdir() if p.is_dir()
         and p.name.startswith("iter_")),
        key=lambda p: int(p.name.split("_")[-1]) if p.name.split("_")[-1].isdigit() else -1,
    )
    candidates = [p for p in candidates
                  if p.name.startswith("iter_")
                  and p.name.split("_")[-1].isdigit()
                  and int(p.name.split("_")[-1]) < iteration]
    if not candidates:
        return None
    src = candidates[-1]
    dst = checkpoints_dir / f"pre_collapse_iter_{iteration:06d}"
    if dst.exists():
        return dst  # already saved for this iter
    shutil.copytree(src, dst, symlinks=True)
    return dst
