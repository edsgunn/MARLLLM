#!/usr/bin/env python3
"""
Visualise experiment results.

For each experiment:
  - training_curves.png  — loss / return / entropy / kl over iterations

For the whole suite:
  - suite_comparison.png — final-value bar charts across experiments
  - suite_training.png   — all experiments' smoothed return on one axes

Usage:
    python scripts/plot_results.py runs/concordia_experiments
    python scripts/plot_results.py runs/experiments
    python scripts/plot_results.py runs/concordia_experiments --exp 01 09 12
    python scripts/plot_results.py runs/experiments --no-per-exp   # suite only
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "lines.linewidth": 1.4,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

_AGENT_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52",
                 "#8172B2", "#937860", "#DA8BC3", "#8C8C8C"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_metrics(exp_dir: Path) -> dict[str, np.ndarray] | None:
    path = exp_dir / "metrics.jsonl"
    if not path.exists():
        return None
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    if not rows:
        return None

    keys = list(rows[0].keys())
    data: dict[str, np.ndarray] = {}
    for k in keys:
        data[k] = np.array([r.get(k, float("nan")) for r in rows], dtype=float)

    # Discover entity prefixes from keys (e.g. agent_0, player_0, worker_2 ...)
    # Any key of the form "<word>_<int>/<metric>" is an entity key.
    import re as _re
    entity_prefix_re = _re.compile(r"^([a-zA-Z_]+\d+)/(.+)$")
    entity_metrics: dict[str, set[str]] = {}  # prefix -> set of metric names
    for k in keys:
        m = entity_prefix_re.match(k)
        if m:
            prefix, metric = m.group(1), m.group(2)
            entity_metrics.setdefault(prefix, set()).add(metric)

    # Find the ordered list of entity prefixes sharing the same base name
    # e.g. ["player_0", "player_1", "player_2"] or ["agent_0", "agent_1"]
    base_re = _re.compile(r"^([a-zA-Z_]+)(\d+)$")
    base_groups: dict[str, list[tuple[int, str]]] = {}
    for prefix in entity_metrics:
        bm = base_re.match(prefix)
        if bm:
            base, idx = bm.group(1), int(bm.group(2))
            base_groups.setdefault(base, []).append((idx, prefix))

    # Use the largest group as "the agents"
    if base_groups:
        best_base = max(base_groups, key=lambda b: len(base_groups[b]))
        ordered_prefixes = [p for _, p in sorted(base_groups[best_base])]

        all_metrics: set[str] = set()
        for p in ordered_prefixes:
            all_metrics |= entity_metrics.get(p, set())

        for metric in all_metrics:
            series = [data[f"{p}/{metric}"] for p in ordered_prefixes
                      if f"{p}/{metric}" in data]
            if series:
                data[f"agents/{metric}"] = np.stack(series)  # (n_agents, T)

    return data


def smooth(x: np.ndarray, w: int = 10) -> np.ndarray:
    if len(x) < w:
        return x
    kernel = np.ones(w) / w
    pad = np.pad(x, (w // 2, w - w // 2 - 1), mode="edge")
    return np.convolve(pad, kernel, mode="valid")[: len(x)]


def agent_count(data: dict) -> int:
    arr = data.get("agents/mean_return")
    return arr.shape[0] if arr is not None else 0


# ---------------------------------------------------------------------------
# Per-experiment training curves
# ---------------------------------------------------------------------------

def _agent_mean(data: dict, key: str) -> np.ndarray | None:
    """Return the mean of agents/key across agents, or None if missing."""
    arr = data.get(f"agents/{key}")
    return arr.mean(axis=0) if arr is not None else None


def plot_training_curves(exp_dir: Path, data: dict[str, np.ndarray]) -> Path:
    iters = data.get("iteration", np.arange(len(data["total_loss"])))
    has_kl = np.any(data.get("agents/kl", np.zeros(1)) != 0)

    # Layout: 3 rows always; optional KL row appended
    n_rows = 3 + (1 if has_kl else 0)
    fig, axes = plt.subplots(n_rows, 2, figsize=(11, 3.5 * n_rows), squeeze=False)
    fig.suptitle(exp_dir.name, fontsize=11, fontweight="bold")

    # ── Row 0: total loss | loss breakdown (per-component, log scale) ────────
    ax = axes[0, 0]
    ax.plot(iters, smooth(data["total_loss"]), color="#333333", label="total")
    ax.fill_between(iters, data["total_loss"], alpha=0.10, color="#333333")
    ax.set_title("Total loss")
    ax.set_ylabel("loss")

    ax = axes[0, 1]
    _COMP_COLORS = {"perc": "#E64B35", "act": "#4DBBD5", "value": "#F39B7F"}
    _COMP_LABELS = {"perc": "perception (NTP on obs)", "act": "action (REINFORCE)",
                    "value": "value"}
    any_comp = False
    for comp, color in _COMP_COLORS.items():
        mean = _agent_mean(data, f"{comp}_loss")
        if mean is not None:
            ax.plot(iters, smooth(mean), color=color, label=_COMP_LABELS[comp])
            any_comp = True
    if any_comp:
        # Log scale because value_loss dominates by ~3 orders of magnitude
        pos_floor = 1e-4
        ax.set_yscale("symlog", linthresh=pos_floor)
        ax.set_ylabel("loss (symlog scale)")
        ax.legend(loc="upper right")
    ax.set_title("Loss components (mean over agents)")

    # ── Row 1: mean return | policy entropy ──────────────────────────────────
    ax = axes[1, 0]
    if "agents/mean_return" in data:
        all_returns = data["agents/mean_return"]
        mean_ret = all_returns.mean(axis=0)
        ax.axhline(0, color="#aaa", linewidth=0.8, linestyle="--")
        ax.fill_between(iters,
                        all_returns.min(axis=0),
                        all_returns.max(axis=0),
                        alpha=0.15, color="#4C72B0", label="agent range")
        ax.plot(iters, smooth(mean_ret), color="#4C72B0", label="mean")
        for i, series in enumerate(all_returns):
            ax.plot(iters, smooth(series), color=_AGENT_COLORS[i % len(_AGENT_COLORS)],
                    alpha=0.55, linewidth=0.9)
        ax.legend(loc="upper left")
    ax.set_title("Mean return")
    ax.set_ylabel("return")

    ax = axes[1, 1]
    if "agents/entropy" in data:
        for i, series in enumerate(data["agents/entropy"]):
            ax.plot(iters, smooth(series), color=_AGENT_COLORS[i % len(_AGENT_COLORS)],
                    label=f"agent_{i}")
        ax.legend(loc="upper right")
    ax.set_title("Policy entropy (per agent)")
    ax.set_ylabel("entropy (nats)")

    # ── Row 2: action loss (per agent) | value loss (per agent) ──────────────
    ax = axes[2, 0]
    if "agents/act_loss" in data:
        for i, series in enumerate(data["agents/act_loss"]):
            ax.plot(iters, smooth(series), color=_AGENT_COLORS[i % len(_AGENT_COLORS)],
                    label=f"agent_{i}")
        ax.legend(loc="upper right")
    ax.set_title("Action loss — REINFORCE (per agent)")
    ax.set_ylabel("loss")

    ax = axes[2, 1]
    if "agents/value_loss" in data:
        for i, series in enumerate(data["agents/value_loss"]):
            ax.plot(iters, smooth(series), color=_AGENT_COLORS[i % len(_AGENT_COLORS)],
                    label=f"agent_{i}")
        ax.legend(loc="upper right")
    ax.set_title("Value loss (per agent)")
    ax.set_ylabel("loss")

    # ── Row 3 (optional): KL | perception loss ───────────────────────────────
    if has_kl:
        ax = axes[3, 0]
        if "agents/kl" in data:
            for i, series in enumerate(data["agents/kl"]):
                ax.plot(iters, smooth(series), color=_AGENT_COLORS[i % len(_AGENT_COLORS)],
                        label=f"agent_{i}")
            ax.legend()
        ax.set_title("KL from reference (per agent)")
        ax.set_ylabel("KL")

        ax = axes[3, 1]
        if "agents/perc_loss" in data:
            for i, series in enumerate(data["agents/perc_loss"]):
                ax.plot(iters, smooth(series), color=_AGENT_COLORS[i % len(_AGENT_COLORS)],
                        label=f"agent_{i}")
            ax.legend()
        ax.set_title("Perception loss — NTP on obs tokens (per agent)")
        ax.set_ylabel("loss")

    for ax_row in axes:
        for ax in ax_row:
            ax.set_xlabel("iteration")
            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    fig.tight_layout()
    out = exp_dir / "training_curves.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Suite-level: comparison bar chart
# ---------------------------------------------------------------------------

def _final_stats(data: dict) -> dict[str, float]:
    """Return scalar summary of the last ~10% of training."""
    T = len(data["total_loss"])
    tail = max(1, T // 10)

    stats: dict[str, float] = {}
    stats["total_loss"] = float(np.nanmean(data["total_loss"][-tail:]))

    if "agents/mean_return" in data:
        stats["mean_return"] = float(np.nanmean(data["agents/mean_return"][:, -tail:]))
    if "agents/entropy" in data:
        stats["entropy"] = float(np.nanmean(data["agents/entropy"][:, -tail:]))
    if "agents/kl" in data:
        stats["kl"] = float(np.nanmean(data["agents/kl"][:, -tail:]))

    stats["n_iters"] = int(np.nanmax(data.get("iteration", np.array([T]))))
    return stats


def plot_suite_comparison(suite_dir: Path, all_data: dict[str, dict]) -> Path:
    names = list(all_data.keys())
    short_names = [n.split("_", 1)[1] if "_" in n else n for n in names]
    finals = {n: _final_stats(d) for n, d in all_data.items()}

    metrics = ["mean_return", "entropy", "total_loss"]
    labels  = ["Mean return (final 10%)", "Policy entropy (final 10%)", "Total loss (final 10%)"]
    n_metrics = sum(1 for m in metrics if any(m in f for f in finals.values()))

    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, max(4, 0.35 * len(names) + 2)))
    fig.suptitle(f"Suite comparison — {suite_dir.name}", fontsize=11, fontweight="bold")
    if n_metrics == 1:
        axes = [axes]

    ax_idx = 0
    for metric, label in zip(metrics, labels):
        vals = [finals[n].get(metric, float("nan")) for n in names]
        if all(np.isnan(v) for v in vals):
            continue
        ax = axes[ax_idx]; ax_idx += 1

        colors = [_AGENT_COLORS[i % len(_AGENT_COLORS)] for i in range(len(names))]
        bars = ax.barh(short_names, vals, color=colors, alpha=0.8)

        # Value labels
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(val, bar.get_y() + bar.get_height() / 2,
                        f"  {val:.3f}", va="center", fontsize=7)

        ax.set_title(label)
        ax.axvline(0, color="#aaa", linewidth=0.7)
        ax.invert_yaxis()

    fig.tight_layout()
    out = suite_dir / "suite_comparison.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Suite-level: all experiments' return on one plot
# ---------------------------------------------------------------------------

def plot_suite_training(suite_dir: Path, all_data: dict[str, dict]) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Training curves — {suite_dir.name}", fontsize=11, fontweight="bold")

    cmap = plt.get_cmap("tab20")
    names = list(all_data.keys())

    for i, (name, data) in enumerate(all_data.items()):
        iters = data.get("iteration", np.arange(len(data["total_loss"])))
        color = cmap(i / max(len(names) - 1, 1))
        short = name.split("_", 1)[1] if "_" in name else name

        # Total loss
        ax = axes[0]
        ax.plot(iters, smooth(data["total_loss"], 15), color=color,
                label=short, alpha=0.85)

        # Mean return across agents
        ax = axes[1]
        if "agents/mean_return" in data:
            ret = data["agents/mean_return"].mean(axis=0)
            ax.plot(iters, smooth(ret, 15), color=color, label=short, alpha=0.85)

    axes[0].set_title("Total loss")
    axes[0].set_xlabel("iteration")
    axes[0].set_ylabel("loss")
    axes[0].legend(loc="upper right", ncol=2, fontsize=7)

    axes[1].set_title("Mean return (averaged over agents)")
    axes[1].set_xlabel("iteration")
    axes[1].set_ylabel("return")
    axes[1].axhline(0, color="#aaa", linewidth=0.8, linestyle="--")
    axes[1].legend(loc="lower right", ncol=2, fontsize=7)

    fig.tight_layout()
    out = suite_dir / "suite_training.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite_dir", type=Path)
    parser.add_argument("--exp", nargs="+", metavar="PREFIX",
                        help="Only include experiments matching these prefixes")
    parser.add_argument("--no-per-exp", action="store_true",
                        help="Skip per-experiment training_curves.png")
    parser.add_argument("--no-suite", action="store_true",
                        help="Skip suite-level comparison plots")
    args = parser.parse_args()

    suite_dir = args.suite_dir.resolve()
    if not suite_dir.is_dir():
        sys.exit(f"Error: {suite_dir} is not a directory")

    exp_dirs = sorted(d for d in suite_dir.iterdir()
                      if d.is_dir() and d.name != "checkpoints")
    if args.exp:
        exp_dirs = [d for d in exp_dirs if any(d.name.startswith(p) for p in args.exp)]
    if not exp_dirs:
        sys.exit("No experiment directories found")

    # Load all metrics
    all_data: dict[str, dict] = {}
    for exp_dir in exp_dirs:
        data = load_metrics(exp_dir)
        if data is None:
            print(f"  skip {exp_dir.name}  (no metrics.jsonl)")
            continue
        all_data[exp_dir.name] = data

    if not all_data:
        sys.exit("No experiments with metrics found")

    # Per-experiment plots
    if not args.no_per_exp:
        for exp_dir in exp_dirs:
            if exp_dir.name not in all_data:
                continue
            out = plot_training_curves(exp_dir, all_data[exp_dir.name])
            print(f"  {out.relative_to(suite_dir.parent)}")

    # Suite-level plots
    if not args.no_suite and len(all_data) > 1:
        out = plot_suite_comparison(suite_dir, all_data)
        print(f"  {out.relative_to(suite_dir.parent)}")
        out = plot_suite_training(suite_dir, all_data)
        print(f"  {out.relative_to(suite_dir.parent)}")

    print(f"Done — {len(all_data)} experiments plotted.")


if __name__ == "__main__":
    main()
