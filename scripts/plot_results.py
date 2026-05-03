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

    # Discover entity prefixes from keys.
    # An "entity" key has the shape ``<prefix>/<metric...>`` where prefix is
    # an agent name (which may contain spaces, apostrophes, hyphens, dots —
    # e.g. "Silas Varnham", "Thaddeus 'Aurelius' Thorne") and not one of the
    # known non-agent namespaces below.
    import re as _re
    _SKIP_PREFIXES = {
        "env_episodes",  # per-env episode counts
        "cpu", "mem",    # system telemetry
        "time",          # wall-time breakdown
        "rollout",       # rollout-stage stats
        "agents",        # already-stacked aggregates (don't re-discover)
    }
    entity_metrics: dict[str, set[str]] = {}  # prefix -> set of metric names
    for k in keys:
        if "/" not in k:
            continue
        prefix, metric = k.split("/", 1)
        if prefix in _SKIP_PREFIXES or not prefix:
            continue
        entity_metrics.setdefault(prefix, set()).add(metric)

    # Find the ordered list of entity prefixes.
    # Prefer numeric-suffixed groups (agent_0, agent_1 …); fall back to
    # YAML-config order if available (preserves "characters:" ordering),
    # else alphabetical order.
    base_re = _re.compile(r"^([a-zA-Z_]+)(\d+)$")
    base_groups: dict[str, list[tuple[int, str]]] = {}
    for prefix in entity_metrics:
        bm = base_re.match(prefix)
        if bm:
            base, idx = bm.group(1), int(bm.group(2))
            base_groups.setdefault(base, []).append((idx, prefix))

    if base_groups:
        best_base = max(base_groups, key=lambda b: len(base_groups[b]))
        ordered_prefixes = [p for _, p in sorted(base_groups[best_base])]
    elif entity_metrics:
        ordered_prefixes = sorted(entity_metrics.keys())
    else:
        ordered_prefixes = []

    if ordered_prefixes:
        all_metrics: set[str] = set()
        for p in ordered_prefixes:
            all_metrics |= entity_metrics.get(p, set())

        for metric in all_metrics:
            series = [data[f"{p}/{metric}"] for p in ordered_prefixes
                      if f"{p}/{metric}" in data]
            if series:
                data[f"agents/{metric}"] = np.stack(series)  # (n_agents, T)

        data["agent_names"] = ordered_prefixes

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
    agent_names: list[str] = data.get("agent_names", [])

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
                    label=agent_names[i] if i < len(agent_names) else f"agent_{i}")
        ax.legend(loc="upper right")
    ax.set_title("Policy entropy (per agent)")
    ax.set_ylabel("entropy (nats)")

    # ── Row 2: action loss (per agent) | value loss (per agent) ──────────────
    ax = axes[2, 0]
    if "agents/act_loss" in data:
        for i, series in enumerate(data["agents/act_loss"]):
            ax.plot(iters, smooth(series), color=_AGENT_COLORS[i % len(_AGENT_COLORS)],
                    label=agent_names[i] if i < len(agent_names) else f"agent_{i}")
        ax.legend(loc="upper right")
    ax.set_title("Action loss — REINFORCE (per agent)")
    ax.set_ylabel("loss")

    ax = axes[2, 1]
    if "agents/value_loss" in data:
        for i, series in enumerate(data["agents/value_loss"]):
            ax.plot(iters, smooth(series), color=_AGENT_COLORS[i % len(_AGENT_COLORS)],
                    label=agent_names[i] if i < len(agent_names) else f"agent_{i}")
        ax.legend(loc="upper right")
    ax.set_title("Value loss (per agent)")
    ax.set_ylabel("loss")

    # ── Row 3 (optional): KL | perception loss ───────────────────────────────
    if has_kl:
        ax = axes[3, 0]
        if "agents/kl" in data:
            for i, series in enumerate(data["agents/kl"]):
                ax.plot(iters, smooth(series), color=_AGENT_COLORS[i % len(_AGENT_COLORS)],
                        label=agent_names[i] if i < len(agent_names) else f"agent_{i}")
            ax.legend()
        ax.set_title("KL from reference (per agent)")
        ax.set_ylabel("KL")

        ax = axes[3, 1]
        if "agents/perc_loss" in data:
            for i, series in enumerate(data["agents/perc_loss"]):
                ax.plot(iters, smooth(series), color=_AGENT_COLORS[i % len(_AGENT_COLORS)],
                        label=agent_names[i] if i < len(agent_names) else f"agent_{i}")
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
# Per-experiment: compute utilisation
# ---------------------------------------------------------------------------

def plot_compute_utilization(exp_dir: Path, data: dict[str, np.ndarray]) -> Path | None:
    """How efficiently are we training: time breakdown, throughput, memory.

    Returns None if the run has no compute-telemetry keys (older runs).
    """
    iters = data.get("iteration", np.arange(len(data["total_loss"])))

    # Telemetry keys we expect from the trainer's per-iteration logging.
    have_time = any(k in data for k in
                    ["time/iter_s", "time/rollout_s", "time/loss_s"])
    have_thru = any(k in data for k in
                    ["rollout/tokens_per_s", "rollout/gen_calls"])
    have_gpu = any(k.startswith("mem/gpu") for k in data)
    have_cpu = "cpu/percent" in data or "mem/cpu_rss_gb" in data

    if not (have_time or have_thru or have_gpu or have_cpu):
        return None

    fig, axes = plt.subplots(3, 2, figsize=(11, 10), squeeze=False)
    fig.suptitle(f"{exp_dir.name} — compute utilisation",
                 fontsize=11, fontweight="bold")

    # ── Row 0 col 0: stacked time breakdown per iteration ──────────────────
    ax = axes[0, 0]
    components = [
        ("time/rollout_s", "rollout (gen)", "#4DBBD5"),
        ("time/loss_s",    "loss + backward", "#E64B35"),
        ("time/optim_s",   "optimizer step", "#F39B7F"),
        ("time/env_step_s","env step",       "#7E6148"),
    ]
    series = [(label, color, smooth(data[k])) for k, label, color in components if k in data]
    if series:
        labels = [s[0] for s in series]
        colors = [s[1] for s in series]
        arrs = np.stack([s[2] for s in series])
        ax.stackplot(iters, arrs, labels=labels, colors=colors, alpha=0.85)
        if "time/iter_s" in data:
            ax.plot(iters, smooth(data["time/iter_s"]), color="#222222",
                    linewidth=1.2, linestyle="--", label="iter total")
        ax.legend(loc="upper right", fontsize=7)
        ax.set_title("Time breakdown per iteration")
        ax.set_ylabel("seconds")
    else:
        ax.set_visible(False)

    # ── Row 0 col 1: rollout fraction (gen / iter) ─────────────────────────
    ax = axes[0, 1]
    if "rollout/gen_frac" in data:
        ax.plot(iters, smooth(data["rollout/gen_frac"]) * 100,
                color="#4DBBD5", label="generation %")
        ax.set_ylim(0, 100)
        ax.set_title("Generation share of iteration time")
        ax.set_ylabel("% of iter")
        ax.axhline(50, color="grey", linestyle=":", linewidth=0.8)
    elif "time/iter_s" in data and "time/rollout_s" in data:
        frac = data["time/rollout_s"] / np.maximum(data["time/iter_s"], 1e-6)
        ax.plot(iters, smooth(frac) * 100, color="#4DBBD5")
        ax.set_ylim(0, 100)
        ax.set_title("Generation share of iteration time")
        ax.set_ylabel("% of iter")
    else:
        ax.set_visible(False)

    # ── Row 1 col 0: generation throughput ─────────────────────────────────
    ax = axes[1, 0]
    if "rollout/tokens_per_s" in data:
        ax.plot(iters, smooth(data["rollout/tokens_per_s"]),
                color="#4DBBD5", label="generated tokens/s")
        ax.set_title("Generation throughput")
        ax.set_ylabel("tokens / second")
        ax.legend(loc="lower right", fontsize=8)
    else:
        ax.set_visible(False)

    # ── Row 1 col 1: rollout batch sizes (mean / max) ──────────────────────
    ax = axes[1, 1]
    plotted = False
    if "rollout/mean_batch" in data:
        ax.plot(iters, smooth(data["rollout/mean_batch"]),
                color="#4DBBD5", label="mean batch")
        plotted = True
    if "rollout/max_batch" in data:
        ax.plot(iters, smooth(data["rollout/max_batch"]),
                color="#222222", linestyle=":", linewidth=1.0, label="max batch")
        plotted = True
    if "rollout/gen_calls" in data:
        ax2 = ax.twinx()
        ax2.plot(iters, smooth(data["rollout/gen_calls"]),
                 color="#F39B7F", linewidth=1.0, label="gen calls / iter")
        ax2.set_ylabel("gen calls", color="#F39B7F")
        ax2.tick_params(axis="y", labelcolor="#F39B7F")
        ax2.spines["right"].set_visible(True)
        plotted = True
    if plotted:
        ax.set_title("Rollout batching")
        ax.set_ylabel("batch size")
        ax.legend(loc="upper left", fontsize=7)
    else:
        ax.set_visible(False)

    # ── Row 2 col 0: GPU memory ────────────────────────────────────────────
    ax = axes[2, 0]
    gpu_total = None
    plotted = False
    if "mem/gpu0/total_gb" in data and len(data["mem/gpu0/total_gb"]) > 0:
        gpu_total = float(np.nanmax(data["mem/gpu0/total_gb"]))
    pairs = [
        ("mem/gpu0/alloc_gb",      "alloc",         "#4DBBD5"),
        ("mem/gpu0/reserved_gb",   "reserved",      "#F39B7F"),
        ("mem/gpu0/peak_alloc_gb", "peak alloc",    "#E64B35"),
    ]
    for key, label, color in pairs:
        if key in data:
            ax.plot(iters, smooth(data[key]), color=color, label=label)
            plotted = True
    if gpu_total is not None and gpu_total > 0:
        ax.axhline(gpu_total, color="grey", linestyle=":", linewidth=0.8,
                   label=f"GPU total {gpu_total:.0f} GiB")
    if plotted:
        ax.set_title("GPU memory")
        ax.set_ylabel("GiB")
        ax.legend(loc="lower right", fontsize=7)
    else:
        ax.set_visible(False)

    # ── Row 2 col 1: CPU memory + utilisation ──────────────────────────────
    ax = axes[2, 1]
    plotted = False
    if "mem/cpu_rss_gb" in data:
        ax.plot(iters, smooth(data["mem/cpu_rss_gb"]),
                color="#55A868", label="RSS")
        ax.set_ylabel("CPU RSS (GiB)", color="#55A868")
        ax.tick_params(axis="y", labelcolor="#55A868")
        plotted = True
    if "cpu/percent" in data:
        ax2 = ax.twinx()
        ax2.plot(iters, smooth(data["cpu/percent"]),
                 color="#C44E52", linewidth=1.0, label="CPU %")
        ax2.set_ylabel("CPU %", color="#C44E52")
        ax2.tick_params(axis="y", labelcolor="#C44E52")
        ax2.spines["right"].set_visible(True)
        plotted = True
    if plotted:
        ax.set_title("CPU memory + utilisation")
    else:
        ax.set_visible(False)

    # X labels
    for ax_row in axes:
        for ax in ax_row:
            if ax.get_visible():
                ax.set_xlabel("iteration")
                ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    fig.tight_layout()
    out = exp_dir / "compute_utilization.png"
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
            out = plot_compute_utilization(exp_dir, all_data[exp_dir.name])
            if out is not None:
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
