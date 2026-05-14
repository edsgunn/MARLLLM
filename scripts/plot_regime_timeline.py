"""Per-iteration regime classification for a single training run.

Reuses thresholds, CATS, and colours from ``plot_population_regime_figures``.
For each logged iteration we classify every agent into one of the five regimes
and produce three views:

  fig_a  Trajectory scatter in (KL, perception loss) space — one curve per
         agent threading through the regime regions over training, with the
         end-of-run marker styled by final regime.
  fig_b  Stacked-bar regime composition over iterations.
  fig_c  Per-agent regime timeline (heatmap with agents on y, iteration on x,
         cell colour = regime).

Usage:
    python scripts/plot_regime_timeline.py \
        runs/cultural_emergence/run12_32agent_7B_study_group_ashbourne_gc \
        [--out-dir DIR] [--smooth N]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from plot_population_regime_figures import (
    CATS, CAT_COLORS, CAT_LABELS,
    KL_THR, PERC_COHERENT, PERC_DEGEN, ENT_LOW, ENT_HIGH,
    classify,
)

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
})


def load_per_agent_series(run_dir: Path
                          ) -> tuple[list[int], dict[str, dict[str, np.ndarray]]]:
    """Returns (iterations, {agent: {metric: array}}). Missing values -> nan."""
    iters: list[int] = []
    raw: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    metrics_path = run_dir / "metrics.jsonl"
    with metrics_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "iteration" not in rec:
                continue
            it = int(rec["iteration"])
            per_agent: dict[str, dict[str, float]] = defaultdict(dict)
            for k, v in rec.items():
                if "/" not in k:
                    continue
                agent, metric = k.split("/", 1)
                if metric in ("kl", "perc_loss", "entropy"):
                    per_agent[agent][metric] = float(v)
            if not per_agent:
                continue
            iters.append(it)
            seen_agents = set(per_agent.keys()) | set(raw.keys())
            for agent in seen_agents:
                m = per_agent.get(agent, {})
                for metric in ("kl", "perc_loss", "entropy"):
                    raw[agent][metric].append(m.get(metric, float("nan")))
    # Convert to arrays of equal length
    out: dict[str, dict[str, np.ndarray]] = {}
    n = len(iters)
    for agent, mm in raw.items():
        out[agent] = {k: np.asarray(v[:n] + [float("nan")] * (n - len(v)),
                                     dtype=float)
                      for k, v in mm.items()}
    return iters, out


def classify_series(series: dict[str, dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """{agent: array of category strings (one per iteration)}."""
    out: dict[str, np.ndarray] = {}
    for agent, m in series.items():
        kl, perc, ent = m["kl"], m["perc_loss"], m["entropy"]
        cats = np.empty(len(kl), dtype=object)
        for i in range(len(kl)):
            if np.isnan(kl[i]) or np.isnan(perc[i]) or np.isnan(ent[i]):
                cats[i] = None
            else:
                cats[i] = classify(float(kl[i]), float(perc[i]), float(ent[i]))
        out[agent] = cats
    return out


def smooth(a: np.ndarray, w: int) -> np.ndarray:
    if w <= 1 or len(a) < 2:
        return a
    kernel = np.ones(w) / w
    # nan-aware via masked rolling
    a_filled = np.where(np.isnan(a), 0.0, a)
    mask = (~np.isnan(a)).astype(float)
    num = np.convolve(a_filled, kernel, mode="same")
    den = np.convolve(mask, kernel, mode="same")
    out = np.where(den > 0, num / den, np.nan)
    return out


def _shade_regime_regions_kl_perc(ax, x_max: float, y_min: float, y_max: float):
    """Same region shading as fig1 in plot_population_regime_figures."""
    ax.axhspan(PERC_DEGEN, y_max,
               facecolor=CAT_COLORS["text_degenerate"], alpha=0.10, zorder=0)
    ax.axhspan(PERC_COHERENT, PERC_DEGEN,
               facecolor=CAT_COLORS["mid_range"], alpha=0.13, zorder=0)
    ax.fill_between([-0.5, KL_THR], y_min, PERC_COHERENT,
                    facecolor=CAT_COLORS["coherent_low_kl"], alpha=0.10, zorder=0)
    ax.fill_between([KL_THR, x_max + 0.5], y_min, PERC_COHERENT,
                    facecolor=CAT_COLORS["committed"], alpha=0.10, zorder=0)
    ax.axvline(KL_THR, color="0.35", lw=0.7, ls=(0, (4, 2)), zorder=1)
    ax.axhline(PERC_COHERENT, color="0.35", lw=0.7, ls=(0, (4, 2)), zorder=1)
    ax.axhline(PERC_DEGEN, color="0.35", lw=0.7, ls=(0, (1.5, 1.5)), zorder=1)


def fig_trajectories(iters: list[int],
                     series: dict[str, dict[str, np.ndarray]],
                     cats: dict[str, np.ndarray],
                     out_path: Path, *, smooth_w: int) -> None:
    """Per-agent trajectory in (KL, perc_loss) with regime regions shaded.
    Start = small grey ring; end = filled circle outlined by final regime."""
    fig, ax = plt.subplots(figsize=(6.5, 5.5), constrained_layout=True)
    x_max = 8.5
    y_min, y_max = 1.05, 3.10
    _shade_regime_regions_kl_perc(ax, x_max, y_min, y_max)

    agents = sorted(series.keys())
    try:
        import distinctipy
        palette = distinctipy.get_colors(len(agents), rng=0)
    except Exception:
        cmap = plt.get_cmap("turbo")
        palette = [cmap(i / max(len(agents) - 1, 1)) for i in range(len(agents))]

    for i, agent in enumerate(agents):
        kl = np.clip(smooth(series[agent]["kl"], smooth_w), None, x_max)
        perc = smooth(series[agent]["perc_loss"], smooth_w)
        color = palette[i]
        ax.plot(kl, perc, color=color, lw=0.9, alpha=0.55, zorder=2)
        # Start marker
        if not (np.isnan(kl[0]) or np.isnan(perc[0])):
            ax.scatter(kl[0], perc[0], marker="o", s=18,
                       facecolor="white", edgecolor=color, linewidth=0.8,
                       alpha=0.9, zorder=3)
        # End marker — outline = final regime colour
        valid = np.where(~np.isnan(kl) & ~np.isnan(perc))[0]
        if len(valid):
            j = valid[-1]
            cat = cats[agent][j] if cats[agent][j] is not None else "mid_range"
            ax.scatter(kl[j], perc[j], marker="o", s=46,
                       facecolor=color,
                       edgecolor=CAT_COLORS.get(cat, "black"),
                       linewidth=1.2, alpha=0.95, zorder=4)

    ax.set_xlim(-0.3, x_max + 0.2)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("KL from reference (nats)")
    ax.set_ylabel("Perception loss")
    ax.set_title(f"Agent trajectories through regime space  ({len(agents)} agents)",
                 pad=4)

    # Region/regime legend
    cat_handles = [
        plt.Line2D([], [], marker="o", color="w", markerfacecolor="0.85",
                   markeredgecolor=CAT_COLORS[c], markeredgewidth=1.0,
                   markersize=7, label=CAT_LABELS[c])
        for c in CATS
    ]
    misc_handles = [
        plt.Line2D([], [], marker="o", color="w", markerfacecolor="white",
                   markeredgecolor="0.4", markersize=5, label="start (iter 0)"),
        plt.Line2D([], [], marker="o", color="w", markerfacecolor="0.55",
                   markeredgecolor="0.2", markersize=7, label="end (final iter)"),
    ]
    leg1 = ax.legend(handles=cat_handles, loc="upper right",
                     title="final regime (outline)", title_fontsize=7.5,
                     frameon=True, framealpha=0.92, edgecolor="0.85",
                     handletextpad=0.3, borderpad=0.4, labelspacing=0.3)
    ax.add_artist(leg1)
    ax.legend(handles=misc_handles, loc="lower right",
              frameon=True, framealpha=0.92, edgecolor="0.85",
              handletextpad=0.3, borderpad=0.4)

    fig.savefig(out_path, format="pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(out_path.with_suffix(".png"), format="png", dpi=220,
                bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"[ok] wrote {out_path} (+ .png)")


def fig_composition(iters: list[int], cats: dict[str, np.ndarray],
                    out_path: Path) -> None:
    """Stacked bars showing regime composition at each iteration."""
    agents = sorted(cats.keys())
    n_it = len(iters)
    counts = {c: np.zeros(n_it, dtype=int) for c in CATS}
    for j in range(n_it):
        for a in agents:
            cat = cats[a][j]
            if cat in counts:
                counts[cat][j] += 1
    totals = np.array([sum(counts[c][j] for c in CATS) for j in range(n_it)])
    totals = np.where(totals == 0, 1, totals)
    fracs = {c: counts[c] / totals for c in CATS}

    fig, ax = plt.subplots(figsize=(7.0, 3.4), constrained_layout=True)
    bottom = np.zeros(n_it)
    x = np.asarray(iters)
    # Use fill_between for a smooth stacked-area look.
    for c in CATS:
        top = bottom + fracs[c]
        ax.fill_between(x, bottom, top, facecolor=CAT_COLORS[c],
                        edgecolor="white", linewidth=0.0,
                        label=CAT_LABELS[c], alpha=0.95)
        bottom = top
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", "0.25", "0.5", "0.75", "1"])
    ax.set_xlabel("iteration")
    ax.set_ylabel("fraction of agents")
    ax.set_title(f"Regime composition over training  ({len(agents)} agents)",
                 pad=4)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5),
              frameon=False, handlelength=1.2, borderpad=0.3,
              labelspacing=0.35)
    fig.savefig(out_path, format="pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(out_path.with_suffix(".png"), format="png", dpi=220,
                bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"[ok] wrote {out_path} (+ .png)")


def fig_per_agent_timeline(iters: list[int], cats: dict[str, np.ndarray],
                           out_path: Path) -> None:
    """Heatmap: rows = agents, columns = iterations, cell = regime colour.
    Agents are sorted by their final regime, then by name within regime."""
    cat_to_idx = {c: i for i, c in enumerate(CATS)}
    agents = list(cats.keys())
    # Sort: final regime (in CATS order), then alphabetic
    def sort_key(a):
        final = next((c for c in cats[a][::-1] if c is not None), "mid_range")
        return (cat_to_idx.get(final, len(CATS)), a)
    agents.sort(key=sort_key)

    n_it = len(iters)
    grid = np.full((len(agents), n_it), -1, dtype=int)
    for r, a in enumerate(agents):
        for j in range(n_it):
            c = cats[a][j]
            if c in cat_to_idx:
                grid[r, j] = cat_to_idx[c]

    # Build a discrete colormap: index 0..len(CATS)-1; -1 = nan (light grey).
    colors = [CAT_COLORS[c] for c in CATS]
    cmap = mpl.colors.ListedColormap(colors)
    cmap.set_under("#eeeeee")
    norm = mpl.colors.BoundaryNorm(np.arange(-0.5, len(CATS) + 0.5, 1), cmap.N)

    h = max(3.0, 0.18 * len(agents) + 1.0)
    fig, ax = plt.subplots(figsize=(7.5, h), constrained_layout=True)
    im = ax.imshow(grid, aspect="auto", cmap=cmap, norm=norm,
                   interpolation="nearest",
                   extent=[iters[0], iters[-1], len(agents) - 0.5, -0.5])
    ax.set_yticks(range(len(agents)))
    ax.set_yticklabels(agents, fontsize=6.5)
    ax.set_xlabel("iteration")
    ax.set_title(f"Per-agent regime over training  ({len(agents)} agents, "
                 f"sorted by final regime)", pad=4)
    ax.tick_params(axis="y", length=0)

    # Legend (regime swatches)
    handles = [mpl.patches.Patch(facecolor=CAT_COLORS[c], edgecolor="white",
                                 label=CAT_LABELS[c]) for c in CATS]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5),
              frameon=False, handlelength=1.2, borderpad=0.3,
              labelspacing=0.35)

    fig.savefig(out_path, format="pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(out_path.with_suffix(".png"), format="png", dpi=220,
                bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"[ok] wrote {out_path} (+ .png)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Default: <run_dir>/regime_figures")
    ap.add_argument("--smooth", type=int, default=5,
                    help="Window for smoothing trajectories (iterations).")
    args = ap.parse_args()

    out_dir = args.out_dir or (args.run_dir / "regime_figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    iters, series = load_per_agent_series(args.run_dir)
    cats = classify_series(series)
    print(f"Loaded {len(series)} agents × {len(iters)} iterations "
          f"from {args.run_dir.name}")

    # Final-iteration summary
    final_counts = {c: 0 for c in CATS}
    for a, arr in cats.items():
        for c in arr[::-1]:
            if c is not None:
                final_counts[c] = final_counts.get(c, 0) + 1
                break
    print("Final-iteration regime counts:")
    for c in CATS:
        print(f"  {c:<18}  {final_counts[c]}")

    fig_trajectories(iters, series, cats,
                     out_dir / "regime_trajectories_kl_perc.pdf",
                     smooth_w=args.smooth)
    fig_composition(iters, cats,
                    out_dir / "regime_composition_over_iters.pdf")
    fig_per_agent_timeline(iters, cats,
                           out_dir / "regime_per_agent_timeline.pdf")


if __name__ == "__main__":
    main()
