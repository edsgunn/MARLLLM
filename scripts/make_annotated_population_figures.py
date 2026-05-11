"""Annotated versions of the iter-200 population scatters, designed to let
you match every point to the corresponding agent + run for trace inspection.

Two layouts per metric pair:

- *small_multiples* — 8 sub-panels (one per substrate-population run).
  Every point gets a label. The fewest-clutter view, and the one to use
  when you want to look up a specific agent.
- *overlay* — single panel like the original figure, but with every point
  labelled. Crowded, but preserves the cross-population view.

Outputs go to `runs/cultural_emergence/_population_figures_annotated/`,
side-by-side with the original (un-annotated) figures so the originals stay.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "runs" / "cultural_emergence" / "_population_figures"
OUT = REPO / "runs" / "cultural_emergence" / "_population_figures_annotated"

# Pull the iter-200 classification table the original script wrote.
CSV_CANDIDATES = sorted(SRC.glob("classification_table_iter200_*.csv"))
if not CSV_CANDIDATES:
    raise SystemExit("No classification_table_iter200_*.csv found in "
                     f"{SRC}; run plot_population_regime_figures.py first")
CSV = CSV_CANDIDATES[-1]

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
})

CAT_COLORS = {
    "coherent_low_kl": "#117733",
    "committed":       "#332288",
    "mid_range":       "#DDCC77",
    "policy_collapse": "#88CCEE",
    "text_degenerate": "#CC6677",
}
SUBSTRATE_COLORS = {
    "study_group_ashbourne_gc":      "#0072B2",
    "study_group_strathearn_server": "#D55E00",
}
SUBSTRATE_SHORT = {
    "study_group_ashbourne_gc":      "ashbourne",
    "study_group_strathearn_server": "strathearn",
}
POP_MARKERS = {2: "D", 4: "o", 8: "s", 16: "^"}

KL_THR, PERC_COHERENT, PERC_DEGEN = 1.0, 1.7, 2.0
ENT_LOW, ENT_HIGH = 0.2, 1.8


def short_first_name(full: str) -> str:
    """Return a short label for an agent — just the first name suffices since
    each panel only has one run, so duplicates inside a panel are rare."""
    return full.split()[0].rstrip(",")


def label_offset(point_idx: int, n_points: int) -> tuple[float, float]:
    """Cycle through small offsets so labels don't all sit in the same spot."""
    offsets = [(6, 4), (-6, 4), (6, -8), (-6, -8), (10, 0), (-10, 0)]
    return offsets[point_idx % len(offsets)]


def regions_kl_perc(ax, xmax=8.5, ymin=1.05, ymax=3.55):
    ax.axhspan(PERC_DEGEN, ymax, facecolor=CAT_COLORS["text_degenerate"],
               alpha=0.10, zorder=0)
    ax.axhspan(PERC_COHERENT, PERC_DEGEN, facecolor=CAT_COLORS["mid_range"],
               alpha=0.13, zorder=0)
    ax.fill_between([-0.5, KL_THR], ymin, PERC_COHERENT,
                    facecolor=CAT_COLORS["coherent_low_kl"], alpha=0.10, zorder=0)
    ax.fill_between([KL_THR, xmax + 0.5], ymin, PERC_COHERENT,
                    facecolor=CAT_COLORS["committed"], alpha=0.10, zorder=0)
    ax.axvline(KL_THR, color="0.35", lw=0.7, ls=(0, (4, 2)), zorder=1)
    ax.axhline(PERC_COHERENT, color="0.35", lw=0.7, ls=(0, (4, 2)), zorder=1)
    ax.axhline(PERC_DEGEN, color="0.35", lw=0.7, ls=(0, (1.5, 1.5)), zorder=1)
    ax.set_xlim(-0.3, xmax + 0.2)
    ax.set_ylim(ymin, ymax)


def regions_kl_entropy(ax, xmax=8.5, ymin=0.0, ymax=2.2):
    ax.axhspan(ymin, ENT_LOW, facecolor=CAT_COLORS["policy_collapse"],
               alpha=0.18, zorder=0)
    ax.axhspan(ENT_HIGH, ymax, facecolor=CAT_COLORS["policy_collapse"],
               alpha=0.18, zorder=0)
    ax.axvline(KL_THR, color="0.35", lw=0.7, ls=(0, (4, 2)), zorder=1)
    ax.axhline(ENT_LOW, color="0.35", lw=0.7, ls=(0, (1.5, 1.5)), zorder=1)
    ax.axhline(ENT_HIGH, color="0.35", lw=0.7, ls=(0, (1.5, 1.5)), zorder=1)
    ax.set_xlim(-0.3, xmax + 0.2)
    ax.set_ylim(ymin, ymax)


def regions_perc_entropy(ax, xmin=0.0, xmax=2.2, ymin=0.5, ymax=3.10):
    ax.axhspan(PERC_DEGEN, ymax, facecolor=CAT_COLORS["text_degenerate"],
               alpha=0.10, zorder=0)
    ax.axhspan(PERC_COHERENT, PERC_DEGEN, facecolor=CAT_COLORS["mid_range"],
               alpha=0.13, zorder=0)
    ax.axvspan(xmin, ENT_LOW, facecolor=CAT_COLORS["policy_collapse"],
               alpha=0.18, zorder=0)
    ax.axvspan(ENT_HIGH, xmax, facecolor=CAT_COLORS["policy_collapse"],
               alpha=0.18, zorder=0)
    ax.axvline(ENT_LOW, color="0.35", lw=0.7, ls=(0, (1.5, 1.5)), zorder=1)
    ax.axvline(ENT_HIGH, color="0.35", lw=0.7, ls=(0, (1.5, 1.5)), zorder=1)
    ax.axhline(PERC_COHERENT, color="0.35", lw=0.7, ls=(0, (4, 2)), zorder=1)
    ax.axhline(PERC_DEGEN, color="0.35", lw=0.7, ls=(0, (1.5, 1.5)), zorder=1)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)


def plot_smallmultiples(df: pd.DataFrame, x_col: str, y_col: str,
                        xlabel: str, ylabel: str,
                        regions_fn, out_path: Path, title: str,
                        x_clip: float | None = 8.5):
    """8-panel grid (4 pop sizes × 2 substrates). Each point labelled."""
    pop_sizes = sorted(df["pop_size"].unique())
    substrates = sorted(df["substrate"].unique())
    fig, axes = plt.subplots(len(substrates), len(pop_sizes),
                             figsize=(2.4 * len(pop_sizes), 2.6 * len(substrates)),
                             sharex=True, sharey=True, constrained_layout=True)
    axes = np.atleast_2d(axes)
    for i, sub in enumerate(substrates):
        for j, pop in enumerate(pop_sizes):
            ax = axes[i, j]
            regions_fn(ax)
            cell = df[(df["substrate"] == sub) & (df["pop_size"] == pop)]
            cell = cell.sort_values("trace_confirmed_coherent")  # starred last
            color = SUBSTRATE_COLORS[sub]
            marker = POP_MARKERS.get(int(pop), "x")
            for k, (_, r) in enumerate(cell.iterrows()):
                xv = r[x_col]
                if x_clip is not None:
                    xv = min(xv, x_clip)
                edge = CAT_COLORS.get(r.get("category", ""), "white")
                ax.scatter(xv, r[y_col], marker=marker, s=48,
                           facecolor=color, edgecolor=edge, linewidth=0.9,
                           alpha=0.95, zorder=3)
                if r["trace_confirmed_coherent"]:
                    ax.scatter(xv, r[y_col], marker="*", s=210,
                               facecolor="gold", edgecolor="black",
                               linewidth=0.9, zorder=10)
                dx, dy = label_offset(k, len(cell))
                ax.annotate(short_first_name(r["agent"]),
                            xy=(xv, r[y_col]), xytext=(dx, dy),
                            textcoords="offset points",
                            fontsize=6.5, color="0.15",
                            path_effects=[
                                path_effects.withStroke(
                                    linewidth=2.2, foreground="white")
                            ],
                            zorder=5)
            ax.text(0.02, 0.96, f"n={len(cell)}",
                    transform=ax.transAxes, ha="left", va="top",
                    fontsize=6.5, color="0.4")
            if i == 0:
                ax.set_title(f"{pop}-agent", fontsize=9)
            if j == 0:
                ax.set_ylabel(f"{SUBSTRATE_SHORT[sub]}\n{ylabel}",
                              fontsize=8)
            if i == len(substrates) - 1:
                ax.set_xlabel(xlabel)
            ax.yaxis.grid(True, color="0.93", lw=0.5); ax.set_axisbelow(True)
    fig.suptitle(title, fontsize=10)
    fig.savefig(out_path, format="pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(out_path.with_suffix(".png"), format="png", dpi=220,
                bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"[ok] {out_path.name}")


def plot_overlay(df: pd.DataFrame, x_col: str, y_col: str,
                 xlabel: str, ylabel: str, regions_fn,
                 out_path: Path, title: str, x_clip: float | None = 8.5):
    """Single-panel like the original, but with every point labelled."""
    fig, ax = plt.subplots(figsize=(7.0, 5.4), constrained_layout=True)
    regions_fn(ax)
    df = df.sort_values("trace_confirmed_coherent")
    for k, (_, r) in enumerate(df.iterrows()):
        xv = r[x_col]
        if x_clip is not None:
            xv = min(xv, x_clip)
        marker = POP_MARKERS.get(int(r["pop_size"]), "x")
        color = SUBSTRATE_COLORS[r["substrate"]]
        edge = CAT_COLORS.get(r.get("category", ""), "white")
        ax.scatter(xv, r[y_col], marker=marker, s=48,
                   facecolor=color, edgecolor=edge, linewidth=0.9,
                   alpha=0.95, zorder=3)
        if r["trace_confirmed_coherent"]:
            ax.scatter(xv, r[y_col], marker="*", s=240,
                       facecolor="gold", edgecolor="black", linewidth=0.9,
                       zorder=10)
        # Compact tag: first-name + pop_size + sub-letter
        tag = f"{short_first_name(r['agent'])} {int(r['pop_size'])}{SUBSTRATE_SHORT[r['substrate']][0].upper()}"
        dx, dy = label_offset(k, len(df))
        ax.annotate(tag, xy=(xv, r[y_col]), xytext=(dx, dy),
                    textcoords="offset points",
                    fontsize=6, color="0.15",
                    path_effects=[
                        path_effects.withStroke(
                            linewidth=2.0, foreground="white")
                    ],
                    zorder=5)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.yaxis.grid(True, color="0.93", lw=0.5); ax.set_axisbelow(True)

    sub_handles = [plt.Line2D([], [], marker="o", color="w",
                              markerfacecolor=c, markeredgecolor="white",
                              markersize=7, label=SUBSTRATE_SHORT[s])
                   for s, c in SUBSTRATE_COLORS.items()]
    pop_handles = [plt.Line2D([], [], marker=m, color="w",
                              markerfacecolor="0.35",
                              markeredgecolor="white", markersize=7,
                              label=f"{n}-agent")
                   for n, m in POP_MARKERS.items() if n in df["pop_size"].unique()]
    leg1 = ax.legend(handles=sub_handles, loc="upper left", frameon=True,
                     framealpha=0.92, edgecolor="0.85", fancybox=False,
                     borderpad=0.4)
    leg1.get_frame().set_linewidth(0.5)
    ax.add_artist(leg1)
    leg2 = ax.legend(handles=pop_handles, loc="upper right", frameon=True,
                     framealpha=0.92, edgecolor="0.85", fancybox=False,
                     borderpad=0.4)
    leg2.get_frame().set_linewidth(0.5)
    ax.add_artist(leg2)

    cats_present = [c for c in CAT_COLORS if c in set(df["category"])]
    cat_handles = [plt.Line2D([], [], marker="o", color="w",
                              markerfacecolor="0.85",
                              markeredgecolor=CAT_COLORS[c],
                              markeredgewidth=0.9, markersize=8,
                              label=c.replace("_", " "))
                   for c in cats_present]
    leg3 = ax.legend(handles=cat_handles, loc="lower right", frameon=True,
                     framealpha=0.92, edgecolor="0.85", fancybox=False,
                     borderpad=0.4, fontsize=7,
                     title="regime (outline)", title_fontsize=7)
    leg3.get_frame().set_linewidth(0.5)

    fig.savefig(out_path, format="pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(out_path.with_suffix(".png"), format="png", dpi=220,
                bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"[ok] {out_path.name}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CSV)
    print(f"loaded {len(df)} agents from {CSV.name}")

    # --- KL × perception loss ---
    plot_smallmultiples(
        df, "kl", "perc_loss",
        "KL from reference (nats)", "Perception loss",
        regions_kl_perc, OUT / "fig1_annotated_smallmultiples_kl_vs_percloss.pdf",
        "Annotated: KL vs perception loss at iter 200 (small multiples)",
    )
    plot_overlay(
        df, "kl", "perc_loss",
        "KL from reference (nats)", "Perception loss",
        regions_kl_perc, OUT / "fig1_annotated_overlay_kl_vs_percloss.pdf",
        "Annotated: KL vs perception loss at iter 200 (all agents)",
    )

    # --- KL × entropy ---
    plot_smallmultiples(
        df, "kl", "entropy",
        "KL from reference (nats)", "Policy entropy",
        regions_kl_entropy, OUT / "fig3_annotated_smallmultiples_kl_vs_entropy.pdf",
        "Annotated: KL vs entropy at iter 200 (small multiples)",
    )

    # --- entropy × perception ---
    plot_smallmultiples(
        df, "entropy", "perc_loss",
        "Policy entropy", "Perception loss",
        regions_perc_entropy,
        OUT / "fig4_annotated_smallmultiples_entropy_vs_percloss.pdf",
        "Annotated: entropy vs perception loss at iter 200 (small multiples)",
        x_clip=None,
    )

    # Index csv for quick lookup
    idx = df[["agent", "substrate", "pop_size", "kl", "perc_loss", "entropy",
              "category", "final_iter", "run", "trace_confirmed_coherent"]]
    idx = idx.sort_values(["substrate", "pop_size", "category", "agent"])
    idx.to_csv(OUT / "agent_index.csv", index=False)
    print(f"[ok] agent_index.csv  ({len(idx)} agents)")


if __name__ == "__main__":
    main()
