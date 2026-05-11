"""Build the two population-size figures for the NeurIPS submission.

Figure 1: scatter of (KL, perception loss) across all (agent, end-of-training)
          pairs, with classification region overlays.
Figure 2: stacked bars of regime composition vs population size, one panel per
          substrate.

Reads `metrics.jsonl` from each configured run, takes per-agent metrics from
the final logged iteration, classifies each agent, writes a CSV, and saves
both figures as PDFs.

Usage:
    python scripts/plot_population_regime_figures.py [--out-dir DIR]
                                                     [--include-2agent]
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# NeurIPS-friendly typography. Times-like serif, 9pt body to match \small in
# the template (figures usually read at 8-9pt for axis labels).
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
    "legend.title_fontsize": 7.5,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

REPO = Path(__file__).resolve().parent.parent
RUNS = REPO / "runs" / "cultural_emergence"


@dataclass
class RunSpec:
    name: str
    substrate: str   # study_group_ashbourne_gc | study_group_strathearn_server
    pop_size: int


# Canonical runs for the figures. Add/remove here to redo with different runs.
RUN_SPECS: list[RunSpec] = [
    RunSpec("run10_2agent_7B_study_group_ashbourne_gc",      "study_group_ashbourne_gc",     2),
    RunSpec("run10_2agent_7B_study_group_strathearn_server", "study_group_strathearn_server", 2),
    RunSpec("run10_4agent_7B_study_group_ashbourne_gc",      "study_group_ashbourne_gc",     4),
    RunSpec("run10_4agent_7B_study_group_strathearn_server", "study_group_strathearn_server", 4),
    RunSpec("run7_8agent_7B_study_group_ashbourne_gc",       "study_group_ashbourne_gc",     8),
    RunSpec("run7_8agent_7B_study_group_strathearn_server",  "study_group_strathearn_server", 8),
    RunSpec("run8_16agent_7B_study_group_ashbourne_gc",      "study_group_ashbourne_gc",     16),
    RunSpec("run8_16agent_7B_study_group_strathearn_server", "study_group_strathearn_server", 16),
]


# Trace-confirmed coherent agents (high-KL but verified coherent in trace).
# Format: (run_name, agent_name).
TRACE_CONFIRMED_COHERENT: set[tuple[str, str]] = {
    ("run8_16agent_7B_study_group_ashbourne_gc", "Priya Shah"),
}

# Classification thresholds (calibrated against observed trajectories).
KL_THR = 1.0
PERC_COHERENT = 1.7
PERC_DEGEN = 2.0
ENT_LOW = 0.2
ENT_HIGH = 1.8

CATS = ["coherent_low_kl", "committed", "mid_range",
        "policy_collapse", "text_degenerate"]
# Colorblind-friendly palette (Wong 2011, adjusted). Used for stacked bars
# and region shading.
CAT_COLORS = {
    "coherent_low_kl": "#117733",  # teal-green
    "committed":       "#332288",  # deep blue/purple
    "mid_range":       "#DDCC77",  # sand
    "policy_collapse": "#88CCEE",  # light blue (entropy collapse, surface text often fine)
    "text_degenerate": "#CC6677",  # muted red (perc-driven; visibly broken text)
}
CAT_LABELS = {
    "coherent_low_kl": "coherent (low KL)",
    "committed":       "committed",
    "mid_range":       "mid-range",
    "policy_collapse": "policy collapse",
    "text_degenerate": "text degenerate",
}
SUBSTRATE_COLORS = {
    "study_group_ashbourne_gc":      "#0072B2",  # blue
    "study_group_strathearn_server": "#D55E00",  # vermillion
}
SUBSTRATE_LABELS = {
    "study_group_ashbourne_gc":      "ashbourne",
    "study_group_strathearn_server": "strathearn",
}
POP_MARKERS = {2: "D", 4: "o", 8: "s", 16: "^"}


def classify(kl: float, perc: float, entropy: float) -> str:
    """5-way classification.

    `text_degenerate` (perc >= 2.0) is taken first because broken text
    dominates whatever else is going on. `policy_collapse` (entropy out of
    band, perc still < 2.0) is the surface-coherent-but-distribution-collapsed
    failure mode. The remaining three categories partition the alive region.
    """
    if perc >= PERC_DEGEN:
        return "text_degenerate"
    if entropy < ENT_LOW or entropy > ENT_HIGH:
        return "policy_collapse"
    if perc < PERC_COHERENT and kl < KL_THR:
        return "coherent_low_kl"
    if perc < PERC_COHERENT and kl >= KL_THR:
        return "committed"
    return "mid_range"


_AGENT_METRIC_RE = re.compile(r"^(?P<agent>.+?)/(?P<metric>kl|perc_loss|entropy|act_loss|value_loss)$")


def extract_agents_from_record(rec: dict) -> dict[str, dict[str, float]]:
    """Return {agent_name: {metric: value}} from a single metrics record."""
    out: dict[str, dict[str, float]] = defaultdict(dict)
    for k, v in rec.items():
        m = _AGENT_METRIC_RE.match(k)
        if not m:
            continue
        out[m.group("agent")][m.group("metric")] = v
    return dict(out)


def load_iteration(run_dir: Path, target_iter: int | None
                   ) -> tuple[int, dict[str, dict[str, float]]]:
    """Return (iteration, agent_metrics) for the record closest to target_iter
    without exceeding it; if all records exceed target_iter, use the earliest;
    if target_iter is None, use the last record. Records loaded lazily so we
    don't keep the whole jsonl in memory."""
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    best = None  # closest record at-or-below target
    earliest = None
    last = None
    with metrics_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "iteration" not in rec:
                continue
            it = int(rec["iteration"])
            last = (it, rec)
            if earliest is None:
                earliest = (it, rec)
            if target_iter is not None and it <= target_iter:
                if best is None or it > best[0]:
                    best = (it, rec)
    if last is None:
        raise RuntimeError(f"No iteration records in {metrics_path}")
    if target_iter is None:
        chosen = last
    elif best is not None:
        chosen = best
    else:
        chosen = earliest  # all records past target; fall back to first
    return chosen[0], extract_agents_from_record(chosen[1])


def build_table(specs: list[RunSpec], target_iter: int | None) -> pd.DataFrame:
    rows = []
    missing = []
    for spec in specs:
        run_dir = RUNS / spec.name
        if not run_dir.exists():
            missing.append(f"{spec.name} (dir missing)")
            continue
        try:
            iter_num, agents = load_iteration(run_dir, target_iter)
        except (FileNotFoundError, RuntimeError) as e:
            missing.append(f"{spec.name} ({e})")
            continue
        for agent, m in agents.items():
            kl, perc, ent = m.get("kl"), m.get("perc_loss"), m.get("entropy")
            gaps = [k for k, v in [("kl", kl), ("perc_loss", perc), ("entropy", ent)] if v is None]
            if gaps:
                missing.append(f"{spec.name}/{agent} missing {gaps}")
                continue
            rows.append({
                "run": spec.name,
                "substrate": spec.substrate,
                "pop_size": spec.pop_size,
                "agent": agent,
                "final_iter": iter_num,
                "kl": kl,
                "perc_loss": perc,
                "entropy": ent,
                "act_loss": m.get("act_loss", float("nan")),
                "value_loss": m.get("value_loss", float("nan")),
                "category": classify(kl, perc, ent),
                "trace_confirmed_coherent": (spec.name, agent) in TRACE_CONFIRMED_COHERENT,
            })
    if missing:
        print("[warn] gaps surfaced (not silently dropped):")
        for s in missing:
            print(f"   - {s}")
    df = pd.DataFrame(rows)
    return df


# ---------------- plotting ----------------

def fig1_scatter(df: pd.DataFrame, out_path: Path) -> None:
    # Two-column-equivalent width on a single-column NeurIPS figure: ~3.25in
    # works but is cramped with this much info. Use 3.4 x 3.2 for the single-
    # column slot, with care taken to keep typography legible.
    fig, ax = plt.subplots(figsize=(3.4, 3.2), constrained_layout=True)

    x_max = 8.5
    y_min, y_max = 1.10, 3.55  # headroom above PERC_DEGEN for legends

    # Soft region shading (no overlap; mid-range and text_degenerate stack
    # along y, coherent vs committed split along x within the safe-perception
    # band). Policy-collapse can't be encoded as a region on these axes; ✕
    # markers flag those points individually.
    ax.axhspan(PERC_DEGEN, y_max, xmin=0, xmax=1,
               facecolor=CAT_COLORS["text_degenerate"], alpha=0.10, zorder=0)
    ax.axhspan(PERC_COHERENT, PERC_DEGEN, xmin=0, xmax=1,
               facecolor=CAT_COLORS["mid_range"], alpha=0.13, zorder=0)
    # split the safe band at KL_THR: coherent (left) vs committed (right)
    ax.fill_between([-0.5, KL_THR], y_min, PERC_COHERENT,
                    facecolor=CAT_COLORS["coherent_low_kl"], alpha=0.10, zorder=0)
    ax.fill_between([KL_THR, x_max + 0.5], y_min, PERC_COHERENT,
                    facecolor=CAT_COLORS["committed"], alpha=0.10, zorder=0)

    # Threshold lines
    ax.axvline(KL_THR, color="0.35", lw=0.7, ls=(0, (4, 2)), zorder=1)
    ax.axhline(PERC_COHERENT, color="0.35", lw=0.7, ls=(0, (4, 2)), zorder=1)
    ax.axhline(PERC_DEGEN,    color="0.35", lw=0.7, ls=(0, (1.5, 1.5)), zorder=1)

    # Plot points; use marker for pop-size, color for substrate.
    # Render trace-confirmed-coherent agents last so the star sits above all
    # other overlays (substrate dots, ✕ markers, region shading).
    plotted = df.copy()
    plotted["kl_plot"] = plotted["kl"].clip(upper=x_max)
    plotted = plotted.sort_values("trace_confirmed_coherent")  # False first, True last
    for _, r in plotted.iterrows():
        marker = POP_MARKERS.get(int(r["pop_size"]), "x")
        color = SUBSTRATE_COLORS.get(r["substrate"], "k")
        # Fill = substrate, edge = regime. The edge colour does the work of the
        # old ✕ overlay: policy-collapse points sit in the green/blue alive
        # region but have a light-blue outline; text-degenerate has a red
        # outline; coherent / committed / mid-range are outlined in their
        # respective region colours.
        edge = CAT_COLORS.get(r["category"], "white")
        ax.scatter(
            r["kl_plot"], r["perc_loss"],
            marker=marker, s=44,
            facecolor=color, edgecolor=edge, linewidth=0.9,
            alpha=0.95, zorder=3,
        )
        if r["kl"] > x_max:
            ax.annotate("", xy=(x_max + 0.1, r["perc_loss"]),
                        xytext=(x_max - 0.4, r["perc_loss"]),
                        arrowprops=dict(arrowstyle="->", color="black", lw=0.6),
                        annotation_clip=False)
        if r["trace_confirmed_coherent"]:
            # Star sits at the highest z so it stays visible above any other
            # marker that lands at the same point.
            ax.scatter(r["kl_plot"], r["perc_loss"], marker="*", s=210,
                       facecolor="gold", edgecolor="black", linewidth=0.9,
                       zorder=10)

    ax.set_xlim(-0.3, x_max + 0.2)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel(r"KL from reference (nats)")
    ax.set_ylabel(r"Perception loss")
    ax.set_title("Commitment vs. degeneration", pad=4)

    # Region labels — placed in low-density corners, with light white halo so
    # they remain readable if a point lands near them.
    bbox_kw = dict(boxstyle="round,pad=0.18", facecolor="white",
                   edgecolor="none", alpha=0.78)
    label_kw = dict(fontsize=7, fontstyle="italic", bbox=bbox_kw)
    ax.text(0.05, 1.15, "coherent", color=CAT_COLORS["coherent_low_kl"],
            ha="left", va="bottom", **label_kw)
    # "committed" label is omitted here — the regime legend in the lower-right
    # already names it, and an italic label would either collide with the
    # legend or with the dense agent cluster around (4-8, 1.5).
    ax.text(x_max - 0.15, 1.83, "mid-range", color="#8a7a3d",
            ha="right", va="center", **label_kw)
    ax.text(4.2, 2.50, "text degenerate", color=CAT_COLORS["text_degenerate"],
            ha="center", va="center", **label_kw)

    # Two legends, side-by-side at bottom-center to keep the plot area clean.
    sub_handles = [
        plt.Line2D([], [], marker="o", color="w", markerfacecolor=c,
                   markeredgecolor="white", markeredgewidth=0.5,
                   markersize=6, label=SUBSTRATE_LABELS[s])
        for s, c in SUBSTRATE_COLORS.items()
    ]
    pop_handles = [
        plt.Line2D([], [], marker=m, color="w", markerfacecolor="0.35",
                   markeredgecolor="white", markeredgewidth=0.5,
                   markersize=6, label=f"{n} agents")
        for n, m in POP_MARKERS.items() if n in df["pop_size"].unique()
    ]
    # Three legends inside the plot:
    #   top-left  → pop-size (marker shape)
    #   top-right → regime (marker outline colour)
    #   below it  → substrate (marker fill colour)
    leg_pop = ax.legend(handles=pop_handles, loc="upper left",
                        frameon=True, framealpha=0.92, edgecolor="0.85",
                        fancybox=False, handletextpad=0.3,
                        borderpad=0.3, labelspacing=0.25, fontsize=7)
    leg_pop.get_frame().set_linewidth(0.5)
    ax.add_artist(leg_pop)

    cats_present = [c for c in CATS if c in set(df["category"])]
    cat_handles = [
        plt.Line2D([], [], marker="o", color="w", markerfacecolor="0.85",
                   markeredgecolor=CAT_COLORS[c], markeredgewidth=0.9,
                   markersize=6, label=CAT_LABELS[c])
        for c in cats_present
    ]
    leg_reg = ax.legend(handles=cat_handles, loc="upper right",
                        frameon=True, framealpha=0.92, edgecolor="0.85",
                        fancybox=False, handletextpad=0.3,
                        borderpad=0.3, labelspacing=0.25,
                        title="regime (outline)", title_fontsize=6.5,
                        fontsize=6.5)
    leg_reg.get_frame().set_linewidth(0.5)
    ax.add_artist(leg_reg)
    # Force a draw so the regime legend's bounding box is known, then anchor
    # the substrate legend immediately below it.
    fig.canvas.draw()
    reg_bbox = leg_reg.get_window_extent().transformed(ax.transAxes.inverted())
    leg_sub = ax.legend(handles=sub_handles, loc="upper right",
                        bbox_to_anchor=(reg_bbox.x1, reg_bbox.y0 - 0.01),
                        bbox_transform=ax.transAxes,
                        frameon=True, framealpha=0.92, edgecolor="0.85",
                        fancybox=False, handletextpad=0.3,
                        borderpad=0.3, labelspacing=0.25,
                        title="substrate (fill)", title_fontsize=6.5,
                        fontsize=6.5)
    leg_sub.get_frame().set_linewidth(0.5)

    fig.savefig(out_path, format="pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_path.with_suffix(".png"), format="png", dpi=220,
                bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"[ok] wrote {out_path} (+ .png)")


def fig2_stacked_bars(df: pd.DataFrame, out_path: Path) -> None:
    substrates = sorted(df["substrate"].unique())
    # Two-panel layout sized for full-text-width in NeurIPS (~5.5in usable).
    fig, axes = plt.subplots(1, len(substrates),
                             figsize=(5.5, 2.9),
                             sharey=True)
    fig.subplots_adjust(left=0.09, right=0.99, top=0.88, bottom=0.27,
                        wspace=0.08)
    if len(substrates) == 1:
        axes = [axes]

    pop_sizes = sorted(df["pop_size"].unique())
    x = np.arange(len(pop_sizes))
    width = 0.55

    for i, (ax, sub) in enumerate(zip(axes, substrates)):
        sub_df = df[df["substrate"] == sub]
        bottom = np.zeros(len(pop_sizes))
        totals = [int((sub_df["pop_size"] == p).sum()) for p in pop_sizes]
        for cat in CATS:
            fracs = []
            for p in pop_sizes:
                m = sub_df[sub_df["pop_size"] == p]
                fracs.append((m["category"] == cat).sum() / max(len(m), 1))
            fracs = np.array(fracs)
            ax.bar(x, fracs, width, bottom=bottom,
                   label=CAT_LABELS[cat],
                   color=CAT_COLORS[cat], edgecolor="white", linewidth=0.6)
            # Inline percentage labels for segments large enough to read
            for xi, frac, b in zip(x, fracs, bottom):
                if frac >= 0.05:
                    ax.text(xi, b + frac / 2, f"{frac*100:.0f}%",
                            ha="center", va="center",
                            fontsize=7, color="white", fontweight="bold")
            bottom += fracs
        for xi, n in zip(x, totals):
            ax.text(xi, 1.025, rf"$n=\,{n}$", ha="center", va="bottom",
                    fontsize=7.5, color="0.25")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{p}" for p in pop_sizes])
        ax.set_xlabel("population size")
        ax.set_ylim(0, 1.10)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_title(SUBSTRATE_LABELS[sub], pad=4)
        ax.tick_params(axis="x", length=0)
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color="0.92", lw=0.5, zorder=0)

    axes[0].set_ylabel("fraction of agents")
    axes[0].set_yticklabels(["0", "0.25", "0.5", "0.75", "1"])

    # Single shared legend below the panels.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc="lower center", bbox_to_anchor=(0.5, 0.0),
               ncol=len(CATS), frameon=False, handlelength=1.2,
               columnspacing=1.6, handletextpad=0.4)

    fig.savefig(out_path, format="pdf", pad_inches=0.04)
    fig.savefig(out_path.with_suffix(".png"), format="png", dpi=220,
                pad_inches=0.04)
    plt.close(fig)
    print(f"[ok] wrote {out_path} (+ .png)")


def _scatter_substrate_pop(ax, df, x_col, y_col, x_clip=None):
    """Common scatter routine: substrate -> color, pop_size -> marker."""
    plotted = df.copy()
    if x_clip is not None:
        plotted["x"] = plotted[x_col].clip(upper=x_clip)
    else:
        plotted["x"] = plotted[x_col]
    # Render trace-confirmed agents last so the star sits above other markers.
    plotted = plotted.sort_values("trace_confirmed_coherent")
    for _, r in plotted.iterrows():
        marker = POP_MARKERS.get(int(r["pop_size"]), "x")
        color = SUBSTRATE_COLORS.get(r["substrate"], "k")
        edge = CAT_COLORS.get(r.get("category", ""), "white")
        ax.scatter(r["x"], r[y_col], marker=marker, s=44,
                   facecolor=color, edgecolor=edge, linewidth=0.9,
                   alpha=0.95, zorder=3)
        if x_clip is not None and r[x_col] > x_clip:
            ax.annotate("", xy=(x_clip + 0.1, r[y_col]),
                        xytext=(x_clip - 0.4, r[y_col]),
                        arrowprops=dict(arrowstyle="->", color="black", lw=0.6),
                        annotation_clip=False)
        if r["trace_confirmed_coherent"]:
            ax.scatter(r["x"], r[y_col], marker="*", s=210,
                       facecolor="gold", edgecolor="black", linewidth=0.9,
                       zorder=10)


def _legend_substrate_pop(ax, df):
    """Three-legend layout: top-left = pop-size shapes, top-right = regime
    outlines, immediately below = substrate fills."""
    sub_handles = [
        plt.Line2D([], [], marker="o", color="w", markerfacecolor=c,
                   markeredgecolor="white", markeredgewidth=0.5,
                   markersize=6, label=SUBSTRATE_LABELS[s])
        for s, c in SUBSTRATE_COLORS.items()
    ]
    pop_handles = [
        plt.Line2D([], [], marker=m, color="w", markerfacecolor="0.35",
                   markeredgecolor="white", markeredgewidth=0.5,
                   markersize=6, label=f"{n} agents")
        for n, m in POP_MARKERS.items() if n in df["pop_size"].unique()
    ]
    cats_present = [c for c in CATS if c in set(df["category"])] \
        if "category" in df.columns else []
    cat_handles = [
        plt.Line2D([], [], marker="o", color="w", markerfacecolor="0.85",
                   markeredgecolor=CAT_COLORS[c], markeredgewidth=0.9,
                   markersize=6, label=CAT_LABELS[c])
        for c in cats_present
    ]

    leg_pop = ax.legend(handles=pop_handles, loc="upper left",
                        frameon=True, framealpha=0.92, edgecolor="0.85",
                        fancybox=False, handletextpad=0.3,
                        borderpad=0.3, labelspacing=0.25, fontsize=7)
    leg_pop.get_frame().set_linewidth(0.5)
    ax.add_artist(leg_pop)

    if cat_handles:
        leg_reg = ax.legend(handles=cat_handles, loc="upper right",
                            frameon=True, framealpha=0.92, edgecolor="0.85",
                            fancybox=False, handletextpad=0.3,
                            borderpad=0.3, labelspacing=0.25,
                            title="regime (outline)", title_fontsize=6.5,
                            fontsize=6.5)
        leg_reg.get_frame().set_linewidth(0.5)
        ax.add_artist(leg_reg)
        ax.figure.canvas.draw()
        reg_bbox = leg_reg.get_window_extent().transformed(ax.transAxes.inverted())
        leg_sub = ax.legend(handles=sub_handles, loc="upper right",
                            bbox_to_anchor=(reg_bbox.x1, reg_bbox.y0 - 0.01),
                            bbox_transform=ax.transAxes,
                            frameon=True, framealpha=0.92, edgecolor="0.85",
                            fancybox=False, handletextpad=0.3,
                            borderpad=0.3, labelspacing=0.25,
                            title="substrate (fill)", title_fontsize=6.5,
                            fontsize=6.5)
        leg_sub.get_frame().set_linewidth(0.5)
    else:
        leg_sub = ax.legend(handles=sub_handles, loc="upper right",
                            frameon=True, framealpha=0.92, edgecolor="0.85",
                            fancybox=False, handletextpad=0.3,
                            borderpad=0.3, labelspacing=0.25, fontsize=7)
        leg_sub.get_frame().set_linewidth(0.5)


def fig3_kl_vs_entropy(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(3.4, 3.2), constrained_layout=True)
    x_max = 8.5
    y_min, y_max = 0.0, 2.2

    # Entropy "alive" band: [ENT_LOW, ENT_HIGH]. Outside = policy_collapse
    # (entropy is the policy-distribution metric, so out-of-band here is the
    # policy-collapse failure mode, not the perc-driven text-degenerate one).
    ax.axhspan(y_min, ENT_LOW, facecolor=CAT_COLORS["policy_collapse"], alpha=0.18, zorder=0)
    ax.axhspan(ENT_HIGH, y_max, facecolor=CAT_COLORS["policy_collapse"], alpha=0.18, zorder=0)
    ax.axvline(KL_THR, color="0.35", lw=0.7, ls=(0, (4, 2)), zorder=1)
    ax.axhline(ENT_LOW, color="0.35", lw=0.7, ls=(0, (1.5, 1.5)), zorder=1)
    ax.axhline(ENT_HIGH, color="0.35", lw=0.7, ls=(0, (1.5, 1.5)), zorder=1)

    _scatter_substrate_pop(ax, df, "kl", "entropy", x_clip=x_max)

    ax.set_xlim(-0.3, x_max + 0.2)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel(r"KL from reference (nats)")
    ax.set_ylabel(r"Policy entropy (nats)")
    ax.set_title("Commitment vs. entropy collapse", pad=4)

    bbox_kw = dict(boxstyle="round,pad=0.18", facecolor="white",
                   edgecolor="none", alpha=0.78)
    label_kw = dict(fontsize=7, fontstyle="italic", bbox=bbox_kw)
    ax.text(x_max - 0.15, ENT_LOW - 0.04, "policy collapse (low entropy)",
            color="#1f6f8a", ha="right", va="top", **label_kw)
    ax.text(x_max - 0.15, ENT_HIGH + 0.04, "policy collapse (high entropy)",
            color="#1f6f8a", ha="right", va="bottom", **label_kw)

    _legend_substrate_pop(ax, df)
    fig.savefig(out_path, format="pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(out_path.with_suffix(".png"), format="png", dpi=220,
                bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"[ok] wrote {out_path} (+ .png)")


def fig4_perc_vs_entropy(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(3.4, 3.2), constrained_layout=True)
    x_min, x_max = 0.0, 2.2
    y_min, y_max = 0.5, 3.10

    # Degenerate by perception (top band) and by entropy (left/right edges).
    ax.axhspan(PERC_DEGEN, y_max, facecolor=CAT_COLORS["text_degenerate"], alpha=0.10, zorder=0)
    ax.axhspan(PERC_COHERENT, PERC_DEGEN, facecolor=CAT_COLORS["mid_range"], alpha=0.13, zorder=0)
    ax.axvspan(x_min, ENT_LOW, facecolor=CAT_COLORS["policy_collapse"], alpha=0.18, zorder=0)
    ax.axvspan(ENT_HIGH, x_max, facecolor=CAT_COLORS["policy_collapse"], alpha=0.18, zorder=0)
    ax.axvline(ENT_LOW, color="0.35", lw=0.7, ls=(0, (1.5, 1.5)), zorder=1)
    ax.axvline(ENT_HIGH, color="0.35", lw=0.7, ls=(0, (1.5, 1.5)), zorder=1)
    ax.axhline(PERC_COHERENT, color="0.35", lw=0.7, ls=(0, (4, 2)), zorder=1)
    ax.axhline(PERC_DEGEN, color="0.35", lw=0.7, ls=(0, (1.5, 1.5)), zorder=1)

    _scatter_substrate_pop(ax, df, "entropy", "perc_loss")

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel(r"Policy entropy (nats)")
    ax.set_ylabel(r"Perception loss")
    ax.set_title("Perception loss vs. entropy", pad=4)

    bbox_kw = dict(boxstyle="round,pad=0.18", facecolor="white",
                   edgecolor="none", alpha=0.78)
    label_kw = dict(fontsize=7, fontstyle="italic", bbox=bbox_kw)
    ax.text(x_max - 0.04, PERC_DEGEN + 0.04, "text degenerate",
            color=CAT_COLORS["text_degenerate"], ha="right", va="bottom", **label_kw)
    ax.text(ENT_LOW - 0.02, y_max - 0.1, "policy collapse",
            color="#1f6f8a", ha="right", va="top",
            rotation=90, **label_kw)

    _legend_substrate_pop(ax, df)
    fig.savefig(out_path, format="pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(out_path.with_suffix(".png"), format="png", dpi=220,
                bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"[ok] wrote {out_path} (+ .png)")


def fig5_metric_grid(df: pd.DataFrame, out_path: Path) -> None:
    """Box+strip plot grid: rows = metrics (KL, perc_loss, entropy),
    cols = substrates, x = pop_size."""
    substrates = sorted(df["substrate"].unique())
    metrics = [
        ("kl",        "KL from reference",  None),
        ("perc_loss", "Perception loss",    [(PERC_COHERENT, "0.35", (0,(4,2))),
                                              (PERC_DEGEN,    "0.35", (0,(1.5,1.5)))]),
        ("entropy",   "Policy entropy",     [(ENT_LOW, "0.35", (0,(1.5,1.5))),
                                              (ENT_HIGH, "0.35", (0,(1.5,1.5)))]),
    ]
    n_rows, n_cols = len(metrics), len(substrates)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5.5, 5.4),
                             sharex="col", constrained_layout=True)
    pop_sizes = sorted(df["pop_size"].unique())
    x_pos = {p: i for i, p in enumerate(pop_sizes)}
    rng = np.random.default_rng(0)

    for r, (col, ylabel, hlines) in enumerate(metrics):
        for c, sub in enumerate(substrates):
            ax = axes[r, c]
            sub_df = df[df["substrate"] == sub]
            data_per_pop = [sub_df[sub_df["pop_size"] == p][col].values for p in pop_sizes]
            # Box plot
            bp = ax.boxplot(data_per_pop, positions=list(x_pos.values()),
                            widths=0.55, showfliers=False, patch_artist=True,
                            medianprops=dict(color="black", lw=0.9))
            for patch in bp["boxes"]:
                patch.set(facecolor=SUBSTRATE_COLORS[sub], alpha=0.18,
                          edgecolor=SUBSTRATE_COLORS[sub], linewidth=0.7)
            for ln in bp["whiskers"] + bp["caps"]:
                ln.set(color=SUBSTRATE_COLORS[sub], linewidth=0.7)
            # Jittered points
            for p, vals in zip(pop_sizes, data_per_pop):
                if len(vals) == 0:
                    continue
                jitter = rng.uniform(-0.12, 0.12, size=len(vals))
                ax.scatter(np.full(len(vals), x_pos[p]) + jitter, vals,
                           s=14, alpha=0.85,
                           facecolor=SUBSTRATE_COLORS[sub], edgecolor="white",
                           linewidth=0.4, zorder=3)
            if hlines:
                for y, color, ls in hlines:
                    ax.axhline(y, color=color, lw=0.5, ls=ls, zorder=1)
            ax.set_xticks(list(x_pos.values()))
            ax.set_xticklabels([str(p) for p in pop_sizes])
            ax.tick_params(axis="x", length=0)
            ax.yaxis.grid(True, color="0.93", lw=0.5, zorder=0)
            ax.set_axisbelow(True)
            if c == 0:
                ax.set_ylabel(ylabel)
            if r == 0:
                ax.set_title(SUBSTRATE_LABELS[sub], pad=4)
            if r == n_rows - 1:
                ax.set_xlabel("population size")

    fig.savefig(out_path, format="pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(out_path.with_suffix(".png"), format="png", dpi=220,
                bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"[ok] wrote {out_path} (+ .png)")


# ---------------- main ----------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path,
                    default=REPO / "runs" / "cultural_emergence" / "_population_figures")
    ap.add_argument("--iter", type=int, default=200,
                    help="Target iteration; per-run we use the closest "
                         "logged iteration <= target. Pass 0 (or omit and "
                         "use --final) for end-of-training.")
    ap.add_argument("--final", action="store_true",
                    help="Use the last logged iteration of each run instead.")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()
    target = None if args.final else args.iter
    tag = "final" if target is None else f"iter{target}"

    specs = list(RUN_SPECS)

    df = build_table(specs, target)
    if df.empty:
        raise SystemExit("No data loaded")

    # Sanity check: surface obviously-wrong classifications.
    for _, r in df.iterrows():
        if r["trace_confirmed_coherent"] and r["category"] in (
                "text_degenerate", "policy_collapse"):
            print(f"[sanity] trace-confirmed coherent agent classified {r['category']}: "
                  f"{r['run']}/{r['agent']} (kl={r['kl']:.2f}, perc={r['perc_loss']:.2f}, "
                  f"ent={r['entropy']:.2f}) -- thresholds may need review")

    # Surface any per-run iteration mismatches.
    print(f"\nIterations sampled (target={target}):")
    for (run, popsz), g in df.groupby(["run", "pop_size"]):
        it = int(g["final_iter"].iloc[0])
        marker = "" if (target is None or it == target) else f"  (≠{target})"
        print(f"  {run}  pop={popsz}  iter={it}{marker}")

    csv_path = args.out_dir / f"classification_table_{tag}_{today}.csv"
    df.sort_values(["substrate", "pop_size", "run", "agent"]).to_csv(csv_path, index=False)
    print(f"[ok] wrote {csv_path}  ({len(df)} agents across {df['run'].nunique()} runs)")

    print("\nClassification counts by (substrate, pop_size):")
    print(df.groupby(["substrate", "pop_size", "category"]).size().unstack(fill_value=0))

    print("\nAgents by category:")
    for cat in CATS:
        sub = df[df["category"] == cat].sort_values(["substrate", "pop_size", "run", "agent"])
        print(f"\n[{cat}]  n={len(sub)}")
        if sub.empty:
            print("  (none)")
            continue
        for _, r in sub.iterrows():
            star = " *" if r["trace_confirmed_coherent"] else ""
            print(f"  {r['agent']:<22}  kl={r['kl']:6.3f}  perc={r['perc_loss']:5.3f}  "
                  f"ent={r['entropy']:5.3f}  iter={r['final_iter']:>4}  "
                  f"<- {r['run']}{star}")

    fig1_path = args.out_dir / f"fig1_scatter_kl_vs_percloss_{tag}_{today}.pdf"
    fig2_path = args.out_dir / f"fig2_regime_by_popsize_{tag}_{today}.pdf"
    fig3_path = args.out_dir / f"fig3_scatter_kl_vs_entropy_{tag}_{today}.pdf"
    fig4_path = args.out_dir / f"fig4_scatter_perc_vs_entropy_{tag}_{today}.pdf"
    fig5_path = args.out_dir / f"fig5_metric_grid_by_popsize_{tag}_{today}.pdf"
    fig1_scatter(df, fig1_path)
    fig2_stacked_bars(df, fig2_path)
    fig3_kl_vs_entropy(df, fig3_path)
    fig4_perc_vs_entropy(df, fig4_path)
    fig5_metric_grid(df, fig5_path)


if __name__ == "__main__":
    main()
