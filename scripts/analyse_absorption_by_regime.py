"""Regime-stratified re-analysis of the per-token PMI absorption data.

Loads the existing absorption CSVs (cross-environment + 5-seed inner-loop),
classifies each (run, character) pair into a regime using the same thresholds
as `plot_population_regime_figures.py`, then computes regime-stratified
trajectories, shrinkage, and a Robotic-Athanor zoom.

Outputs land in `runs/cultural_emergence/_absorption_by_regime/`.
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from plot_population_regime_figures import (
    classify, ENT_LOW, ENT_HIGH, KL_THR, PERC_COHERENT, PERC_DEGEN,
    extract_agents_from_record,
)

RUNS = REPO / "runs" / "cultural_emergence"
OUT = RUNS / "_absorption_by_regime"
ABSORB_CSV = RUNS / "_absorption_summary" / "absorption_all.csv"
SEEDS_CSV = RUNS / "_seed_aggregate_inner_loop" / "absorption_all_seeds.csv"

# Five distinct categories now: split the legacy `degenerate` into
# perception-driven (text gibberish) vs entropy-driven (policy collapse with
# coherent surface text), per the trace verification report.
def classify_split(kl, perc, ent):
    if perc >= PERC_DEGEN:
        return "text_degenerate"
    if ent < ENT_LOW or ent > ENT_HIGH:
        return "policy_collapse"
    if perc < PERC_COHERENT and kl < KL_THR:
        return "coherent_low_kl"
    if perc < PERC_COHERENT and kl >= KL_THR:
        return "committed"
    return "mid_range"


REGIMES = ["coherent_low_kl", "committed", "mid_range",
           "policy_collapse", "text_degenerate"]
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


# ------------------------ regime classification ------------------------

def iter_records(metrics_path: Path):
    with metrics_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "iteration" in rec:
                yield rec


def classify_all_iters(run_dir: Path) -> pd.DataFrame:
    """Return a DataFrame with per-(iter, agent) metrics + regime, for every
    logged iteration of the run."""
    rows = []
    metrics = run_dir / "metrics.jsonl"
    if not metrics.exists():
        return pd.DataFrame(columns=["run", "iter", "agent", "kl", "perc_loss",
                                     "entropy", "regime"])
    for rec in iter_records(metrics):
        it = int(rec["iteration"])
        for agent, m in extract_agents_from_record(rec).items():
            kl, perc, ent = m.get("kl"), m.get("perc_loss"), m.get("entropy")
            if kl is None or perc is None or ent is None:
                continue
            rows.append(dict(run=str(run_dir), iter=it, agent=agent,
                             kl=kl, perc_loss=perc, entropy=ent,
                             regime=classify_split(kl, perc, ent)))
    return pd.DataFrame(rows)


def end_of_training_regime(run_dir: Path) -> pd.DataFrame:
    df = classify_all_iters(run_dir)
    if df.empty:
        return df
    last_it = df["iter"].max()
    end = df[df["iter"] == last_it].copy()
    end["final_iter"] = last_it
    return end


# ------------------------ data assembly ------------------------

def load_absorption_with_regime() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (cross_env_df, seeds_df, regime_table). regime_table holds the
    end-of-training regime per (run, character)."""
    cross = pd.read_csv(ABSORB_CSV)
    seeds = pd.read_csv(SEEDS_CSV)

    # Build regime table over union of run dirs in both CSVs.
    run_dirs = pd.concat([cross["run_dir"], seeds["run_dir"]]).unique()
    regime_rows = []
    full_traj_frames = []
    for rd in run_dirs:
        run_dir = Path(rd)
        end = end_of_training_regime(run_dir)
        if end.empty:
            print(f"[warn] no metrics for {run_dir.name}")
            continue
        regime_rows.append(end)
        full_traj_frames.append(classify_all_iters(run_dir))
    regime_table = pd.concat(regime_rows, ignore_index=True)
    full_traj = pd.concat(full_traj_frames, ignore_index=True)

    # Merge on (run_dir, character) -> regime.
    key = regime_table[["run", "agent", "regime", "final_iter",
                        "kl", "perc_loss", "entropy"]].rename(
        columns={"run": "run_dir", "agent": "character"})
    cross = cross.merge(key, on=["run_dir", "character"], how="left")
    seeds = seeds.merge(key, on=["run_dir", "character"], how="left")

    # Also attach contemporaneous regime + the underlying training metrics
    # (KL, perception loss, entropy) at the same iter the PMI was measured.
    contemp = full_traj[["run", "iter", "agent", "regime", "kl", "perc_loss",
                         "entropy"]].rename(columns={
        "run": "run_dir", "agent": "character",
        "regime": "contemp_regime",
        "kl": "kl_at_iter", "perc_loss": "perc_loss_at_iter",
        "entropy": "entropy_at_iter",
    })
    cross = cross.merge(contemp, on=["run_dir", "iter", "character"], how="left")
    seeds = seeds.merge(contemp, on=["run_dir", "iter", "character"], how="left")

    # Surface unmatched
    n_unmatched_cross = cross["regime"].isna().sum()
    n_unmatched_seeds = seeds["regime"].isna().sum()
    if n_unmatched_cross or n_unmatched_seeds:
        print(f"[warn] {n_unmatched_cross} cross-env / {n_unmatched_seeds} seed "
              "rows had no matching metrics; dropping them")
        cross = cross.dropna(subset=["regime"])
        seeds = seeds.dropna(subset=["regime"])
    return cross, seeds, regime_table, full_traj


# ------------------------ trajectories ------------------------

def trajectory_table(df: pd.DataFrame, group_keys: list[str]) -> pd.DataFrame:
    """Mean ± SE of gap per (group_keys, iter). Aggregates over (character,
    episode) — i.e., per-agent variability is folded in."""
    agg = (df.groupby(group_keys + ["iter"])
             .agg(gap_mean=("gap", "mean"),
                  gap_se=("gap", lambda s: s.std(ddof=1) / np.sqrt(len(s))
                          if len(s) > 1 else np.nan),
                  n=("gap", "size"),
                  n_agents=("character", "nunique"))
             .reset_index())
    return agg


def regime_endpoint_stats(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """First-iter and last-iter mean gap per group, plus shrinkage."""
    rows = []
    for keys, g in df.groupby(by):
        if isinstance(keys, tuple):
            kd = dict(zip(by, keys))
        else:
            kd = {by[0]: keys}
        iters = sorted(g["iter"].unique())
        if len(iters) < 2:
            continue
        first, last = iters[0], iters[-1]
        gf = g[g["iter"] == first]["gap"]
        gl = g[g["iter"] == last]["gap"]
        rows.append({
            **kd,
            "first_iter": first, "last_iter": last,
            "first_gap_mean": gf.mean(),
            "last_gap_mean": gl.mean(),
            "shrinkage": gf.mean() - gl.mean(),
            "shrinkage_frac": (gf.mean() - gl.mean()) / gf.mean()
                              if gf.mean() else np.nan,
            # n_agents counts unique character *names*; in pooled groupings
            # the same name can come from multiple training runs (different
            # seeds, or same character in different substrates), so it
            # under-counts. n_instances counts unique (run_dir, character)
            # pairs and reflects the number of independent trained agents.
            "n_agents": g["character"].nunique(),
            "n_instances": g[["run_dir", "character"]].drop_duplicates().shape[0],
            "n_episodes_first": len(gf),
            "n_episodes_last": len(gl),
        })
    return pd.DataFrame(rows)


# ------------------------ plotting ------------------------

def plot_trajectories_by_regime(df: pd.DataFrame, out_path: Path,
                                title: str, facet: str | None = None):
    """Mean gap trajectory per regime. If facet provided (e.g. 'env'),
    one panel per facet value."""
    if facet is None:
        fig, ax = plt.subplots(figsize=(5.2, 3.2), constrained_layout=True)
        axes = [ax]
        facet_values = [None]
    else:
        facet_values = sorted(df[facet].unique())
        ncols = min(4, len(facet_values))
        nrows = (len(facet_values) + ncols - 1) // ncols
        fig, axes_arr = plt.subplots(nrows, ncols,
                                     figsize=(min(2.0 * ncols, 7.5),
                                              max(2.4 * nrows, 2.6)),
                                     sharey=True, constrained_layout=True)
        axes = np.atleast_1d(axes_arr).flatten()

    for ax, fv in zip(axes, facet_values):
        sub = df if fv is None else df[df[facet] == fv]
        for regime in REGIMES:
            r = sub[sub["regime"] == regime]
            if r.empty:
                continue
            traj = trajectory_table(r, [])
            n_agents = r["character"].nunique()
            ax.errorbar(traj["iter"], traj["gap_mean"],
                        yerr=traj["gap_se"].fillna(0),
                        marker="o", ms=3.5, lw=1.2, capsize=2,
                        color=REGIME_COLORS[regime],
                        label=f"{REGIME_LABELS[regime]} (n={n_agents})")
        ax.axhline(0, color="0.7", lw=0.5, ls=":")
        ax.set_xlabel("training iteration")
        ax.set_ylabel("PMI gap (nats / token)")
        if fv is not None:
            ax.set_title(str(fv), fontsize=9)
        ax.yaxis.grid(True, color="0.93", lw=0.5, zorder=0)
        ax.set_axisbelow(True)
    # Hide unused axes if facetted
    if facet is not None:
        for extra in axes[len(facet_values):]:
            extra.set_visible(False)

    # Single legend on first axis for non-facetted, else shared below.
    if facet is None:
        axes[0].legend(loc="upper right", frameon=False)
    else:
        # Combine handles across panels.
        seen = set()
        handles, labels = [], []
        for ax in axes[:len(facet_values)]:
            for h, l in zip(*ax.get_legend_handles_labels()):
                # Strip per-panel n
                base = l.split(" (")[0]
                if base in seen:
                    continue
                seen.add(base)
                handles.append(h)
                labels.append(base)
        fig.legend(handles, labels,
                   loc="lower center", bbox_to_anchor=(0.5, -0.04),
                   ncol=len(labels), frameon=False)

    fig.suptitle(title, fontsize=10)
    fig.savefig(out_path, format="pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(out_path.with_suffix(".png"), format="png", dpi=220,
                bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"[ok] {out_path.name}")


def plot_shrinkage_bars(stats: pd.DataFrame, out_path: Path,
                        title: str, by: list[str]):
    """Bar chart of shrinkage per (substrate, regime). by = ['env', 'regime'] etc."""
    envs = sorted(stats[by[0]].unique())
    fig, ax = plt.subplots(figsize=(min(7.5, 1.0 * len(envs) + 2.5), 3.2),
                           constrained_layout=True)
    width = 0.16
    x = np.arange(len(envs))
    for i, regime in enumerate(REGIMES):
        ys, ns = [], []
        for env in envs:
            row = stats[(stats[by[0]] == env) & (stats["regime"] == regime)]
            ys.append(row["shrinkage"].iloc[0] if not row.empty else np.nan)
            ns.append(int(row["n_agents"].iloc[0]) if not row.empty else 0)
        bars = ax.bar(x + (i - 2) * width, ys, width,
                      color=REGIME_COLORS[regime], edgecolor="white",
                      linewidth=0.4, label=REGIME_LABELS[regime])
        for xi, y, n in zip(x + (i - 2) * width, ys, ns):
            if n > 0 and not np.isnan(y):
                ax.text(xi, y + (0.002 if y >= 0 else -0.006), f"n{n}",
                        ha="center",
                        va="bottom" if y >= 0 else "top", fontsize=6,
                        color="0.25")
    ax.axhline(0, color="0.4", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(envs, rotation=20, ha="right")
    ax.set_ylabel("shrinkage (first–last gap, nats/token)")
    ax.yaxis.grid(True, color="0.93", lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    ax.set_title(title, fontsize=10)
    fig.savefig(out_path, format="pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(out_path.with_suffix(".png"), format="png", dpi=220,
                bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"[ok] {out_path.name}")


def plot_robotic_athanor_zoom(df: pd.DataFrame, full_traj: pd.DataFrame,
                              out_path: Path):
    sub = df[df["env"] == "robotic_athanor"].copy()
    if sub.empty:
        return
    chars = sorted(sub["character"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2),
                             constrained_layout=True, sharey=False)
    ax_l, ax_r = axes

    # LEFT: per-agent absorption gap, coloured by regime.
    for ch in chars:
        cd = sub[sub["character"] == ch].sort_values("iter")
        regime = cd["regime"].iloc[0]
        per_iter = cd.groupby("iter")["gap"].mean().reset_index()
        ax_l.plot(per_iter["iter"], per_iter["gap"], "-o", ms=3.5, lw=1.0,
                  color=REGIME_COLORS[regime], alpha=0.85,
                  label=f"{ch} ({REGIME_LABELS[regime]})")
    ax_l.axhline(0, color="0.7", lw=0.5, ls=":")
    ax_l.set_xlabel("training iteration")
    ax_l.set_ylabel("PMI gap (nats / token)")
    ax_l.set_title("Per-agent PMI absorption", fontsize=9)
    ax_l.yaxis.grid(True, color="0.93", lw=0.5, zorder=0); ax_l.set_axisbelow(True)

    # RIGHT: regime trajectory of the robotic_athanor agents — show
    # entropy vs perc_loss across iters per agent.
    ra_run = sub["run_dir"].iloc[0]
    traj = full_traj[full_traj["run"] == ra_run]
    for ch in chars:
        ad = traj[traj["agent"] == ch].sort_values("iter")
        if ad.empty:
            continue
        regime_end = ad.iloc[-1]
        ax_r.plot(ad["entropy"], ad["perc_loss"], "-", lw=0.8,
                  color=REGIME_COLORS[regime_end["regime"]], alpha=0.5)
        ax_r.scatter(ad["entropy"].iloc[-1], ad["perc_loss"].iloc[-1],
                     s=30, color=REGIME_COLORS[regime_end["regime"]],
                     edgecolor="white", linewidth=0.6, zorder=3)
    ax_r.axvline(ENT_LOW, color="0.4", lw=0.5, ls=(0, (1.5, 1.5)))
    ax_r.axvline(ENT_HIGH, color="0.4", lw=0.5, ls=(0, (1.5, 1.5)))
    ax_r.axhline(PERC_DEGEN, color="0.4", lw=0.5, ls=(0, (1.5, 1.5)))
    ax_r.set_xlabel("policy entropy")
    ax_r.set_ylabel("perception loss")
    ax_r.set_title("Regime trajectory in (entropy, perc.) space", fontsize=9)
    ax_r.yaxis.grid(True, color="0.93", lw=0.5, zorder=0); ax_r.set_axisbelow(True)

    fig.suptitle("Robotic Athanor: regime context for the absorption profile",
                 fontsize=10)
    fig.savefig(out_path, format="pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(out_path.with_suffix(".png"), format="png", dpi=220,
                bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"[ok] {out_path.name}")


def plot_seed_trajectories(df: pd.DataFrame, out_path: Path):
    """Inner-loop seed runs: PMI gap by regime, also faceted by seed."""
    seeds = sorted(df["seed"].unique())
    ncols = len(seeds)
    fig, axes = plt.subplots(1, ncols, figsize=(min(2.0 * ncols, 9), 2.7),
                             sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, s in zip(axes, seeds):
        sub = df[df["seed"] == s]
        for regime in REGIMES:
            r = sub[sub["regime"] == regime]
            if r.empty:
                continue
            traj = trajectory_table(r, [])
            n_agents = r["character"].nunique()
            ax.errorbar(traj["iter"], traj["gap_mean"],
                        yerr=traj["gap_se"].fillna(0),
                        marker="o", ms=3, lw=1.0, capsize=2,
                        color=REGIME_COLORS[regime],
                        label=f"{REGIME_LABELS[regime]} (n={n_agents})")
        ax.axhline(0, color="0.7", lw=0.5, ls=":")
        ax.set_title(f"seed {s}", fontsize=9)
        ax.set_xlabel("iter")
        ax.yaxis.grid(True, color="0.93", lw=0.5); ax.set_axisbelow(True)
    axes[0].set_ylabel("PMI gap")
    # Combined legend
    seen, handles, labels = set(), [], []
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            base = l.split(" (")[0]
            if base in seen: continue
            seen.add(base); handles.append(h); labels.append(base)
    fig.legend(handles, labels, loc="lower center",
               bbox_to_anchor=(0.5, -0.06), ncol=len(labels), frameon=False)
    fig.suptitle("Inner Loop seeds: PMI absorption by end-of-training regime",
                 fontsize=10)
    fig.savefig(out_path, format="pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(out_path.with_suffix(".png"), format="png", dpi=220,
                bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"[ok] {out_path.name}")


# ------------------------ metric / gap co-evolution ------------------------

def plot_metric_trajectories(df: pd.DataFrame, out_path: Path,
                             title: str, regime_col: str = "regime"):
    """Three stacked panels: PMI gap, KL, entropy — each as mean across
    agents within a regime, vs training iteration. Shows what the model is
    doing while the gap shrinks."""
    fig, axes = plt.subplots(3, 1, figsize=(5.4, 6.4),
                             sharex=True, constrained_layout=True)
    metric_cols = [
        ("gap", "PMI gap (nats / token)"),
        ("kl_at_iter", "KL from reference (nats)"),
        ("entropy_at_iter", "policy entropy (nats)"),
    ]
    for ax, (col, ylabel) in zip(axes, metric_cols):
        for regime in REGIMES:
            r = df[df[regime_col] == regime]
            if r.empty or col not in r.columns:
                continue
            agg = (r.groupby("iter")
                    .agg(mean=(col, "mean"),
                         se=(col, lambda s: s.std(ddof=1)/np.sqrt(len(s))
                             if len(s) > 1 else np.nan),
                         n_obs=(col, "size"))
                    .reset_index())
            n_agents = r["character"].nunique()
            ax.errorbar(agg["iter"], agg["mean"], yerr=agg["se"].fillna(0),
                        marker="o", ms=3.5, lw=1.2, capsize=2,
                        color=REGIME_COLORS[regime],
                        label=f"{REGIME_LABELS[regime]} (n={n_agents})")
        ax.set_ylabel(ylabel)
        ax.yaxis.grid(True, color="0.93", lw=0.5); ax.set_axisbelow(True)
        if col == "gap":
            ax.axhline(0, color="0.7", lw=0.5, ls=":")
        if col == "entropy_at_iter":
            ax.axhline(ENT_LOW, color="0.6", lw=0.5, ls=(0, (1.5, 1.5)))
            ax.axhline(ENT_HIGH, color="0.6", lw=0.5, ls=(0, (1.5, 1.5)))
        if col == "kl_at_iter":
            ax.axhline(KL_THR, color="0.6", lw=0.5, ls=(0, (4, 2)))
    axes[-1].set_xlabel("training iteration")
    axes[0].legend(loc="upper right", frameon=False, fontsize=7)
    fig.suptitle(title, fontsize=10)
    fig.savefig(out_path, format="pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(out_path.with_suffix(".png"), format="png", dpi=220,
                bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"[ok] {out_path.name}")


def plot_endpoint_relationships(df: pd.DataFrame, out_path: Path, title: str):
    """Per-agent: shrinkage (first - last gap) vs end-of-training KL, perc,
    entropy. Three panels. Marker = end regime."""
    # Compute per-agent shrinkage from absorption data + read end-of-training
    # metrics from the regime columns (already merged from `regime_table`).
    rows = []
    for (run, character), g in df.groupby(["run_dir", "character"]):
        iters = sorted(g["iter"].unique())
        if len(iters) < 2:
            continue
        first, last = iters[0], iters[-1]
        gf = g[g["iter"] == first]["gap"].mean()
        gl = g[g["iter"] == last]["gap"].mean()
        end_kl = g["kl"].iloc[0]
        end_perc = g["perc_loss"].iloc[0]
        end_ent = g["entropy"].iloc[0]
        regime = g["regime"].iloc[0]
        rows.append(dict(run_dir=run, character=character,
                         shrinkage=gf - gl, first_gap=gf, last_gap=gl,
                         end_kl=end_kl, end_perc=end_perc, end_ent=end_ent,
                         regime=regime))
    end_df = pd.DataFrame(rows)
    end_df.to_csv(out_path.with_suffix(".csv"), index=False)

    fig, axes = plt.subplots(1, 3, figsize=(8.2, 3.0),
                             constrained_layout=True)
    panels = [("end_kl", "KL from reference (end of training)", True),
              ("end_perc", "perception loss (end of training)", False),
              ("end_ent", "policy entropy (end of training)", False)]
    for ax, (col, xlabel, log_x) in zip(axes, panels):
        for regime in REGIMES:
            sub = end_df[end_df["regime"] == regime]
            if sub.empty:
                continue
            ax.scatter(sub[col], sub["shrinkage"], s=32,
                       facecolor=REGIME_COLORS[regime], edgecolor="white",
                       linewidth=0.6, alpha=0.9, label=REGIME_LABELS[regime])
        ax.axhline(0, color="0.7", lw=0.5, ls=":")
        if col == "end_kl":
            ax.axvline(KL_THR, color="0.6", lw=0.5, ls=(0, (4, 2)))
            ax.set_xscale("symlog", linthresh=0.5)
        elif col == "end_perc":
            ax.axvline(PERC_COHERENT, color="0.6", lw=0.5, ls=(0, (4, 2)))
            ax.axvline(PERC_DEGEN, color="0.6", lw=0.5, ls=(0, (1.5, 1.5)))
        elif col == "end_ent":
            ax.axvline(ENT_LOW, color="0.6", lw=0.5, ls=(0, (1.5, 1.5)))
            ax.axvline(ENT_HIGH, color="0.6", lw=0.5, ls=(0, (1.5, 1.5)))
        ax.set_xlabel(xlabel)
        ax.yaxis.grid(True, color="0.93", lw=0.5); ax.set_axisbelow(True)
    axes[0].set_ylabel("PMI gap shrinkage (first $-$ last)")
    # Single legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               bbox_to_anchor=(0.5, -0.06),
               ncol=len(labels), frameon=False)
    fig.suptitle(title, fontsize=10)
    fig.savefig(out_path, format="pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(out_path.with_suffix(".png"), format="png", dpi=220,
                bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"[ok] {out_path.name}")


def plot_phase_trajectories(df: pd.DataFrame, out_path: Path, title: str):
    """Per-agent trajectories in (KL, gap) and (entropy, gap) space.
    Each line connects an agent's iters in order; coloured by end regime."""
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.6),
                             constrained_layout=True)
    pairs = [("kl_at_iter", "KL from reference (at iter)", "log"),
             ("entropy_at_iter", "policy entropy (at iter)", "linear")]
    for ax, (xcol, xlabel, xscale) in zip(axes, pairs):
        for (run, char), g in df.groupby(["run_dir", "character"]):
            g = g.sort_values("iter")
            mean_per_iter = g.groupby("iter").agg(
                x=(xcol, "first"), y=("gap", "mean")).reset_index()
            regime = g["regime"].iloc[0]
            ax.plot(mean_per_iter["x"], mean_per_iter["y"], "-",
                    lw=0.6, alpha=0.4, color=REGIME_COLORS[regime])
            # Mark start and end
            ax.scatter(mean_per_iter["x"].iloc[0], mean_per_iter["y"].iloc[0],
                       s=10, marker="o", facecolor="white",
                       edgecolor=REGIME_COLORS[regime], linewidth=0.5,
                       alpha=0.7, zorder=3)
            ax.scatter(mean_per_iter["x"].iloc[-1], mean_per_iter["y"].iloc[-1],
                       s=22, marker="o", facecolor=REGIME_COLORS[regime],
                       edgecolor="white", linewidth=0.5, zorder=4)
        ax.axhline(0, color="0.7", lw=0.5, ls=":")
        if xcol == "kl_at_iter":
            ax.axvline(KL_THR, color="0.6", lw=0.5, ls=(0, (4, 2)))
            ax.set_xscale("symlog", linthresh=0.1)
        if xcol == "entropy_at_iter":
            ax.axvline(ENT_LOW, color="0.6", lw=0.5, ls=(0, (1.5, 1.5)))
            ax.axvline(ENT_HIGH, color="0.6", lw=0.5, ls=(0, (1.5, 1.5)))
        ax.set_xlabel(xlabel)
        ax.set_ylabel("PMI gap (nats / token)")
        ax.set_ylim(-0.4, 0.8)  # clip rare extreme outliers for legibility
        ax.yaxis.grid(True, color="0.93", lw=0.5); ax.set_axisbelow(True)
    # Build legend
    legend_handles = [
        plt.Line2D([], [], marker="o", color="w", markerfacecolor=c,
                   markeredgecolor="white", markersize=6,
                   label=REGIME_LABELS[r])
        for r, c in REGIME_COLORS.items()
        if r in df["regime"].unique()
    ]
    legend_handles.append(
        plt.Line2D([], [], marker="o", color="w", markerfacecolor="white",
                   markeredgecolor="0.4", markersize=6, label="iter start")
    )
    legend_handles.append(
        plt.Line2D([], [], marker="o", color="w", markerfacecolor="0.4",
                   markeredgecolor="white", markersize=6, label="iter end")
    )
    fig.legend(handles=legend_handles, loc="lower center",
               bbox_to_anchor=(0.5, -0.08),
               ncol=len(legend_handles), frameon=False, fontsize=7)
    fig.suptitle(title, fontsize=10)
    fig.savefig(out_path, format="pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(out_path.with_suffix(".png"), format="png", dpi=220,
                bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"[ok] {out_path.name}")


# ------------------------ regime trajectory consistency ------------------------

def regime_consistency(absorption_iters: list[int],
                       full_traj: pd.DataFrame,
                       end_regime: pd.DataFrame) -> pd.DataFrame:
    """For each (run, agent), compare regime at each absorption iter to
    end-of-training regime."""
    rows = []
    for (run, agent), end in end_regime.set_index(
            ["run", "agent"])["regime"].items():
        ad = full_traj[(full_traj["run"] == run) & (full_traj["agent"] == agent)]
        regimes_seen = set()
        for it in absorption_iters:
            r = ad[ad["iter"] == it]
            if r.empty:
                continue
            regimes_seen.add(r["regime"].iloc[0])
        rows.append({
            "run": Path(run).name, "agent": agent, "end_regime": end,
            "regimes_during_absorption": sorted(regimes_seen),
            "stable": (regimes_seen == {end}) and len(regimes_seen) > 0,
        })
    return pd.DataFrame(rows)


# ------------------------ main ------------------------

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cross, seeds, regime_table, full_traj = load_absorption_with_regime()

    # Save merged regime table
    regime_table.to_csv(OUT / "regime_table_eot.csv", index=False)

    # Save merged absorption with regime label
    cross.to_csv(OUT / "absorption_cross_with_regime.csv", index=False)
    seeds.to_csv(OUT / "absorption_seeds_with_regime.csv", index=False)

    # ---- counts ----
    counts_cross = (cross.drop_duplicates(["run_dir", "character"])
                         .groupby(["env", "regime"]).size().unstack(fill_value=0))
    counts_seeds = (seeds.drop_duplicates(["run_dir", "character"])
                         .groupby(["seed", "regime"]).size().unstack(fill_value=0))
    print("\nCross-environment regime counts (unique agents per env):")
    print(counts_cross)
    print("\nInner-loop seed regime counts (unique agents per seed):")
    print(counts_seeds)
    counts_cross.to_csv(OUT / "regime_counts_cross.csv")
    counts_seeds.to_csv(OUT / "regime_counts_seeds.csv")

    # ---- trajectories ----
    plot_trajectories_by_regime(cross, OUT / "fig_traj_cross_pooled.pdf",
                                "PMI absorption by regime — cross-env (pooled)")
    plot_trajectories_by_regime(cross, OUT / "fig_traj_cross_by_env.pdf",
                                "PMI absorption by regime, faceted by substrate",
                                facet="env")
    plot_seed_trajectories(seeds, OUT / "fig_traj_inner_loop_seeds.pdf")

    # ---- shrinkage stats ----
    cross_stats = regime_endpoint_stats(cross, ["env", "regime"])
    seeds_stats = regime_endpoint_stats(seeds, ["seed", "regime"])
    cross_stats.to_csv(OUT / "stats_shrinkage_cross.csv", index=False)
    seeds_stats.to_csv(OUT / "stats_shrinkage_seeds.csv", index=False)
    print("\nCross-env shrinkage by (substrate, regime):")
    print(cross_stats.sort_values(["env", "regime"]).to_string(index=False))
    print("\nInner-loop seeds shrinkage by (seed, regime):")
    print(seeds_stats.sort_values(["seed", "regime"]).to_string(index=False))

    plot_shrinkage_bars(cross_stats, OUT / "fig_shrinkage_cross.pdf",
                        "Absorption shrinkage by regime (cross-env)",
                        ["env", "regime"])

    # ---- robotic athanor zoom ----
    plot_robotic_athanor_zoom(cross, full_traj,
                              OUT / "fig_robotic_athanor_zoom.pdf")

    # ---- regime stability during absorption window ----
    cross_iters = sorted(cross["iter"].unique())
    cons = regime_consistency(cross_iters, full_traj,
                              regime_table.rename(columns={"agent": "agent"}))
    cons.to_csv(OUT / "regime_consistency_cross.csv", index=False)
    n_unstable = (~cons["stable"]).sum()
    print(f"\nCross-env: {n_unstable} / {len(cons)} agents passed through "
          f"multiple regimes during the absorption window.")

    # ---- pooled by regime (cross-env, all envs combined) ----
    pooled_stats = regime_endpoint_stats(cross, ["regime"])
    pooled_stats.to_csv(OUT / "stats_shrinkage_pooled.csv", index=False)
    print("\nPooled (cross-env, all substrates) shrinkage by regime:")
    print(pooled_stats.sort_values("regime").to_string(index=False))

    # ---- pooled inner-loop seeds by regime ----
    pooled_seeds = regime_endpoint_stats(seeds, ["regime"])
    pooled_seeds.to_csv(OUT / "stats_shrinkage_pooled_seeds.csv", index=False)
    print("\nPooled inner-loop (5 seeds) shrinkage by regime:")
    print(pooled_seeds.sort_values("regime").to_string(index=False))

    # ---- contemporaneous-regime stratification (gap binned by the
    # regime the agent was actually in at the PMI iter) ----
    contemp_pool = (cross.groupby(["contemp_regime", "iter"])
                         .agg(gap_mean=("gap", "mean"),
                              gap_se=("gap", lambda s: s.std(ddof=1)/np.sqrt(len(s))
                                      if len(s)>1 else np.nan),
                              n=("gap", "size")).reset_index())
    contemp_pool.to_csv(OUT / "contemp_regime_traj_cross.csv", index=False)
    print("\nContemporaneous regime PMI gaps (cross-env, per iter):")
    print(contemp_pool.sort_values(["contemp_regime", "iter"]).to_string(index=False))

    # Plot contemporaneous-regime trajectory
    fig, ax = plt.subplots(figsize=(5.4, 3.2), constrained_layout=True)
    for regime in REGIMES:
        r = contemp_pool[contemp_pool["contemp_regime"] == regime]
        if r.empty:
            continue
        n_obs_total = int(r["n"].sum())
        ax.errorbar(r["iter"], r["gap_mean"], yerr=r["gap_se"].fillna(0),
                    marker="o", ms=3.5, lw=1.2, capsize=2,
                    color=REGIME_COLORS[regime],
                    label=f"{REGIME_LABELS[regime]} ({n_obs_total} obs.)")
    ax.axhline(0, color="0.7", lw=0.5, ls=":")
    ax.set_xlabel("training iteration"); ax.set_ylabel("PMI gap (nats / token)")
    ax.set_title("PMI absorption by contemporaneous regime (cross-env, pooled)",
                 fontsize=10)
    ax.yaxis.grid(True, color="0.93", lw=0.5); ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False)
    fig.savefig(OUT / "fig_traj_contemp_regime.pdf", format="pdf",
                bbox_inches="tight", pad_inches=0.04)
    fig.savefig(OUT / "fig_traj_contemp_regime.png", format="png", dpi=220,
                bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print("[ok] fig_traj_contemp_regime")

    # ---- co-evolution figures ----
    plot_metric_trajectories(
        cross, OUT / "fig_metric_traj_cross.pdf",
        "PMI gap / KL / entropy co-evolution by end-of-training regime "
        "(cross-env)")
    plot_metric_trajectories(
        seeds, OUT / "fig_metric_traj_inner_loop_seeds.pdf",
        "PMI gap / KL / entropy co-evolution (inner-loop, 5 seeds pooled)")
    plot_metric_trajectories(
        cross, OUT / "fig_metric_traj_contemp_cross.pdf",
        "Same, by contemporaneous regime (cross-env)",
        regime_col="contemp_regime")

    plot_endpoint_relationships(
        cross, OUT / "fig_endpoint_shrinkage_vs_metrics_cross.pdf",
        "Per-agent PMI shrinkage vs end-of-training metrics (cross-env)")
    plot_endpoint_relationships(
        seeds, OUT / "fig_endpoint_shrinkage_vs_metrics_seeds.pdf",
        "Per-agent PMI shrinkage vs end-of-training metrics (inner-loop, "
        "5 seeds pooled)")

    plot_phase_trajectories(
        cross, OUT / "fig_phase_trajectories_cross.pdf",
        "Per-agent trajectories in (KL, gap) and (entropy, gap) space "
        "(cross-env)")


if __name__ == "__main__":
    main()
