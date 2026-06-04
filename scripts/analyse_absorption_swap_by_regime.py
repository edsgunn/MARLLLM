"""Regime-stratified analysis of the swapped-character PMI control.

Loads the swap-absorption JSONL shards from one or more directories, attaches
contemporaneous + end-of-training regime labels using the same machinery as
`analyse_absorption_by_regime.py`, then produces:

  - A regime-stratified table with own-vs-none and swap-vs-none endpoints
    plus differential shrinkage (own - swap).
  - Two-panel Figure-2 analogue (own-vs-none | swap-vs-none) by
    contemporaneous regime.
  - Per-agent scatter: end-of-training own-vs-none shrinkage vs swap-vs-none
    shrinkage, coloured by end regime.

Usage
-----
  uv run python scripts/analyse_absorption_swap_by_regime.py \\
      --out runs/cultural_emergence/_absorption_swap_by_regime \\
      runs/cultural_emergence/run7_8agent_7B_margin_notes/absorption_swap/JOBID \\
      runs/cultural_emergence/run7_8agent_7B_study_group/absorption_swap/JOBID \\
      ...
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from analyse_absorption_by_regime import (  # noqa: E402
    REGIMES, REGIME_COLORS, REGIME_LABELS,
    classify_all_iters, end_of_training_regime,
)

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42,
})

_ENV_RE = re.compile(r"run\d+_\d+agent_\d+B_(.+?)(?:_seed\d+)?$")
_SEED_RE = re.compile(r"_seed(\d+)$")


def env_from_run(run_dir: Path) -> str:
    m = _ENV_RE.match(run_dir.name)
    return m.group(1) if m else run_dir.name


def seed_from_run(run_dir: Path) -> int:
    m = _SEED_RE.search(run_dir.name)
    return int(m.group(1)) if m else 1


def load_swap_dirs(dirs: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    for d in dirs:
        for f in sorted(glob.glob(str(Path(d) / "absorption_swap_rank*.jsonl"))):
            with open(f) as fh:
                rows.extend(json.loads(line) for line in fh if line.strip())
    if not rows:
        raise SystemExit("no swap data loaded")
    df = pd.DataFrame(rows)
    df["env"] = df["run_dir"].apply(lambda r: env_from_run(Path(r)))
    df["seed"] = df["run_dir"].apply(lambda r: seed_from_run(Path(r)))
    return df


def attach_regime(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Merge end-of-training and contemporaneous regime labels."""
    run_dirs = df["run_dir"].unique()
    end_frames, traj_frames = [], []
    for rd in run_dirs:
        end = end_of_training_regime(Path(rd))
        if end.empty:
            print(f"[warn] no metrics for {Path(rd).name}")
            continue
        end_frames.append(end)
        traj_frames.append(classify_all_iters(Path(rd)))
    end_table = pd.concat(end_frames, ignore_index=True)
    traj = pd.concat(traj_frames, ignore_index=True)

    end_key = end_table[["run", "agent", "regime", "final_iter",
                         "kl", "perc_loss", "entropy"]].rename(
        columns={"run": "run_dir", "agent": "character"})
    df = df.merge(end_key, on=["run_dir", "character"], how="left")

    contemp = traj[["run", "iter", "agent", "regime",
                    "kl", "perc_loss", "entropy"]].rename(columns={
        "run": "run_dir", "agent": "character",
        "regime": "contemp_regime",
        "kl": "kl_at_iter", "perc_loss": "perc_loss_at_iter",
        "entropy": "entropy_at_iter"})
    df = df.merge(contemp, on=["run_dir", "iter", "character"], how="left")

    n_missing = df["regime"].isna().sum()
    if n_missing:
        print(f"[warn] dropping {n_missing} rows w/o regime")
        df = df.dropna(subset=["regime"])
    return df, traj


# ------------------------ trajectory ------------------------

def traj_by_regime(df: pd.DataFrame, gap_col: str, regime_col: str) -> pd.DataFrame:
    return (df.groupby([regime_col, "iter"])
              .agg(mean=(gap_col, "mean"),
                   se=(gap_col, lambda s: s.std(ddof=1) / np.sqrt(len(s))
                       if len(s) > 1 else np.nan),
                   n=(gap_col, "size"),
                   n_agents=("character", "nunique"))
              .reset_index())


def plot_two_panel(df: pd.DataFrame, out_path: Path, regime_col: str, title: str):
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.3),
                             sharey=True, constrained_layout=True)
    cfg = [("gap",      "PMI: own − none (nats/token)",   axes[0]),
           ("gap_swap", "PMI: swap − none (nats/token)", axes[1])]
    for gap_col, ylabel, ax in cfg:
        tr = traj_by_regime(df, gap_col, regime_col)
        for regime in REGIMES:
            r = tr[tr[regime_col] == regime]
            if r.empty:
                continue
            n_agents = df[df[regime_col] == regime]["character"].nunique()
            ax.errorbar(r["iter"], r["mean"], yerr=r["se"].fillna(0),
                        marker="o", ms=3.5, lw=1.2, capsize=2,
                        color=REGIME_COLORS[regime],
                        label=f"{REGIME_LABELS[regime]} (n={n_agents})")
        ax.axhline(0, color="0.7", lw=0.5, ls=":")
        ax.set_xlabel("training iteration")
        ax.set_ylabel(ylabel)
        ax.yaxis.grid(True, color="0.93", lw=0.5); ax.set_axisbelow(True)
    axes[0].legend(loc="upper right", frameon=False, fontsize=7)
    fig.suptitle(title, fontsize=10)
    fig.savefig(out_path, format="pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(out_path.with_suffix(".png"), dpi=220, bbox_inches="tight",
                pad_inches=0.04)
    plt.close(fig)
    print(f"[ok] {out_path.name}")


def per_agent_endpoints(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (run, ch), g in df.groupby(["run_dir", "character"]):
        iters = sorted(g["iter"].unique())
        if len(iters) < 2:
            continue
        first, last = iters[0], iters[-1]
        gf = g[g["iter"] == first]
        gl = g[g["iter"] == last]
        rows.append({
            "run_dir": run, "character": ch,
            "env": g["env"].iloc[0], "seed": g["seed"].iloc[0],
            "regime": g["regime"].iloc[0],
            "end_kl": g["kl"].iloc[0],
            "end_perc": g["perc_loss"].iloc[0],
            "end_ent": g["entropy"].iloc[0],
            "first_gap_own": gf["gap"].mean(),
            "last_gap_own":  gl["gap"].mean(),
            "first_gap_swap": gf["gap_swap"].mean(),
            "last_gap_swap":  gl["gap_swap"].mean(),
            "shrink_own":   gf["gap"].mean() - gl["gap"].mean(),
            "shrink_swap":  gf["gap_swap"].mean() - gl["gap_swap"].mean(),
        })
    out = pd.DataFrame(rows)
    out["shrink_diff"] = out["shrink_own"] - out["shrink_swap"]
    return out


def plot_scatter(end_df: pd.DataFrame, out_path: Path, title: str):
    fig, ax = plt.subplots(figsize=(4.4, 4.0), constrained_layout=True)
    for regime in REGIMES:
        sub = end_df[end_df["regime"] == regime]
        if sub.empty:
            continue
        ax.scatter(sub["shrink_own"], sub["shrink_swap"], s=32,
                   facecolor=REGIME_COLORS[regime], edgecolor="white",
                   linewidth=0.6, alpha=0.9, label=REGIME_LABELS[regime])
    lims = [
        min(end_df["shrink_own"].min(), end_df["shrink_swap"].min()) - 0.05,
        max(end_df["shrink_own"].max(), end_df["shrink_swap"].max()) + 0.05,
    ]
    ax.plot(lims, lims, ls="--", color="0.6", lw=0.7,
            label="y = x (generic drift)")
    ax.axhline(0, color="0.85", lw=0.5)
    ax.axvline(0, color="0.85", lw=0.5)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("own-vs-none shrinkage (nats/token)")
    ax.set_ylabel("swap-vs-none shrinkage (nats/token)")
    ax.legend(loc="upper left", frameon=False, fontsize=7)
    ax.set_title(title, fontsize=10)
    ax.yaxis.grid(True, color="0.93", lw=0.5); ax.set_axisbelow(True)
    fig.savefig(out_path, format="pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(out_path.with_suffix(".png"), dpi=220, bbox_inches="tight",
                pad_inches=0.04)
    plt.close(fig)
    print(f"[ok] {out_path.name}")


def endpoint_table_by_regime(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    rows = []
    for keys, g in df.groupby(by):
        kd = dict(zip(by, keys)) if isinstance(keys, tuple) else {by[0]: keys}
        iters = sorted(g["iter"].unique())
        if len(iters) < 2:
            continue
        first, last = iters[0], iters[-1]
        gf = g[g["iter"] == first]
        gl = g[g["iter"] == last]
        own_shrink  = gf["gap"].mean()       - gl["gap"].mean()
        swap_shrink = gf["gap_swap"].mean()  - gl["gap_swap"].mean()
        rows.append({
            **kd,
            "first_iter": first, "last_iter": last,
            "first_gap_own":  gf["gap"].mean(),
            "last_gap_own":   gl["gap"].mean(),
            "first_gap_swap": gf["gap_swap"].mean(),
            "last_gap_swap":  gl["gap_swap"].mean(),
            "shrink_own":  own_shrink,
            "shrink_swap": swap_shrink,
            "differential_shrinkage": own_shrink - swap_shrink,
            "n_agents":   g["character"].nunique(),
            "n_instances": g[["run_dir", "character"]].drop_duplicates().shape[0],
            "n_episodes_first": len(gf),
            "n_episodes_last":  len(gl),
        })
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("dirs", nargs="+",
                   help="Swap absorption output dirs (each has absorption_swap_rank*.jsonl).")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    df = load_swap_dirs(args.dirs)
    print(f"loaded {len(df)} swap rows from {len(args.dirs)} dirs")
    df, traj = attach_regime(df)
    df.to_csv(out_dir / "absorption_swap_with_regime.csv", index=False)
    print(f"wrote {out_dir/'absorption_swap_with_regime.csv'}  ({len(df)} rows)")

    # Two-panel by contemporaneous regime (cross + seeds combined view).
    plot_two_panel(df, out_dir / "fig_traj_own_vs_swap_contemp.pdf",
                   regime_col="contemp_regime",
                   title="PMI absorption by contemporaneous regime — own vs swapped character")

    # Two-panel by end-of-training regime.
    plot_two_panel(df, out_dir / "fig_traj_own_vs_swap_endregime.pdf",
                   regime_col="regime",
                   title="PMI absorption by end-of-training regime — own vs swapped character")

    # Per-agent scatter.
    end_df = per_agent_endpoints(df)
    end_df.to_csv(out_dir / "endpoint_shrinkage_own_vs_swap.csv", index=False)
    plot_scatter(end_df, out_dir / "fig_scatter_own_vs_swap.pdf",
                 "End-of-training shrinkage: own vs swapped character\n"
                 "(off-diagonal = character-specific absorption; on-diagonal = generic drift)")

    # Regime-stratified endpoint tables.
    for by, name in [(["regime"], "by_endregime"),
                     (["contemp_regime"], "by_contempregime"),
                     (["env", "regime"], "by_env_endregime")]:
        tbl = endpoint_table_by_regime(df, by)
        tbl.to_csv(out_dir / f"endpoint_table_{name}.csv", index=False)
        print(f"\nEndpoint table {name}:")
        print(tbl.sort_values(by).to_string(index=False))


if __name__ == "__main__":
    main()
