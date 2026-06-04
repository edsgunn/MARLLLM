"""Aggregate and analyse the PMI decomposition outputs.

Reads per-rank JSONL shards produced by analyze_pmi_decomposition.py,
computes the seven decomposition gaps per row, merges contemporaneous /
end-of-training regime labels, and produces:

  - Headline table by end-of-training regime (Inner Loop) with all gaps
  - Cultural-learning trajectory: gap_history_specific_to_run vs iter,
    stratified by contemporaneous regime
  - Decomposition stacked-bar: char_specific, env_specific, history_specific
    components per regime
  - Per-agent scatter: gap_history_specific_to_run vs gap_char_specific
  - Cross-substrate env-swap sanity check: gap_env_specific by substrate

Usage
-----
  uv run python scripts/analyse_pmi_decomposition.py \\
      --out runs/cultural_emergence/_pmi_decomposition \\
      --inner-loop-dir runs/cultural_emergence/run7_8agent_7B_inner_loop/pmi_decomp/<jobid> \\
      --cross-env-dir  runs/cultural_emergence/run7_8agent_7B_margin_notes/pmi_decomp/<jobid>
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


def load_dirs(dirs: list[str]) -> pd.DataFrame:
    rows = []
    for d in dirs:
        for f in sorted(glob.glob(str(Path(d) / "pmi_decomp_rank*.jsonl"))):
            with open(f) as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["env"] = df["run_dir"].apply(lambda r: env_from_run(Path(r)))
    df["seed"] = df["run_dir"].apply(lambda r: seed_from_run(Path(r)))
    return df


def compute_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """Add gap columns. Signed so positive ⇒ the named slot contributes."""
    def col(name):
        c = f"nll_{name}"
        return df[c] if c in df.columns else pd.Series([np.nan]*len(df), index=df.index)
    df["gap_char_specific"] = col("swap_char_within") - col("own")
    df["gap_char_any"]      = col("none_char") - col("swap_char_within")
    df["gap_char_total"]    = col("none_char") - col("own")
    df["gap_env_specific"]  = col("swap_env") - col("own")
    df["gap_env_any"]       = col("none_env") - col("swap_env")
    df["gap_env_total"]     = col("none_env") - col("own")
    df["gap_history_specific_to_run"] = col("swap_history_cross_run") - col("own")
    df["gap_history_specific_to_episode"] = col("swap_history_within_run") - col("own")
    df["gap_history_any"]   = col("none_history") - col("swap_history_cross_run")
    df["gap_history_total"] = col("none_history") - col("own")
    return df


def attach_regime(df: pd.DataFrame) -> pd.DataFrame:
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
    n_miss = df["regime"].isna().sum()
    if n_miss:
        print(f"[warn] dropping {n_miss} rows w/o regime")
        df = df.dropna(subset=["regime"])
    return df


# ------------------------ headline tables ------------------------

GAP_COLS = [
    "gap_char_specific", "gap_char_any",
    "gap_env_specific", "gap_env_any",
    "gap_history_specific_to_run",
    "gap_history_specific_to_episode",
    "gap_history_any",
]


def endpoint_by_regime(df: pd.DataFrame, regime_col: str) -> pd.DataFrame:
    """Last-iter mean of each gap per regime."""
    rows = []
    for regime, g in df.groupby(regime_col):
        last_iter = g["iter"].max()
        gl = g[g["iter"] == last_iter]
        row = {"regime": regime, "last_iter": int(last_iter),
               "n_instances": gl[["run_dir", "character"]].drop_duplicates().shape[0],
               "n_episodes": len(gl)}
        for c in GAP_COLS:
            row[c] = gl[c].dropna().mean() if c in gl.columns else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def trajectory(df: pd.DataFrame, gap_col: str, regime_col: str) -> pd.DataFrame:
    return (df.dropna(subset=[gap_col])
              .groupby([regime_col, "iter"])
              .agg(mean=(gap_col, "mean"),
                   se=(gap_col, lambda s: s.std(ddof=1)/np.sqrt(len(s))
                       if len(s) > 1 else np.nan),
                   n=(gap_col, "size"))
              .reset_index())


# ------------------------ figures ------------------------

def plot_culture_trajectory(df: pd.DataFrame, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4),
                             sharey=True, constrained_layout=True)
    for ax, col, ttl in zip(
        axes,
        ["gap_history_specific_to_run", "gap_history_specific_to_episode"],
        ["cross-run history swap (cultural learning)",
         "within-run history swap (episode-specific)"],
    ):
        tr = trajectory(df, col, "contemp_regime")
        for regime in REGIMES:
            r = tr[tr["contemp_regime"] == regime]
            if r.empty:
                continue
            n_obs = int(r["n"].sum())
            ax.errorbar(r["iter"], r["mean"], yerr=r["se"].fillna(0),
                        marker="o", ms=3.5, lw=1.2, capsize=2,
                        color=REGIME_COLORS[regime],
                        label=f"{REGIME_LABELS[regime]} ({n_obs} obs.)")
        ax.axhline(0, color="0.7", lw=0.5, ls=":")
        ax.set_xlabel("training iteration")
        ax.set_ylabel("PMI gap (nats/token)")
        ax.set_title(ttl, fontsize=9)
        ax.yaxis.grid(True, color="0.93", lw=0.5); ax.set_axisbelow(True)
    axes[0].legend(loc="upper left", frameon=False, fontsize=7)
    fig.suptitle("Cultural-learning component of PMI absorption over training",
                 fontsize=10)
    fig.savefig(out, format="pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(out.with_suffix(".png"), dpi=220, bbox_inches="tight",
                pad_inches=0.04)
    plt.close(fig)
    print(f"[ok] {out.name}")


def plot_decomposition_bars(end_tbl: pd.DataFrame, out: Path,
                            title: str = "Decomposition by regime"):
    components = [
        ("gap_char_specific",            "char-specific",     "#332288"),
        ("gap_env_specific",             "env-specific",      "#117733"),
        ("gap_history_specific_to_run",  "history-specific",  "#CC6677"),
        ("gap_char_any",                 "any-character",     "#88CCEE"),
        ("gap_env_any",                  "any-environment",   "#DDCC77"),
    ]
    regimes_present = [r for r in REGIMES if r in end_tbl["regime"].values]
    fig, ax = plt.subplots(figsize=(7.4, 3.6), constrained_layout=True)
    x = np.arange(len(regimes_present))
    # Separate positives (stacked above 0) from negatives (stacked below)
    pos = np.zeros(len(x)); neg = np.zeros(len(x))
    for col, label, color in components:
        vals = np.array([end_tbl.set_index("regime").loc[r, col] if r in end_tbl["regime"].values else 0
                         for r in regimes_present])
        vals = np.nan_to_num(vals)
        bottoms = np.where(vals >= 0, pos, neg)
        ax.bar(x, vals, bottom=bottoms, width=0.7,
               color=color, edgecolor="white", linewidth=0.5,
               label=label)
        pos = pos + np.where(vals >= 0, vals, 0)
        neg = neg + np.where(vals <  0, vals, 0)
    ax.axhline(0, color="0.4", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([REGIME_LABELS[r] for r in regimes_present],
                       rotation=20, ha="right")
    ax.set_ylabel("PMI gap (nats / token)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False)
    ax.yaxis.grid(True, color="0.93", lw=0.5); ax.set_axisbelow(True)
    ax.set_title(title, fontsize=10)
    fig.savefig(out, format="pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(out.with_suffix(".png"), dpi=220, bbox_inches="tight",
                pad_inches=0.04)
    plt.close(fig)
    print(f"[ok] {out.name}")


def plot_culture_vs_char_scatter(df: pd.DataFrame, out: Path):
    """Per-agent end-of-training: history-specific vs char-specific."""
    rows = []
    for (run, ch), g in df.groupby(["run_dir", "character"]):
        last = g["iter"].max()
        gl = g[g["iter"] == last]
        rows.append({
            "run_dir": run, "character": ch,
            "regime": g["regime"].iloc[0],
            "char_spec": gl["gap_char_specific"].mean(),
            "hist_spec": gl["gap_history_specific_to_run"].mean(),
        })
    end_df = pd.DataFrame(rows).dropna(subset=["char_spec", "hist_spec"])
    fig, ax = plt.subplots(figsize=(4.6, 4.0), constrained_layout=True)
    for regime in REGIMES:
        s = end_df[end_df["regime"] == regime]
        if s.empty: continue
        ax.scatter(s["char_spec"], s["hist_spec"], s=32,
                   facecolor=REGIME_COLORS[regime], edgecolor="white",
                   linewidth=0.6, alpha=0.9, label=REGIME_LABELS[regime])
    ax.axhline(0, color="0.85", lw=0.5)
    ax.axvline(0, color="0.85", lw=0.5)
    ax.set_xlabel("char-specific PMI (nats/token)")
    ax.set_ylabel("history-specific PMI (nats/token)")
    ax.legend(loc="best", frameon=False, fontsize=7)
    ax.set_title("End-of-training: history vs char absorption", fontsize=10)
    ax.yaxis.grid(True, color="0.93", lw=0.5); ax.set_axisbelow(True)
    fig.savefig(out, format="pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(out.with_suffix(".png"), dpi=220, bbox_inches="tight",
                pad_inches=0.04)
    plt.close(fig)
    print(f"[ok] {out.name}")
    end_df.to_csv(out.with_suffix(".csv"), index=False)


def plot_env_swap_by_substrate(df_crossenv: pd.DataFrame, out: Path):
    """End-of-training gap_env_specific by substrate."""
    rows = []
    for (env, run, ch), g in df_crossenv.groupby(["env", "run_dir", "character"]):
        last = g["iter"].max()
        gl = g[g["iter"] == last]
        rows.append({"env": env, "run_dir": run, "character": ch,
                     "gap_env_specific": gl["gap_env_specific"].mean(),
                     "gap_env_any":      gl["gap_env_any"].mean()})
    end_df = pd.DataFrame(rows).dropna(subset=["gap_env_specific"])
    envs = sorted(end_df["env"].unique())
    fig, ax = plt.subplots(figsize=(min(7.0, 1.0*len(envs)+2.5), 3.2),
                           constrained_layout=True)
    rng = np.random.default_rng(0)
    for i, env in enumerate(envs):
        s = end_df[end_df["env"] == env]
        xj = i + rng.uniform(-0.18, 0.18, size=len(s))
        ax.scatter(xj, s["gap_env_specific"], s=30, color="#117733",
                   alpha=0.7, edgecolor="white", linewidth=0.5)
        mean = s["gap_env_specific"].mean()
        se = s["gap_env_specific"].std(ddof=1) / np.sqrt(max(len(s), 1)) if len(s) > 1 else 0
        ax.errorbar(i, mean, yerr=se, fmt="D", color="black",
                    capsize=4, ms=7, zorder=10)
    ax.axhline(0, color="0.7", lw=0.5, ls=":")
    ax.set_xticks(range(len(envs)))
    ax.set_xticklabels(envs, rotation=25, ha="right")
    ax.set_ylabel("gap_env_specific (nats/token)")
    ax.set_title("Env-specific absorption by substrate (end of training)",
                 fontsize=10)
    ax.yaxis.grid(True, color="0.93", lw=0.5); ax.set_axisbelow(True)
    fig.savefig(out, format="pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(out.with_suffix(".png"), dpi=220, bbox_inches="tight",
                pad_inches=0.04)
    plt.close(fig)
    print(f"[ok] {out.name}")
    end_df.to_csv(out.with_suffix(".csv"), index=False)


# ------------------------ main ------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inner-loop-dir", required=True,
                   help="Output dir of inner-loop pmi_decomp job (contains rank shards).")
    p.add_argument("--cross-env-dir", required=True,
                   help="Output dir of cross-env pmi_decomp job (contains rank shards).")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    df_inner = load_dirs([args.inner_loop_dir])
    df_cross = load_dirs([args.cross_env_dir])
    print(f"loaded inner-loop: {len(df_inner)} rows; cross-env: {len(df_cross)} rows")

    if df_inner.empty:
        raise SystemExit("no inner-loop rows; aborting")

    df_inner = compute_gaps(df_inner)
    df_cross = compute_gaps(df_cross) if not df_cross.empty else df_cross

    df_inner = attach_regime(df_inner)
    if not df_cross.empty:
        df_cross = attach_regime(df_cross)

    df_inner.to_csv(out_dir / "pmi_decomp_inner_with_regime.csv", index=False)
    if not df_cross.empty:
        df_cross.to_csv(out_dir / "pmi_decomp_cross_with_regime.csv", index=False)

    # ---------- headline tables ----------
    tbl_end = endpoint_by_regime(df_inner, "regime")
    tbl_end.to_csv(out_dir / "table_endregime_inner.csv", index=False)
    print("\nInner Loop — end-of-training by end regime:")
    print(tbl_end.to_string(index=False))

    tbl_contemp = endpoint_by_regime(df_inner, "contemp_regime")
    tbl_contemp.to_csv(out_dir / "table_contempregime_inner.csv", index=False)
    print("\nInner Loop — end-of-training by contemporaneous regime:")
    print(tbl_contemp.to_string(index=False))

    if not df_cross.empty:
        tbl_cross = endpoint_by_regime(df_cross, "contemp_regime")
        tbl_cross.to_csv(out_dir / "table_contempregime_cross.csv", index=False)
        print("\nCross-env — end-of-training by contemporaneous regime (no history):")
        print(tbl_cross.to_string(index=False))

    # ---------- figures ----------
    plot_culture_trajectory(df_inner, out_dir / "fig_culture_trajectory.pdf")
    plot_decomposition_bars(tbl_contemp,
                            out_dir / "fig_decomposition_bars_inner.pdf",
                            "Inner Loop: PMI decomposition by contemporaneous regime")
    plot_culture_vs_char_scatter(df_inner,
                                 out_dir / "fig_scatter_history_vs_char.pdf")
    if not df_cross.empty:
        plot_env_swap_by_substrate(df_cross,
                                   out_dir / "fig_env_swap_by_substrate.pdf")

    # ---------- iter-25 baseline check ----------
    early = df_inner[df_inner["iter"] == df_inner["iter"].min()]
    print(f"\nIter-{int(df_inner['iter'].min())} baseline (cultural-learning gap):")
    if "gap_history_specific_to_run" in early.columns:
        b = early["gap_history_specific_to_run"].dropna()
        print(f"  mean={b.mean():.4f}, n={len(b)}, std={b.std():.4f}")

if __name__ == "__main__":
    main()
