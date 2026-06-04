"""Aggregate trajectory-KL shards into pairwise divergence matrices.

Each shard row scores one (target_seed, source_seed, iter, char, episode).
Per-trajectory log-likelihood under a population = sum of NLLs across all
8 character slots' contributions. D̂(source → target) at iter t =
mean over τ ~ source of [LL_source(τ) - LL_target(τ)] = mean of
[NLL_target - NLL_source] (per trajectory, in nats; reported also per token).

Outputs (in --out):
  - per_traj_kl.csv            per (source, target, iter, episode) row
  - pairwise_kl_by_iter.csv    mean and SE per (source, target, iter)
  - fig_kl_trajectory.{pdf,png}
  - fig_kl_matrix_end.{pdf,png}
  - fig_per_character_decomp.{pdf,png}
  - kl_vs_endkl.csv            correlation with per-char end-KL-from-base
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
from analyse_absorption_by_regime import end_of_training_regime  # noqa: E402

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42,
})

_SEED_RE = re.compile(r"_seed(\d+)$")


def seed_label(run_dir: str) -> str:
    name = Path(run_dir).name
    m = _SEED_RE.search(name)
    return f"seed{m.group(1)}" if m else "seed1"


def load_shards(dirs: list[str]) -> pd.DataFrame:
    rows = []
    for d in dirs:
        for f in sorted(glob.glob(str(Path(d) / "trajectory_kl_rank*.jsonl"))):
            with open(f) as fh:
                rows.extend(json.loads(line) for line in fh if line.strip())
    if not rows:
        raise SystemExit("no rows loaded")
    df = pd.DataFrame(rows)
    df["target"] = df["target_run"].apply(seed_label)
    df["source"] = df["source_run"].apply(seed_label)
    return df


def per_trajectory_loglik(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate by (target, source, iter, episode): sum across characters."""
    n_chars_per_pop = df.groupby(["target", "iter"])["character"].nunique().max()
    grouped = df.groupby(["target", "source", "iter", "episode"])
    agg = grouped.agg(nll_sum_total=("nll_sum", "sum"),
                      n_tokens=("n_action_tokens", "sum"),
                      n_chars=("character", "nunique")).reset_index()
    # Drop trajectories where not every character was scored under the
    # target population (incomplete adapter coverage).
    keep_n = int(n_chars_per_pop)
    pre = len(agg)
    agg = agg[agg["n_chars"] == keep_n].copy()
    print(f"per-trajectory rows: {pre} → {len(agg)} (keeping only "
          f"trajectories with full {keep_n}-character coverage)")
    return agg


def pairwise_kl(per_traj: pd.DataFrame) -> pd.DataFrame:
    """Per-trajectory:
       Δ_per_tok = (NLL_target - NLL_source) / n_tokens, where source is the
       population that generated the trajectory τ.
       Aggregate to (source, target, iter): mean ± SE.

       Joined on (source, iter, episode): each row of `per_traj` is one
       (target population evaluates this source's episode). We need pairs.
    """
    # Pivot so rows are (source, iter, episode) and cols are target → NLL,n_tok.
    src = per_traj.rename(columns={"nll_sum_total": "nll", "n_tokens": "ntok"})
    # For each (source, iter, episode), the row where target == source gives
    # NLL_source(τ).
    base = src[src["source"] == src["target"]][
        ["source", "iter", "episode", "nll", "ntok"]
    ].rename(columns={"nll": "nll_source", "ntok": "ntok_source"})
    merged = src.merge(base, on=["source", "iter", "episode"], how="left")
    merged["delta_nll"]    = merged["nll"] - merged["nll_source"]
    merged["delta_pertok"] = merged["delta_nll"] / merged["ntok"]
    # group by (source, target, iter)
    agg = (merged.groupby(["source", "target", "iter"])
                 .agg(mean_pertok=("delta_pertok", "mean"),
                      se_pertok=("delta_pertok",
                                 lambda s: s.std(ddof=1)/np.sqrt(len(s))
                                 if len(s) > 1 else np.nan),
                      mean_per_traj=("delta_nll", "mean"),
                      n_traj=("delta_nll", "size"),
                      mean_ntok=("ntok", "mean"))
                 .reset_index())
    return agg, merged


# ------------------------ plotting ------------------------

def plot_trajectory_kl(pair_df: pd.DataFrame, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4),
                             constrained_layout=True)
    # Left: per-token. Right: per trajectory.
    for ax, ycol, ttl in zip(
        axes,
        ["mean_pertok", "mean_per_traj"],
        ["per-token (nats / token)", "per-trajectory (nats / trajectory)"],
    ):
        # Cross-pop = source != target. Within = source == target (should be 0).
        cross = pair_df[pair_df["source"] != pair_df["target"]]
        within = pair_df[pair_df["source"] == pair_df["target"]]
        # cross: aggregate across pairs
        c = (cross.groupby("iter")
                  .agg(mean=(ycol, "mean"),
                       se=(ycol, lambda s: s.std(ddof=1)/np.sqrt(len(s))
                           if len(s) > 1 else np.nan),
                       n=(ycol, "size"))
                  .reset_index())
        w = (within.groupby("iter")[ycol].mean().reset_index())
        ax.errorbar(c["iter"], c["mean"], yerr=c["se"].fillna(0),
                    marker="o", ms=4, lw=1.4, capsize=3, color="#332288",
                    label=f"cross-population (mean over {len(cross['source'].unique())} ordered pairs)")
        ax.plot(w["iter"], w[ycol], "s--", color="0.5", ms=4, lw=1.0,
                label="within-population (= 0 by construction)")
        ax.axhline(0, color="0.85", lw=0.5)
        ax.set_xlabel("training iteration")
        ax.set_ylabel(f"D̂(source → target)   {ttl}")
        ax.yaxis.grid(True, color="0.93", lw=0.5); ax.set_axisbelow(True)
        ax.legend(loc="upper left", frameon=False, fontsize=7)
    fig.suptitle("Trajectory-level KL between parallel populations over training",
                 fontsize=10)
    fig.savefig(out, format="pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(out.with_suffix(".png"), dpi=220, bbox_inches="tight",
                pad_inches=0.04)
    plt.close(fig)
    print(f"[ok] {out.name}")


def plot_kl_matrix(pair_df: pd.DataFrame, iter_target: int, out: Path):
    """End-of-training pairwise matrix (per-token)."""
    sub = pair_df[pair_df["iter"] == iter_target]
    if sub.empty:
        print(f"[skip] no data at iter {iter_target}")
        return
    seeds = sorted(set(sub["source"]) | set(sub["target"]))
    mat = np.full((len(seeds), len(seeds)), np.nan)
    for i, src in enumerate(seeds):
        for j, tgt in enumerate(seeds):
            r = sub[(sub["source"] == src) & (sub["target"] == tgt)]
            if not r.empty:
                mat[i, j] = r["mean_pertok"].iloc[0]
    fig, ax = plt.subplots(figsize=(4.4, 3.6), constrained_layout=True)
    vmax = max(0.001, float(np.nanmax(mat))) if not np.all(np.isnan(mat)) else 1.0
    im = ax.imshow(mat, cmap="viridis", vmin=0, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(seeds))); ax.set_yticks(range(len(seeds)))
    ax.set_xticklabels(seeds); ax.set_yticklabels(seeds)
    ax.set_xlabel("target population (evaluator)")
    ax.set_ylabel("source population (τ ~ p_source)")
    for i in range(len(seeds)):
        for j in range(len(seeds)):
            v = mat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:+.3f}", ha="center", va="center",
                        color="white" if abs(v) > 0.5 * vmax else "black",
                        fontsize=7)
    cbar = plt.colorbar(im, ax=ax, fraction=0.045)
    cbar.set_label("D̂ per token (nats)")
    ax.set_title(f"Pairwise trajectory KL — iter {iter_target}", fontsize=10)
    fig.savefig(out, format="pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(out.with_suffix(".png"), dpi=220, bbox_inches="tight",
                pad_inches=0.04)
    plt.close(fig)
    print(f"[ok] {out.name}")


def plot_per_character(df: pd.DataFrame, end_iter: int, out: Path):
    """Per-character cross-population KL contribution at end_iter."""
    sub = df[df["iter"] == end_iter]
    if sub.empty:
        print("[skip] no end-iter data for per-char plot")
        return
    # For each (source, target, character): delta NLL averaged over episodes
    # = NLL_target(c, ep) - NLL_source(c, ep)
    src = sub.rename(columns={"nll_sum": "nll", "n_action_tokens": "ntok"})
    base = src[src["source"] == src["target"]][
        ["source", "iter", "character", "episode", "nll", "ntok"]
    ].rename(columns={"nll": "nll_source", "ntok": "ntok_source"})
    merged = src.merge(base, on=["source", "iter", "character", "episode"],
                       how="left")
    merged["delta_pertok"] = (merged["nll"] - merged["nll_source"]) / merged["ntok"]
    cross = merged[merged["source"] != merged["target"]]
    by_char = (cross.groupby("character")["delta_pertok"]
                    .agg(["mean", "std", "count"]).reset_index()
                    .sort_values("mean", ascending=False))
    by_char.to_csv(out.with_suffix(".csv"), index=False)
    fig, ax = plt.subplots(figsize=(7.0, 3.4), constrained_layout=True)
    chars = by_char["character"].tolist()
    x = np.arange(len(chars))
    se = by_char["std"] / np.sqrt(by_char["count"].clip(lower=1))
    ax.bar(x, by_char["mean"], yerr=se, color="#332288",
           edgecolor="white", linewidth=0.5, capsize=3)
    ax.axhline(0, color="0.4", lw=0.5)
    ax.set_xticks(x); ax.set_xticklabels(chars, rotation=30, ha="right",
                                          fontsize=7)
    ax.set_ylabel("per-char Δ NLL per token (nats)")
    ax.set_title(f"Cross-population KL contribution by character slot — iter {end_iter}",
                 fontsize=10)
    ax.yaxis.grid(True, color="0.93", lw=0.5); ax.set_axisbelow(True)
    fig.savefig(out, format="pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(out.with_suffix(".png"), dpi=220, bbox_inches="tight",
                pad_inches=0.04)
    plt.close(fig)
    print(f"[ok] {out.name}")


def correlate_with_endkl(df: pd.DataFrame, seed_dirs: list[Path],
                          end_iter: int, out: Path):
    """Per-(seed, character) correlation between (a) end-of-training KL from
    base (regime-classification metric) and (b) per-char contribution to
    cross-pop trajectory KL.

    (a) is read from each seed's metrics.jsonl via end_of_training_regime.
    (b) is computed from `df` (the raw per-char shard rows).
    """
    sub = df[df["iter"] == end_iter]
    if sub.empty:
        return
    src = sub.rename(columns={"nll_sum": "nll", "n_action_tokens": "ntok"})
    base = src[src["source"] == src["target"]][
        ["source", "iter", "character", "episode", "nll", "ntok"]
    ].rename(columns={"nll": "nll_source", "ntok": "ntok_source"})
    merged = src.merge(base, on=["source", "iter", "character", "episode"],
                       how="left")
    merged["delta_pertok"] = (merged["nll"] - merged["nll_source"]) / merged["ntok"]
    cross = merged[merged["source"] != merged["target"]]
    # Group by (source seed, character) — "how anomalous this seed's char is
    # under other seeds' weights" (mean over (target, episode))
    per_seed_char = (cross.groupby(["source", "character"])["delta_pertok"]
                          .mean().reset_index()
                          .rename(columns={"delta_pertok": "kl_pertok"}))
    # Now get end-KL-from-base per (seed, char) by scanning each seed's runs.
    end_kls = []
    for sd in seed_dirs:
        end = end_of_training_regime(sd)
        if end.empty:
            continue
        for _, r in end.iterrows():
            end_kls.append({"source": seed_label(str(sd)),
                            "character": r["agent"],
                            "end_kl_from_base": r["kl"],
                            "regime": r["regime"]})
    end_kl_df = pd.DataFrame(end_kls)
    joined = per_seed_char.merge(end_kl_df, on=["source", "character"], how="left")
    joined.to_csv(out, index=False)
    if joined["end_kl_from_base"].notna().sum() < 3:
        print(f"[skip] insufficient end-KL data for correlation")
        return
    x = joined["end_kl_from_base"].astype(float).values
    y = joined["kl_pertok"].astype(float).values
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return
    r = np.corrcoef(x[mask], y[mask])[0, 1]
    n = int(mask.sum())
    print(f"\nPer-(seed, char) correlation: end-KL-from-base vs cross-pop "
          f"trajectory-KL contribution: r={r:.3f} (n={n})")
    # Scatter
    fig, ax = plt.subplots(figsize=(4.4, 3.6), constrained_layout=True)
    REGIME_COLORS = {"coherent_low_kl": "#117733", "committed": "#332288",
                     "mid_range": "#DDCC77", "policy_collapse": "#88CCEE",
                     "text_degenerate": "#CC6677"}
    for regime, color in REGIME_COLORS.items():
        s = joined[joined["regime"] == regime]
        if s.empty: continue
        ax.scatter(s["end_kl_from_base"], s["kl_pertok"], s=32,
                   facecolor=color, edgecolor="white", linewidth=0.6,
                   alpha=0.85, label=regime.replace("_", " "))
    ax.set_xlabel("end-of-training KL from base (nats)")
    ax.set_ylabel("cross-pop trajectory-KL contribution (nats/token)")
    ax.set_xscale("symlog", linthresh=0.5)
    ax.axhline(0, color="0.85", lw=0.5)
    ax.set_title(f"Per-(seed, char): r = {r:.3f}, n = {n}", fontsize=10)
    ax.yaxis.grid(True, color="0.93", lw=0.5); ax.set_axisbelow(True)
    ax.legend(loc="best", frameon=False, fontsize=7)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.04)
    fig.savefig(out.with_suffix(".png"), dpi=220, bbox_inches="tight",
                pad_inches=0.04)
    plt.close(fig)
    print(f"[ok] {out.stem}_scatter")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dirs", nargs="+",
                   help="Shard output dirs containing trajectory_kl_rank*.jsonl")
    p.add_argument("--out", required=True)
    p.add_argument("--seed-run-dir", action="append", required=True,
                   help="The 5 inner-loop seed runs, for correlation analysis.")
    args = p.parse_args()
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    df = load_shards(args.dirs)
    print(f"loaded {len(df)} rows")
    df.to_csv(out_dir / "raw_per_char.csv", index=False)

    per_traj = per_trajectory_loglik(df)
    per_traj.to_csv(out_dir / "per_traj_kl.csv", index=False)

    pair_df, merged = pairwise_kl(per_traj)
    pair_df.to_csv(out_dir / "pairwise_kl_by_iter.csv", index=False)
    print("\nPairwise mean KL per-token by iter (cross-population):")
    cross = pair_df[pair_df["source"] != pair_df["target"]]
    summary = (cross.groupby("iter")
                    .agg(mean=("mean_pertok", "mean"),
                         min=("mean_pertok", "min"),
                         max=("mean_pertok", "max"),
                         n_pairs=("mean_pertok", "size"))
                    .reset_index())
    print(summary.to_string(index=False))

    # Within-pop sanity check (should be 0)
    within = pair_df[pair_df["source"] == pair_df["target"]]
    print(f"\nWithin-pop diagonal (mean of pertok over all (seed, iter)): "
          f"{within['mean_pertok'].mean():.6f} (should be 0)")

    plot_trajectory_kl(pair_df, out_dir / "fig_kl_trajectory.pdf")
    # Use latest available iter that has full panel
    iters_with_full_panel = sorted(
        i for i in pair_df["iter"].unique()
        if (pair_df[(pair_df["iter"] == i) & (pair_df["source"] != pair_df["target"])]
            .shape[0]) >= 12)
    if iters_with_full_panel:
        end_iter = max(iters_with_full_panel)
        plot_kl_matrix(pair_df, end_iter,
                       out_dir / f"fig_kl_matrix_iter{end_iter}.pdf")
        plot_per_character(df, end_iter,
                           out_dir / f"fig_per_character_iter{end_iter}.pdf")
        seed_dirs = [Path(s) for s in args.seed_run_dir]
        correlate_with_endkl(df, seed_dirs, end_iter,
                             out_dir / "kl_vs_endkl.csv")
    else:
        print("[warn] no iter with full 12-pair panel; skipping end plots")


if __name__ == "__main__":
    main()
