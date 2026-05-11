"""
Cross-seed aggregate of character-prompt absorption — for the same env run
under multiple seeds, average over seeds to get statistically robust curves.

Each absorption directory should belong to the *same* environment under a
different seed. The seed label is extracted from the run directory name
(e.g. `..._seed2` → seed=2; the unsuffixed base run is treated as seed=1).

Usage
-----
  uv run python scripts/aggregate_seed_replicates.py \\
      --env-label inner_loop \\
      --out runs/cultural_emergence/_seed_aggregate_inner_loop \\
      runs/cultural_emergence/run7_8agent_7B_inner_loop/absorption/4458639 \\
      runs/cultural_emergence/run7_8agent_7B_inner_loop_seed2/absorption/<jobid> \\
      runs/cultural_emergence/run7_8agent_7B_inner_loop_seed3/absorption/<jobid> \\
      runs/cultural_emergence/run7_8agent_7B_inner_loop_seed4/absorption/<jobid> \\
      runs/cultural_emergence/run7_8agent_7B_inner_loop_seed5/absorption/<jobid>
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


_SEED_RE = re.compile(r"_seed(\d+)$")


def extract_seed(run_dir: Path) -> int:
    m = _SEED_RE.search(run_dir.name)
    return int(m.group(1)) if m else 1


def load_abs_dir(abs_dir: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(str(abs_dir / "absorption_rank*.jsonl"))):
        with open(f) as fh:
            rows.extend(json.loads(line) for line in fh if line.strip())
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["seed"] = extract_seed(Path(rows[0]["run_dir"]))
    return df


def per_seed_iter_agg(df: pd.DataFrame) -> pd.DataFrame:
    """Token-weighted per-(seed, iter) aggregate of `gap` and component NLLs."""
    rows = []
    for (seed, it), g in df.groupby(["seed", "iter"]):
        w = g["n_action_tokens"].values
        gap = g["gap"].values
        rows.append({
            "seed": seed, "iter": it,
            "gap":              np.average(gap, weights=w),
            "gap_se_within":    (gap.std(ddof=1) / np.sqrt(len(gap))) if len(gap) > 1 else 0.0,
            "mean_nll_with":    np.average(g["mean_nll_with"],    weights=w),
            "mean_nll_without": np.average(g["mean_nll_without"], weights=w),
            "n_episodes":       len(g),
            "n_tokens":         int(w.sum()),
            "n_chars":          g["character"].nunique(),
        })
    return pd.DataFrame(rows).sort_values(["seed", "iter"])


def cross_seed_agg(per_seed: pd.DataFrame) -> pd.DataFrame:
    """Mean ± SE across seeds at each iter (only iters present in ≥2 seeds)."""
    rows = []
    for it, g in per_seed.groupby("iter"):
        if len(g) < 1:
            continue
        rows.append({
            "iter": it,
            "n_seeds": len(g),
            "gap_mean":            g["gap"].mean(),
            "gap_se_across_seeds": g["gap"].std(ddof=1) / np.sqrt(len(g)) if len(g) > 1 else 0.0,
            "gap_std_across_seeds": g["gap"].std(ddof=1) if len(g) > 1 else 0.0,
            "nll_with_mean":       g["mean_nll_with"].mean(),
            "nll_without_mean":    g["mean_nll_without"].mean(),
        })
    return pd.DataFrame(rows).sort_values("iter")


def linear_slope_test(per_seed: pd.DataFrame) -> dict:
    """Fit gap ~ iter + seed_intercept; report slope and t-stat."""
    # OLS with seed dummies. Center iter for cleaner intercept.
    df = per_seed.copy()
    seeds = sorted(df["seed"].unique())
    df["iter_c"] = df["iter"] - df["iter"].mean()
    # Design: intercept per seed + shared slope.
    n = len(df)
    p = len(seeds) + 1  # one intercept per seed + slope
    X = np.zeros((n, p))
    for i, row in enumerate(df.itertuples(index=False)):
        X[i, seeds.index(row.seed)] = 1.0
        X[i, -1] = row.iter_c
    y = df["gap"].values
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = max(n - p, 1)
    sigma2 = float(resid @ resid) / dof
    cov = sigma2 * np.linalg.pinv(X.T @ X)
    slope = float(beta[-1])
    slope_se = float(np.sqrt(max(cov[-1, -1], 0.0)))
    t = slope / slope_se if slope_se > 0 else float("nan")
    return {
        "slope_per_iter": slope,
        "slope_se": slope_se,
        "t_stat": t,
        "n_observations": n,
        "n_seeds": len(seeds),
        "dof": dof,
    }


def plot_summary(df: pd.DataFrame, env_label: str, out_path: Path) -> dict:
    per_seed = per_seed_iter_agg(df)
    cross = cross_seed_agg(per_seed)
    slope_info = linear_slope_test(per_seed)

    seeds = sorted(per_seed["seed"].unique())
    cmap = plt.get_cmap("tab10")
    seed_colors = {s: cmap(i % 10) for i, s in enumerate(seeds)}

    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.32)

    # ── (1) Per-seed traces + cross-seed mean ──────────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    for seed in seeds:
        sub = per_seed[per_seed["seed"] == seed]
        ax1.plot(sub["iter"], sub["gap"], "o-", color=seed_colors[seed],
                 alpha=0.55, lw=1.4, ms=5, label=f"seed {seed}")
    # Cross-seed mean
    ax1.errorbar(cross["iter"], cross["gap_mean"], yerr=cross["gap_se_across_seeds"],
                 fmt="o-", color="black", lw=3, capsize=4, ms=7, zorder=10,
                 label=f"mean across {len(seeds)} seeds ±SE")
    ax1.fill_between(cross["iter"],
                     cross["gap_mean"] - cross["gap_se_across_seeds"],
                     cross["gap_mean"] + cross["gap_se_across_seeds"],
                     color="black", alpha=0.15)
    ax1.axhline(0, ls="--", c="grey", lw=1)
    ax1.set_xlabel("training iteration")
    ax1.set_ylabel("PMI per token (nats):  NLL(no character) − NLL(with character)")
    ax1.set_title(f"Character absorption — {env_label} ({len(seeds)} seeds)\n"
                  f"slope = {slope_info['slope_per_iter']:+.5f} nats/token/iter "
                  f"(t = {slope_info['t_stat']:+.2f}, dof = {slope_info['dof']})")
    ax1.legend(loc="best", fontsize=9)
    ax1.grid(alpha=0.3)

    # ── (2) NLL with vs without across seeds ──────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    for seed in seeds:
        sub = per_seed[per_seed["seed"] == seed]
        ax2.plot(sub["iter"], sub["mean_nll_with"],    "-",  color=seed_colors[seed], alpha=0.5, lw=1)
        ax2.plot(sub["iter"], sub["mean_nll_without"], "--", color=seed_colors[seed], alpha=0.5, lw=1)
    ax2.plot(cross["iter"], cross["nll_with_mean"],    "o-", color="C0", lw=3, label="with character (mean)")
    ax2.plot(cross["iter"], cross["nll_without_mean"], "s-", color="C3", lw=3, label="without character (mean)")
    ax2.set_xlabel("training iteration")
    ax2.set_ylabel("mean NLL per action token (nats)")
    ax2.set_title("Action-token surprise — components\n(faint: per-seed; bold: cross-seed mean)")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    # ── (3) Per-character absorption (averaged across seeds) ──────────────
    ax3 = fig.add_subplot(gs[1, :2])
    char_iter = (
        df.groupby(["character", "iter"]).apply(
            lambda g: pd.Series({
                "gap": np.average(g["gap"], weights=g["n_action_tokens"]),
                "n_seeds": g["seed"].nunique(),
                "se": g["gap"].std(ddof=1) / np.sqrt(len(g)) if len(g) > 1 else 0.0,
            }),
            include_groups=False,
        ).reset_index()
    )
    chars = sorted(char_iter["character"].unique())
    cmap2 = plt.get_cmap("tab20")
    for i, ch in enumerate(chars):
        sub = char_iter[char_iter["character"] == ch].sort_values("iter")
        ax3.errorbar(sub["iter"], sub["gap"], yerr=sub["se"],
                     marker="o", capsize=2, color=cmap2(i % 20),
                     alpha=0.85, label=ch)
    ax3.axhline(0, ls="--", c="grey", lw=1)
    ax3.set_xlabel("training iteration")
    ax3.set_ylabel("PMI gap (nats / token), seed-averaged")
    ax3.set_title("Per-character absorption (averaged across seeds)")
    ax3.legend(fontsize=7, ncol=2, loc="best")
    ax3.grid(alpha=0.3)

    # ── (4) Per-seed boxplot at first vs last iter ────────────────────────
    ax4 = fig.add_subplot(gs[1, 2])
    iters = sorted(df["iter"].unique())
    first_it, last_it = iters[0], iters[-1]
    box_data = []
    box_labels = []
    box_colors = []
    for seed in seeds:
        for it, suffix in [(first_it, f"i{first_it}"), (last_it, f"i{last_it}")]:
            d = df[(df["seed"] == seed) & (df["iter"] == it)]["gap"].values
            if len(d) == 0:
                continue
            box_data.append(d); box_labels.append(f"s{seed} {suffix}")
            box_colors.append(seed_colors[seed])
    bp = ax4.boxplot(box_data, patch_artist=True, showfliers=False, widths=0.6)
    for patch, c in zip(bp["boxes"], box_colors):
        patch.set_facecolor(c); patch.set_alpha(0.55)
    ax4.set_xticklabels(box_labels, rotation=70, ha="right", fontsize=7)
    ax4.axhline(0, ls="--", c="grey", lw=1)
    ax4.set_ylabel("per-episode gap (nats/token)")
    ax4.set_title("Per-seed first vs last iter\n(episode-level distribution)")
    ax4.grid(alpha=0.3, axis="y")

    fig.suptitle(
        f"Character-prompt absorption — {env_label} seed replicates\n"
        f"{len(seeds)} seeds · {len(df)} episodes · {df['character'].nunique()} characters",
        fontsize=14, y=0.995,
    )
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"wrote {out_path}")
    return slope_info


def plot_headline(df: pd.DataFrame, env_label: str, out_path: Path) -> None:
    per_seed = per_seed_iter_agg(df)
    cross = cross_seed_agg(per_seed)
    seeds = sorted(per_seed["seed"].unique())
    cmap = plt.get_cmap("tab10")
    seed_colors = {s: cmap(i % 10) for i, s in enumerate(seeds)}

    fig, ax = plt.subplots(figsize=(11, 6.5))
    for seed in seeds:
        sub = per_seed[per_seed["seed"] == seed]
        ax.plot(sub["iter"], sub["gap"], "o-", color=seed_colors[seed],
                alpha=0.45, lw=1.3, ms=5, label=f"seed {seed}")
    ax.errorbar(cross["iter"], cross["gap_mean"], yerr=cross["gap_se_across_seeds"],
                fmt="o-", color="black", lw=3, capsize=4, ms=7, zorder=10,
                label=f"mean ± SE across {len(seeds)} seeds")
    ax.fill_between(cross["iter"],
                    cross["gap_mean"] - cross["gap_se_across_seeds"],
                    cross["gap_mean"] + cross["gap_se_across_seeds"],
                    color="black", alpha=0.15)
    ax.axhline(0, ls="--", c="grey", lw=1)
    ax.set_xlabel("training iteration")
    ax.set_ylabel("PMI per token (nats):  NLL(no character) − NLL(with character)")
    ax.set_title(f"Character absorption — {env_label} ({len(seeds)} seeds)")
    ax.legend(loc="best", fontsize=10)
    ax.grid(alpha=0.3)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"wrote {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Cross-seed absorption aggregate.")
    p.add_argument("absorption_dirs", nargs="+",
                   help="One or more absorption directories, all from the same env.")
    p.add_argument("--env-label", required=True, help="Display label for the env.")
    p.add_argument("--out", required=True, help="Output directory.")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    dfs = []
    for d in args.absorption_dirs:
        sub = load_abs_dir(Path(d))
        if sub.empty:
            print(f"  WARN: no data in {d}")
            continue
        seed = sub["seed"].iloc[0]
        print(f"  seed {seed}: loaded {len(sub):4d} rows from {d}")
        dfs.append(sub)
    if not dfs:
        raise SystemExit("no data loaded")

    df = pd.concat(dfs, ignore_index=True)
    df.to_csv(out_dir / "absorption_all_seeds.csv", index=False)
    print(f"wrote {out_dir / 'absorption_all_seeds.csv'} ({len(df)} rows)")

    slope = plot_summary(df, args.env_label, out_dir / "absorption_seed_panels.png")
    plot_headline(df, args.env_label, out_dir / "absorption_seed_headline.png")

    cross = cross_seed_agg(per_seed_iter_agg(df))
    cross.to_csv(out_dir / "absorption_cross_seed_iter.csv", index=False)
    print(f"\ncross-seed aggregate:\n{cross.to_string(index=False)}")
    print(f"\nslope test (gap ~ iter + seed_intercept):\n  {slope}")


if __name__ == "__main__":
    main()
