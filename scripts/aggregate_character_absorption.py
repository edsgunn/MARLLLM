"""
Cross-environment aggregate of character-prompt absorption.

Walks one or more absorption directories (each containing a per-rank shard
set) and produces:

  - one CSV with all rows tagged by env name
  - one multi-panel PNG comparing absorption curves across envs

Usage
-----
  uv run python scripts/aggregate_character_absorption.py \\
      --out runs/cultural_emergence/_absorption_summary \\
      runs/cultural_emergence/run7_8agent_7B_margin_notes/absorption/4458450 \\
      runs/cultural_emergence/run7_8agent_7B_study_group/absorption/4458638 \\
      runs/cultural_emergence/run7_8agent_7B_inner_loop/absorption/4458639 \\
      runs/cultural_emergence/run7_8agent_7B_conjecture/absorption/4458640 \\
      runs/cultural_emergence/run7_8agent_7B_robotic_athanor/absorption/4458641
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


_ENV_RE = re.compile(r"run\d+_\d+agent_\d+B_(.+)$")


def env_name_from_run(run_dir: Path) -> str:
    """Extract env name from a run dir like `run7_8agent_7B_margin_notes`."""
    m = _ENV_RE.match(run_dir.name)
    return m.group(1) if m else run_dir.name


def load_abs_dir(abs_dir: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(str(abs_dir / "absorption_rank*.jsonl"))):
        with open(f) as fh:
            rows.extend(json.loads(line) for line in fh if line.strip())
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["env"] = env_name_from_run(Path(rows[0]["run_dir"]))
    return df


def per_env_iter_agg(df: pd.DataFrame) -> pd.DataFrame:
    """Token-weighted mean & SE of `gap` per (env, iter)."""
    rows = []
    for (env, it), g in df.groupby(["env", "iter"]):
        w = g["n_action_tokens"].values
        gap = g["gap"].values
        nll_w = g["mean_nll_with"].values
        nll_wo = g["mean_nll_without"].values
        rows.append({
            "env": env, "iter": it,
            "gap":               np.average(gap, weights=w),
            "gap_se":            (gap.std(ddof=1) / np.sqrt(len(gap))) if len(gap) > 1 else 0.0,
            "mean_nll_with":     np.average(nll_w, weights=w),
            "mean_nll_without":  np.average(nll_wo, weights=w),
            "n_episodes":        len(g),
            "n_tokens":          int(w.sum()),
        })
    return pd.DataFrame(rows).sort_values(["env", "iter"])


def plot_summary(df: pd.DataFrame, out_path: Path) -> None:
    agg = per_env_iter_agg(df)
    envs = sorted(agg["env"].unique())
    cmap = plt.get_cmap("tab10")
    colors = {e: cmap(i % 10) for i, e in enumerate(envs)}

    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.42, wspace=0.32)

    # ── (1) Headline: gap vs iter, one line per env ────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    for env in envs:
        sub = agg[agg["env"] == env]
        ax1.errorbar(sub["iter"], sub["gap"], yerr=sub["gap_se"],
                     marker="o", capsize=3, lw=2, color=colors[env],
                     label=f"{env} (n={sub['n_episodes'].sum()})")
    ax1.axhline(0, ls="--", c="grey", lw=1)
    ax1.set_xlabel("training iteration")
    ax1.set_ylabel("PMI per token (nats):  NLL(no character) − NLL(with character)")
    ax1.set_title("Character-prompt PMI gap across environments\n(decreasing → character absorbed; flat → prompt still load-bearing)")
    ax1.legend(loc="best", fontsize=9)
    ax1.grid(alpha=0.3)

    # ── (2) Effect-size summary: first vs last iter per env ────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    summary = []
    for env in envs:
        sub = agg[agg["env"] == env].sort_values("iter")
        summary.append({
            "env": env,
            "first_iter": int(sub["iter"].iloc[0]),
            "last_iter":  int(sub["iter"].iloc[-1]),
            "first_gap":  float(sub["gap"].iloc[0]),
            "last_gap":   float(sub["gap"].iloc[-1]),
            "shrinkage":  float(sub["gap"].iloc[0] - sub["gap"].iloc[-1]),
        })
    s_df = pd.DataFrame(summary)
    x = np.arange(len(s_df))
    w = 0.38
    ax2.bar(x - w / 2, s_df["first_gap"], width=w, color="C3", alpha=0.85,
            label=[f"first (i{i})" for i in s_df["first_iter"]][0] + " etc.")
    ax2.bar(x + w / 2, s_df["last_gap"], width=w, color="C0", alpha=0.85,
            label=[f"last (i{i})" for i in s_df["last_iter"]][0] + " etc.")
    for i, row in s_df.iterrows():
        ax2.text(i - w / 2, row["first_gap"], f"i{row['first_iter']}",
                 ha="center", va="bottom" if row["first_gap"] >= 0 else "top", fontsize=7)
        ax2.text(i + w / 2, row["last_gap"], f"i{row['last_iter']}",
                 ha="center", va="bottom" if row["last_gap"] >= 0 else "top", fontsize=7)
    ax2.set_xticks(x); ax2.set_xticklabels(s_df["env"], rotation=30, ha="right")
    ax2.axhline(0, ls="--", c="grey", lw=1)
    ax2.set_ylabel("PMI gap (nats / token)")
    ax2.set_title("Endpoint comparison\n(first vs last checkpointed iter)")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3, axis="y")

    # ── (3) NLL with vs without per env ────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    for env in envs:
        sub = agg[agg["env"] == env]
        ax3.plot(sub["iter"], sub["mean_nll_with"], "o-", color=colors[env],
                 lw=2, label=f"{env} • with")
        ax3.plot(sub["iter"], sub["mean_nll_without"], "s--", color=colors[env],
                 lw=1.4, alpha=0.7)
    ax3.set_xlabel("training iteration")
    ax3.set_ylabel("mean NLL per action token (nats)")
    ax3.set_title("Action-token surprise\n(solid = with character, dashed = without)")
    ax3.legend(fontsize=7, loc="best")
    ax3.grid(alpha=0.3)

    # ── (4) Per-env per-character spaghetti at last iter ───────────────────
    ax4 = fig.add_subplot(gs[1, 1:])
    last_per_env = (
        df.sort_values("iter").groupby("env").apply(
            lambda g: g[g["iter"] == g["iter"].max()],
            include_groups=False,
        ).reset_index().drop(columns=["level_1"], errors="ignore")
    )
    env_positions = {e: i for i, e in enumerate(envs)}
    rng = np.random.default_rng(0)
    for env in envs:
        sub = last_per_env[last_per_env["env"] == env]
        char_gap = sub.groupby("character")["gap"].mean()
        x_pos = env_positions[env] + rng.uniform(-0.18, 0.18, size=len(char_gap))
        ax4.scatter(x_pos, char_gap.values, s=70, alpha=0.65,
                    color=colors[env], edgecolor="black", lw=0.6)
        ax4.errorbar([env_positions[env]], [char_gap.mean()],
                     yerr=[char_gap.std(ddof=1) / np.sqrt(len(char_gap))] if len(char_gap) > 1 else [0],
                     fmt="D", color="black", capsize=4, ms=8, zorder=10)
    ax4.set_xticks(list(env_positions.values()))
    ax4.set_xticklabels(envs, rotation=30, ha="right")
    ax4.axhline(0, ls="--", c="grey", lw=1)
    ax4.set_ylabel("per-character mean gap at final iter (nats / token)")
    ax4.set_title("Final-checkpoint per-character spread\n(dots = characters; black diamond = env mean ± SE)")
    ax4.grid(alpha=0.3, axis="y")

    # ── (5) Heatmap env × iter (filled where data present) ────────────────
    ax5 = fig.add_subplot(gs[2, :2])
    all_iters = sorted(df["iter"].unique())
    grid = np.full((len(envs), len(all_iters)), np.nan)
    for i, env in enumerate(envs):
        for j, it in enumerate(all_iters):
            sub = agg[(agg["env"] == env) & (agg["iter"] == it)]
            if not sub.empty:
                grid[i, j] = sub["gap"].iloc[0]
    vmax = float(np.nanmax(np.abs(grid))) if not np.all(np.isnan(grid)) else 1.0
    im = ax5.imshow(grid, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax5.set_yticks(range(len(envs))); ax5.set_yticklabels(envs)
    ax5.set_xticks(range(len(all_iters))); ax5.set_xticklabels(all_iters, rotation=45)
    ax5.set_xlabel("training iteration")
    ax5.set_title("PMI gap per (env, iter) — red: prompt still helps; blue: absorbed-or-better")
    cbar = plt.colorbar(im, ax=ax5, fraction=0.025); cbar.set_label("gap (nats / token)")
    for i in range(len(envs)):
        for j in range(len(all_iters)):
            v = grid[i, j]
            if not np.isnan(v):
                ax5.text(j, i, f"{v:+.2f}", ha="center", va="center",
                         color="black" if abs(v) < 0.5 * vmax else "white", fontsize=7)

    # ── (6) Episode-level distribution at first vs last iter, all envs ─────
    ax6 = fig.add_subplot(gs[2, 2])
    box_data = []
    box_labels = []
    box_colors = []
    for env in envs:
        first_it = df[df["env"] == env]["iter"].min()
        last_it  = df[df["env"] == env]["iter"].max()
        for it, suffix in [(first_it, "first"), (last_it, "last")]:
            d = df[(df["env"] == env) & (df["iter"] == it)]["gap"].values
            box_data.append(d)
            box_labels.append(f"{env}\n{suffix}")
            box_colors.append(colors[env])
    bp = ax6.boxplot(box_data, patch_artist=True, showfliers=False, widths=0.6)
    for patch, c in zip(bp["boxes"], box_colors):
        patch.set_facecolor(c); patch.set_alpha(0.55)
    ax6.set_xticklabels(box_labels, rotation=60, ha="right", fontsize=7)
    ax6.axhline(0, ls="--", c="grey", lw=1)
    ax6.set_ylabel("per-episode gap (nats / token)")
    ax6.set_title("First vs last iter — episode-level distributions")
    ax6.grid(alpha=0.3, axis="y")

    fig.suptitle(
        f"Character-prompt absorption — cross-environment summary  "
        f"({len(envs)} envs · {len(df)} episode-evaluations)",
        fontsize=14, y=0.995,
    )
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"wrote {out_path}")
    return s_df


def plot_headline(df: pd.DataFrame, out_path: Path) -> None:
    agg = per_env_iter_agg(df)
    envs = sorted(agg["env"].unique())
    cmap = plt.get_cmap("tab10")
    colors = {e: cmap(i % 10) for i, e in enumerate(envs)}

    fig, ax = plt.subplots(figsize=(11, 6.5))
    for env in envs:
        sub = agg[agg["env"] == env]
        ax.errorbar(sub["iter"], sub["gap"], yerr=sub["gap_se"],
                    marker="o", capsize=3, lw=2.2, color=colors[env],
                    label=f"{env}")
    ax.axhline(0, ls="--", c="grey", lw=1)
    ax.set_xlabel("training iteration")
    ax.set_ylabel("PMI per token (nats):  NLL(no character) − NLL(with character)")
    ax.set_title("Character-prompt absorption across 5 forum environments\n"
                 "(per-character LoRA, Qwen2.5-7B-Instruct)")
    ax.legend(loc="best", fontsize=10)
    ax.grid(alpha=0.3)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"wrote {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Cross-env absorption summary.")
    p.add_argument("absorption_dirs", nargs="+",
                   help="One or more absorption directories.")
    p.add_argument("--out", required=True,
                   help="Output directory for the aggregate CSV + PNGs.")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    dfs = []
    for d in args.absorption_dirs:
        sub = load_abs_dir(Path(d))
        if sub.empty:
            print(f"  WARN: no data in {d}")
            continue
        env = sub["env"].iloc[0]
        print(f"  loaded {len(sub):4d} rows from {env:20s}  ({d})")
        dfs.append(sub)
    if not dfs:
        raise SystemExit("no data loaded")
    df = pd.concat(dfs, ignore_index=True)
    df.to_csv(out_dir / "absorption_all.csv", index=False)
    print(f"wrote {out_dir / 'absorption_all.csv'}  ({len(df)} rows)")

    summary = plot_summary(df, out_dir / "absorption_summary_panels.png")
    plot_headline(df, out_dir / "absorption_summary_headline.png")
    summary.to_csv(out_dir / "absorption_endpoints.csv", index=False)
    print(f"\nendpoint summary:\n{summary.to_string(index=False)}")


if __name__ == "__main__":
    main()
