"""
Visualise character-prompt absorption over training.

Reads per-rank JSONL shards produced by analyze_character_absorption.py and
emits a multi-panel PNG (and a one-panel headline PNG) into the same dir.

Usage
-----
  uv run python scripts/plot_character_absorption.py \\
      runs/cultural_emergence/run7_8agent_7B_margin_notes/absorption/4458450
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_shards(absorption_dir: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(str(absorption_dir / "absorption_rank*.jsonl"))):
        with open(f) as fh:
            rows.extend(json.loads(line) for line in fh if line.strip())
    if not rows:
        raise SystemExit(f"No data in {absorption_dir}")
    df = pd.DataFrame(rows)
    df["pmi_per_token"] = df["mean_nll_without"] - df["mean_nll_with"]
    return df


def aggregate(df: pd.DataFrame, weight: str = "tokens") -> pd.DataFrame:
    """Per (iter, character) aggregate, weighted by n_action_tokens."""
    groups = df.groupby(["iter", "character"])
    if weight == "tokens":
        agg = groups.apply(
            lambda g: pd.Series({
                "mean_nll_with":    np.average(g["mean_nll_with"],    weights=g["n_action_tokens"]),
                "mean_nll_without": np.average(g["mean_nll_without"], weights=g["n_action_tokens"]),
                "gap":              np.average(g["gap"],              weights=g["n_action_tokens"]),
                "gap_se":           g["gap"].std(ddof=1) / np.sqrt(len(g)) if len(g) > 1 else 0.0,
                "n_episodes":       len(g),
                "n_tokens":         int(g["n_action_tokens"].sum()),
            }),
            include_groups=False,
        ).reset_index()
    else:
        agg = groups[["mean_nll_with", "mean_nll_without", "gap"]].mean().reset_index()
    return agg


def plot_all(df: pd.DataFrame, out_path: Path) -> None:
    agg = aggregate(df)
    iters = sorted(df["iter"].unique())
    chars = sorted(df["character"].unique())
    cmap = plt.get_cmap("tab10")
    colors = {c: cmap(i % 10) for i, c in enumerate(chars)}

    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.42, wspace=0.32)

    # ── (1) Headline: gap vs iter, per character + population mean ─────────
    ax1 = fig.add_subplot(gs[0, :2])
    for ch in chars:
        sub = agg[agg["character"] == ch].sort_values("iter")
        ax1.errorbar(sub["iter"], sub["gap"], yerr=sub["gap_se"],
                     marker="o", capsize=3, color=colors[ch], label=ch, alpha=0.85)
    pop = agg.groupby("iter")["gap"].agg(["mean", "sem"]).reset_index()
    ax1.plot(pop["iter"], pop["mean"], "k-", lw=3, label="population mean", zorder=10)
    ax1.fill_between(pop["iter"], pop["mean"] - pop["sem"], pop["mean"] + pop["sem"],
                     color="k", alpha=0.15)
    ax1.axhline(0, ls="--", c="grey", lw=1)
    ax1.set_xlabel("training iteration")
    ax1.set_ylabel("PMI per token (nats):  NLL(no character) − NLL(with character)")
    ax1.set_title("Character-prompt PMI gap over training\n(decreasing → character absorbed into adapter weights)")
    ax1.legend(loc="best", fontsize=8, ncol=2)
    ax1.grid(alpha=0.3)

    # ── (2) NLL with vs without (population means) ─────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    pop_w = df.groupby("iter").apply(
        lambda g: pd.Series({
            "with":    np.average(g["mean_nll_with"],    weights=g["n_action_tokens"]),
            "without": np.average(g["mean_nll_without"], weights=g["n_action_tokens"]),
        }),
        include_groups=False,
    ).reset_index()
    ax2.plot(pop_w["iter"], pop_w["with"],    "o-", label="with character",   color="C0")
    ax2.plot(pop_w["iter"], pop_w["without"], "s-", label="without character", color="C3")
    ax2.set_xlabel("training iteration")
    ax2.set_ylabel("mean NLL per action token (nats)")
    ax2.set_title("Action-token surprise\n(token-weighted population mean)")
    ax2.legend()
    ax2.grid(alpha=0.3)

    # ── (3) Heatmap: PMI gap (iter × character) ────────────────────────────
    ax3 = fig.add_subplot(gs[1, :2])
    pivot = agg.pivot(index="character", columns="iter", values="gap").reindex(chars)
    vmax = float(np.nanmax(np.abs(pivot.values)))
    im = ax3.imshow(pivot.values, aspect="auto", cmap="RdBu_r",
                    vmin=-vmax, vmax=vmax,
                    extent=[min(iters) - 12.5, max(iters) + 12.5, -0.5, len(chars) - 0.5],
                    origin="lower")
    ax3.set_yticks(range(len(chars))); ax3.set_yticklabels(chars)
    ax3.set_xticks(iters)
    ax3.set_xlabel("training iteration"); ax3.set_ylabel("")
    ax3.set_title("PMI gap heat-map (red = prompt still helps; blue = absorbed-or-better)")
    cbar = plt.colorbar(im, ax=ax3, fraction=0.025); cbar.set_label("gap (nats / token)")
    # annotate
    for i, ch in enumerate(chars):
        for it in iters:
            v = pivot.loc[ch, it] if it in pivot.columns else np.nan
            if not np.isnan(v):
                ax3.text(it, i, f"{v:+.2f}", ha="center", va="center",
                         color="black" if abs(v) < 0.5 * vmax else "white", fontsize=7)

    # ── (4) Per-episode gap distributions per iter (strip + box) ───────────
    ax4 = fig.add_subplot(gs[1, 2])
    iter_to_gaps = [df[df["iter"] == it]["gap"].values for it in iters]
    bp = ax4.boxplot(iter_to_gaps, positions=iters, widths=12, patch_artist=True,
                     showfliers=False)
    for patch in bp["boxes"]:
        patch.set_facecolor("lightgrey"); patch.set_alpha(0.6)
    rng = np.random.default_rng(0)
    for it, gaps in zip(iters, iter_to_gaps):
        jitter = rng.uniform(-4, 4, size=len(gaps))
        ax4.scatter(np.full_like(gaps, it, dtype=float) + jitter, gaps,
                    s=15, alpha=0.5, color="C0")
    ax4.axhline(0, ls="--", c="grey", lw=1)
    ax4.set_xlabel("training iteration"); ax4.set_ylabel("per-episode gap (nats/token)")
    ax4.set_title("Per-episode gap distribution")
    ax4.grid(alpha=0.3)

    # ── (5) Per-character small multiples: NLL with/without over iter ──────
    n_chars = len(chars)
    ncols = min(4, n_chars)
    nrows = (n_chars + ncols - 1) // ncols
    inner = gs[2, :].subgridspec(nrows, ncols, hspace=0.55, wspace=0.32)
    for i, ch in enumerate(chars):
        ax = fig.add_subplot(inner[i // ncols, i % ncols])
        sub = agg[agg["character"] == ch].sort_values("iter")
        ax.plot(sub["iter"], sub["mean_nll_with"],    "o-", color="C0", label="with",    ms=4)
        ax.plot(sub["iter"], sub["mean_nll_without"], "s-", color="C3", label="without", ms=4)
        ax.set_title(ch, fontsize=9)
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(fontsize=7, loc="best")
        if i // ncols == nrows - 1:
            ax.set_xlabel("iter")
        if i % ncols == 0:
            ax.set_ylabel("NLL/tok")

    fig.suptitle(
        f"Character-prompt absorption — {Path(df.iloc[0]['run_dir']).name}\n"
        f"{len(df)} episodes · {n_chars} characters · iters {min(iters)}–{max(iters)}",
        fontsize=13, y=0.995,
    )
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"wrote {out_path}")


def plot_headline(df: pd.DataFrame, out_path: Path) -> None:
    """Single-panel version of the top-left plot for quick sharing."""
    agg = aggregate(df)
    chars = sorted(df["character"].unique())
    cmap = plt.get_cmap("tab10")
    colors = {c: cmap(i % 10) for i, c in enumerate(chars)}

    fig, ax = plt.subplots(figsize=(10, 6))
    for ch in chars:
        sub = agg[agg["character"] == ch].sort_values("iter")
        ax.errorbar(sub["iter"], sub["gap"], yerr=sub["gap_se"],
                    marker="o", capsize=3, color=colors[ch], label=ch, alpha=0.8)
    pop = agg.groupby("iter")["gap"].agg(["mean", "sem"]).reset_index()
    ax.plot(pop["iter"], pop["mean"], "k-", lw=3, label="population mean", zorder=10)
    ax.fill_between(pop["iter"], pop["mean"] - pop["sem"], pop["mean"] + pop["sem"],
                    color="k", alpha=0.15)
    ax.axhline(0, ls="--", c="grey", lw=1)
    ax.set_xlabel("training iteration")
    ax.set_ylabel("NLL(no character) − NLL(with character) per action token (nats)")
    ax.set_title(f"Character-prompt PMI gap — {Path(df.iloc[0]['run_dir']).name}")
    ax.legend(loc="best", fontsize=9, ncol=2)
    ax.grid(alpha=0.3)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"wrote {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Plot character-absorption diagnostic.")
    p.add_argument("absorption_dir",
                   help="Dir containing absorption_rank*.jsonl shards.")
    p.add_argument("--out-prefix", default="absorption",
                   help="Filename prefix for the PNGs (default: absorption).")
    args = p.parse_args()

    abs_dir = Path(args.absorption_dir)
    df = load_shards(abs_dir)
    print(f"loaded {len(df)} episode rows from {abs_dir}")
    print(df.groupby("iter")["gap"].describe()[["count", "mean", "std", "min", "max"]])

    plot_all(df, abs_dir / f"{args.out_prefix}_panels.png")
    plot_headline(df, abs_dir / f"{args.out_prefix}_headline.png")
    df.to_csv(abs_dir / f"{args.out_prefix}.csv", index=False)
    print(f"wrote {abs_dir / (args.out_prefix + '.csv')}")


if __name__ == "__main__":
    main()
