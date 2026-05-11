"""Analyse the 4x4 kl/beta sweep on the 8-agent ashbourne_gc env.

Aggregates per-character metrics across 8 agents per cell and reports:
- early (iters 0-49) vs late (iters 250-299) means for entropy, kl,
  mean_surprise, success_rate, total_loss
- 4x4 heatmaps over (kl_coef, beta) of the late-window means
- per-cell curves for entropy and KL
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("runs/cultural_emergence/kl_beta_sweep_8agent_ashbourne_gc")
OUT = ROOT / "analysis"
OUT.mkdir(exist_ok=True)

KL_VALUES = [0.0, 0.01, 0.1, 0.5]
BETA_VALUES = [0.0, 0.01, 0.1, 0.5]
CHARACTERS = [
    "Priya Shah", "Tom Whitaker", "Hana Yilmaz", "Olu Adeyemi",
    "Beatrice Okafor", "Sam Pritchard", "Imogen Carter", "Marcus Webb",
]
PER_CHAR_METRICS = ["entropy", "kl", "mean_surprise", "act_loss", "perc_loss"]
GLOBAL_METRICS = ["success_rate", "total_loss"]

def fmt(v: float) -> str:
    return str(v).replace(".", "p")

def load_run(kl: float, beta: float) -> pd.DataFrame:
    p = ROOT / f"kl{fmt(kl)}_beta{fmt(beta)}" / "metrics.jsonl"
    rows = [json.loads(l) for l in p.read_text().splitlines()]
    df = pd.DataFrame(rows)
    # average per-char metrics across agents -> bare column name
    for m in PER_CHAR_METRICS:
        cols = [f"{c}/{m}" for c in CHARACTERS if f"{c}/{m}" in df.columns]
        df[f"pop_{m}"] = df[cols].mean(axis=1)
    return df

def window_mean(df: pd.DataFrame, col: str, lo: int, hi: int) -> float:
    sub = df[(df["iteration"] >= lo) & (df["iteration"] < hi)]
    return float(sub[col].mean())

def main() -> None:
    summary_rows = []
    runs: dict[tuple[float, float], pd.DataFrame] = {}
    for kl in KL_VALUES:
        for beta in BETA_VALUES:
            df = load_run(kl, beta)
            runs[(kl, beta)] = df
            row = {"kl_coef": kl, "beta": beta}
            for m in PER_CHAR_METRICS:
                row[f"early_{m}"] = window_mean(df, f"pop_{m}", 0, 50)
                row[f"late_{m}"] = window_mean(df, f"pop_{m}", 250, 300)
            for m in GLOBAL_METRICS:
                if m in df.columns:
                    row[f"early_{m}"] = window_mean(df, m, 0, 50)
                    row[f"late_{m}"] = window_mean(df, m, 250, 300)
            # cross-agent entropy spread late (collapse diagnostic)
            ent_cols = [f"{c}/entropy" for c in CHARACTERS]
            late = df[(df["iteration"] >= 250)][ent_cols]
            row["late_entropy_std_across_agents"] = float(late.mean(axis=0).std())
            summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "summary.csv", index=False)
    print("=== Summary (one row per cell) ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # ---- heatmaps of late-window means ----
    def heatmap(metric_col: str, title: str, fname: str, cmap="viridis") -> None:
        grid = np.zeros((len(KL_VALUES), len(BETA_VALUES)))
        for i, kl in enumerate(KL_VALUES):
            for j, beta in enumerate(BETA_VALUES):
                v = summary[(summary.kl_coef == kl) & (summary.beta == beta)][metric_col].iloc[0]
                grid[i, j] = v
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        im = ax.imshow(grid, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(BETA_VALUES))); ax.set_xticklabels([str(b) for b in BETA_VALUES])
        ax.set_yticks(range(len(KL_VALUES))); ax.set_yticklabels([str(k) for k in KL_VALUES])
        ax.set_xlabel("beta (entropy coef)")
        ax.set_ylabel("kl_coef")
        ax.set_title(title)
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                ax.text(j, i, f"{grid[i,j]:.3g}", ha="center", va="center", color="w", fontsize=8)
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(OUT / fname, dpi=130)
        plt.close(fig)

    heatmap("late_entropy", "Late policy entropy (mean iters 250-299)", "heatmap_late_entropy.png")
    heatmap("late_kl", "Late KL(π||π_ref) (mean iters 250-299)", "heatmap_late_kl.png", cmap="magma")
    heatmap("late_mean_surprise", "Late mean surprise (iters 250-299)", "heatmap_late_surprise.png", cmap="cividis")
    heatmap("late_act_loss", "Late act_loss (iters 250-299)", "heatmap_late_act_loss.png", cmap="magma")
    heatmap("late_perc_loss", "Late perc_loss (iters 250-299)", "heatmap_late_perc_loss.png", cmap="cividis")
    if "late_success_rate" in summary.columns:
        heatmap("late_success_rate", "Late success_rate (iters 250-299)", "heatmap_late_success.png")

    # ---- curves: entropy & kl per cell ----
    fig, axes = plt.subplots(len(KL_VALUES), len(BETA_VALUES), figsize=(14, 10), sharex=True, sharey=False)
    for i, kl in enumerate(KL_VALUES):
        for j, beta in enumerate(BETA_VALUES):
            df = runs[(kl, beta)]
            ax = axes[i, j]
            ax.plot(df["iteration"], df["pop_entropy"], color="tab:blue", lw=1.0, label="entropy")
            ax2 = ax.twinx()
            ax2.plot(df["iteration"], df["pop_kl"], color="tab:red", lw=1.0, label="kl")
            ax.set_title(f"kl={kl}, beta={beta}", fontsize=9)
            if i == len(KL_VALUES) - 1: ax.set_xlabel("iter")
            ax.tick_params(axis="y", labelcolor="tab:blue", labelsize=7)
            ax2.tick_params(axis="y", labelcolor="tab:red", labelsize=7)
    fig.suptitle("Entropy (blue) and KL (red) per cell — 8-agent ashbourne_gc")
    fig.tight_layout()
    fig.savefig(OUT / "curves_entropy_kl_grid.png", dpi=130)
    plt.close(fig)

    # ---- success_rate curves overlaid by beta, faceted by kl ----
    fig, axes = plt.subplots(1, len(KL_VALUES), figsize=(16, 4), sharey=True)
    for i, kl in enumerate(KL_VALUES):
        ax = axes[i]
        for beta in BETA_VALUES:
            df = runs[(kl, beta)]
            if "success_rate" in df:
                ax.plot(df["iteration"], df["success_rate"].rolling(10, min_periods=1).mean(),
                        lw=1.2, label=f"beta={beta}")
        ax.set_title(f"kl={kl}")
        ax.set_xlabel("iter")
        if i == 0: ax.set_ylabel("success_rate (10-iter MA)")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "curves_success_rate.png", dpi=130)
    plt.close(fig)

    print(f"\nWrote: {OUT}/")
    for p in sorted(OUT.iterdir()):
        print(" ", p.name)

if __name__ == "__main__":
    main()
