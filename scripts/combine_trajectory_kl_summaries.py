"""Side-by-side comparison of trajectory-KL across the three measured datasets:
Inner Loop 8-agent, Margin Notes 8-agent, Ashbourne GC 16-agent.
"""
from __future__ import annotations

from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "runs" / "cultural_emergence" / "_trajectory_kl_combined"
OUT.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42,
})

DATASETS = [
    ("Inner Loop 8-agent",
     "_trajectory_kl_summary",
     "#332288"),
    ("Margin Notes 8-agent",
     "_trajectory_kl_summary_margin_notes",
     "#CC6677"),
    ("Ashbourne GC 16-agent",
     "_trajectory_kl_summary_ashbourne16",
     "#117733"),
]

# Trajectory plot
fig, ax = plt.subplots(figsize=(6.8, 4.0), constrained_layout=True)
for label, sub, color in DATASETS:
    df = pd.read_csv(REPO/"runs"/"cultural_emergence"/sub/"pairwise_kl_by_iter.csv")
    cross = df[df.source != df.target]
    agg = (cross.groupby("iter")
                .agg(mean=("mean_pertok", "mean"),
                     se=("mean_pertok", lambda s: s.std(ddof=1)/np.sqrt(len(s))
                         if len(s)>1 else 0),
                     n=("mean_pertok", "size"))
                .reset_index())
    ax.errorbar(agg["iter"], agg["mean"], yerr=agg["se"].fillna(0),
                marker="o", ms=5, lw=1.6, capsize=3, color=color,
                label=f"{label} (max n={agg['n'].max()} pairs)")
ax.axhline(0, color="0.7", lw=0.5, ls=":")
ax.set_xlabel("training iteration")
ax.set_ylabel("cross-population $\\hat{D}$ per token (nats)")
ax.set_title("Trajectory-level cross-population divergence — three datasets")
ax.legend(loc="upper left", frameon=False)
ax.yaxis.grid(True, color="0.93", lw=0.5); ax.set_axisbelow(True)
out_pdf = OUT/"fig_combined_trajectory.pdf"
fig.savefig(out_pdf, format="pdf", bbox_inches="tight", pad_inches=0.04)
fig.savefig(out_pdf.with_suffix(".png"), dpi=220, bbox_inches="tight",
            pad_inches=0.04)
plt.close(fig)
print(f"[ok] {out_pdf.name}")

# end-KL scatter combined
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), constrained_layout=True,
                         sharey=False)
REGIME_COLORS = {"coherent_low_kl": "#117733", "committed": "#332288",
                 "mid_range": "#DDCC77", "policy_collapse": "#88CCEE",
                 "text_degenerate": "#CC6677"}
for ax, (label, sub, _) in zip(axes, DATASETS):
    df = pd.read_csv(REPO/"runs"/"cultural_emergence"/sub/"kl_vs_endkl.csv")
    for regime, color in REGIME_COLORS.items():
        s = df[df.regime == regime]
        if s.empty: continue
        ax.scatter(s.end_kl_from_base, s.kl_pertok, s=28,
                   facecolor=color, edgecolor="white", linewidth=0.5,
                   alpha=0.85, label=regime.replace("_", " "))
    r = np.corrcoef(df.end_kl_from_base.dropna(),
                    df.kl_pertok.dropna())[0,1]
    ax.set_xscale("symlog", linthresh=0.5)
    ax.axhline(0, color="0.85", lw=0.5)
    ax.set_xlabel("end-of-training KL from base (nats)")
    if ax is axes[0]:
        ax.set_ylabel("cross-pop traj-KL per char (nats/token)")
    ax.set_title(f"{label}\nr = {r:.3f},  n = {len(df)}", fontsize=9)
    ax.yaxis.grid(True, color="0.93", lw=0.5); ax.set_axisbelow(True)
axes[-1].legend(loc="lower left", frameon=False, fontsize=7)
out_pdf = OUT/"fig_combined_kl_vs_endkl.pdf"
fig.savefig(out_pdf, format="pdf", bbox_inches="tight", pad_inches=0.04)
fig.savefig(out_pdf.with_suffix(".png"), dpi=220, bbox_inches="tight",
            pad_inches=0.04)
plt.close(fig)
print(f"[ok] {out_pdf.name}")

# Summary table CSV
rows = []
for label, sub, _ in DATASETS:
    df = pd.read_csv(REPO/"runs"/"cultural_emergence"/sub/"pairwise_kl_by_iter.csv")
    cross = df[df.source != df.target]
    kvk = pd.read_csv(REPO/"runs"/"cultural_emergence"/sub/"kl_vs_endkl.csv")
    r = float(np.corrcoef(kvk.end_kl_from_base.dropna(),
                          kvk.kl_pertok.dropna())[0,1])
    for it, g in cross.groupby("iter"):
        rows.append({"dataset": label, "iter": it,
                     "mean_pertok": g.mean_pertok.mean(),
                     "n_pairs": len(g),
                     "min_pertok": g.mean_pertok.min(),
                     "max_pertok": g.mean_pertok.max(),
                     "r_endkl_corr_n": int(kvk.shape[0]),
                     "r_endkl": r})
pd.DataFrame(rows).to_csv(OUT/"combined_summary.csv", index=False)
print(f"[ok] combined_summary.csv")
