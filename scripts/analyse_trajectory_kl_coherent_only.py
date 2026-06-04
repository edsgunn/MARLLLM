"""Diagnostic 1: re-aggregate trajectory-KL with coherent-only filtering.

For each dataset, loads the per-character shard rows, attaches end-of-training
regime to both source and target characters, and recomputes the trajectory
KL using:
 - 'full'             — all characters (matches the existing headline)
 - 'source_coherent'  — only contributions from chars coherent in source pop
 - 'strict_coherent'  — only contributions from chars coherent in BOTH source
                         and target populations
"""
from __future__ import annotations

import argparse
import glob
import json
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
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42,
})

import re
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
    return pd.DataFrame(rows)


def get_regimes(seed_dirs: list[Path]) -> pd.DataFrame:
    """End-of-training regime per (seed_label, character)."""
    rows = []
    for sd in seed_dirs:
        end = end_of_training_regime(sd)
        if end.empty:
            continue
        for _, r in end.iterrows():
            rows.append({"seed": seed_label(str(sd)), "character": r["agent"],
                         "regime": r["regime"], "end_kl": r["kl"]})
    return pd.DataFrame(rows)


def per_traj_kl_with_filter(df: pd.DataFrame, regimes: pd.DataFrame,
                             mode: str) -> pd.DataFrame:
    """Compute per-trajectory KL with regime filtering.

    mode = 'full' | 'source_coherent' | 'strict_coherent'
    Returns per (source, target, iter, episode) row with delta_pertok
    and n_tokens, restricted to per-char contributions matching the filter.
    """
    df = df.copy()
    df["target"] = df["target_run"].apply(seed_label)
    df["source"] = df["source_run"].apply(seed_label)
    # Attach regime labels for source and target chars (same char slot, but
    # the agent in source pop vs target pop can be in different regimes).
    reg = regimes.set_index(["seed", "character"])["regime"].to_dict()
    df["regime_source"] = [reg.get((s, c)) for s, c in zip(df["source"], df["character"])]
    df["regime_target"] = [reg.get((t, c)) for t, c in zip(df["target"], df["character"])]
    # Diagonal per-(source, character, iter, episode) NLL — this is the base
    # for per-character delta_nll under that source. Always available since
    # the analyzer scored each adapter on its own seed's episodes.
    diag = (df[df["source"] == df["target"]]
              [["source", "character", "iter", "episode", "nll_sum",
                "n_action_tokens"]]
              .rename(columns={"nll_sum": "nll_source",
                               "n_action_tokens": "n_tok_source"}))
    df = df.merge(diag, on=["source", "character", "iter", "episode"], how="left")
    df = df.dropna(subset=["nll_source"])
    df["delta_nll"] = df["nll_sum"] - df["nll_source"]
    if mode == "source_coherent":
        df = df[df["regime_source"] == "coherent_low_kl"]
    elif mode == "strict_coherent":
        df = df[(df["regime_source"] == "coherent_low_kl") &
                (df["regime_target"] == "coherent_low_kl")]
    # Sum filtered per-char deltas per trajectory
    grouped = df.groupby(["target", "source", "iter", "episode"]).agg(
        delta_nll=("delta_nll", "sum"),
        n_tokens=("n_action_tokens", "sum"),
        n_chars=("character", "nunique"),
    ).reset_index()
    grouped["delta_pertok"] = grouped["delta_nll"] / grouped["n_tokens"].replace(0, np.nan)
    return grouped.rename(columns={"delta_nll": "delta_nll_total",
                                    "n_tokens": "n_tokens"})


def aggregate_by_iter(merged: pd.DataFrame) -> pd.DataFrame:
    """Cross-pop only, aggregate to per-iter mean."""
    cross = merged[merged["source"] != merged["target"]].copy()
    pair_means = (cross.groupby(["source", "target", "iter"])
                       ["delta_pertok"].mean().reset_index()
                       .rename(columns={"delta_pertok": "pair_mean"}))
    iter_summary = (pair_means.groupby("iter")
                              .agg(mean=("pair_mean", "mean"),
                                   se=("pair_mean", lambda s: s.std(ddof=1)/np.sqrt(len(s))
                                       if len(s)>1 else 0),
                                   n_pairs=("pair_mean", "size"))
                              .reset_index())
    return iter_summary


DATASETS = [
    {
        "name": "Inner Loop 8a",
        "key": "innerloop",
        "shards_dir": "_trajectory_kl/4824410",
        "seed_runs": [
            "run7_8agent_7B_inner_loop",
            "run7_8agent_7B_inner_loop_seed2",
            "run7_8agent_7B_inner_loop_seed3",
            "run7_8agent_7B_inner_loop_seed4",
            "run7_8agent_7B_inner_loop_seed5",
        ],
        "color": "#332288",
    },
    {
        "name": "Margin Notes 8a",
        "key": "marginnotes",
        "shards_dir": "_trajectory_kl/4825302",
        "seed_runs": [
            "run7_8agent_7B_margin_notes",
            "run7_8agent_7B_margin_notes_seed2",
            "run7_8agent_7B_margin_notes_seed3",
            "run7_8agent_7B_margin_notes_seed4",
            "run7_8agent_7B_margin_notes_seed5",
        ],
        "color": "#CC6677",
    },
    {
        "name": "Ashbourne 16a",
        "key": "ashbourne16",
        "shards_dir": "_trajectory_kl/4825303",
        "seed_runs": [
            "run8_16agent_7B_study_group_ashbourne_gc",
            "run8_16agent_7B_study_group_ashbourne_gc_seed2",
            "run8_16agent_7B_study_group_ashbourne_gc_seed3",
            "run8_16agent_7B_study_group_ashbourne_gc_seed4",
            "run8_16agent_7B_study_group_ashbourne_gc_seed5",
        ],
        "color": "#117733",
    },
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="runs/cultural_emergence/_diagnostics_d1_coherent_only")
    args = p.parse_args()
    out_dir = REPO / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    all_summaries = []
    for ds in DATASETS:
        base = REPO / "runs" / "cultural_emergence"
        df = load_shards([str(base / ds["shards_dir"])])
        seed_dirs = [base / r for r in ds["seed_runs"]]
        regimes = get_regimes(seed_dirs)
        regimes.to_csv(out_dir / f"regimes_{ds['key']}.csv", index=False)
        print(f"=== {ds['name']} ===")
        print(f"  shards rows: {len(df)}; regime entries: {len(regimes)}")

        rows = []
        for mode in ("full", "source_coherent", "strict_coherent"):
            merged = per_traj_kl_with_filter(df, regimes, mode)
            summ = aggregate_by_iter(merged)
            summ["mode"] = mode
            summ["dataset"] = ds["name"]
            print(f"  {mode}:")
            print(summ[["iter", "mean", "n_pairs"]].to_string(index=False))
            all_summaries.append(summ)
            rows.append(summ.assign(mode=mode))
        per_ds = pd.concat(rows, ignore_index=True)
        per_ds.to_csv(out_dir / f"summary_{ds['key']}.csv", index=False)

    combined = pd.concat(all_summaries, ignore_index=True)
    combined.to_csv(out_dir / "combined_d1_summary.csv", index=False)

    # Headline figure: full vs strict_coherent per dataset, 3-panel
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8),
                             constrained_layout=True, sharey=False)
    for ax, ds in zip(axes, DATASETS):
        sub = combined[combined["dataset"] == ds["name"]]
        for mode, lstyle, lab in [
            ("full", "-", "all characters"),
            ("source_coherent", "--", "source-coherent only"),
            ("strict_coherent", ":", "strict coherent (both)"),
        ]:
            s = sub[sub["mode"] == mode].sort_values("iter")
            ax.errorbar(s["iter"], s["mean"], yerr=s["se"].fillna(0),
                        marker="o", ms=4, lw=1.4, capsize=3,
                        linestyle=lstyle, color=ds["color"], label=lab)
        ax.axhline(0, color="0.7", lw=0.5, ls=":")
        ax.set_xlabel("training iteration")
        if ax is axes[0]:
            ax.set_ylabel("cross-pop $\\hat{D}$ per token (nats)")
        ax.set_title(ds["name"], fontsize=10)
        ax.yaxis.grid(True, color="0.93", lw=0.5); ax.set_axisbelow(True)
        ax.legend(loc="upper left", frameon=False, fontsize=7)
    fig.suptitle("Diagnostic 1: coherent-only headline trajectory", fontsize=10)
    fig.savefig(out_dir / "fig_d1_coherent_only.pdf", format="pdf",
                bbox_inches="tight", pad_inches=0.04)
    fig.savefig(out_dir / "fig_d1_coherent_only.png", dpi=220,
                bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"[ok] fig_d1_coherent_only")


if __name__ == "__main__":
    main()
