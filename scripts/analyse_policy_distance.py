"""Aggregate policy-distance + entropy shards into D2 and D3 results."""
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

DATASET_INFO = {
    "innerloop": {
        "name": "Inner Loop 8a", "color": "#332288",
        "seed_runs": [
            "run7_8agent_7B_inner_loop",
            "run7_8agent_7B_inner_loop_seed2",
            "run7_8agent_7B_inner_loop_seed3",
            "run7_8agent_7B_inner_loop_seed4",
            "run7_8agent_7B_inner_loop_seed5",
        ],
    },
    "marginnotes": {
        "name": "Margin Notes 8a", "color": "#CC6677",
        "seed_runs": [
            "run7_8agent_7B_margin_notes",
            "run7_8agent_7B_margin_notes_seed2",
            "run7_8agent_7B_margin_notes_seed3",
            "run7_8agent_7B_margin_notes_seed4",
            "run7_8agent_7B_margin_notes_seed5",
        ],
    },
    "ashbourne16": {
        "name": "Ashbourne 16a", "color": "#117733",
        "seed_runs": [
            "run8_16agent_7B_study_group_ashbourne_gc",
            "run8_16agent_7B_study_group_ashbourne_gc_seed2",
            "run8_16agent_7B_study_group_ashbourne_gc_seed3",
            "run8_16agent_7B_study_group_ashbourne_gc_seed4",
            "run8_16agent_7B_study_group_ashbourne_gc_seed5",
        ],
    },
}

REGIME_COLORS = {
    "coherent_low_kl": "#117733", "committed": "#332288",
    "mid_range": "#DDCC77", "policy_collapse": "#88CCEE",
    "text_degenerate": "#CC6677",
}


def seed_label_from_path(p: str) -> str:
    import re
    name = Path(p).name
    m = re.search(r"_seed(\d+)$", name)
    return f"seed{m.group(1)}" if m else "seed1"


def load_shards(d: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for f in sorted(glob.glob(str(Path(d) / "policy_distance_rank*.jsonl"))):
        with open(f) as fh:
            rows.extend(json.loads(line) for line in fh if line.strip())
    df = pd.DataFrame(rows)
    ent = df[df["type"] == "entropy"].copy()
    kl = df[df["type"] == "sym_kl"].copy()
    return ent, kl


def get_regimes_per_dataset() -> dict[str, pd.DataFrame]:
    base = REPO / "runs" / "cultural_emergence"
    out = {}
    for ds_key, info in DATASET_INFO.items():
        rows = []
        for r in info["seed_runs"]:
            sd = base / r
            end = end_of_training_regime(sd)
            if end.empty:
                continue
            seed = seed_label_from_path(str(sd))
            for _, e in end.iterrows():
                rows.append({"dataset": ds_key, "seed": seed,
                             "character": e["agent"],
                             "regime": e["regime"],
                             "end_kl": e["kl"]})
        out[ds_key] = pd.DataFrame(rows)
    return out


# ------------------------ D2 ------------------------

def diagnostic_2(ent: pd.DataFrame, regimes: dict[str, pd.DataFrame],
                  out_dir: Path):
    """For each (dataset, source_seed, character) compute own-adapter
    entropy vs sibling-adapter entropy at source_seed's own action positions.

    Also pull the per-character kl_pertok from the existing trajectory_kl
    summaries (if available) and produce the scatter plot the spec asks for.
    """
    # Own entropy: rows with eval_source == adapter
    own = ent[ent["eval_source"] == ent["adapter"]].copy()
    own = own.rename(columns={"mean_entropy": "ent_own",
                              "adapter": "source_seed"})
    own_agg = (own.groupby(["dataset", "source_seed", "character", "iter"])
                  .agg(ent_own=("ent_own", "mean"),
                       n_tokens=("n_action_tokens", "mean"),
                       n_episodes=("ent_own", "size"))
                  .reset_index())

    # Sibling entropy: rows where eval_source == agent_seed but adapter != agent_seed
    sib = ent[(ent["eval_source"] != ent["adapter"])].copy()
    sib = sib.rename(columns={"mean_entropy": "ent_sib",
                              "eval_source": "source_seed"})
    sib_agg = (sib.groupby(["dataset", "source_seed", "character", "iter"])
                   .agg(ent_sib=("ent_sib", "mean"),
                        n_sib_obs=("ent_sib", "size"))
                   .reset_index())

    merged = own_agg.merge(sib_agg,
                           on=["dataset", "source_seed", "character", "iter"],
                           how="left")
    merged["ent_diff"] = merged["ent_own"] - merged["ent_sib"]

    # Attach regime
    reg_combined = pd.concat([df.assign(dataset=k) for k, df in regimes.items()
                              if not df.empty], ignore_index=True)
    reg_lookup = reg_combined.set_index(
        ["dataset", "seed", "character"])[["regime", "end_kl"]].rename_axis(
        index={"seed": "source_seed"})
    merged = merged.merge(reg_lookup,
                          on=["dataset", "source_seed", "character"],
                          how="left")
    merged.to_csv(out_dir / "d2_entropy_diff_per_seed_char_iter.csv",
                  index=False)

    # End-of-training row per (dataset, seed, char)
    eot = merged.loc[merged.groupby(["dataset", "source_seed", "character"])
                           ["iter"].idxmax()].copy()
    eot.to_csv(out_dir / "d2_entropy_diff_eot.csv", index=False)

    # Attach kl_pertok at end-of-training from existing summaries
    kvk_lookup = {}
    for ds_key, info in DATASET_INFO.items():
        sub_label = ds_key.replace("innerloop", "summary").replace(
            "marginnotes", "summary_margin_notes").replace(
            "ashbourne16", "summary_ashbourne16")
        # Existing summary CSVs:
        candidates = {
            "innerloop":   "_trajectory_kl_summary/kl_vs_endkl.csv",
            "marginnotes": "_trajectory_kl_summary_margin_notes/kl_vs_endkl.csv",
            "ashbourne16": "_trajectory_kl_summary_ashbourne16/kl_vs_endkl.csv",
        }
        p = REPO / "runs" / "cultural_emergence" / candidates[ds_key]
        if p.exists():
            kvk_lookup[ds_key] = pd.read_csv(p)

    rows = []
    for _, r in eot.iterrows():
        ds = r["dataset"]
        if ds not in kvk_lookup:
            continue
        sub = kvk_lookup[ds]
        match = sub[(sub["source"] == r["source_seed"]) &
                    (sub["character"] == r["character"])]
        if match.empty:
            continue
        rows.append({**r.to_dict(),
                     "kl_pertok": float(match.iloc[0]["kl_pertok"])})
    joined = pd.DataFrame(rows)
    joined.to_csv(out_dir / "d2_joined_kl_pertok_entropy.csv", index=False)

    # Per-regime entropy diff
    if not joined.empty and "regime" in joined.columns:
        per_reg = (joined.groupby("regime")
                          .agg(mean_ent_diff=("ent_diff", "mean"),
                               std_ent_diff=("ent_diff", "std"),
                               n=("ent_diff", "size"))
                          .reset_index())
        per_reg.to_csv(out_dir / "d2_entropy_diff_by_regime.csv", index=False)
        print("\nD2 — per-regime entropy diff (own − sibling) at end of training:")
        print(per_reg.to_string(index=False))

    # Headline scatter: per-(seed, char) kl_pertok vs ent_diff
    fig, ax = plt.subplots(figsize=(5.4, 4.0), constrained_layout=True)
    if not joined.empty:
        for regime, c in REGIME_COLORS.items():
            s = joined[joined["regime"] == regime]
            if s.empty: continue
            ax.scatter(s["kl_pertok"], s["ent_diff"], s=36,
                       facecolor=c, edgecolor="white", linewidth=0.6,
                       alpha=0.85, label=regime.replace("_", " "))
        # Mark extreme outliers
        outliers = joined[joined["kl_pertok"] < -3]
        for _, r in outliers.iterrows():
            ax.annotate(f"{r['source_seed']} {r['character']}",
                        (r["kl_pertok"], r["ent_diff"]),
                        fontsize=6, alpha=0.7,
                        xytext=(4, 0), textcoords="offset points")
    ax.axhline(0, color="0.85", lw=0.5)
    ax.axvline(0, color="0.85", lw=0.5)
    ax.set_xlabel("kl_pertok (per-char cross-pop traj-KL contribution, nats/token)")
    ax.set_ylabel("own − sibling entropy at own samples (nats)")
    ax.set_title("D2: entropy difference vs per-char traj-KL contribution",
                 fontsize=10)
    ax.legend(loc="lower right", frameon=False, fontsize=7)
    ax.yaxis.grid(True, color="0.93", lw=0.5); ax.set_axisbelow(True)
    fig.savefig(out_dir / "fig_d2_scatter.pdf", format="pdf",
                bbox_inches="tight", pad_inches=0.04)
    fig.savefig(out_dir / "fig_d2_scatter.png", dpi=220,
                bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"[ok] fig_d2_scatter")
    return joined


# ------------------------ D3 ------------------------

def diagnostic_3(kl: pd.DataFrame, regimes: dict[str, pd.DataFrame],
                  out_dir: Path):
    """Aggregate symmetric KL to per-(dataset, char, iter, pair) means.
    Then compare with per-char end-KL-from-base."""
    # Pair-level mean over (eval_source, episode)
    pair_agg = (kl.groupby(["dataset", "character", "iter",
                            "adapter_a", "adapter_b"])
                  .agg(mean_sym_kl=("mean_sym_kl", "mean"),
                       n=("mean_sym_kl", "size"))
                  .reset_index())
    pair_agg.to_csv(out_dir / "d3_sym_kl_pairs.csv", index=False)

    # Per-(dataset, character, iter): symmetric "policy distance" averaged
    # across all pairs (the "how distinct is this char slot across seeds")
    per_char = (pair_agg.groupby(["dataset", "character", "iter"])
                         .agg(policy_dist=("mean_sym_kl", "mean"),
                              n_pairs=("mean_sym_kl", "size"))
                         .reset_index())
    per_char.to_csv(out_dir / "d3_policy_dist_per_char.csv", index=False)

    # Per-iter trajectory of policy distance vs trajectory KL
    per_iter = (per_char.groupby(["dataset", "iter"])
                          .agg(mean_pd=("policy_dist", "mean"),
                               se_pd=("policy_dist",
                                      lambda s: s.std(ddof=1)/np.sqrt(len(s))
                                      if len(s)>1 else 0),
                               n_chars=("policy_dist", "size"))
                          .reset_index())
    per_iter.to_csv(out_dir / "d3_policy_dist_per_iter.csv", index=False)
    print("\nD3 — mean policy distance per (dataset, iter):")
    print(per_iter.to_string(index=False))

    # End-of-training merge with regime+end_kl
    reg_combined = pd.concat([df.assign(dataset=k) for k, df in regimes.items()
                              if not df.empty], ignore_index=True)
    # End-iter per dataset
    end_iter = per_char.groupby("dataset")["iter"].max().to_dict()
    rows = []
    for ds_key, end_it in end_iter.items():
        sub = per_char[(per_char["dataset"] == ds_key) &
                       (per_char["iter"] == end_it)]
        for _, r in sub.iterrows():
            reg_rows = reg_combined[
                (reg_combined["dataset"] == ds_key) &
                (reg_combined["character"] == r["character"])]
            for _, rr in reg_rows.iterrows():
                rows.append({"dataset": ds_key, "character": r["character"],
                             "iter": end_it,
                             "policy_dist": r["policy_dist"],
                             "seed": rr["seed"],
                             "regime": rr["regime"],
                             "end_kl": rr["end_kl"]})
    end_df = pd.DataFrame(rows)
    end_df.to_csv(out_dir / "d3_end_policy_dist_with_regime.csv", index=False)

    # Correlation per dataset
    print("\nD3 — correlation: end-KL-from-base vs policy_dist at end of training:")
    for ds_key in DATASET_INFO:
        sub = end_df[end_df["dataset"] == ds_key]
        sub = sub.dropna(subset=["end_kl", "policy_dist"])
        if len(sub) < 3:
            continue
        r = np.corrcoef(sub["end_kl"].astype(float),
                        sub["policy_dist"].astype(float))[0, 1]
        print(f"  {DATASET_INFO[ds_key]['name']:25s}  r={r:.3f}  n={len(sub)}")

    # Plot: trajectory of policy distance per dataset
    fig, ax = plt.subplots(figsize=(6.0, 3.6), constrained_layout=True)
    for ds_key, info in DATASET_INFO.items():
        sub = per_iter[per_iter["dataset"] == ds_key].sort_values("iter")
        if sub.empty: continue
        ax.errorbar(sub["iter"], sub["mean_pd"], yerr=sub["se_pd"].fillna(0),
                    marker="o", ms=5, lw=1.6, capsize=3,
                    color=info["color"], label=info["name"])
    ax.set_xlabel("training iteration")
    ax.set_ylabel("mean per-char policy distance (sym KL, nats/token)")
    ax.set_title("D3: per-character policy distance over training")
    ax.yaxis.grid(True, color="0.93", lw=0.5); ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=False)
    fig.savefig(out_dir / "fig_d3_trajectory.pdf", format="pdf",
                bbox_inches="tight", pad_inches=0.04)
    fig.savefig(out_dir / "fig_d3_trajectory.png", dpi=220,
                bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"[ok] fig_d3_trajectory")

    # Scatter: end-KL vs policy distance, per dataset
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6),
                             constrained_layout=True, sharey=False)
    for ax, (ds_key, info) in zip(axes, DATASET_INFO.items()):
        sub = end_df[end_df["dataset"] == ds_key]
        for regime, c in REGIME_COLORS.items():
            s = sub[sub["regime"] == regime]
            if s.empty: continue
            ax.scatter(s["end_kl"], s["policy_dist"], s=32,
                       facecolor=c, edgecolor="white", linewidth=0.6,
                       alpha=0.85, label=regime.replace("_", " "))
        if not sub.dropna(subset=["end_kl", "policy_dist"]).empty:
            d = sub.dropna(subset=["end_kl", "policy_dist"])
            r = np.corrcoef(d["end_kl"].astype(float),
                            d["policy_dist"].astype(float))[0, 1]
            ax.set_title(f"{info['name']} — r={r:.3f}, n={len(d)}", fontsize=9)
        ax.set_xlabel("end-of-training KL from base (nats)")
        if ax is axes[0]:
            ax.set_ylabel("policy distance (sym KL, nats/token)")
        ax.set_xscale("symlog", linthresh=0.5)
        ax.yaxis.grid(True, color="0.93", lw=0.5); ax.set_axisbelow(True)
    axes[-1].legend(loc="best", frameon=False, fontsize=7)
    fig.suptitle("D3: policy distance vs per-character end-KL-from-base", fontsize=10)
    fig.savefig(out_dir / "fig_d3_scatter.pdf", format="pdf",
                bbox_inches="tight", pad_inches=0.04)
    fig.savefig(out_dir / "fig_d3_scatter.png", dpi=220,
                bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"[ok] fig_d3_scatter")

    return end_df


# ------------------------ cross-diagnostic ------------------------

def cross_diagnostic(d2_joined: pd.DataFrame, d3_end: pd.DataFrame,
                      out_dir: Path):
    """Combine D2 entropy diff + D3 policy dist + trajectory kl_pertok."""
    if d2_joined.empty or d3_end.empty:
        return
    # d2 has per (dataset, source_seed, character) row; d3 has per
    # (dataset, character, seed=for-each-seed) — same char across seeds has
    # one policy_dist value (averaged across pairs)
    merged = d2_joined.merge(
        d3_end[["dataset", "character", "seed", "policy_dist"]],
        left_on=["dataset", "source_seed", "character"],
        right_on=["dataset", "seed", "character"], how="left",
    )
    merged.to_csv(out_dir / "cross_diagnostic.csv", index=False)
    # Plot
    fig, ax = plt.subplots(figsize=(5.6, 4.2), constrained_layout=True)
    for regime, c in REGIME_COLORS.items():
        s = merged[merged["regime"] == regime]
        if s.empty: continue
        # Size encodes |kl_pertok|
        ax.scatter(s["policy_dist"], s["ent_diff"],
                   s=20 + 8*s["kl_pertok"].abs(),
                   facecolor=c, edgecolor="white", linewidth=0.6,
                   alpha=0.85, label=regime.replace("_", " "))
    ax.axhline(0, color="0.85", lw=0.5)
    ax.set_xlabel("D3 policy distance (sym KL nats/token)")
    ax.set_ylabel("D2 entropy diff (own − sibling, nats)")
    ax.set_title("Cross-diagnostic alignment (size ∝ |kl_pertok|)",
                 fontsize=10)
    ax.legend(loc="best", frameon=False, fontsize=7)
    ax.yaxis.grid(True, color="0.93", lw=0.5); ax.set_axisbelow(True)
    fig.savefig(out_dir / "fig_cross_diagnostic.pdf", format="pdf",
                bbox_inches="tight", pad_inches=0.04)
    fig.savefig(out_dir / "fig_cross_diagnostic.png", dpi=220,
                bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"[ok] fig_cross_diagnostic")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("shards_dir")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    ent, kl = load_shards(args.shards_dir)
    print(f"loaded entropy rows: {len(ent)}; sym_kl rows: {len(kl)}")
    ent.to_csv(out_dir / "raw_entropy.csv", index=False)
    kl.to_csv(out_dir / "raw_sym_kl.csv", index=False)

    regimes = get_regimes_per_dataset()
    d2 = diagnostic_2(ent, regimes, out_dir)
    d3 = diagnostic_3(kl, regimes, out_dir)
    cross_diagnostic(d2, d3, out_dir)


if __name__ == "__main__":
    main()
