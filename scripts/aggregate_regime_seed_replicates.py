"""Cross-seed aggregation of population regime outcomes, for the rebuttal.

Addresses the reviewer concern that Experiment 1 used one population per
(pop size, substrate) condition. For every condition where we have independent
seed replicates (identical config + roster, training seed varied), this script:

  1. classifies every agent in every replicate with the *same* thresholds and
     code path as the submitted figures (imported from
     plot_population_regime_figures.py, not re-implemented);
  2. treats the population (seed) as the unit of analysis and reports
     regime-composition mean +/- SD and 95% t-CI across seeds;
  3. reports population-level binary outcomes (e.g. "contains >=1 committed
     agent") with exact Clopper-Pearson 95% CIs;
  4. quantifies within-population dependence via one-way ICC(1) and the
     implied design effect / effective sample size for each metric;
  5. measures per-character regime consistency across seeds (Fleiss kappa) --
     with the roster held fixed, this separates "roster composition" from
     seed-level randomness;
  6. sweeps the classification thresholds to show the headline fractions are
     not artifacts of the specific cutoffs;
  7. reports per-agent classification agreement between the common evaluation
     iteration and each run's final iteration (timepoint robustness).

Outputs (CSVs, figures, SUMMARY.md) go to
runs/cultural_emergence/_seed_replicate_regimes/ by default.

Usage:
    .venv/bin/python scripts/aggregate_regime_seed_replicates.py [--out-dir DIR]
"""

from __future__ import annotations

import argparse
import itertools
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parent))
# Import the exact thresholds, palette, rcParams, and loaders used for the
# submitted figures so nothing here can silently diverge from the paper.
from plot_population_regime_figures import (  # noqa: E402
    CAT_COLORS,
    CAT_LABELS,
    CATS,
    ENT_HIGH,
    ENT_LOW,
    KL_THR,
    PERC_COHERENT,
    PERC_DEGEN,
    RUNS,
    classify,
    load_iteration,
)

import matplotlib.pyplot as plt  # noqa: E402  (after rcParams set on import)

ALIVE_CATS = {"coherent_low_kl", "committed", "mid_range"}


@dataclass
class ReplicateSet:
    label: str            # short name used in filenames / tables
    title: str            # display title
    pop_size: int
    runs: dict[int, str]  # seed index (1 = unsuffixed base) -> run dir name
    extra_iters: list[int] = field(default_factory=list)  # robustness timepoints


REPLICATE_SETS: list[ReplicateSet] = [
    ReplicateSet(
        label="ashbourne16",
        title="Study Group (Ashbourne)\n16 agents",
        pop_size=16,
        runs={
            1: "run8_16agent_7B_study_group_ashbourne_gc",
            2: "run8_16agent_7B_study_group_ashbourne_gc_seed2",
            3: "run8_16agent_7B_study_group_ashbourne_gc_seed3",
            4: "run8_16agent_7B_study_group_ashbourne_gc_seed4",
            5: "run8_16agent_7B_study_group_ashbourne_gc_seed5",
        },
    ),
    ReplicateSet(
        label="inner_loop8",
        title="Inner Loop\n8 agents",
        pop_size=8,
        runs={
            1: "run7_8agent_7B_inner_loop",
            2: "run7_8agent_7B_inner_loop_seed2",
            3: "run7_8agent_7B_inner_loop_seed3",
            4: "run7_8agent_7B_inner_loop_seed4",
            5: "run7_8agent_7B_inner_loop_seed5",
        },
        extra_iters=[100],
    ),
    ReplicateSet(
        label="margin_notes8",
        title="Margin Notes\n8 agents",
        pop_size=8,
        runs={
            1: "run7_8agent_7B_margin_notes",
            2: "run7_8agent_7B_margin_notes_seed2",
            3: "run7_8agent_7B_margin_notes_seed3",
            4: "run7_8agent_7B_margin_notes_seed4",
            5: "run7_8agent_7B_margin_notes_seed5",
        },
        extra_iters=[150],
    ),
]

SYSTEM_AGENTS = {"cpu", "gen", "mem", "env_episodes"}


# ---------------- loading ----------------

def last_logged_iter(run_dir: Path) -> int:
    it, _ = load_iteration(run_dir, None)
    return it


def load_set_table(rset: ReplicateSet, target_iter: int | None) -> pd.DataFrame:
    """Per-agent classification table for one replicate set at target_iter.

    target_iter=None means each run's final logged iteration.
    Runs whose last logged iteration is below target_iter are skipped (they
    would otherwise be silently classified at an earlier timepoint)."""
    rows = []
    for seed, run_name in sorted(rset.runs.items()):
        run_dir = RUNS / run_name
        if target_iter is not None and last_logged_iter(run_dir) < target_iter:
            continue
        iter_num, agents = load_iteration(run_dir, target_iter)
        for agent, m in agents.items():
            if agent in SYSTEM_AGENTS:
                continue
            kl, perc, ent = m.get("kl"), m.get("perc_loss"), m.get("entropy")
            if kl is None or perc is None or ent is None:
                continue
            rows.append({
                "set": rset.label,
                "run": run_name,
                "seed": seed,
                "pop_size": rset.pop_size,
                "agent": agent,
                "iter": iter_num,
                "kl": kl,
                "perc_loss": perc,
                "entropy": ent,
                "category": classify(kl, perc, ent),
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        expected = {rset.pop_size}
        counts = set(df.groupby("seed").size())
        if counts != expected:
            print(f"[warn] {rset.label}: per-seed agent counts {counts} != {expected}")
    return df


# ---------------- statistics ----------------

def t_ci(vals: np.ndarray, conf: float = 0.95) -> tuple[float, float, float]:
    """(mean, lo, hi) t-interval; degenerate for n<2."""
    vals = np.asarray(vals, dtype=float)
    n = len(vals)
    m = float(vals.mean())
    if n < 2:
        return m, m, m
    half = sps.t.ppf(0.5 + conf / 2, n - 1) * vals.std(ddof=1) / np.sqrt(n)
    return m, m - half, m + half


def clopper_pearson(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    a = 1 - conf
    lo = sps.beta.ppf(a / 2, k, n - k + 1) if k > 0 else 0.0
    hi = sps.beta.ppf(1 - a / 2, k + 1, n - k) if k < n else 1.0
    return float(lo), float(hi)


def composition_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per-seed regime fractions plus cross-seed mean/sd/95% CI rows."""
    seeds = sorted(df["seed"].unique())
    frac = {
        s: df[df["seed"] == s]["category"].value_counts(normalize=True)
        for s in seeds
    }
    out = pd.DataFrame(
        {s: [frac[s].get(c, 0.0) for c in CATS] for s in seeds}, index=CATS
    ).T
    out.index.name = "seed"
    out["alive"] = out[list(ALIVE_CATS)].sum(axis=1)
    stats_rows = {}
    for col in out.columns:
        m, lo, hi = t_ci(out[col].values)
        stats_rows[col] = {
            "mean": m, "sd": out[col].std(ddof=1),
            "ci_lo": max(lo, 0.0), "ci_hi": min(hi, 1.0),
        }
    stats_df = pd.DataFrame(stats_rows).T
    return out, stats_df


def population_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    """Binary population-level outcomes with exact 95% CIs across seeds."""
    seeds = sorted(df["seed"].unique())
    n = len(seeds)

    def per_seed(fn) -> int:
        return sum(bool(fn(df[df["seed"] == s])) for s in seeds)

    outcomes = {
        ">=1 committed agent":
            per_seed(lambda g: (g["category"] == "committed").any()),
        ">=1 coherent (low-KL) agent":
            per_seed(lambda g: (g["category"] == "coherent_low_kl").any()),
        "majority alive (>50%)":
            per_seed(lambda g: (g["category"].isin(ALIVE_CATS)).mean() > 0.5),
        ">=1 policy-collapse agent":
            per_seed(lambda g: (g["category"] == "policy_collapse").any()),
        ">=1 text-degenerate agent":
            per_seed(lambda g: (g["category"] == "text_degenerate").any()),
        "whole population collapsed/degenerate":
            per_seed(lambda g: (~g["category"].isin(ALIVE_CATS)).all()),
    }
    rows = []
    for name, k in outcomes.items():
        lo, hi = clopper_pearson(k, n)
        rows.append({"outcome": name, "k": k, "n": n,
                     "fraction": k / n, "ci95_lo": lo, "ci95_hi": hi})
    return pd.DataFrame(rows)


def icc_oneway(df: pd.DataFrame, metric: str) -> dict:
    """ICC(1) from one-way ANOVA with seed as the grouping factor, plus the
    design effect for a cluster of size m (the population size)."""
    groups = [g[metric].values for _, g in df.groupby("seed")]
    k = len(groups)
    m = int(np.mean([len(g) for g in groups]))
    grand = np.concatenate(groups).mean()
    n_tot = sum(len(g) for g in groups)
    ssb = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ssw = sum(((g - g.mean()) ** 2).sum() for g in groups)
    msb = ssb / (k - 1)
    msw = ssw / (n_tot - k)
    icc = (msb - msw) / (msb + (m - 1) * msw) if (msb + (m - 1) * msw) > 0 else 0.0
    icc = max(icc, 0.0)
    deff = 1 + (m - 1) * icc
    return {
        "metric": metric, "icc1": icc, "design_effect": deff,
        "n_agents": n_tot, "n_populations": k,
        "effective_n": n_tot / deff,
        "F": msb / msw if msw > 0 else np.inf,
    }


def fleiss_kappa(df: pd.DataFrame) -> tuple[float, float]:
    """Fleiss kappa treating characters as items and seeds as raters.

    Returns (kappa, mean modal agreement). Only valid when the roster is
    shared across seeds (it is, within a replicate set)."""
    pivot = df.pivot_table(index="agent", columns="seed", values="category",
                           aggfunc="first")
    pivot = pivot.dropna()
    n_raters = pivot.shape[1]
    counts = np.zeros((len(pivot), len(CATS)))
    for i, (_, row) in enumerate(pivot.iterrows()):
        for c in row.values:
            counts[i, CATS.index(c)] += 1
    p_i = ((counts ** 2).sum(axis=1) - n_raters) / (n_raters * (n_raters - 1))
    p_bar = p_i.mean()
    p_j = counts.sum(axis=0) / counts.sum()
    p_e = (p_j ** 2).sum()
    kappa = (p_bar - p_e) / (1 - p_e) if p_e < 1 else np.nan
    modal = (counts.max(axis=1) / n_raters).mean()
    return float(kappa), float(modal)


def threshold_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-seed mean alive/committed fractions under perturbed thresholds."""
    rows = []
    for kl_thr, perc_coh in itertools.product(
            [0.5, 0.75, 1.0, 1.25, 1.5], [1.6, 1.7, 1.8]):
        def cls(r):
            if r["perc_loss"] >= PERC_DEGEN:
                return "text_degenerate"
            if r["entropy"] < ENT_LOW or r["entropy"] > ENT_HIGH:
                return "policy_collapse"
            if r["perc_loss"] < perc_coh and r["kl"] < kl_thr:
                return "coherent_low_kl"
            if r["perc_loss"] < perc_coh:
                return "committed"
            return "mid_range"
        cats = df.apply(cls, axis=1)
        per_seed_alive = df.assign(c=cats).groupby("seed")["c"] \
            .apply(lambda s: s.isin(ALIVE_CATS).mean())
        per_seed_comm = df.assign(c=cats).groupby("seed")["c"] \
            .apply(lambda s: (s == "committed").mean())
        rows.append({
            "kl_thr": kl_thr, "perc_coherent": perc_coh,
            "alive_mean": per_seed_alive.mean(), "alive_sd": per_seed_alive.std(ddof=1),
            "committed_mean": per_seed_comm.mean(), "committed_sd": per_seed_comm.std(ddof=1),
            "is_paper_thresholds": (kl_thr == KL_THR and perc_coh == PERC_COHERENT),
        })
    return pd.DataFrame(rows)


def timepoint_agreement(rset: ReplicateSet, common_df: pd.DataFrame) -> pd.DataFrame:
    """Per-run agreement between classification at the common iteration and at
    that run's final logged iteration."""
    final_df = load_set_table(rset, None)
    merged = common_df.merge(
        final_df[["seed", "agent", "iter", "category"]],
        on=["seed", "agent"], suffixes=("_common", "_final"))
    rows = []
    for seed, g in merged.groupby("seed"):
        rows.append({
            "seed": seed,
            "iter_common": int(g["iter_common"].iloc[0]),
            "iter_final": int(g["iter_final"].iloc[0]),
            "n_agents": len(g),
            "agreement": (g["category_common"] == g["category_final"]).mean(),
        })
    return pd.DataFrame(rows)


# ---------------- figures ----------------

def fig_composition_by_seed(tables: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
                            titles: dict[str, str],
                            out_path: Path) -> None:
    """Stacked regime bars per seed, plus a cross-seed mean bar, one panel per
    replicate set. Mirrors the style of the submitted fig2."""
    labels = list(tables.keys())
    fig, axes = plt.subplots(1, len(labels), figsize=(5.5, 2.9), sharey=True)
    fig.subplots_adjust(left=0.09, right=0.99, top=0.86, bottom=0.30, wspace=0.08)
    if len(labels) == 1:
        axes = [axes]

    for ax, lab in zip(axes, labels):
        per_seed, cstats = tables[lab]
        seeds = list(per_seed.index)
        x = np.arange(len(seeds) + 1)
        width = 0.62
        bottom = np.zeros(len(x))
        for cat in CATS:
            fracs = np.array([per_seed.loc[s, cat] for s in seeds]
                             + [cstats.loc[cat, "mean"]])
            ax.bar(x, fracs, width, bottom=bottom, label=CAT_LABELS[cat],
                   color=CAT_COLORS[cat], edgecolor="white", linewidth=0.6)
            bottom += fracs
        # 95% CI whisker on the mean bar's alive fraction boundary
        alive_mean = cstats.loc["alive", "mean"]
        ax.errorbar(x[-1], alive_mean,
                    yerr=[[alive_mean - cstats.loc["alive", "ci_lo"]],
                          [cstats.loc["alive", "ci_hi"] - alive_mean]],
                    fmt="none", ecolor="black", elinewidth=0.9, capsize=2.5,
                    zorder=5)
        ax.axvline(len(seeds) - 0.5, color="0.75", lw=0.6, ls=":")
        ax.set_xticks(x)
        ax.set_xticklabels([f"s{s}" for s in seeds] + ["mean"], fontsize=7.5)
        ax.set_ylim(0, 1.02)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_title(titles[lab], pad=4, fontsize=8.5)
        ax.tick_params(axis="x", length=0)
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color="0.92", lw=0.5, zorder=0)
        ax.set_xlabel("population (seed)")

    axes[0].set_ylabel("fraction of agents")
    axes[0].set_yticklabels(["0", "0.25", "0.5", "0.75", "1"])
    handles, leg_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, leg_labels, loc="lower center",
               bbox_to_anchor=(0.5, 0.0), ncol=len(CATS), frameon=False,
               handlelength=1.2, columnspacing=1.4, handletextpad=0.4,
               fontsize=6.8)
    fig.savefig(out_path, format="pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(out_path.with_suffix(".png"), format="png", dpi=220,
                bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"[ok] wrote {out_path} (+ .png)")


SEED_MARKERS = {1: "o", 2: "s", 3: "D", 4: "^", 5: "v"}


def fig_scatter_all_seeds(dfs: dict[str, pd.DataFrame],
                          titles: dict[str, str],
                          out_path: Path) -> None:
    """KL vs perception-loss scatter with all seeds pooled, one panel per set.
    Same axes/regions as the submitted fig1; marker shape encodes seed."""
    labels = list(dfs.keys())
    fig, axes = plt.subplots(1, len(labels), figsize=(5.5, 2.6), sharey=True)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.86, bottom=0.30, wspace=0.08)
    if len(labels) == 1:
        axes = [axes]
    x_max = 8.5
    y_min, y_max = 1.10, 3.35

    for ax, lab in zip(axes, labels):
        df = dfs[lab]
        ax.axhspan(PERC_DEGEN, y_max, facecolor=CAT_COLORS["text_degenerate"],
                   alpha=0.10, zorder=0)
        ax.axhspan(PERC_COHERENT, PERC_DEGEN, facecolor=CAT_COLORS["mid_range"],
                   alpha=0.13, zorder=0)
        ax.fill_between([-0.5, KL_THR], y_min, PERC_COHERENT,
                        facecolor=CAT_COLORS["coherent_low_kl"], alpha=0.10, zorder=0)
        ax.fill_between([KL_THR, x_max + 0.5], y_min, PERC_COHERENT,
                        facecolor=CAT_COLORS["committed"], alpha=0.10, zorder=0)
        ax.axvline(KL_THR, color="0.35", lw=0.7, ls=(0, (4, 2)), zorder=1)
        ax.axhline(PERC_COHERENT, color="0.35", lw=0.7, ls=(0, (4, 2)), zorder=1)
        ax.axhline(PERC_DEGEN, color="0.35", lw=0.7, ls=(0, (1.5, 1.5)), zorder=1)
        for _, r in df.iterrows():
            ax.scatter(min(r["kl"], x_max), r["perc_loss"],
                       marker=SEED_MARKERS.get(int(r["seed"]), "x"), s=26,
                       facecolor="0.45",
                       edgecolor=CAT_COLORS.get(r["category"], "k"),
                       linewidth=0.9, alpha=0.9, zorder=3)
            if r["kl"] > x_max:
                ax.annotate("", xy=(x_max + 0.1, r["perc_loss"]),
                            xytext=(x_max - 0.4, r["perc_loss"]),
                            arrowprops=dict(arrowstyle="->", color="black", lw=0.6),
                            annotation_clip=False)
        ax.set_xlim(-0.3, x_max + 0.2)
        ax.set_ylim(y_min, y_max)
        ax.set_title(titles[lab], pad=4, fontsize=8.5)
        ax.set_xlabel("KL from reference (nats)")
    axes[0].set_ylabel("Perception loss")

    seed_handles = [
        plt.Line2D([], [], marker=m, color="w", markerfacecolor="0.45",
                   markeredgecolor="white", markeredgewidth=0.5, markersize=5.5,
                   label=f"seed {s}")
        for s, m in SEED_MARKERS.items()
    ]
    cat_handles = [
        plt.Line2D([], [], marker="o", color="w", markerfacecolor="0.85",
                   markeredgecolor=CAT_COLORS[c], markeredgewidth=0.9,
                   markersize=5.5, label=CAT_LABELS[c])
        for c in CATS
    ]
    fig.legend(handles=seed_handles + cat_handles, loc="lower center",
               bbox_to_anchor=(0.5, 0.0), ncol=5, frameon=False,
               handlelength=1.0, columnspacing=1.0, handletextpad=0.3,
               fontsize=6.5)
    fig.savefig(out_path, format="pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(out_path.with_suffix(".png"), format="png", dpi=220,
                bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"[ok] wrote {out_path} (+ .png)")


# ---------------- main ----------------

def fmt_pct(x: float) -> str:
    return f"{100 * x:.0f}\\%" if False else f"{100 * x:.0f}%"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path,
                    default=RUNS / "_seed_replicate_regimes")
    args = ap.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()

    all_agents = []
    comp_tables: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    scatter_dfs: dict[str, pd.DataFrame] = {}
    titles: dict[str, str] = {}
    summary_lines: list[str] = [
        f"# Cross-seed regime replication summary ({today})\n",
        f"Classification thresholds identical to the submitted figures: "
        f"KL={KL_THR}, perc_coherent={PERC_COHERENT}, perc_degen={PERC_DEGEN}, "
        f"entropy band=[{ENT_LOW}, {ENT_HIGH}].\n",
    ]

    for rset in REPLICATE_SETS:
        common_iter = min(last_logged_iter(RUNS / r) for r in rset.runs.values())
        df = load_set_table(rset, common_iter)
        if df.empty:
            print(f"[warn] no data for {rset.label}")
            continue
        n_seeds = df["seed"].nunique()
        titles[rset.label] = rset.title
        scatter_dfs[rset.label] = df
        all_agents.append(df)

        per_seed, cstats = composition_table(df)
        comp_tables[rset.label] = (per_seed, cstats)
        outcomes = population_outcomes(df)
        iccs = pd.DataFrame([icc_oneway(df, m) for m in
                             ("kl", "perc_loss", "entropy")])
        kappa, modal = fleiss_kappa(df)
        sens = threshold_sensitivity(df)
        agree = timepoint_agreement(rset, df)

        per_seed.to_csv(out / f"{rset.label}_composition_per_seed.csv")
        cstats.to_csv(out / f"{rset.label}_composition_cross_seed.csv")
        outcomes.to_csv(out / f"{rset.label}_population_outcomes.csv", index=False)
        iccs.to_csv(out / f"{rset.label}_icc.csv", index=False)
        sens.to_csv(out / f"{rset.label}_threshold_sensitivity.csv", index=False)
        agree.to_csv(out / f"{rset.label}_timepoint_agreement.csv", index=False)

        # Robustness at extra timepoints (subset of seeds may qualify)
        extra_txt = []
        for it in rset.extra_iters:
            edf = load_set_table(rset, it)
            if edf.empty:
                continue
            eps, ecs = composition_table(edf)
            eps.to_csv(out / f"{rset.label}_composition_per_seed_iter{it}.csv")
            extra_txt.append(
                f"  - at iter {it} (n={edf['seed'].nunique()} seeds): alive "
                f"{ecs.loc['alive', 'mean']:.2f} ± {ecs.loc['alive', 'sd']:.2f} (sd)")

        a = cstats.loc["alive"]
        c = cstats.loc["committed"]
        sens_alive = sens["alive_mean"]
        summary_lines += [
            f"\n## {rset.title.replace(chr(10), ', ')}  ({n_seeds} independent populations, "
            f"evaluated at iteration {common_iter})\n",
            f"- Alive fraction (coherent+committed+mid-range): "
            f"mean {a['mean']:.2f} ± {a['sd']:.2f} (sd), "
            f"95% CI [{a['ci_lo']:.2f}, {a['ci_hi']:.2f}] across seeds.",
            f"- Committed fraction: mean {c['mean']:.2f} ± {c['sd']:.2f} (sd), "
            f"95% CI [{c['ci_lo']:.2f}, {c['ci_hi']:.2f}].",
            "- Population-level outcomes (k/n populations, exact 95% CI):",
        ]
        for _, r in outcomes.iterrows():
            summary_lines.append(
                f"    - {r['outcome']}: {int(r['k'])}/{int(r['n'])} "
                f"[{r['ci95_lo']:.2f}, {r['ci95_hi']:.2f}]")
        summary_lines += [
            "- Within-population dependence (one-way ICC across seeds): " +
            ", ".join(f"{r['metric']} ICC={r['icc1']:.2f} "
                      f"(design effect {r['design_effect']:.1f}, "
                      f"effective n {r['effective_n']:.0f}/{int(r['n_agents'])})"
                      for _, r in iccs.iterrows()),
            f"- Per-character regime consistency across seeds (roster fixed): "
            f"Fleiss kappa = {kappa:.2f}, mean modal agreement = {modal:.2f}.",
            f"- Threshold sensitivity (KL in [0.5,1.5], perc in [1.6,1.8]): "
            f"cross-seed alive mean spans "
            f"[{sens_alive.min():.2f}, {sens_alive.max():.2f}].",
            f"- Timepoint robustness: common-iter vs final-iter per-agent "
            f"agreement " +
            ", ".join(f"s{int(r['seed'])}={r['agreement']:.2f}"
                      for _, r in agree.iterrows()) + ".",
        ]
        summary_lines += extra_txt

        print(f"\n=== {rset.title.replace(chr(10), ', ')} "
              f"(common iter {common_iter}, {n_seeds} seeds) ===")
        print(per_seed.round(3).to_string())
        print(cstats.round(3).to_string())
        print(outcomes.round(3).to_string(index=False))
        print(iccs.round(3).to_string(index=False))
        print(f"Fleiss kappa={kappa:.3f} modal agreement={modal:.3f}")
        print(agree.round(3).to_string(index=False))

    big = pd.concat(all_agents, ignore_index=True)
    big.to_csv(out / f"agent_classifications_all_seeds_{today}.csv", index=False)
    print(f"\n[ok] wrote per-agent table ({len(big)} agents, "
          f"{big.groupby('set')['seed'].nunique().to_dict()} seeds per set)")

    fig_composition_by_seed(comp_tables, titles,
                            out / f"figS_regime_by_seed_{today}.pdf")
    fig_scatter_all_seeds(scatter_dfs, titles,
                          out / f"figS_scatter_all_seeds_{today}.pdf")

    (out / "SUMMARY.md").write_text("\n".join(summary_lines) + "\n")
    print(f"[ok] wrote {out / 'SUMMARY.md'}")


if __name__ == "__main__":
    main()
