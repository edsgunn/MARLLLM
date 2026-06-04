"""Robustness of the absorption / regime contrast to classification thresholds.

Reads the merged absorption-with-regime CSVs (cross-env + inner-loop seeds)
produced by `analyse_absorption_by_regime.py`, which carry the per-agent
contemporaneous metrics (kl_at_iter, perc_loss_at_iter, entropy_at_iter)
plus end-of-training metrics (kl, perc_loss, entropy). Recomputes regime
labels under a sweep of thresholds; also fits the threshold-free
shrinkage-vs-KL regression. No GPU required.

Outputs go to `runs/cultural_emergence/_absorption_threshold_sensitivity/`.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42,
})

REPO = Path(__file__).resolve().parent.parent
DEFAULT_CROSS = REPO / "runs" / "cultural_emergence" / "_absorption_by_regime" / "absorption_cross_with_regime.csv"
DEFAULT_SEEDS = REPO / "runs" / "cultural_emergence" / "_absorption_by_regime" / "absorption_seeds_with_regime.csv"

# Fixed (non-swept) thresholds match the primary values in
# plot_population_regime_figures.py.
DEFAULT_KL  = 1.0
DEFAULT_PC  = 1.7    # coherent boundary
DEFAULT_PD  = 2.0    # text-degenerate boundary
DEFAULT_ENT_LO = 0.2
DEFAULT_ENT_HI = 1.8


def classify_split(kl, perc, ent, kl_thr, perc_coh, perc_deg, ent_lo, ent_hi):
    if perc >= perc_deg:                  return "text_degenerate"
    if ent < ent_lo or ent > ent_hi:      return "policy_collapse"
    if perc < perc_coh and kl <  kl_thr:  return "coherent_low_kl"
    if perc < perc_coh and kl >= kl_thr:  return "committed"
    return "mid_range"


def first_last_shrinkage(df: pd.DataFrame, regime_col: str) -> pd.DataFrame:
    """Mean first-last shrinkage per regime stratum."""
    rows = []
    for regime, g in df.groupby(regime_col):
        iters = sorted(g["iter"].unique())
        if len(iters) < 2:
            continue
        first, last = iters[0], iters[-1]
        gf = g[g["iter"] == first]["gap"].mean()
        gl = g[g["iter"] == last]["gap"].mean()
        rows.append({
            "regime": regime,
            "first_gap": gf, "last_gap": gl,
            "shrinkage": gf - gl,
            "n_agents": g["character"].nunique(),
            "n_instances": g[["run_dir", "character"]].drop_duplicates().shape[0],
        })
    return pd.DataFrame(rows)


# ------------------------ Part A: KL sweep ------------------------

def sweep_kl(df_cross: pd.DataFrame, df_seeds: pd.DataFrame,
             kl_grid: list[float], out_dir: Path):
    rows = []
    for kl_thr in kl_grid:
        for src_name, df in [("cross", df_cross), ("seeds", df_seeds)]:
            d = df.copy()
            d["contemp_regime_swept"] = [
                classify_split(k, p, e,
                               kl_thr, DEFAULT_PC, DEFAULT_PD,
                               DEFAULT_ENT_LO, DEFAULT_ENT_HI)
                for k, p, e in zip(d["kl_at_iter"], d["perc_loss_at_iter"],
                                   d["entropy_at_iter"])
            ]
            stats = first_last_shrinkage(d, "contemp_regime_swept")
            for _, r in stats.iterrows():
                rows.append({"source": src_name, "kl_thr": kl_thr, **r.to_dict()})
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "sweep_kl.csv", index=False)
    print(f"[ok] sweep_kl.csv ({len(out)} rows)")

    # Plot: KL threshold vs shrinkage, separate lines for committed vs coherent.
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2),
                             sharey=False, constrained_layout=True)
    for ax, src in zip(axes, ["cross", "seeds"]):
        sub = out[out["source"] == src]
        for regime, color in [("committed", "#332288"),
                              ("coherent_low_kl", "#117733")]:
            r = sub[sub["regime"] == regime].sort_values("kl_thr")
            ax.plot(r["kl_thr"], r["shrinkage"], "-o", color=color, lw=1.4,
                    ms=4, label=regime.replace("_", " "))
            for _, row in r.iterrows():
                ax.text(row["kl_thr"], row["shrinkage"], f" n={int(row['n_agents'])}",
                        fontsize=6, color=color, va="center")
        # Ratio annotation
        ax_ratio = ax.twinx()
        ratio = []
        for kl in sorted(sub["kl_thr"].unique()):
            r_comm = sub[(sub["regime"] == "committed") & (sub["kl_thr"] == kl)]
            r_coh  = sub[(sub["regime"] == "coherent_low_kl") & (sub["kl_thr"] == kl)]
            if not r_comm.empty and not r_coh.empty and r_coh["shrinkage"].iloc[0] != 0:
                ratio.append((kl, r_comm["shrinkage"].iloc[0] / r_coh["shrinkage"].iloc[0]))
        if ratio:
            xs, ys = zip(*ratio)
            ax_ratio.plot(xs, ys, "s--", color="C3", lw=1.0, ms=4,
                          label="committed/coherent ratio")
            ax_ratio.set_ylabel("committed / coherent shrinkage ratio",
                                color="C3", fontsize=8)
            ax_ratio.tick_params(axis="y", labelcolor="C3", labelsize=7)
            ax_ratio.axhline(1.0, color="C3", ls=":", lw=0.6)
            ax_ratio.axhline(3.0, color="C3", ls=(0, (1, 1.5)), lw=0.6, alpha=0.5)
        ax.axvline(DEFAULT_KL, color="0.5", lw=0.5, ls=(0, (4, 2)),
                   label=f"primary (KL={DEFAULT_KL})")
        ax.set_xlabel("KL threshold (nats)")
        ax.set_ylabel("first − last shrinkage (nats / token)")
        ax.set_title(f"{src}", fontsize=9)
        ax.legend(loc="upper right", frameon=False, fontsize=7)
        ax.yaxis.grid(True, color="0.93", lw=0.5); ax.set_axisbelow(True)
    fig.suptitle("Sensitivity to KL threshold: committed vs coherent absorption shrinkage",
                 fontsize=10)
    fig.savefig(out_dir / "fig_sweep_kl.pdf", format="pdf",
                bbox_inches="tight", pad_inches=0.04)
    fig.savefig(out_dir / "fig_sweep_kl.png", dpi=220,
                bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print("[ok] fig_sweep_kl")


def sweep_perc(df_cross: pd.DataFrame, df_seeds: pd.DataFrame,
               perc_grid: list[float], out_dir: Path):
    rows = []
    for perc_coh in perc_grid:
        for src_name, df in [("cross", df_cross), ("seeds", df_seeds)]:
            d = df.copy()
            d["contemp_regime_swept"] = [
                classify_split(k, p, e,
                               DEFAULT_KL, perc_coh, DEFAULT_PD,
                               DEFAULT_ENT_LO, DEFAULT_ENT_HI)
                for k, p, e in zip(d["kl_at_iter"], d["perc_loss_at_iter"],
                                   d["entropy_at_iter"])
            ]
            stats = first_last_shrinkage(d, "contemp_regime_swept")
            for _, r in stats.iterrows():
                rows.append({"source": src_name, "perc_coh": perc_coh, **r.to_dict()})
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "sweep_perc.csv", index=False)
    print(f"[ok] sweep_perc.csv ({len(out)} rows)")

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.0),
                             sharey=False, constrained_layout=True)
    for ax, src in zip(axes, ["cross", "seeds"]):
        sub = out[out["source"] == src]
        for regime, color in [("committed", "#332288"),
                              ("coherent_low_kl", "#117733"),
                              ("mid_range", "#DDCC77")]:
            r = sub[sub["regime"] == regime].sort_values("perc_coh")
            if r.empty:
                continue
            ax.plot(r["perc_coh"], r["shrinkage"], "-o", color=color, lw=1.4,
                    ms=4, label=regime.replace("_", " "))
        ax.axvline(DEFAULT_PC, color="0.5", lw=0.5, ls=(0, (4, 2)))
        ax.set_xlabel("perception-loss boundary (coherent / mid)")
        ax.set_ylabel("first − last shrinkage")
        ax.set_title(src, fontsize=9)
        ax.yaxis.grid(True, color="0.93", lw=0.5); ax.set_axisbelow(True)
        ax.legend(loc="best", frameon=False, fontsize=7)
    fig.suptitle("Sensitivity to perception-loss threshold", fontsize=10)
    fig.savefig(out_dir / "fig_sweep_perc.pdf", format="pdf",
                bbox_inches="tight", pad_inches=0.04)
    fig.savefig(out_dir / "fig_sweep_perc.png", dpi=220,
                bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print("[ok] fig_sweep_perc")


# ------------------------ Part B: threshold-free ------------------------

def per_agent_endpoints(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (run, ch), g in df.groupby(["run_dir", "character"]):
        iters = sorted(g["iter"].unique())
        if len(iters) < 2:
            continue
        first, last = iters[0], iters[-1]
        gf = g[g["iter"] == first]["gap"].mean()
        gl = g[g["iter"] == last]["gap"].mean()
        rows.append({
            "run_dir": run, "character": ch,
            "shrinkage": gf - gl,
            "end_kl":   g["kl"].iloc[0],
            "end_perc": g["perc_loss"].iloc[0],
            "end_ent":  g["entropy"].iloc[0],
            "regime":   g["regime"].iloc[0],
        })
    return pd.DataFrame(rows)


def ols_fit(x: np.ndarray, y: np.ndarray) -> dict:
    """Simple OLS slope + intercept + R² + 95% CI on slope."""
    n = len(x)
    if n < 3:
        return {}
    xm, ym = x.mean(), y.mean()
    sxx = ((x - xm) ** 2).sum()
    sxy = ((x - xm) * (y - ym)).sum()
    slope = sxy / sxx
    intercept = ym - slope * xm
    yhat = slope * x + intercept
    resid = y - yhat
    ss_res = (resid ** 2).sum()
    ss_tot = ((y - ym) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot else np.nan
    sigma2 = ss_res / (n - 2)
    se_slope = np.sqrt(sigma2 / sxx)
    # t-critical 95% two-sided, df=n-2; use 1.96 as adequate for n large enough
    from math import sqrt
    t_crit = 1.96
    ci = (slope - t_crit * se_slope, slope + t_crit * se_slope)
    return dict(slope=slope, intercept=intercept, r2=r2, se=se_slope,
                ci_lo=ci[0], ci_hi=ci[1], n=n)


def huber_fit(x: np.ndarray, y: np.ndarray, c: float = 1.345,
              n_iter: int = 30) -> dict:
    """IRLS Huber regression on (x, y) with intercept. Returns slope/intercept."""
    n = len(x)
    if n < 3:
        return {}
    X = np.column_stack([np.ones_like(x), x])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    for _ in range(n_iter):
        r = y - X @ beta
        s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-9
        z = r / s
        w = np.where(np.abs(z) <= c, 1.0, c / np.maximum(np.abs(z), 1e-9))
        W = np.diag(w)
        try:
            beta_new = np.linalg.solve(X.T @ W @ X, X.T @ W @ y)
        except np.linalg.LinAlgError:
            break
        if np.allclose(beta_new, beta, atol=1e-6):
            beta = beta_new
            break
        beta = beta_new
    intercept, slope = beta
    return dict(slope=slope, intercept=intercept, n=n)


def threshold_free(df: pd.DataFrame, label: str, out_dir: Path):
    end_df = per_agent_endpoints(df)
    end_df.to_csv(out_dir / f"endpoint_{label}.csv", index=False)

    # Symlog-friendly KL: add a tiny offset for log-x.
    def fit_and_plot(sub_df, suffix, restrict_perc):
        x_kl = sub_df["end_kl"].values.astype(float)
        y    = sub_df["shrinkage"].values.astype(float)
        # log-x regression for cleaner interpretability; clip KL>0 floor.
        x_log = np.log10(np.clip(x_kl, 1e-3, None))
        ols    = ols_fit(x_log, y)
        huber  = huber_fit(x_log, y)
        # Also do linear-x fits for comparison.
        ols_lin   = ols_fit(x_kl, y)
        huber_lin = huber_fit(x_kl, y)
        return dict(ols_logx=ols, huber_logx=huber,
                    ols_linx=ols_lin, huber_linx=huber_lin,
                    n=len(sub_df), suffix=suffix, restrict_perc=restrict_perc)

    full = fit_and_plot(end_df, "all", False)
    coherent_only = fit_and_plot(end_df[end_df["end_perc"] < DEFAULT_PD],
                                 "perc_lt_2", True)

    # Save stats as JSON-ish
    stats_rows = []
    for tag, res in [("all", full), ("perc_lt_2", coherent_only)]:
        for fit_name, r in res.items():
            if fit_name in ("n", "suffix", "restrict_perc"):
                continue
            if not r:
                continue
            stats_rows.append({"subset": tag, "fit": fit_name, **r})
    pd.DataFrame(stats_rows).to_csv(
        out_dir / f"thresholdfree_stats_{label}.csv", index=False)
    print(f"[ok] thresholdfree_stats_{label}.csv")

    # Plot scatter + lines. Two panels: all agents | perc < 2.
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6),
                             sharey=True, constrained_layout=True)
    REGIME_COLORS_LOCAL = {
        "coherent_low_kl": "#117733", "committed": "#332288",
        "mid_range": "#DDCC77", "policy_collapse": "#88CCEE",
        "text_degenerate": "#CC6677",
    }
    for ax, (sub, title, res) in zip(axes, [
        (end_df, "all agents", full),
        (end_df[end_df["end_perc"] < DEFAULT_PD],
         f"perception loss < {DEFAULT_PD}", coherent_only),
    ]):
        for regime, color in REGIME_COLORS_LOCAL.items():
            s = sub[sub["regime"] == regime]
            if s.empty:
                continue
            ax.scatter(s["end_kl"], s["shrinkage"], s=32,
                       facecolor=color, edgecolor="white",
                       linewidth=0.6, alpha=0.9,
                       label=regime.replace("_", " "))
        # Draw OLS and Huber on log axis.
        x_grid_log = np.linspace(
            np.log10(max(1e-3, sub["end_kl"].min())),
            np.log10(max(1e-3, sub["end_kl"].max())), 50)
        if res["ols_logx"]:
            r = res["ols_logx"]
            ax.plot(10 ** x_grid_log,
                    r["slope"] * x_grid_log + r["intercept"],
                    "-", color="black", lw=1.2,
                    label=f"OLS log10 KL: slope={r['slope']:.2f} "
                          f"[{r['ci_lo']:.2f},{r['ci_hi']:.2f}], R²={r['r2']:.2f}")
        if res["huber_logx"]:
            r = res["huber_logx"]
            ax.plot(10 ** x_grid_log,
                    r["slope"] * x_grid_log + r["intercept"],
                    "--", color="0.4", lw=1.0,
                    label=f"Huber log10 KL: slope={r['slope']:.2f}")
        ax.axhline(0, color="0.7", lw=0.5, ls=":")
        ax.axvline(DEFAULT_KL, color="0.6", lw=0.5, ls=(0, (4, 2)))
        ax.set_xscale("symlog", linthresh=0.1)
        ax.set_xlabel("end-of-training KL from reference (nats)")
        ax.set_ylabel("PMI shrinkage (first − last)")
        ax.set_title(title, fontsize=9)
        ax.yaxis.grid(True, color="0.93", lw=0.5); ax.set_axisbelow(True)
        ax.legend(loc="best", frameon=False, fontsize=6.5)
    fig.suptitle(f"Threshold-free: PMI shrinkage scales with KL movement ({label})",
                 fontsize=10)
    fig.savefig(out_dir / f"fig_thresholdfree_{label}.pdf", format="pdf",
                bbox_inches="tight", pad_inches=0.04)
    fig.savefig(out_dir / f"fig_thresholdfree_{label}.png", dpi=220,
                bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"[ok] fig_thresholdfree_{label}")


# ------------------------ main ------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cross", default=str(DEFAULT_CROSS))
    p.add_argument("--seeds", default=str(DEFAULT_SEEDS))
    p.add_argument("--out", required=True)
    args = p.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    cross = pd.read_csv(args.cross)
    seeds = pd.read_csv(args.seeds)
    print(f"loaded cross: {len(cross)} rows; seeds: {len(seeds)} rows")

    # Part A sweeps.
    sweep_kl(cross, seeds,
             kl_grid=[0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
             out_dir=out_dir)
    sweep_perc(cross, seeds,
               perc_grid=[1.5, 1.6, 1.7, 1.8, 1.9, 2.0],
               out_dir=out_dir)

    # Part B threshold-free fits.
    threshold_free(cross, "cross", out_dir)
    threshold_free(seeds, "seeds", out_dir)


if __name__ == "__main__":
    main()
