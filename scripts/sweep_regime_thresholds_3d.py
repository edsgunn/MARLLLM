"""Full threshold sweep + clustering alternative for the regime taxonomy.

Addresses the reviewer request for threshold sensitivity of the
KL / perception-loss / entropy regime labels. Where the earlier
analyse_threshold_sensitivity.py swept KL and perception loss one at a time,
this script:

  1. sweeps the *joint* grid over all five classification parameters
     (kl_thr, perc_coherent, perc_degen, ent_lo, ent_hi);
  2. for every grid point recomputes
       a. per-agent regime labels for the submitted Figure-1 agents (iter 200)
          and for the 15 seed-replicate populations, giving a label-stability
          distribution per agent (uncertainty over labels);
       b. regime composition (alive / committed fractions) per condition;
       c. the headline committed-vs-coherent PMI shrinkage contrast, pooled
          cross-env and pooled inner-loop seeds (end-of-training labels, same
          pooling as the submitted analysis);
  3. runs an unsupervised alternative: k-means and Gaussian-mixture clustering
     on standardised (log10 KL, perception loss, entropy) with no thresholds,
     reports agreement with the taxonomy (ARI) and re-tests the PMI contrast
     using cluster labels instead of threshold labels;
  4. splits `policy_collapse` into its low-entropy and high-entropy branches
     to quantify how much of that regime is actually near-deterministic.

Inputs (all pre-existing CSVs; no GPU):
  - _population_figures/classification_table_iter200_2026-05-07.csv
  - _seed_replicate_regimes/agent_classifications_all_seeds_*.csv
  - _absorption_threshold_sensitivity/endpoint_{cross,seeds}.csv

Outputs to runs/cultural_emergence/_regime_threshold_sweep_3d/.

Usage:
    .venv/bin/python scripts/sweep_regime_thresholds_3d.py
"""

from __future__ import annotations

import argparse
import glob
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
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
)

import matplotlib.pyplot as plt  # noqa: E402

ALIVE = {"coherent_low_kl", "committed", "mid_range"}

# ---------------- the grid ----------------
KL_GRID = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
PERC_COH_GRID = [1.5, 1.6, 1.7, 1.8, 1.9]
PERC_DEG_GRID = [1.9, 2.0, 2.1, 2.2]
ENT_LO_GRID = [0.1, 0.15, 0.2, 0.25, 0.3]
ENT_HI_GRID = [1.5, 1.6, 1.7, 1.8, 1.9]

PAPER = dict(kl=KL_THR, pc=PERC_COHERENT, pd=PERC_DEGEN,
             elo=ENT_LOW, ehi=ENT_HIGH)


def grid_combos() -> list[dict]:
    combos = []
    for kl, pc, pdg, elo, ehi in itertools.product(
            KL_GRID, PERC_COH_GRID, PERC_DEG_GRID, ENT_LO_GRID, ENT_HI_GRID):
        if pdg <= pc:
            continue
        combos.append(dict(kl=kl, pc=pc, pd=pdg, elo=elo, ehi=ehi))
    return combos


def classify_vec(kl: np.ndarray, perc: np.ndarray, ent: np.ndarray,
                 p: dict) -> np.ndarray:
    """Vectorised 5-way classification, same precedence as the paper's
    classify(): text_degenerate first, then entropy band, then KL split."""
    return np.select(
        [perc >= p["pd"],
         (ent < p["elo"]) | (ent > p["ehi"]),
         (perc < p["pc"]) & (kl < p["kl"]),
         perc < p["pc"]],
        ["text_degenerate", "policy_collapse", "coherent_low_kl", "committed"],
        default="mid_range",
    )


# ---------------- data loading ----------------

def load_paper_table() -> pd.DataFrame:
    path = RUNS / "_population_figures" / "classification_table_iter200_2026-05-07.csv"
    df = pd.read_csv(path)
    df["condition"] = (df["substrate"].str.replace("study_group_", "", regex=False)
                       .str.replace("_server", "", regex=False).str.replace("_gc", "", regex=False)
                       + "-" + df["pop_size"].astype(str))
    df["dataset"] = "figure1_iter200"
    return df


def load_replicates() -> pd.DataFrame:
    paths = sorted(glob.glob(str(
        RUNS / "_seed_replicate_regimes" / "agent_classifications_all_seeds_*.csv")))
    df = pd.read_csv(paths[-1])
    df["condition"] = df["set"]
    df["dataset"] = "seed_replicates"
    return df


def load_endpoints() -> dict[str, pd.DataFrame]:
    base = RUNS / "_absorption_threshold_sensitivity"
    out = {}
    for name in ("cross", "seeds"):
        d = pd.read_csv(base / f"endpoint_{name}.csv")
        d = d.rename(columns={"end_kl": "kl", "end_perc": "perc_loss",
                              "end_ent": "entropy"})
        out[name] = d
    return out


# ---------------- sweep ----------------

def run_sweep(paper: pd.DataFrame, reps: pd.DataFrame,
              endpoints: dict[str, pd.DataFrame], out: Path):
    combos = grid_combos()
    print(f"grid: {len(combos)} valid threshold combinations")

    label_df = pd.concat([
        paper[["dataset", "condition", "run", "agent", "kl", "perc_loss", "entropy"]],
        reps.rename(columns={"seed": "rep_seed"})[
            ["dataset", "condition", "run", "agent", "kl", "perc_loss", "entropy"]],
    ], ignore_index=True)
    kl_a = label_df["kl"].values
    pe_a = label_df["perc_loss"].values
    en_a = label_df["entropy"].values

    # label counts per agent across the grid
    counts = {c: np.zeros(len(label_df), dtype=int) for c in CATS}
    comp_rows, pmi_rows = [], []

    ep = {k: (v["kl"].values, v["perc_loss"].values, v["entropy"].values,
              v["shrinkage"].values) for k, v in endpoints.items()}

    for p in combos:
        labels = classify_vec(kl_a, pe_a, en_a, p)
        for c in CATS:
            counts[c] += labels == c
        is_paper = all(abs(p[k] - PAPER[k]) < 1e-9 for k in p)

        # composition per condition
        tmp = label_df.assign(lab=labels)
        for cond, g in tmp.groupby("condition"):
            comp_rows.append({
                **p, "condition": cond, "n": len(g),
                "alive_frac": g["lab"].isin(ALIVE).mean(),
                "committed_frac": (g["lab"] == "committed").mean(),
                "coherent_frac": (g["lab"] == "coherent_low_kl").mean(),
                "collapse_frac": (g["lab"] == "policy_collapse").mean(),
                "is_paper": is_paper,
            })

        # PMI contrast, end-of-training labels, pooled
        for src, (k, pe, en, shr) in ep.items():
            lab = classify_vec(k, pe, en, p)
            comm, coh = shr[lab == "committed"], shr[lab == "coherent_low_kl"]
            pmi_rows.append({
                **p, "source": src,
                "n_committed": len(comm), "n_coherent": len(coh),
                "committed_shrinkage": comm.mean() if len(comm) else np.nan,
                "coherent_shrinkage": coh.mean() if len(coh) else np.nan,
                "difference": (comm.mean() - coh.mean())
                              if len(comm) and len(coh) else np.nan,
                "is_paper": is_paper,
            })

    stab = label_df.copy()
    total = sum(counts.values())[0] if len(label_df) else 0
    for c in CATS:
        stab[f"frac_{c}"] = counts[c] / len(combos)
    paper_lab = classify_vec(kl_a, pe_a, en_a, PAPER)
    stab["paper_label"] = paper_lab
    stab["stability"] = [counts[l][i] / len(combos)
                         for i, l in enumerate(paper_lab)]
    stab.to_csv(out / "label_stability_per_agent.csv", index=False)

    comp = pd.DataFrame(comp_rows)
    comp.to_csv(out / "composition_grid.csv", index=False)
    pmi = pd.DataFrame(pmi_rows)
    pmi.to_csv(out / "pmi_contrast_grid.csv", index=False)
    return stab, comp, pmi, len(combos)


# ---------------- clustering ----------------

def features(df: pd.DataFrame) -> np.ndarray:
    X = np.column_stack([
        np.log10(np.clip(df["kl"].values, 1e-3, None)),
        df["perc_loss"].values,
        df["entropy"].values,
    ])
    return StandardScaler().fit_transform(X)


def cluster_analysis(df: pd.DataFrame, label_col: str, tag: str,
                     out: Path, shrink_col: str | None = None) -> list[str]:
    """KMeans (silhouette-selected k) + GMM (BIC-selected k); ARI against the
    threshold labels; optional PMI contrast between committed-like and
    coherent-like clusters."""
    X = features(df)
    y = df[label_col].values
    lines = [f"\n### Clustering: {tag} (n={len(df)})\n"]
    rows = []
    sil = {}
    for k in range(2, 9):
        km = KMeans(n_clusters=k, n_init=50, random_state=0).fit(X)
        sil[k] = silhouette_score(X, km.labels_)
        rows.append({"method": "kmeans", "k": k, "silhouette": sil[k],
                     "ari_vs_labels": adjusted_rand_score(y, km.labels_)})
    bic = {}
    for k in range(1, 9):
        gm = GaussianMixture(n_components=k, covariance_type="full",
                             random_state=0, n_init=5).fit(X)
        bic[k] = gm.bic(X)
        if k >= 2:
            rows.append({"method": "gmm", "k": k, "bic": bic[k],
                         "ari_vs_labels": adjusted_rand_score(y, gm.predict(X))})
    pd.DataFrame(rows).to_csv(out / f"clustering_scores_{tag}.csv", index=False)

    k_sil = max(sil, key=sil.get)
    k_bic = min(bic, key=bic.get)
    km = KMeans(n_clusters=k_sil, n_init=50, random_state=0).fit(X)
    ari_sil = adjusted_rand_score(y, km.labels_)
    km5 = KMeans(n_clusters=5, n_init=50, random_state=0).fit(X)
    ari5 = adjusted_rand_score(y, km5.labels_)
    lines.append(
        f"- KMeans silhouette selects k={k_sil} (ARI vs 5-way labels "
        f"{ari_sil:.2f}); forced k=5 ARI {ari5:.2f}. GMM BIC selects k={k_bic}.")

    # cross-tab of chosen clustering vs labels
    ct = pd.crosstab(pd.Series(km.labels_, name=f"kmeans_k{k_sil}"),
                     pd.Series(y, name="threshold_label"))
    ct.to_csv(out / f"clustering_crosstab_{tag}.csv")
    lines.append(f"- cross-tab written to clustering_crosstab_{tag}.csv")

    df_out = df.copy()
    df_out[f"kmeans_k{k_sil}"] = km.labels_
    df_out["kmeans_k5"] = km5.labels_

    if shrink_col is not None:
        # committed-like = alive-region cluster with highest mean KL;
        # coherent-like = alive-region cluster with lowest mean KL.
        cent = df_out.groupby(f"kmeans_k{k_sil}").agg(
            kl=("kl", "mean"), perc=("perc_loss", "mean"),
            ent=("entropy", "mean"), shrinkage=(shrink_col, "mean"),
            n=("kl", "size"))
        cent.to_csv(out / f"clustering_centroids_{tag}.csv")
        alive_cl = cent[cent["perc"] < PERC_DEGEN]
        if len(alive_cl) >= 2:
            c_comm = alive_cl["kl"].idxmax()
            c_coh = alive_cl["kl"].idxmin()
            s_comm = df_out[df_out[f"kmeans_k{k_sil}"] == c_comm][shrink_col]
            s_coh = df_out[df_out[f"kmeans_k{k_sil}"] == c_coh][shrink_col]
            u = sps.mannwhitneyu(s_comm, s_coh, alternative="greater")
            lines.append(
                f"- PMI contrast with cluster labels (no thresholds): "
                f"high-KL cluster (n={len(s_comm)}, mean KL "
                f"{alive_cl.loc[c_comm, 'kl']:.2f}) shrinkage "
                f"{s_comm.mean():+.3f} vs low-KL cluster (n={len(s_coh)}, "
                f"mean KL {alive_cl.loc[c_coh, 'kl']:.2f}) "
                f"{s_coh.mean():+.3f}; one-sided Mann-Whitney "
                f"U={u.statistic:.0f}, p={u.pvalue:.3f}.")
    df_out.to_csv(out / f"clustering_assignments_{tag}.csv", index=False)
    return lines


# ---------------- dose-response: linear vs step, KL bimodality ----------------

def dose_response_analysis(endpoints: dict[str, pd.DataFrame], out: Path
                           ) -> list[str]:
    """Is the committed/coherent boundary a discontinuity in the
    shrinkage-vs-KL response (step), or a bin on a continuous dose-response
    whose covariate happens to be bimodal? Fits linear / step / step+linear
    models on the alive-region agents and a 1-D GMM on log10 KL."""
    dfs = []
    for src, d in endpoints.items():
        d = d.copy()
        d["lab"] = classify_vec(d["kl"].values, d["perc_loss"].values,
                                d["entropy"].values, PAPER)
        d["src"] = src
        dfs.append(d)
    alive = pd.concat(dfs, ignore_index=True)
    alive = alive[alive["lab"].isin(ALIVE)]

    def fit_models(d: pd.DataFrame, label: str) -> list[dict]:
        x = d["kl"].values
        y = d["shrinkage"].values
        n = len(d)
        step = (x >= KL_THR).astype(float)
        designs = {
            "const":       np.ones((n, 1)),
            "linear_kl":   np.column_stack([np.ones(n), x]),
            "step_kl1":    np.column_stack([np.ones(n), step]),
            "step+linear": np.column_stack([np.ones(n), step, x]),
        }
        rows = []
        for name, X in designs.items():
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            resid = y - X @ beta
            rss = float(resid @ resid)
            k = X.shape[1] + 1
            hat = X @ np.linalg.pinv(X.T @ X) @ X.T
            loo = float(np.mean((resid / (1 - np.diag(hat))) ** 2))
            rows.append({
                "subset": label, "model": name, "n": n,
                "aic": n * np.log(rss / n) + 2 * k,
                "bic": n * np.log(rss / n) + k * np.log(n),
                "r2": 1 - rss / np.sum((y - y.mean()) ** 2),
                "loocv_mse": loo,
                "coefs": np.round(beta, 4).tolist(),
            })
        return rows

    model_rows = fit_models(alive, "pooled")
    for src, g in alive.groupby("src"):
        model_rows += fit_models(g, src)
    models = pd.DataFrame(model_rows)
    models.to_csv(out / "dose_response_models.csv", index=False)

    gmm_rows = []
    for label, g in [("pooled", alive)] + list(alive.groupby("src")):
        lx = np.log10(np.clip(g["kl"].values, 1e-3, None)).reshape(-1, 1)
        bics = {k: GaussianMixture(k, random_state=0, n_init=10).fit(lx).bic(lx)
                for k in (1, 2, 3)}
        best = min(bics, key=bics.get)
        row = {"subset": label, "n": len(g), "best_k": best,
               **{f"bic_k{k}": v for k, v in bics.items()}}
        gm2 = GaussianMixture(2, random_state=0, n_init=10).fit(lx)
        mu = np.sort(gm2.means_.ravel())
        row["mode_lo_kl"], row["mode_hi_kl"] = 10 ** mu[0], 10 ** mu[1]
        row["weights"] = np.round(gm2.weights_, 2).tolist()
        gmm_rows.append(row)
    gmm = pd.DataFrame(gmm_rows)
    gmm.to_csv(out / "kl_bimodality_gmm.csv", index=False)

    # figure: dose-response scatter + fits | log-KL marginal with GMM
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), constrained_layout=True)
    ax = axes[0]
    for lab in ("coherent_low_kl", "committed", "mid_range"):
        g = alive[alive["lab"] == lab]
        ax.scatter(np.clip(g["kl"], 1e-3, None), g["shrinkage"], s=26,
                   facecolor=CAT_COLORS[lab], edgecolor="white", lw=0.5,
                   label=f"{CAT_LABELS[lab]} (n={len(g)})", zorder=3)
    x_grid = np.linspace(alive["kl"].min(), alive["kl"].max(), 200)
    lin = models[(models["subset"] == "pooled") & (models["model"] == "linear_kl")]["coefs"].iloc[0]
    stp = models[(models["subset"] == "pooled") & (models["model"] == "step_kl1")]["coefs"].iloc[0]
    ax.plot(x_grid, lin[0] + lin[1] * x_grid, "-", color="black", lw=1.2,
            label=f"linear: {lin[1]:+.3f}/nat")
    ax.plot(x_grid, stp[0] + stp[1] * (x_grid >= KL_THR), "--", color="0.45",
            lw=1.0, label="step at KL=1")
    ax.axvline(KL_THR, color="0.6", lw=0.6, ls=(0, (4, 2)))
    ax.axhline(0, color="0.8", lw=0.5, ls=":")
    ax.set_xscale("symlog", linthresh=0.1)
    ax.set_xlim(left=0.8 * alive["kl"].min())
    ax.set_xlabel("end-of-training KL (nats)")
    ax.set_ylabel("PMI shrinkage (nats/token)")
    ax.set_title("Continuous dose-response", fontsize=9)
    ax.legend(loc="upper left", frameon=False, fontsize=6)

    ax = axes[1]
    lx = np.log10(np.clip(alive["kl"].values, 1e-3, None))
    bins = np.linspace(lx.min() - 0.1, lx.max() + 0.1, 26)
    ax.hist(lx, bins=bins, density=True, color="0.8", edgecolor="white", lw=0.5)
    gm2 = GaussianMixture(2, random_state=0, n_init=10).fit(lx.reshape(-1, 1))
    xg = np.linspace(bins[0], bins[-1], 300)
    dens = np.exp(gm2.score_samples(xg.reshape(-1, 1)))
    ax.plot(xg, dens, color="#0072B2", lw=1.4, label="2-comp GMM")
    for m in gm2.means_.ravel():
        ax.axvline(m, color="#0072B2", lw=0.7, ls=":")
    ax.axvline(np.log10(KL_THR), color="0.35", lw=0.8, ls=(0, (4, 2)),
               label="KL threshold")
    ax.set_xlabel("log10 end-of-training KL")
    ax.set_ylabel("density")
    ax.set_title("Bimodal KL occupancy (alive agents)", fontsize=9)
    ax.legend(loc="upper right", frameon=False, fontsize=6)
    fig.savefig(out / "fig_dose_response.pdf", format="pdf",
                bbox_inches="tight", pad_inches=0.04)
    fig.savefig(out / "fig_dose_response.png", dpi=220,
                bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print("[ok] wrote fig_dose_response (+ csvs)")

    lines = ["\n## Dose-response: linear vs step, and KL bimodality\n"]
    for sub in ("pooled", "seeds", "cross"):
        m = models[models["subset"] == sub].set_index("model")
        lines.append(
            f"- {sub}: linear R2={m.loc['linear_kl', 'r2']:.2f} "
            f"(slope {m.loc['linear_kl', 'coefs'][1]:+.3f}/nat), "
            f"step R2={m.loc['step_kl1', 'r2']:.2f}; adding a step to the "
            f"linear model moves R2 to {m.loc['step+linear', 'r2']:.2f} "
            f"(step coef {m.loc['step+linear', 'coefs'][1]:+.3f}). "
            f"LOOCV: linear {m.loc['linear_kl', 'loocv_mse']:.4f} vs "
            f"step {m.loc['step_kl1', 'loocv_mse']:.4f}.")
    for _, r in gmm.iterrows():
        lines.append(
            f"- GMM on log10 KL ({r['subset']}, n={r['n']}): best k={r['best_k']}; "
            f"2-comp modes at {r['mode_lo_kl']:.2f} / {r['mode_hi_kl']:.2f} "
            f"nats, weights {r['weights']}.")
    return lines


# ---------------- figures ----------------

def fig_pmi_sweep(pmi: pd.DataFrame, out_path: Path):
    """ECDF of the committed-coherent difference over the full grid, plus
    marginal sweeps for each threshold parameter (others at paper values)."""
    params = [("kl", "KL threshold (nats)", KL_THR),
              ("pc", "perc-loss coherent cut", PERC_COHERENT),
              ("pd", "perc-loss degenerate cut", PERC_DEGEN),
              ("elo", "entropy lower bound", ENT_LOW),
              ("ehi", "entropy upper bound", ENT_HIGH)]
    fig, axes = plt.subplots(2, 3, figsize=(8.4, 5.0), constrained_layout=True)
    src_colors = {"cross": "#0072B2", "seeds": "#D55E00"}
    src_names = {"cross": "cross-env (7 configs)", "seeds": "inner-loop (5 seeds)"}

    ax = axes[0, 0]
    for src, col in src_colors.items():
        d = pmi[(pmi["source"] == src)]["difference"].dropna().sort_values()
        ax.step(d, np.arange(1, len(d) + 1) / len(d), color=col, lw=1.4,
                label=src_names[src])
        frac_pos = (d > 0).mean()
        base = pmi[(pmi["source"] == src) & pmi["is_paper"]]["difference"]
        if not base.empty:
            ax.axvline(base.iloc[0], color=col, lw=0.8, ls=(0, (4, 2)))
        ax.text(0.03, 0.92 - 0.09 * (src == "seeds"),
                f"{src_names[src]}: {100 * frac_pos:.1f}% of grid > 0",
                transform=ax.transAxes, fontsize=7, color=col)
    ax.axvline(0, color="0.4", lw=0.7)
    ax.set_xlabel("committed − coherent shrinkage (nats/token)")
    ax.set_ylabel("ECDF over grid")
    ax.set_title("Full-grid contrast", fontsize=9)
    ax.legend(loc="lower right", frameon=False, fontsize=6.5)
    ax.yaxis.grid(True, color="0.93", lw=0.5)
    ax.set_axisbelow(True)

    for ax, (key, xlabel, paper_val) in zip(axes.flat[1:], params):
        others = [k for k in ("kl", "pc", "pd", "elo", "ehi") if k != key]
        m = pmi.copy()
        for o in others:
            m = m[np.isclose(m[o], PAPER[o])]
        for src, col in src_colors.items():
            s = m[m["source"] == src].sort_values(key)
            ax.plot(s[key], s["committed_shrinkage"], "-o", ms=3.5, lw=1.2,
                    color=CAT_COLORS["committed"],
                    alpha=1.0 if src == "cross" else 0.55,
                    ls="-" if src == "cross" else "--")
            ax.plot(s[key], s["coherent_shrinkage"], "-o", ms=3.5, lw=1.2,
                    color=CAT_COLORS["coherent_low_kl"],
                    alpha=1.0 if src == "cross" else 0.55,
                    ls="-" if src == "cross" else "--")
            for _, r in s.iterrows():
                ax.text(r[key], r["committed_shrinkage"],
                        f" {int(r['n_committed'])}", fontsize=5,
                        color=CAT_COLORS["committed"], va="bottom")
        ax.axvline(paper_val, color="0.5", lw=0.6, ls=(0, (4, 2)))
        ax.axhline(0, color="0.7", lw=0.5, ls=":")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("shrinkage (nats/token)")
        ax.yaxis.grid(True, color="0.93", lw=0.5)
        ax.set_axisbelow(True)

    handles = [
        plt.Line2D([], [], color=CAT_COLORS["committed"], lw=1.4, label="committed"),
        plt.Line2D([], [], color=CAT_COLORS["coherent_low_kl"], lw=1.4, label="coherent (low KL)"),
        plt.Line2D([], [], color="0.3", lw=1.2, ls="-", label="cross-env"),
        plt.Line2D([], [], color="0.3", lw=1.2, ls="--", alpha=0.55, label="inner-loop seeds"),
    ]
    axes[0, 1].legend(handles=handles, loc="upper right", frameon=False,
                      fontsize=6.2)
    fig.suptitle("Committed vs coherent PMI shrinkage across the full "
                 "threshold grid (small numbers: committed n)", fontsize=10)
    fig.savefig(out_path, format="pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(out_path.with_suffix(".png"), dpi=220,
                bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"[ok] wrote {out_path} (+ .png)")


def fig_label_stability(stab: pd.DataFrame, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), constrained_layout=True)
    for ax, (ds, g) in zip(axes, stab.groupby("dataset")):
        for cat in CATS:
            s = g[g["paper_label"] == cat]["stability"]
            if s.empty:
                continue
            ax.hist(s, bins=np.linspace(0, 1, 21), histtype="stepfilled",
                    alpha=0.55, color=CAT_COLORS[cat],
                    label=f"{CAT_LABELS[cat]} (n={len(s)})")
        ax.set_xlabel("fraction of grid keeping the paper label")
        ax.set_title(ds, fontsize=9)
        ax.legend(loc="upper left", frameon=False, fontsize=6)
        ax.yaxis.grid(True, color="0.93", lw=0.5)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("agents")
    fig.savefig(out_path, format="pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(out_path.with_suffix(".png"), dpi=220,
                bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"[ok] wrote {out_path} (+ .png)")


def fig_clustering(assign_path: Path, out_path: Path):
    df = pd.read_csv(assign_path)
    kcol = [c for c in df.columns if c.startswith("kmeans_k") and c != "kmeans_k5"][0]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), constrained_layout=True)
    ax = axes[0]
    cl_colors = plt.get_cmap("tab10")
    for cl, g in df.groupby(kcol):
        ax.scatter(np.clip(g["kl"], 1e-3, None), g["perc_loss"], s=30,
                   facecolor=cl_colors(int(cl) % 10), edgecolor="white",
                   lw=0.5, label=f"cluster {cl} (n={len(g)})")
    ax.set_xscale("symlog", linthresh=0.1)
    ax.axvline(KL_THR, color="0.5", lw=0.6, ls=(0, (4, 2)))
    ax.axhline(PERC_COHERENT, color="0.5", lw=0.6, ls=(0, (4, 2)))
    ax.axhline(PERC_DEGEN, color="0.5", lw=0.6, ls=(0, (1.5, 1.5)))
    ax.set_xlabel("end-of-training KL (nats)")
    ax.set_ylabel("perception loss")
    ax.set_title(f"Unsupervised {kcol} on (log KL, perc, entropy)", fontsize=8.5)
    ax.legend(loc="upper left", frameon=False, fontsize=6)

    ax = axes[1]
    order = df.groupby(kcol)["kl"].mean().sort_values().index
    xs = np.arange(len(order))
    means = [df[df[kcol] == c]["shrinkage"].mean() for c in order]
    ax.bar(xs, means, 0.6,
           color=[cl_colors(int(c) % 10) for c in order],
           edgecolor="white", lw=0.6)
    rng = np.random.default_rng(0)
    for x, c in zip(xs, order):
        vals = df[df[kcol] == c]["shrinkage"].values
        ax.scatter(np.full(len(vals), x) + rng.uniform(-0.15, 0.15, len(vals)),
                   vals, s=12, facecolor="0.25", edgecolor="white", lw=0.3,
                   zorder=3)
    ax.axhline(0, color="0.6", lw=0.6)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"cl {c}\nKL={df[df[kcol]==c]['kl'].mean():.1f}"
                        for c in order], fontsize=7)
    ax.set_ylabel("PMI shrinkage (nats/token)")
    ax.set_title("Shrinkage by cluster (sorted by mean KL)", fontsize=8.5)
    ax.yaxis.grid(True, color="0.93", lw=0.5)
    ax.set_axisbelow(True)
    fig.savefig(out_path, format="pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(out_path.with_suffix(".png"), dpi=220,
                bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"[ok] wrote {out_path} (+ .png)")


# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path,
                    default=RUNS / "_regime_threshold_sweep_3d")
    ap.add_argument("--dose-only", action="store_true",
                    help="Only run the dose-response / bimodality analysis.")
    args = ap.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    paper = load_paper_table()
    reps = load_replicates()
    endpoints = load_endpoints()

    if args.dose_only:
        lines = dose_response_analysis(endpoints, out)
        (out / "DOSE_RESPONSE.md").write_text("\n".join(lines) + "\n")
        print("\n".join(lines))
        return

    stab, comp, pmi, n_combos = run_sweep(paper, reps, endpoints, out)

    lines = [
        "# Joint threshold sweep + clustering alternative",
        f"\nGrid: KL {KL_GRID}; perc_coherent {PERC_COH_GRID}; "
        f"perc_degen {PERC_DEG_GRID}; ent_lo {ENT_LO_GRID}; "
        f"ent_hi {ENT_HI_GRID}; {n_combos} valid combinations "
        "(paper values: KL=1.0, 1.7, 2.0, 0.2, 1.8).",
    ]

    # -- label stability summary
    lines.append("\n## Label stability (uncertainty over labels)\n")
    for ds, g in stab.groupby("dataset"):
        med = g["stability"].median()
        q = (g["stability"] >= 0.9).mean()
        lines.append(f"- {ds}: n={len(g)} agents; median label stability "
                     f"{med:.2f}; {100 * q:.0f}% of agents keep their paper "
                     f"label in >=90% of the grid.")
        for cat, gc in g.groupby("paper_label"):
            lines.append(f"    - {cat}: n={len(gc)}, median {gc['stability'].median():.2f}, "
                         f"min {gc['stability'].min():.2f}")

    # -- composition robustness
    lines.append("\n## Regime-composition robustness per condition\n")
    lines.append("| condition | n | alive min/med/max | committed min/med/max |")
    lines.append("|---|---|---|---|")
    for cond, g in comp.groupby("condition"):
        a, c = g["alive_frac"], g["committed_frac"]
        lines.append(f"| {cond} | {int(g['n'].iloc[0])} "
                     f"| {a.min():.2f} / {a.median():.2f} / {a.max():.2f} "
                     f"| {c.min():.2f} / {c.median():.2f} / {c.max():.2f} |")

    # -- PMI contrast robustness
    lines.append("\n## Committed-vs-coherent PMI contrast across the grid\n")
    for src, g in pmi.groupby("source"):
        d = g["difference"].dropna()
        base = g[g["is_paper"]]["difference"]
        n_empty = g["difference"].isna().sum()
        lines.append(
            f"- {src}: paper-threshold difference "
            f"{base.iloc[0]:+.3f} nats/token; across {len(d)} grid points "
            f"with both groups non-empty (of {len(g)}; {n_empty} had an empty "
            f"group): {100 * (d > 0).mean():.1f}% positive, "
            f"median {d.median():+.3f}, IQR [{d.quantile(0.25):+.3f}, "
            f"{d.quantile(0.75):+.3f}], min {d.min():+.3f}.")
        nc = g["n_committed"]
        lines.append(f"    - committed group size across grid: min {nc.min()}, "
                     f"median {nc.median():.0f}, max {nc.max()}.")

    # -- policy-collapse branch split (entropy definitional point)
    lines.append("\n## policy_collapse: low- vs high-entropy branch\n")
    both = pd.concat([paper, reps], ignore_index=True)
    pc = both[classify_vec(both["kl"].values, both["perc_loss"].values,
                           both["entropy"].values, PAPER) == "policy_collapse"]
    lo = (pc["entropy"] < ENT_LOW).sum()
    hi = (pc["entropy"] > ENT_HIGH).sum()
    lines.append(f"- Of {len(pc)} policy_collapse agents (Figure-1 set + "
                 f"replicates) at paper thresholds: {lo} via entropy < "
                 f"{ENT_LOW} (near-deterministic), {hi} via entropy > "
                 f"{ENT_HIGH} (high-entropy divergence).")
    if len(pc):
        pc[["dataset", "condition", "run", "agent", "kl", "perc_loss",
            "entropy"]].to_csv(out / "policy_collapse_agents.csv", index=False)

    # -- inner-loop committed count accounting
    lines.append("\n## Inner Loop committed-agent accounting\n")
    eps = endpoints["seeds"].copy()
    eps["lab"] = classify_vec(eps["kl"].values, eps["perc_loss"].values,
                              eps["entropy"].values, PAPER)
    comm = eps[eps["lab"] == "committed"]
    lines.append(f"- End-of-training committed in the 5 inner-loop seeds: "
                 f"{len(comm)} (run, character) instances, "
                 f"{comm['character'].nunique()} unique characters "
                 f"({', '.join(sorted(comm['character'].unique()))}).")
    per_run = comm.groupby("run_dir").size()
    lines.append("- Per seed: " + "; ".join(
        f"{Path(r).name.replace('run7_8agent_7B_', '')}: {n}"
        for r, n in per_run.items()) + ".")

    # -- clustering
    lines.append("\n## Unsupervised clustering alternative\n")
    # (a) taxonomy agreement on the full labelled sets
    both_lab = both.assign(lab=classify_vec(
        both["kl"].values, both["perc_loss"].values,
        both["entropy"].values, PAPER))
    lines += cluster_analysis(both_lab, "lab", "fig1_plus_replicates", out)
    # (b) PMI contrast via clusters on the endpoint sets
    for src in ("cross", "seeds"):
        d = endpoints[src].copy()
        d["lab"] = classify_vec(d["kl"].values, d["perc_loss"].values,
                                d["entropy"].values, PAPER)
        lines += cluster_analysis(d, "lab", f"endpoints_{src}", out,
                                  shrink_col="shrinkage")
    d_all = pd.concat([endpoints["cross"], endpoints["seeds"]],
                      ignore_index=True)
    d_all["lab"] = classify_vec(d_all["kl"].values, d_all["perc_loss"].values,
                                d_all["entropy"].values, PAPER)
    lines += cluster_analysis(d_all, "lab", "endpoints_pooled", out,
                              shrink_col="shrinkage")

    lines += dose_response_analysis(endpoints, out)

    fig_pmi_sweep(pmi, out / "fig_pmi_contrast_grid.pdf")
    fig_label_stability(stab, out / "fig_label_stability.pdf")
    fig_clustering(out / "clustering_assignments_endpoints_pooled.csv",
                   out / "fig_clustering_pmi.pdf")

    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(f"[ok] wrote {out / 'REPORT.md'}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
