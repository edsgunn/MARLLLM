#!/usr/bin/env python
"""Plot the output of scripts/rollout_hidden_norms.py against action entropy.

For each measured run, writes into its output directory:

  trajectories.png     shared-x stack: action entropy (logged in training, line;
                       measured on the fresh rollouts, markers), ||h|| after and
                       before the final RMSNorm at ACT positions, and the logit
                       scale. Solid = on-policy rollouts, dashed = the iter-0
                       trajectories re-scored (inputs fixed, weights vary).
  norm_vs_entropy.png  one point per (agent, checkpoint): final-hidden norm vs
                       logged action entropy, with pooled and within-agent
                       Spearman correlations.
  token_level.png      within a checkpoint: mean token entropy in bins of
                       ||h_post||, one line per checkpoint.
  correlations.csv     the correlations behind norm_vs_entropy.png.

Usage:
    .venv/bin/python scripts/plot_hidden_norms.py runs/cultural_emergence/_hidden_norms/<run> [...]
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
          "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, MUTED, GRID = "#2b2b2b", "#6b6b66", "#e6e6e3"
ENT_WINDOW = 2   # logged entropy at a checkpoint = mean over iters it±2

plt.rcParams.update({
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "legend.frameon": False,
})


def load(out_dir: Path):
    meta = json.loads((out_dir / "meta.json").read_text())
    # rows[(data, subset)][agent][iter] -> list of episode rows
    rows = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for line in open(out_dir / "episodes.jsonl"):
        r = json.loads(line)
        if r["n"] > 0:
            rows[(r["data"], r["subset"])][r["agent"]][r["iter"]].append(r)
    logged = defaultdict(dict)
    for line in open(Path(meta["run_dir"]) / "metrics.jsonl"):
        d = json.loads(line)
        it = int(d["iteration"])
        for a in meta["agents"]:
            if f"{a}/entropy" in d:
                logged[a][it] = d[f"{a}/entropy"]
    return meta, rows, logged


def token_weighted(eps: list[dict], key: str) -> float:
    n = np.array([e["n"] for e in eps], float)
    return float(np.dot(n, [e[key] for e in eps]) / n.sum())


def series(rows, agent, key):
    its = sorted(rows[agent])
    return np.array(its), np.array([token_weighted(rows[agent][i], key) for i in its])


def logged_at(logged: dict, it: int) -> float:
    vals = [v for i, v in logged.items() if abs(i - it) <= ENT_WINDOW and i >= 1]
    return float(np.mean(vals)) if vals else float("nan")


def smooth(xs, ys, w=5):
    ys = np.asarray(ys, float)
    k = np.ones(w) / w
    pad = np.pad(ys, (w // 2, w - 1 - w // 2), mode="edge")
    return xs, np.convolve(pad, k, mode="valid")


def run_label(meta) -> str:
    return f"{Path(meta['run_dir']).name}  ({len(meta['agents'])} agents)"


def plot_trajectories(out_dir, meta, rows, logged):
    agents = meta["agents"]
    on, fx = rows[("onpolicy", "act")], rows[("fixed", "act")]
    panels = [
        ("entropy", "action entropy (nats/token)"),
        ("norm_post", "‖h‖ final hidden\n(after final norm)"),
        ("norm_pre", "‖h‖ residual\n(before final norm)"),
        ("logit_std", "logit std over vocab"),
    ]
    fig, axes = plt.subplots(len(panels), 1, figsize=(8, 10), sharex=True)
    for ci, a in enumerate(agents):
        c = SERIES[ci % len(SERIES)]
        for ax, (key, ylab) in zip(axes, panels):
            if key == "entropy":
                xs = sorted(logged[a])
                ax.plot(*smooth(np.array(xs), [logged[a][i] for i in xs]),
                        color=c, lw=1.2, alpha=0.85, label=a)
                x, y = series(on, a, "entropy_mean")
                ax.plot(x, y, "o", color=c, ms=4, mec="white", mew=0.8)
            else:
                x, y = series(on, a, f"{key}_mean")
                ax.plot(x, y, "-o", color=c, lw=1.6, ms=3.5, mec="white", mew=0.6)
                if fx.get(a):
                    x, y = series(fx, a, f"{key}_mean")
                    ax.plot(x, y, "--", color=c, lw=1.0, alpha=0.7)
            ax.set_ylabel(ylab)
    axes[0].set_yscale("log")
    axes[0].set_title("line: logged in training (5-iter mean)   ·   dots: measured on fresh rollouts",
                      fontsize=8, color=MUTED, loc="left")
    axes[1].set_title("solid: on-policy rollouts   ·   dashed: iter-0 trajectories re-scored (inputs fixed)",
                      fontsize=8, color=MUTED, loc="left")
    axes[-1].set_xlabel("training iteration")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center",
               ncol=min(4, len(agents)), bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(run_label(meta) + " — ACT positions", y=1.03, fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_dir / "trajectories.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def within_agent_rho(pts):
    rhos = [spearmanr(x, y)[0] for x, y in pts.values() if len(x) >= 4]
    return float(np.nanmean(rhos)) if rhos else float("nan")


def plot_scatter(out_dir, meta, rows, logged):
    agents = meta["agents"]
    on, fx = rows[("onpolicy", "act")], rows[("fixed", "act")]
    keys = [("norm_post", "‖h‖ final hidden (after final norm)"),
            ("norm_pre", "‖h‖ residual (before final norm)")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    out_rows = []
    for ax, (key, xlab) in zip(axes, keys):
        for data, src in (("onpolicy", on), ("fixed", fx)):
            per_agent = {}
            for ci, a in enumerate(agents):
                if not src.get(a):
                    continue
                x, nrm = series(src, a, f"{key}_mean")
                ent = np.array([logged_at(logged[a], i) for i in x])
                _, meas = series(on, a, "entropy_mean")
                ok = np.isfinite(ent)
                per_agent[a] = (nrm[ok], ent[ok])
                if data == "onpolicy":
                    ax.scatter(nrm[ok], ent[ok], s=22, color=SERIES[ci % len(SERIES)],
                               edgecolor="white", linewidth=0.6, label=a, zorder=3)
                    ax.plot(nrm[ok], ent[ok], color=SERIES[ci % len(SERIES)], lw=0.6, alpha=0.4)
                    out_rows.append({"metric": key, "data": data, "agent": a,
                                     "spearman_vs_logged": spearmanr(nrm[ok], ent[ok])[0],
                                     "spearman_vs_measured": spearmanr(nrm, meas)[0]})
            allx = np.concatenate([v[0] for v in per_agent.values()])
            ally = np.concatenate([v[1] for v in per_agent.values()])
            pooled, within = spearmanr(allx, ally)[0], within_agent_rho(per_agent)
            out_rows.append({"metric": key, "data": data, "agent": "ALL (pooled)",
                             "spearman_vs_logged": pooled})
            out_rows.append({"metric": key, "data": data, "agent": "MEAN within-agent",
                             "spearman_vs_logged": within})
            if data == "onpolicy":
                ax.text(0.02, 0.03, f"Spearman ρ pooled {pooled:+.2f} · within-agent mean {within:+.2f}",
                        transform=ax.transAxes, fontsize=8, color=INK)
        ax.set_xlabel(xlab)
        ax.set_yscale("log")
    axes[0].set_ylabel("logged action entropy at checkpoint (nats/token)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center",
               ncol=min(8, len(agents)), bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(run_label(meta) + " — one point per (agent, checkpoint), on-policy, ACT positions",
                 y=1.09, fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(out_dir / "norm_vs_entropy.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    with open(out_dir / "correlations.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "data", "agent",
                                          "spearman_vs_logged", "spearman_vs_measured"])
        w.writeheader()
        w.writerows(out_rows)
    return out_rows


def plot_token_level(out_dir, meta):
    agents = meta["agents"]
    iters = [i for i in meta["iters"] if (out_dir / f"tokens_iter{i:06d}.npz").exists()]
    cmap = plt.get_cmap("Blues")
    ncol = min(4, len(agents))
    nrow = int(np.ceil(len(agents) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.8 * nrow),
                             sharex=True, sharey=True, squeeze=False)
    edges = None
    for k, it in enumerate(iters):
        z = np.load(out_dir / f"tokens_iter{it:06d}.npz")
        if edges is None:   # bins fixed from iter 0 so curves are comparable
            alln = np.concatenate([z[f"{a}|norm_post"] for a in agents if f"{a}|norm_post" in z])
            edges = np.quantile(alln, np.linspace(0, 1, 13))
            edges[0], edges[-1] = 0, np.inf
        col = cmap(0.3 + 0.7 * k / max(1, len(iters) - 1))
        for ai, a in enumerate(agents):
            ax = axes[ai // ncol][ai % ncol]
            if f"{a}|norm_post" not in z:
                continue
            n, h = z[f"{a}|norm_post"], z[f"{a}|entropy"]
            idx = np.digitize(n, edges) - 1
            xs, ys = [], []
            for b in range(len(edges) - 1):
                m = idx == b
                if m.sum() >= 30:
                    xs.append(np.median(n[m]))
                    ys.append(h[m].mean())
            ax.plot(xs, ys, "-", color=col, lw=1.3, label=f"iter {it}" if ai == 0 else None)
            ax.set_title(a, fontsize=9, color=INK)
    for ax in axes.flat:
        ax.set_yscale("log")
    for ax in axes[-1]:
        ax.set_xlabel("‖h‖ final hidden (token)")
    for row in axes:
        row[0].set_ylabel("mean token entropy")
    fig.legend(*axes[0][0].get_legend_handles_labels(), loc="center left",
               bbox_to_anchor=(1.0, 0.5), title="checkpoint")
    fig.suptitle(run_label(meta) + " — ACT tokens binned by final-hidden norm (light → dark = early → late)",
                 fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(out_dir / "token_level.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("out_dirs", nargs="+")
    args = p.parse_args()
    for d in map(Path, args.out_dirs):
        meta, rows, logged = load(d)
        plot_trajectories(d, meta, rows, logged)
        corr = plot_scatter(d, meta, rows, logged)
        plot_token_level(d, meta)
        print(f"== {d.name}")
        for r in corr:
            if r["agent"].startswith(("ALL", "MEAN")):
                print(f"  {r['metric']:10s} {r['data']:9s} {r['agent']:18s} ρ={r['spearman_vs_logged']:+.2f}")


if __name__ == "__main__":
    main()
