"""Rung 1 — is sanctioning targeted (vs. mood)?

Does the *same behaviour type* reliably draw the *same valence*? Deterministic
aggregation over event records + post-hoc behaviour clusters. Gated on a
passing validation run for the same (provider, model, prompt_version).

Outputs:
  - per-cluster valence distribution + targeting-consistency table
  - I(behaviour_cluster ; valence) with a permutation-shuffle null (+ CI)
  - MI-vs-iteration plot per substrate (should rise if targeting emerges)
  - an MI sensitivity sweep over the clustering threshold (researcher d.o.f.)
"""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mutual_info_score

from . import PROMPT_VERSION
from .cluster import cluster_vectors, collect_events, embed_texts, medoid_labels
from .persistence import ClusterStore
from .validate import gate_status, report_path


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------
def check_gate(cfg, force=False) -> bool:
    ok, msg = gate_status(cfg)
    print(f"[rung1] validation gate: {msg}")
    if ok or force:
        if force and not ok:
            print("[rung1] --force set: emitting Rung-1 results despite failing/missing gate.")
        return True
    print("[rung1] REFUSING to emit Rung-1 results. Run `validate` to a passing "
          "report for this (provider, model, prompt_version), or pass --force.")
    return False


# ---------------------------------------------------------------------------
# MI with permutation null
# ---------------------------------------------------------------------------
def mi_with_null(cluster_ids: np.ndarray, valences: np.ndarray, n_perm=1000, seed=0):
    mask = cluster_ids != -1
    c = cluster_ids[mask]
    v = valences[mask]
    if len(c) < 2 or len(set(c)) < 2 or len(set(v)) < 2:
        return dict(mi=0.0, null_mean=0.0, null_lo=0.0, null_hi=0.0, delta=0.0, n=int(len(c)))
    mi = mutual_info_score(c, v)
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    vv = v.copy()
    for i in range(n_perm):
        rng.shuffle(vv)
        null[i] = mutual_info_score(c, vv)
    return dict(
        mi=float(mi), null_mean=float(null.mean()),
        null_lo=float(np.percentile(null, 2.5)), null_hi=float(np.percentile(null, 97.5)),
        delta=float(mi - null.mean()), n=int(len(c)),
    )


# ---------------------------------------------------------------------------
# Per-cluster consistency
# ---------------------------------------------------------------------------
def consistency_table(df: pd.DataFrame, group_cols: list) -> pd.DataFrame:
    rows = []
    for keys, sub in df.groupby(group_cols + ["behaviour_cluster_id"]):
        *grp, cid = keys
        if cid == -1:
            continue
        n = len(sub)
        p = {v: (sub["valence"] == v).mean() for v in ("approve", "disapprove", "neutral")}
        dominant = max(p, key=p.get)
        row = dict(zip(group_cols, grp))
        row.update(
            behaviour_cluster_id=cid, n_events=n,
            p_approve=p["approve"], p_disapprove=p["disapprove"], p_neutral=p["neutral"],
            dominant_valence=dominant, targeting_consistency=p[dominant],
            cluster_label=sub["cluster_label"].iloc[0],
        )
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def _load_joined(cfg, substrates, run_ids, iter_min, iter_max) -> pd.DataFrame:
    ids, texts, meta = collect_events(cfg, substrates, run_ids, iter_min, iter_max)
    assigns = ClusterStore(cfg.paths.clusters).load()
    if not assigns:
        print("[rung1] no cluster assignments found. Run `cluster` first.")
        return pd.DataFrame()
    rows = []
    for i, eid in enumerate(ids):
        a = assigns.get(eid)
        if a is None:
            continue  # event not in the current cluster run (e.g. clustered on a subset)
        m = meta[i]
        rows.append(dict(
            event_id=eid, behaviour_cluster_id=a["behaviour_cluster_id"],
            cluster_label=a["cluster_label"], valence=m["valence"],
            substrate=m["substrate"], run_id=m["run_id"], iteration=m["iteration"],
        ))
    return pd.DataFrame(rows)


def _mi_sweep(cfg, substrates, run_ids, iter_min, iter_max) -> pd.DataFrame:
    """Re-cluster at each sweep threshold and report cluster count + MI."""
    ids, texts, meta = collect_events(cfg, substrates, run_ids, iter_min, iter_max)
    if not ids:
        return pd.DataFrame()
    X = embed_texts(texts, cfg)
    valences = np.array([m["valence"] for m in meta])
    rows = []
    for thr in cfg.clustering.sweep:
        labels = cluster_vectors(X, cfg, distance_threshold=thr)
        n_clusters = len([c for c in set(labels) if c != -1])
        res = mi_with_null(labels, valences, n_perm=300)
        rows.append(dict(
            distance_threshold=thr, min_cluster_size=cfg.clustering.min_cluster_size,
            n_clusters=n_clusters, n_noise=int((labels == -1).sum()),
            mi=res["mi"], mi_minus_null=res["delta"],
        ))
    return pd.DataFrame(rows)


def run_rung1(cfg, substrates=None, run_ids=None, iter_min=None, iter_max=None, force=False):
    cfg.make_dirs()
    if not check_gate(cfg, force=force):
        return
    df = _load_joined(cfg, substrates, run_ids, iter_min, iter_max)
    if df.empty:
        return

    # 1) per-cluster consistency (pooled over runs, per substrate+iteration; and per-run)
    pooled = consistency_table(df, ["substrate", "iteration"])
    per_run = consistency_table(df, ["substrate", "run_id", "iteration"])
    p1 = os.path.join(cfg.paths.output_dir, "rung1_consistency_pooled.csv")
    p2 = os.path.join(cfg.paths.output_dir, "rung1_consistency_per_run.csv")
    pooled.to_csv(p1, index=False)
    per_run.to_csv(p2, index=False)
    print(f"[rung1] wrote {p1} ({len(pooled)} rows), {p2} ({len(per_run)} rows)")

    # 2) MI(behaviour ; valence) per substrate+iteration (pooled runs)
    mi_rows = []
    for (substrate, iteration), sub in df.groupby(["substrate", "iteration"]):
        res = mi_with_null(sub["behaviour_cluster_id"].to_numpy(),
                           sub["valence"].to_numpy(), n_perm=500)
        res.update(substrate=substrate, iteration=iteration)
        mi_rows.append(res)
    mi_df = pd.DataFrame(mi_rows).sort_values(["substrate", "iteration"])
    p3 = os.path.join(cfg.paths.output_dir, "rung1_mi_by_iteration.csv")
    mi_df.to_csv(p3, index=False)
    print(f"[rung1] wrote {p3}")

    # MI-vs-iteration plot
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for substrate, sub in mi_df.groupby("substrate"):
        ax.plot(sub["iteration"], sub["delta"], marker="o", label=f"{substrate} (MI − null)")
        ax.fill_between(sub["iteration"], sub["null_lo"] - sub["null_mean"],
                        sub["null_hi"] - sub["null_mean"], alpha=0.12)
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xlabel("training iteration (checkpoint)")
    ax.set_ylabel("I(behaviour ; valence) − permutation null")
    ax.set_title("Rung 1 — targeted-sanctioning signal vs iteration")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig_path = os.path.join(cfg.paths.figures, "rung1_mi_vs_iteration.png")
    fig.savefig(fig_path, dpi=130)
    plt.close(fig)
    print(f"[rung1] wrote {fig_path}")

    # 3) clustering-sensitivity sweep
    sweep = _mi_sweep(cfg, substrates, run_ids, iter_min, iter_max)
    if not sweep.empty:
        p4 = os.path.join(cfg.paths.output_dir, "rung1_mi_threshold_sweep.csv")
        sweep.to_csv(p4, index=False)
        print(f"[rung1] wrote {p4}")
        print("\n=== MI sensitivity to clustering threshold (corpus-wide) ===")
        print(sweep.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
