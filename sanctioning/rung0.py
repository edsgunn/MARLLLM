"""Rung 0 — does sanctioning occur at all?

Deterministic aggregation over persisted event records. No LLM calls.
Self-directed events are retained in the store but excluded here (sanctioning
is between agents). Approval and disapproval rates are reported separately:
the chatbot-base hypothesis predicts disapproval ~ 0 early, and a flat-near-zero
disapproval rate across iterations is a reportable finding, not a bug.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .persistence import EventStore


def _row_for_record(d: dict) -> dict:
    n_approve = n_disapprove = n_neutral = 0
    c_approve = c_disapprove = 0.0
    for e in d["events"]:
        if e.get("target_is_self"):
            continue  # excluded from sanctioning aggregation (retained in store)
        v = e["valence"]
        if v == "approve":
            n_approve += 1
            c_approve += float(e.get("confidence", 0.0))
        elif v == "disapprove":
            n_disapprove += 1
            c_disapprove += float(e.get("confidence", 0.0))
        else:
            n_neutral += 1
    reacting = (n_approve + n_disapprove) > 0
    return dict(
        substrate=d["substrate"], run_id=d["run_id"], iteration=d["iteration"],
        n_approve=n_approve, n_disapprove=n_disapprove, n_neutral=n_neutral,
        c_approve=c_approve, c_disapprove=c_disapprove, reacting=int(reacting),
    )


def build_table(cfg, substrates=None, run_ids=None, iter_min=None, iter_max=None) -> pd.DataFrame:
    store = EventStore(cfg.paths.events)
    rows = [
        _row_for_record(d)
        for d in store.iter_records(substrates, run_ids, iter_min, iter_max)
    ]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    g = df.groupby(["substrate", "run_id", "iteration"], as_index=False).agg(
        n_posts=("reacting", "size"),
        n_reacting_posts=("reacting", "sum"),
        n_approve=("n_approve", "sum"),
        n_disapprove=("n_disapprove", "sum"),
        n_neutral=("n_neutral", "sum"),
        c_approve=("c_approve", "sum"),
        c_disapprove=("c_disapprove", "sum"),
    )
    g["n_events"] = g["n_approve"] + g["n_disapprove"] + g["n_neutral"]
    g["sanction_rate"] = (g["n_approve"] + g["n_disapprove"]) / g["n_posts"]
    g["approval_rate"] = g["n_approve"] / g["n_posts"]
    g["disapproval_rate"] = g["n_disapprove"] / g["n_posts"]
    # confidence-weighted variants (secondary)
    g["approval_rate_cw"] = g["c_approve"] / g["n_posts"]
    g["disapproval_rate_cw"] = g["c_disapprove"] / g["n_posts"]
    return g.sort_values(["substrate", "run_id", "iteration"]).reset_index(drop=True)


def _plot_rate(df, rate_col, title, out_path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for substrate, sub in df.groupby("substrate"):
        agg = sub.groupby("iteration")[rate_col].agg(["mean", "std", "count"])
        err = agg["std"].fillna(0.0)
        ax.errorbar(agg.index, agg["mean"], yerr=err, marker="o", capsize=3, label=substrate)
    ax.set_xlabel("training iteration (checkpoint)")
    ax.set_ylabel(rate_col)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def run_rung0(cfg, substrates=None, run_ids=None, iter_min=None, iter_max=None):
    cfg.make_dirs()
    df = build_table(cfg, substrates, run_ids, iter_min, iter_max)
    if df.empty:
        print("[rung0] no event records found. Run `extract` first.")
        return
    csv_path = os.path.join(cfg.paths.output_dir, "rung0_table.csv")
    df.to_csv(csv_path, index=False)
    print(f"[rung0] wrote {csv_path}  ({len(df)} rows)")

    # mean over runs per substrate (error bars across runs)
    _plot_rate(df, "disapproval_rate", "Rung 0 — disapproval rate vs iteration",
               os.path.join(cfg.paths.figures, "rung0_disapproval_rate.png"))
    _plot_rate(df, "approval_rate", "Rung 0 — approval rate vs iteration",
               os.path.join(cfg.paths.figures, "rung0_approval_rate.png"))
    _plot_rate(df, "sanction_rate", "Rung 0 — overall sanctioning rate vs iteration",
               os.path.join(cfg.paths.figures, "rung0_sanction_rate.png"))
    print(f"[rung0] wrote plots to {cfg.paths.figures}/")

    # console summary per substrate
    print("\n=== Rung 0 summary (mean over runs & iterations) ===")
    summ = df.groupby("substrate")[
        ["sanction_rate", "approval_rate", "disapproval_rate"]
    ].mean()
    print(summ.to_string(float_format=lambda x: f"{x:.4f}"))
