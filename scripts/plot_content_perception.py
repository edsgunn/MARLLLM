#!/usr/bin/env python3
"""Plot perception loss split into post content vs environment scaffolding.

Reads the summary written by scripts/analyse_content_perception.py and draws
three panels sharing an x axis of training iteration:

  1. NLL on content tokens, on scaffolding tokens, and the all-token mean that
     `perc_loss` actually reports — with the classifier's 1.7 gate marked.
  2. What the stream is made of: content as a share of observation tokens.
  3. Delivery rate, the substrate-level measure that no model loss can game.

The point of the figure is panel 1 against panel 3: if the all-token mean falls
while delivery falls and the content NLL does not, the metric is tracking the
stream getting emptier rather than the agents getting better.

Usage
    .venv/bin/python scripts/plot_content_perception.py \
        [--summary runs/_tier_b/content_perception/summary.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
PERC_COHERENT = 1.7          # the classifier's primary gate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary",
                    default="runs/_tier_b/content_perception/summary.json")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    src = ROOT / a.summary
    rows = sorted(json.loads(src.read_text()).values(),
                  key=lambda r: r["iteration"])
    if not rows:
        raise SystemExit("no rows in the summary")
    out = Path(a.out) if a.out else src.parent / "content_perception.png"

    it = [r["iteration"] for r in rows]
    con = [r["nll_content"] for r in rows]
    sca = [r["nll_scaffold"] for r in rows]
    alls = [r["nll_all"] for r in rows]
    frac = [100 * r["content_frac"] for r in rows]
    deliv = [r["delivery"] for r in rows]
    boundary = 300 if any(r["label"] != rows[0]["label"] for r in rows) else None

    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True,
                             gridspec_kw={"height_ratios": [2.2, 1, 1]})

    ax = axes[0]
    ax.plot(it, con, "o-", color="#CC6677", lw=2, label="post content only")
    ax.plot(it, sca, "s-", color="#88CCEE", lw=1.6, label="environment scaffolding")
    ax.plot(it, alls, "^--", color="#332288", lw=1.8,
            label="all observation tokens  (this is `perc_loss`)")
    ax.axhline(PERC_COHERENT, color="k", ls=":", lw=1.2)
    ax.annotate("classifier's coherence gate, perc_loss < 1.7",
                xy=(it[0], PERC_COHERENT), xytext=(4, 5),
                textcoords="offset points", fontsize=8)
    ax.set_ylabel("negative log-likelihood (nats/token)")
    ax.set_title("Perception loss is two different quantities averaged together")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25)

    ax = axes[1]
    ax.plot(it, frac, "o-", color="#117733", lw=2)
    ax.set_ylabel("content, % of\nobservation tokens")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.25)

    ax = axes[2]
    ax.plot(it, deliv, "o-", color="#AA4499", lw=2)
    ax.set_ylabel("well-formed\npost rate")
    ax.set_xlabel("training iteration")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)

    for ax in axes:
        if boundary:
            ax.axvline(boundary, color="grey", ls="--", lw=1, alpha=0.7)
    axes[0].annotate("resume boundary\n(run7 → extension)", xy=(boundary, axes[0].get_ylim()[1]),
                     xytext=(6, -28), textcoords="offset points",
                     fontsize=8, color="grey") if boundary else None

    fig.suptitle("What falls when `perc_loss` falls — 8-agent Ashbourne reference",
                 y=0.995, fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")

    # A second, smaller figure: the artefact stated as a scatter.
    f2, a2 = plt.subplots(figsize=(5.2, 4.4))
    sc = a2.scatter(deliv, alls, c=it, cmap="viridis", s=70, label="all tokens")
    a2.scatter(deliv, con, facecolors="none", edgecolors="#CC6677", s=70,
               label="content only")
    a2.axhline(PERC_COHERENT, color="k", ls=":", lw=1.2)
    a2.set_xlabel("well-formed post rate")
    a2.set_ylabel("negative log-likelihood (nats/token)")
    a2.set_title("Does the loss fall because the agents improved,\nor because the stream emptied?")
    a2.legend(frameon=False, fontsize=9)
    a2.grid(alpha=0.25)
    f2.colorbar(sc, ax=a2, label="iteration")
    f2.tight_layout()
    out2 = out.with_name("content_perception_scatter.png")
    f2.savefig(out2, dpi=150)
    print(f"wrote {out2}")


if __name__ == "__main__":
    main()
