#!/usr/bin/env python3
"""
Plot the variance-decomposition diagnostic for a training run.

Reads ``{run_dir}/variance_decomposition.jsonl`` (one record per eval) and
produces a 4-panel figure per agent:

  (a) signal over iterations
  (b) noise over iterations
  (c) SNR over iterations on a log y-axis
  (d) signal vs noise scatter, coloured by iteration

When the population has > 4 agents the panels are stacked vertically.

Usage:
    python scripts/plot_variance_decomposition.py runs/my_run
    python scripts/plot_variance_decomposition.py runs/my_run --out custom.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "lines.linewidth": 1.4,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def load_records(path: Path) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def per_agent_series(records: list[dict]) -> dict[str, dict[str, np.ndarray]]:
    """Pivot the JSONL into ``{agent_name: {iter, signal, noise, snr}}`` arrays."""
    by_agent: dict[str, dict[str, list[float]]] = {}
    for rec in records:
        it = rec["iteration"]
        for name, agg in rec.get("agents", {}).items():
            d = by_agent.setdefault(name, {"iter": [], "signal": [], "noise": [], "snr": []})
            d["iter"].append(it)
            d["signal"].append(agg["signal"])
            d["noise"].append(agg["noise"])
            d["snr"].append(agg["snr"])
    return {n: {k: np.asarray(v) for k, v in d.items()} for n, d in by_agent.items()}


def plot_one_agent(axes_row, name: str, series: dict[str, np.ndarray]) -> None:
    ax_sig, ax_noi, ax_snr, ax_scat = axes_row
    it = series["iter"]
    sig = series["signal"]
    noi = series["noise"]
    snr = series["snr"]

    ax_sig.plot(it, sig, marker=".", color="C0")
    ax_sig.set_title(f"{name} — signal")
    ax_sig.set_xlabel("iteration")
    ax_sig.set_ylabel("Var_a[E[G|a]]")

    ax_noi.plot(it, noi, marker=".", color="C1")
    ax_noi.set_title(f"{name} — noise")
    ax_noi.set_xlabel("iteration")
    ax_noi.set_ylabel("E_a[Var[G|a]]")

    ax_snr.plot(it, np.maximum(snr, 1e-12), marker=".", color="C2")
    ax_snr.set_yscale("log")
    ax_snr.set_title(f"{name} — SNR (log)")
    ax_snr.set_xlabel("iteration")
    ax_snr.set_ylabel("signal / noise")
    ax_snr.axhline(1.0, color="k", linestyle="--", linewidth=0.7, alpha=0.5)

    sc = ax_scat.scatter(noi, sig, c=it, cmap="viridis", s=18)
    ax_scat.set_xlabel("noise")
    ax_scat.set_ylabel("signal")
    ax_scat.set_title(f"{name} — signal vs noise")
    # Diagonal: signal = noise (SNR=1).
    if len(noi) and len(sig):
        lo = float(min(noi.min(), sig.min(), 1e-12))
        hi = float(max(noi.max(), sig.max(), lo * 1.01))
        ax_scat.plot([lo, hi], [lo, hi], color="k", linestyle="--", linewidth=0.7, alpha=0.4)
    plt.colorbar(sc, ax=ax_scat, label="iteration")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path, help="Training run directory")
    ap.add_argument("--out", type=Path, default=None, help="Output PNG path")
    args = ap.parse_args()

    jsonl = args.run_dir / "variance_decomposition.jsonl"
    if not jsonl.exists():
        print(f"No variance_decomposition.jsonl at {jsonl}", file=sys.stderr)
        return 1

    records = load_records(jsonl)
    if not records:
        print(f"No records in {jsonl}", file=sys.stderr)
        return 1
    series = per_agent_series(records)
    if not series:
        print(f"No per-agent data in {jsonl}", file=sys.stderr)
        return 1

    n_agents = len(series)
    fig, axes = plt.subplots(
        n_agents, 4,
        figsize=(16, 3.2 * n_agents),
        squeeze=False,
    )
    for row, (name, s) in enumerate(sorted(series.items())):
        plot_one_agent(axes[row], name, s)

    fig.tight_layout()
    out = args.out or (args.run_dir / "variance_decomposition.png")
    fig.savefig(out)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
