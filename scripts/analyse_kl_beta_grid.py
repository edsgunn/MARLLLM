#!/usr/bin/env python
"""Re-read the KL/entropy regularisation grid on DELIVERY and on the CHANGE in
policy entropy, rather than on `alive`.

Motivation
----------
The 4x4 sweep and the beta zoom were both originally read on the `alive`
fraction. Across the 20 cells `alive` is uncorrelated with the well-formed post
rate (Spearman -0.10, p = 0.67), so those readings cannot support claims about
the channel. The delivery rate was recorded all along, as
`tool/post_success_total` / `tool/post_turns_total`; no agent is frozen in any
of these runs, so the population-level rate is valid.

The question this answers: does either regulariser produce a setting where
policy entropy RISES while the perception loss does not collapse?

Usage
-----
    .venv/bin/python scripts/analyse_kl_beta_grid.py

Add run directories with --extra LABEL=PATH to compare other runs (e.g. the
reference and the novelty-search arms) through the same window.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from marlllm.regime_tracking import RegimeThresholds, _parse_agent_metrics  # noqa: E402
from analyse_tier_c import mwu_exact  # noqa: E402

THRESHOLDS = RegimeThresholds()
ALIVE = {"coherent_low_kl", "committed", "mid_range"}

EARLY = (1, 25)
LATE_WIDTH = 25

FAMILIES = {
    "sweep": Path("runs/cultural_emergence/kl_beta_sweep_8agent_ashbourne_gc"),
    "zoom": Path("runs/cultural_emergence/beta_zoom_kl0p5_8agent_ashbourne_gc"),
}


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open()]


def window(rows: list[dict], lo: int, hi: int) -> dict | None:
    """Population means over iterations in [lo, hi], plus the delivery rate.

    Delivery is a ratio of summed counts over the window, not a mean of
    per-iteration ratios: iterations differ in how many post turns they contain.
    """
    sel = [r for r in rows if lo <= r.get("iteration", -1) <= hi]
    if not sel:
        return None
    alive: list[float] = []
    perc: list[float] = []
    ent: list[float] = []
    kl: list[float] = []
    success = turns = 0.0
    for row in sel:
        agents = _parse_agent_metrics(row)
        if agents:
            regimes = [
                THRESHOLDS.classify(
                    a.get("kl", 0.0), a.get("perc_loss", 0.0), a.get("entropy", 0.0)
                )
                for a in agents.values()
            ]
            alive.append(sum(r in ALIVE for r in regimes) / len(regimes))
            perc.append(sum(a.get("perc_loss", 0.0) for a in agents.values()) / len(agents))
            ent.append(sum(a.get("entropy", 0.0) for a in agents.values()) / len(agents))
            kl.append(sum(a.get("kl", 0.0) for a in agents.values()) / len(agents))
        success += row.get("tool/post_success_total", 0.0) or 0.0
        turns += row.get("tool/post_turns_total", 0.0) or 0.0
    mean = lambda v: sum(v) / len(v) if v else float("nan")  # noqa: E731
    return {
        "alive": mean(alive),
        "perc": mean(perc),
        "ent": mean(ent),
        "kl": mean(kl),
        "deliv": success / turns if turns else float("nan"),
        "n": len(sel),
    }


def spearman(xs, ys):
    """Spearman rho with a normal-approximation two-sided p, no scipy needed."""

    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        ranks = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                ranks[order[k]] = avg
            i = j + 1
        return ranks

    rx, ry = rank(list(xs)), rank(list(ys))
    n = len(rx)
    mx, my = sum(rx) / n, sum(ry) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sxx = sum((a - mx) ** 2 for a in rx)
    syy = sum((b - my) ** 2 for b in ry)
    rho = sxy / (sxx * syy) ** 0.5 if sxx and syy else float("nan")
    # t approximation, adequate at n = 20 and only ever used for a sanity read
    if n > 2 and abs(rho) < 1:
        import math

        t = rho * math.sqrt((n - 2) / (1 - rho * rho))
        # two-sided p from the normal approximation to t at n-2 df
        p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))
    else:
        p = float("nan")
    return rho, p


def parse_cell_name(name: str) -> tuple[float, float]:
    kl = re.search(r"kl([0-9p]+)", name)
    beta = re.search(r"beta([0-9p]+)", name)
    if not kl or not beta:
        raise ValueError(f"cannot parse kl/beta from {name!r}")
    f = lambda s: float(s.replace("p", "."))  # noqa: E731
    return f(kl.group(1)), f(beta.group(1))


def collect() -> list[dict]:
    out = []
    for family, root in FAMILIES.items():
        if not root.exists():
            continue
        for d in sorted(root.iterdir()):
            metrics = d / "metrics.jsonl"
            if not metrics.exists():
                continue
            kl_coef, beta = parse_cell_name(d.name)
            rows = load(metrics)
            last = int(max(r.get("iteration", 0) for r in rows))
            early = window(rows, *EARLY)
            late = window(rows, last - LATE_WIDTH + 1, last)
            out.append(
                {
                    "family": family,
                    "cell": d.name,
                    "kl": kl_coef,
                    "beta": beta,
                    "last": last,
                    "d_ent": late["ent"] - early["ent"],
                    "d_perc": late["perc"] - early["perc"],
                    "ent_end": late["ent"],
                    "perc_end": late["perc"],
                    "deliv_early": early["deliv"],
                    "deliv_end": late["deliv"],
                    "alive_end": late["alive"],
                    "kl_end": late["kl"],
                }
            )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--extra",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="additional run dir to report through the same window",
    )
    args = ap.parse_args()

    rows = collect()
    if not rows:
        sys.exit("no grid cells found; run from the repo root")

    print(f"=== GRID, {len(rows)} cells: iterations {EARLY[0]}-{EARLY[1]} vs the last {LATE_WIDTH} ===")
    hdr = (
        f"{'fam':6} {'kl':>5} {'beta':>5} | {'ent end':>7} {'d ent':>6} | "
        f"{'perc end':>8} {'d perc':>7} | {'deliv 1-25':>10} {'deliv end':>9} | {'alive':>5}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(rows, key=lambda r: (r["beta"], r["kl"])):
        print(
            f"{r['family']:6} {r['kl']:5.2f} {r['beta']:5.2f} | {r['ent_end']:7.2f} {r['d_ent']:+6.2f} | "
            f"{r['perc_end']:8.2f} {r['d_perc']:+7.2f} | {r['deliv_early']:10.3f} {r['deliv_end']:9.3f} | "
            f"{r['alive_end']:5.2f}"
        )

    print("\n=== 1. does either knob move the channel? ===")
    for key in ("beta", "kl"):
        rho, p = spearman([r[key] for r in rows], [r["deliv_end"] for r in rows])
        print(f"  delivery(end) vs {key:4}: rho = {rho:+.3f}  p = {p:.3f}")
    dv = [r["deliv_end"] for r in rows]
    print(f"  delivery(end): range {min(dv):.3f}-{max(dv):.3f}, mean {sum(dv)/len(dv):.3f}")
    print(f"  cells above 0.8: {sum(1 for v in dv if v > 0.8)} of {len(dv)}")

    print("\n=== 2. is the entropy bonus a step function? ===")
    lo = [r["d_ent"] for r in rows if r["beta"] <= 0.05]
    hi = [r["d_ent"] for r in rows if r["beta"] >= 0.10]
    print(f"  d entropy, beta <= 0.05 (n={len(lo)}): {[round(v, 3) for v in sorted(lo)]}")
    print(f"  d entropy, beta >= 0.10 (n={len(hi)}): {[round(v, 3) for v in sorted(hi)]}")
    print(f"  MWU exact p = {mwu_exact(lo, hi):.6f}")
    if lo and hi:
        print(f"  separation: max(low) = {max(lo):+.3f} vs min(high) = {min(hi):+.3f}")

    print("\n=== 3. are entropy and perception loss locked together? ===")
    rho, p = spearman([r["d_ent"] for r in rows], [r["d_perc"] for r in rows])
    print(f"  d entropy vs d perception: rho = {rho:+.3f}  p = {p:.2e}  (n = {len(rows)})")
    target = [r for r in rows if r["d_ent"] > 0.10 and r["d_perc"] <= 0.0]
    print(f"  cells with entropy up and perception NOT up: {len(target)}")
    for r in target:
        print(f"    kl={r['kl']:.2f} beta={r['beta']:.2f} [{r['family']}]  "
              f"d ent {r['d_ent']:+.2f}  d perc {r['d_perc']:+.2f}  deliv {r['deliv_end']:.3f}")

    print("\n=== 4. does the KL anchor suppress the entropy rise? (beta = 0.10) ===")
    for r in sorted([r for r in rows if r["beta"] == 0.10], key=lambda r: r["kl"]):
        print(f"  kl = {r['kl']:4.2f} [{r['family']:5}]  d ent {r['d_ent']:+.2f}  "
              f"d perc {r['d_perc']:+.2f}  deliv {r['deliv_end']:.3f}")

    print("\n=== 5. does `alive` track the channel? ===")
    rho, p = spearman([r["alive_end"] for r in rows], [r["deliv_end"] for r in rows])
    print(f"  alive(end) vs delivery(end): rho = {rho:+.3f}  p = {p:.3f}")

    dupes: dict[tuple[float, float], list[dict]] = {}
    for r in rows:
        dupes.setdefault((r["kl"], r["beta"]), []).append(r)
    same = {k: v for k, v in dupes.items() if len(v) > 1}
    if same:
        print("\n=== 6. the noise floor: same config, same seed, run twice ===")
        for (kl_coef, beta), group in same.items():
            print(f"  kl = {kl_coef}, beta = {beta}:")
            for r in group:
                print(f"    [{r['family']:5}] d ent {r['d_ent']:+.3f}  "
                      f"d perc {r['d_perc']:+.3f}  deliv {r['deliv_end']:.3f}")
            for field, label in (("d_ent", "d entropy"), ("d_perc", "d perception"), ("deliv_end", "delivery")):
                vals = [r[field] for r in group]
                print(f"    spread in {label:12}: {max(vals) - min(vals):.3f}")

    if args.extra:
        print("\n=== other runs, same window ===")
        for spec in args.extra:
            label, _, path = spec.partition("=")
            metrics = Path(path) / "metrics.jsonl"
            if not metrics.exists():
                print(f"  {label:28} MISSING {metrics}")
                continue
            r = load(metrics)
            last = int(max(x.get("iteration", 0) for x in r))
            early = window(r, *EARLY)
            late = window(r, last - LATE_WIDTH + 1, last)
            print(
                f"  {label:28} d ent {late['ent'] - early['ent']:+.2f}  "
                f"d perc {late['perc'] - early['perc']:+.2f}  "
                f"deliv {early['deliv']:.3f} -> {late['deliv']:.3f}  (to iter {last})"
            )


if __name__ == "__main__":
    main()
