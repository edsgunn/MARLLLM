#!/usr/bin/env python
"""Novelty-search population-size sweep: delivery and entropy against N, each
arm read against the reference run at the same N.

Pre-registered reading (configs/novelty_search/novelty_size_*.yaml):
  - the well-formed post rate at iteration 300 as a function of N;
  - read on DELIVERY and on the traces, NOT on `alive`;
  - falsifier for "size matters under maximisation": delivery flat across N.

Void conditions, checked here: 300 iterations, and `return_sign` = -1.0 on
every agent.

Usage:  .venv/bin/python scripts/analyse_novelty_size.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from marlllm.regime_tracking import RegimeThresholds, _parse_agent_metrics  # noqa: E402
from analyse_tier_c import mwu_exact  # noqa: E402
from analyse_kl_beta_grid import spearman, window, load  # noqa: E402

THRESHOLDS = RegimeThresholds()

# N -> (reference run, [novelty arms])
ARMS: dict[int, tuple[str, list[str]]] = {
    2: (
        "runs/cultural_emergence/run10_2agent_7B_study_group_ashbourne_gc",
        [
            "runs/novelty_search/novelty_size_02agent_ashbourne_gc_seed1",
            "runs/novelty_search/novelty_size_02agent_ashbourne_gc_seed2",
        ],
    ),
    4: (
        "runs/cultural_emergence/run10_4agent_7B_study_group_ashbourne_gc",
        [
            "runs/novelty_search/novelty_size_04agent_ashbourne_gc_seed1",
            "runs/novelty_search/novelty_size_04agent_ashbourne_gc_seed2",
        ],
    ),
    8: (
        "runs/cultural_emergence/run7_8agent_7B_study_group_ashbourne_gc",
        [
            "runs/novelty_search/c13_novelty_search_8agent_ashbourne_gc_seed1",
            "runs/novelty_search/c13_novelty_search_8agent_ashbourne_gc_seed2",
        ],
    ),
    16: (
        "runs/cultural_emergence/run8_16agent_7B_study_group_ashbourne_gc",
        [
            "runs/novelty_search/novelty_size_16agent_ashbourne_gc_seed1",
            "runs/novelty_search/novelty_size_16agent_ashbourne_gc_seed2",
        ],
    ),
}

CKPTS = [25, 50, 100, 150, 200, 250, 300]


def check_void(path: Path) -> tuple[bool, str]:
    """300 iterations and return_sign == -1.0 on every agent."""
    m = path / "metrics.jsonl"
    if not m.exists():
        return False, "no metrics.jsonl"
    rows = load(m)
    last_iter = int(max(r.get("iteration", 0) for r in rows))
    if last_iter < 300:
        return False, f"only {last_iter} iterations"
    signs = {k: v for k, v in rows[-1].items() if "return_sign" in k}
    if not signs:
        return False, "return_sign absent"
    if set(signs.values()) != {-1.0}:
        return False, f"return_sign = {set(signs.values())}"
    return True, f"{last_iter} iters, return_sign -1.0 on {len(signs)} agents"


def delivery_at(rows: list[dict], centre: int, half: int = 12) -> float:
    """Summed-count delivery rate in a window about `centre`."""
    sel = [r for r in rows if abs(r.get("iteration", -1) - centre) <= half]
    s = sum(r.get("tool/post_success_total", 0.0) or 0.0 for r in sel)
    t = sum(r.get("tool/post_turns_total", 0.0) or 0.0 for r in sel)
    return s / t if t else float("nan")


def main() -> None:
    print("=== VOID CHECK ===")
    usable: dict[int, list[Path]] = {}
    for n, (_ref, arms) in ARMS.items():
        for a in arms:
            p = Path(a)
            ok, why = check_void(p)
            print(f"  {'PASS' if ok else 'SKIP'}  N={n:2}  {p.name:44} {why}")
            if ok:
                usable.setdefault(n, []).append(p)

    print("\n=== DELIVERY AND ENTROPY, novelty arms against their own reference ===")
    hdr = (f"{'N':>3} {'run':44} | {'deliv 1-25':>10} {'deliv end':>9} | "
           f"{'ent 1-25':>8} {'ent end':>7} {'d ent':>6} | {'perc end':>8} {'d perc':>7}")
    print(hdr)
    print("-" * len(hdr))
    table: dict[int, dict] = {}
    for n in sorted(ARMS):
        ref, _ = ARMS[n]
        entries: list[tuple[str, Path]] = [("reference (+1)", Path(ref))]
        entries += [(f"novelty (-1)", p) for p in usable.get(n, [])]
        for label, p in entries:
            m = p / "metrics.jsonl"
            if not m.exists():
                print(f"{n:3} {p.name:44} | MISSING")
                continue
            rows = load(m)
            last = int(max(r.get("iteration", 0) for r in rows))
            e = window(rows, 1, 25)
            l = window(rows, last - 24, last)
            print(f"{n:3} {p.name:44} | {e['deliv']:10.3f} {l['deliv']:9.3f} | "
                  f"{e['ent']:8.2f} {l['ent']:7.2f} {l['ent']-e['ent']:+6.2f} | "
                  f"{l['perc']:8.2f} {l['perc']-e['perc']:+7.2f}")
            rec = table.setdefault(n, {"ref": None, "nov": []})
            if label.startswith("reference"):
                rec["ref"] = (l["deliv"], l["ent"] - e["ent"], l["perc"] - e["perc"])
            else:
                rec["nov"].append((l["deliv"], l["ent"] - e["ent"], l["perc"] - e["perc"]))

    print("\n=== THE PRE-REGISTERED TEST: is delivery flat across N under maximisation? ===")
    ns, devs = [], []
    for n in sorted(table):
        for d, _, _ in table[n]["nov"]:
            ns.append(n)
            devs.append(d)
    if len(set(ns)) > 2:
        rho, p = spearman(ns, devs)
        print(f"  delivery(end) vs N, novelty arms only: rho = {rho:+.3f}  p = {p:.3f}  (n = {len(ns)})")
    print(f"  spread across all novelty arms: {min(devs):.3f} - {max(devs):.3f}")
    print("  NOISE FLOOR for delivery on this rig is 0.173 (same config, same seed;")
    print("  runs/cultural_emergence/kl_beta_sweep_8agent_ashbourne_gc/NOTES.md).")
    print("  Within-N seed spread, which any trend must clear:")
    for n in sorted(table):
        ds = [d for d, _, _ in table[n]["nov"]]
        if len(ds) > 1:
            print(f"    N={n:2}: {ds[0]:.3f} vs {ds[1]:.3f}   spread {abs(ds[0]-ds[1]):.3f}")

    print("\n=== SIGN EFFECT AT EACH N: novelty minus its own reference ===")
    print(f"  {'N':>3} {'ref deliv':>9} {'nov deliv (2 seeds)':>22} {'delta of means':>15}")
    ref_d, nov_d = [], []
    for n in sorted(table):
        rec = table[n]
        if rec["ref"] is None or not rec["nov"]:
            continue
        r = rec["ref"][0]
        vs = [d for d, _, _ in rec["nov"]]
        mean = sum(vs) / len(vs)
        ref_d.append(r)
        nov_d.append(mean)
        print(f"  {n:3} {r:9.3f} {'  '.join(f'{v:.3f}' for v in vs):>22} {mean - r:+15.3f}")
    if len(ref_d) > 1:
        print(f"  MWU exact, ref vs novelty-mean across N: p = {mwu_exact(ref_d, nov_d):.4f}  (n = {len(ref_d)} per side)")
        print("  (paired by N, so this is a weak test with 4 sizes -- report the per-N deltas.)")

    print("\n=== DELIVERY THROUGH TRAINING (novelty arms) ===")
    hdr2 = f"{'N':>3} {'seed':>4} | " + " ".join(f"{c:>6}" for c in CKPTS)
    print(hdr2)
    print("-" * len(hdr2))
    for n in sorted(usable):
        for p in usable[n]:
            rows = load(p / "metrics.jsonl")
            seed = p.name[-1]
            vals = [delivery_at(rows, c) for c in CKPTS]
            print(f"{n:3} {seed:>4} | " + " ".join(f"{v:6.3f}" for v in vals))
        ref = Path(ARMS[n][0])
        if (ref / "metrics.jsonl").exists():
            rows = load(ref / "metrics.jsonl")
            vals = [delivery_at(rows, c) for c in CKPTS]
            print(f"{n:3} {'ref':>4} | " + " ".join(f"{v:6.3f}" for v in vals))


if __name__ == "__main__":
    main()
