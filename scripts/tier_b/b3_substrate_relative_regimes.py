#!/usr/bin/env python3
"""
B3 — IS THE PERCEPTION-LOSS THRESHOLD MEASURING BREAKDOWN, OR SUBSTRATE DIFFICULTY?

WHAT THIS CHANGES
    The classifier calls an agent `text_degenerate` at perception loss >= 2.0
    and `coherent` below 1.7, on every substrate. Those cuts were fixed by
    visual inspection at iteration 200 on the Study Group substrates. This
    measures each substrate's UNTRAINED perception loss (iteration 1, LoRA at
    initialisation, KL from base = 0) and asks whether the absolute cuts mean
    the same thing everywhere.

WHAT WE EXPECT TO LEARN
    The baseline per substrate, and what happens to the classification when the
    threshold is expressed relative to it:
        absolute (paper)  degenerate if perc >= 2.0;      coherent if < 1.7
        relative          degenerate if perc >= baseline;  coherent if < 0.85*baseline
    The relative convention makes "degenerate" mean "no better than the
    untrained model on this substrate", which is what the name implies.

WHAT WOULD FALSIFY THE CONCERN
    If every substrate's untrained perception loss sits well below 1.7, the
    absolute cuts are safe and nothing changes. The concern is specific: if any
    substrate starts ABOVE 2.0, then on that substrate the untrained model is
    already classified degenerate, and "degenerate" there conflates "has broken"
    with "never learned".

WHAT THIS CANNOT SETTLE
    Which convention is right. The relative one is better named; it is not
    obviously better calibrated, and its 0.85 factor is as arbitrary as 1.7 was.
    The value here is the diagnosis and the size of the disagreement, not a new
    set of blessed cuts.

OUTPUT  runs/_tier_b/b3_substrate_relative_regimes/
"""
from __future__ import annotations
import csv, json, os, statistics, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C

COH_FRAC = 0.85
GRID = [25, 100, 200, 300]


def baselines():
    """Substrate -> mean iteration-1 population perception loss."""
    acc = {}
    for rel in C.live_runs():
        recs = C.load_metrics(rel)
        if not recs:
            continue
        env = C.env_of(recs)
        rec = C.record_at(recs, 1, tol=2)
        if env is None or rec is None:
            continue
        per, _ = C.split_record(rec)
        v = C.mean([m.get("perc_loss") for m in per.values()])
        if v is not None:
            acc.setdefault(env, []).append(v)
    return {e: (statistics.mean(v), len(v)) for e, v in acc.items()}


def main():
    out = C.outdir("b3_substrate_relative_regimes")
    base = baselines()

    print(f"{'substrate':<34}{'untrained perc':>16}{'runs':>6}   vs absolute cuts")
    brows = []
    for env, (b, n) in sorted(base.items(), key=lambda kv: -kv[1][0]):
        flag = ("ABOVE the 2.0 degenerate cut" if b >= 2.0 else
                "above the 1.7 coherent cut" if b >= 1.7 else "below both")
        print(f"{env:<34}{b:>16.3f}{n:>6}   {flag}")
        brows.append({"substrate": env, "untrained_perc_loss": round(b, 4),
                      "n_runs": n, "verdict": flag})
    with open(os.path.join(out, "substrate_baselines.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["substrate", "untrained_perc_loss",
                                           "n_runs", "verdict"])
        w.writeheader(); w.writerows(brows)

    rows, disagree, total = [], 0, 0
    for rel in C.live_runs():
        recs = C.load_metrics(rel)
        env = C.env_of(recs)
        if not recs or env not in base:
            continue
        b = base[env][0]
        for t in GRID:
            abs_pop = C.population_at(recs, t)
            rel_pop = C.population_at(recs, t, perc_coh=COH_FRAC * b, perc_degen=b)
            if abs_pop is None or rel_pop is None:
                continue
            am = {a["agent"]: a["regime"] for a in abs_pop["agents"]}
            rm = {a["agent"]: a["regime"] for a in rel_pop["agents"]}
            d = sum(1 for k in am if am[k] != rm.get(k))
            disagree += d; total += len(am)
            rows.append({"run": rel, "env": env, "target": t,
                         "iter": abs_pop["iter"], "n": abs_pop["n"],
                         "alive_absolute": round(abs_pop["alive"], 4),
                         "alive_relative": round(rel_pop["alive"], 4),
                         "n_label_disagreements": d,
                         "baseline": round(b, 4)})
    with open(os.path.join(out, "absolute_vs_relative.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["run", "env", "target", "iter", "n",
                                           "alive_absolute", "alive_relative",
                                           "n_label_disagreements", "baseline"])
        w.writeheader(); w.writerows(rows)

    print(f"\nper-agent label disagreement between conventions: "
          f"{disagree}/{total} = {disagree/total:.1%}")

    print(f"\nThe cell this most affects — iteration 25, when nothing has trained:")
    print(f"{'run':<50}{'env':<32}{'abs':>6}{'rel':>6}")
    for r in rows:
        if r["target"] == 25 and r["alive_absolute"] == 0 and r["alive_relative"] > 0:
            print(f"{r['run'].split('/')[-1][:49]:<50}{r['env'][:31]:<32}"
                  f"{r['alive_absolute']:>6.2f}{r['alive_relative']:>6.2f}")

    n25 = [r for r in rows if r["target"] == 25]
    dead_abs = sum(1 for r in n25 if r["alive_absolute"] == 0)
    dead_rel = sum(1 for r in n25 if r["alive_relative"] == 0)
    print(f"\nat iteration 25: {dead_abs}/{len(n25)} runs read as 0% alive under the "
          f"absolute cuts, {dead_rel}/{len(n25)} under the relative ones")
    json.dump({"baselines": {k: round(v[0], 4) for k, v in base.items()},
               "disagreement_rate": round(disagree / total, 4) if total else None,
               "iter25_zero_alive_absolute": dead_abs,
               "iter25_zero_alive_relative": dead_rel, "iter25_runs": len(n25)},
              open(os.path.join(out, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
