#!/usr/bin/env python3
"""
B2 — IS A REGIME AN OUTCOME, OR A SNAPSHOT ON A TRAJECTORY?

WHAT THIS CHANGES
    The paper reports "end-of-training metrics" at iteration 200 and calls the
    result a taxonomy of TRAINING OUTCOMES. This recomputes the same
    classification at a grid of iterations for every run, so the word "outcome"
    can be checked rather than assumed.

WHAT WE EXPECT TO LEARN
    Whether the alive fraction is stationary after some iteration (in which case
    200 is a converged state and "outcome" is the right word), or whether it has
    a systematic trajectory (in which case every regime claim in the paper is a
    claim about iteration 200 specifically and must say so).

    Reported on a BALANCED PANEL — only runs observed at every grid point — so
    the trend cannot be an artefact of short runs dropping out.

WHAT WOULD FALSIFY THE CONCERN
    A flat mean alive fraction from ~iteration 100 onward, or a mixture of runs
    rising and falling with no net direction, would mean iteration 200 is as good
    a summary as any and the paper needs no change.
    A monotone decline with runs rising as often as falling would mean the mean
    is driven by a few runs, not by a general property.

WHAT WOULD MAKE IT WORSE THAN STATED
    If NO run ever improves after some iteration, the decline is not noise and
    the enabling state is not an attractor.

OUTPUT  runs/_tier_b/b2_regime_by_iteration/
"""
from __future__ import annotations
import csv, json, os, statistics, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C

GRID = [25, 100, 150, 200, 300]


def main():
    out = C.outdir("b2_regime_by_iteration")
    rows, panel = [], {}
    for rel in C.live_runs():
        recs = C.load_metrics(rel)
        if not recs:
            continue
        pts = {}
        for t in GRID:
            pop = C.population_at(recs, t)
            if pop is None:
                continue
            pts[t] = pop
            rows.append({"run": rel, "env": C.env_of(recs), "target": t,
                         "iter": pop["iter"], "n": pop["n"],
                         "alive": round(pop["alive"], 4),
                         **{k: pop["counts"].get(k, 0) for k in C.ORDER}})
        if pts:
            panel[rel] = pts

    with open(os.path.join(out, "regime_by_iteration.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["run", "env", "target", "iter", "n",
                                           "alive", *C.ORDER])
        w.writeheader(); w.writerows(rows)

    ce = {k: v for k, v in panel.items() if k.startswith("runs/cultural_emergence/")}
    bal = {k: v for k, v in ce.items() if all(t in v for t in GRID)}
    print(f"cultural-emergence runs with any point: {len(ce)}")
    print(f"balanced panel (all of {GRID}): n = {len(bal)}\n")
    print(f"{'iteration':>10}{'mean alive':>12}{'median':>9}{'runs at 0':>11}{'runs at 1':>11}")
    means = {}
    for t in GRID:
        xs = [v[t]["alive"] for v in bal.values()]
        means[t] = statistics.mean(xs)
        print(f"{t:>10}{means[t]:>12.3f}{statistics.median(xs):>9.3f}"
              f"{sum(1 for x in xs if x == 0):>11}{sum(1 for x in xs if x == 1):>11}")

    print(f"\n{'transition':>14}{'down':>7}{'same':>7}{'up':>5}{'n':>5}")
    trans = {}
    for a, b in zip(GRID, GRID[1:]):
        d = s = u = 0
        for v in bal.values():
            x, y = v[a]["alive"], v[b]["alive"]
            d, s, u = (d + (y < x), s + (y == x), u + (y > x))
        trans[f"{a}->{b}"] = {"down": d, "same": s, "up": u}
        print(f"{a:>6} -> {b:<5}{d:>7}{s:>7}{u:>5}{len(bal):>5}")

    below = sum(1 for v in bal.values()
                if v[GRID[-1]]["alive"] < max(v[t]["alive"] for t in GRID[:-1]))
    print(f"\nruns ending below their own earlier peak: {below}/{len(bal)}")
    peak = {}
    for v in bal.values():
        b = max(GRID, key=lambda t: (v[t]["alive"], -t))
        peak[b] = peak.get(b, 0) + 1
    print(f"iteration of maximum alive fraction (ties -> earliest): {peak}")

    json.dump({"panel_n": len(bal), "mean_alive": {str(k): round(v, 4) for k, v in means.items()},
               "transitions": trans, "end_below_peak": below, "peak_iteration": peak},
              open(os.path.join(out, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
