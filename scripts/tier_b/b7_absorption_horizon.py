#!/usr/bin/env python3
"""
B7 — IS THE ABSORPTION MEASUREMENT TAKEN WHILE THE POPULATION IS STILL HEALTHY?

WHAT THIS CHANGES
    Nothing about the absorption analysis. It asks where in each population's
    life the PMI measurements sit. B2 shows the alive fraction peaks around
    iteration 100 and falls thereafter, and that no run recovers after 150. The
    absorption window runs from iteration 25 to 300. So some fraction of the
    measurements that produce the headline shrinkage are taken from populations
    that are already coming apart.

    Absorption is measured as a FIRST-MINUS-LAST difference. If the last
    checkpoint of a run is taken deep into that run's decline, the "shrinkage"
    is partly the difference between a healthy population and a failing one —
    which is exactly the artefact the original report identified for the
    text-degenerate stratum and fixed with contemporaneous-regime labelling.
    Contemporaneous labelling fixes the LABEL. It does not stop the last
    checkpoint from being late.

WHAT WE EXPECT TO LEARN
    For each stratum, the share of measurements taken after the run's alive
    fraction has begun to fall, and the shrinkage recomputed using only
    measurements taken at or before the peak.

WHAT WOULD FALSIFY THE CONCERN
    If the committed-vs-coherent contrast has the same sign and comparable
    magnitude on pre-peak measurements only, the headline is not a horizon
    artefact and the concern is disposed of.
    If the contrast shrinks toward zero or reverses, part of what has been
    called absorption is the population declining.

    Note the tension to state either way: restricting to pre-peak measurements
    shortens the window, and absorption needs training to happen in it. A
    smaller contrast on a shorter window is ambiguous between "artefact" and
    "less time to absorb".

INPUT   _absorption_by_regime/absorption_{cross,seeds}_with_regime.csv
OUTPUT  runs/_tier_b/b7_absorption_horizon/
"""
from __future__ import annotations
import csv, json, os, statistics, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C

ABS = os.path.join(C.ROOT, "runs/cultural_emergence/_absorption_by_regime")
GRID = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300]


def alive_curve(run_dir):
    """iteration -> alive fraction, for one run."""
    rel = os.path.relpath(run_dir, C.ROOT) if run_dir.startswith("/") else run_dir
    recs = C.load_metrics(rel)
    if not recs:
        return {}
    out = {}
    for t in GRID:
        pop = C.population_at(recs, t)
        if pop:
            out[t] = pop["alive"]
    return out


def peak_iter(curve):
    return max(curve, key=lambda t: (curve[t], -t)) if curve else None


def below_best(curve, it, tol=6):
    """Is the population strictly below its own best alive fraction at ~it?

    Stricter and better posed than "after the argmax": a run that holds at
    alive = 1.0 from iteration 25 to 200 has its argmax at 25, but nothing has
    declined there.
    """
    if not curve:
        return None
    near = [t for t in curve if abs(t - it) <= tol]
    if not near:
        return None
    return curve[min(near, key=lambda t: abs(t - it))] < max(curve.values()) - 1e-9


def load(name):
    p = os.path.join(ABS, name)
    return list(csv.DictReader(open(p))) if os.path.exists(p) else []


def shrinkage(rows, regime_key):
    """Mean first-iteration gap minus mean last-iteration gap, per stratum.

    Gaps are averaged WITHIN the first and last iteration of each agent before
    differencing, matching stats_shrinkage_pooled.csv. Taking single episodes
    instead biases first low and last high.
    """
    by = {}
    for r in rows:
        try:
            it, gap = int(float(r["iter"])), float(r["gap"])
        except (ValueError, KeyError):
            continue
        by.setdefault((r["run_dir"], r["character"]), []).append((it, gap, r.get(regime_key)))
    firsts, lasts = {}, {}
    for k, v in by.items():
        its = sorted({x[0] for x in v})
        if len(its) < 2:
            continue
        fi, li = its[0], its[-1]
        reg = next((x[2] for x in v if x[0] == li), None)
        if not reg:
            continue
        firsts.setdefault(reg, []).append(statistics.mean(x[1] for x in v if x[0] == fi))
        lasts.setdefault(reg, []).append(statistics.mean(x[1] for x in v if x[0] == li))
    out = {}
    for reg in firsts:
        f, l = statistics.mean(firsts[reg]), statistics.mean(lasts[reg])
        out[reg] = (round(f - l, 4), len(firsts[reg]), round(f, 4), round(l, 4))
    return out


def main():
    out = C.outdir("b7_absorption_horizon")
    curves, peaks = {}, {}
    report = {}

    for tag, fname in (("cross", "absorption_cross_with_regime.csv"),
                       ("seeds", "absorption_seeds_with_regime.csv")):
        rows = load(fname)
        if not rows:
            print(f"  (missing) {fname}")
            continue
        for r in rows:
            rd = r["run_dir"]
            if rd not in curves:
                curves[rd] = alive_curve(rd)
                peaks[rd] = peak_iter(curves[rd])
        tagged, kept = [], []
        for r in rows:
            pk = peaks.get(r["run_dir"])
            try:
                it = int(float(r["iter"]))
            except ValueError:
                continue
            post = below_best(curves.get(r["run_dir"], {}), it)
            post = bool(post)
            r["_post_peak"] = post
            tagged.append(r)
            if not post:
                kept.append(r)
        n_post = sum(1 for r in tagged if r["_post_peak"])
        print(f"=== {tag}: {len(tagged)} PMI measurements, {n_post} "
              f"({n_post/len(tagged):.1%}) taken while the population was already "
              f"below its own best alive fraction")

        full = shrinkage(tagged, "regime")          # end-of-training, matches Table 1
        pre = shrinkage(kept, "regime")
        print(f"{'regime':<20}{'shrinkage (all)':>18}{'shrinkage (while healthy)':>24}")
        for k in C.ORDER:
            a = full.get(k); b = pre.get(k)
            print(f"{k:<20}"
                  f"{(f'{a[0]:+.3f} (n{a[1]})' if a else '—'):>18}"
                  f"{(f'{b[0]:+.3f} (n{b[1]})' if b else '—'):>24}")
        coh_a, com_a = full.get("coherent_low_kl"), full.get("committed")
        coh_b, com_b = pre.get("coherent_low_kl"), pre.get("committed")
        if coh_a and com_a:
            print(f"  committed - coherent, all measurements:  {com_a[0]-coh_a[0]:+.3f}")
        if coh_b and com_b:
            print(f"  committed - coherent, pre-peak only:     {com_b[0]-coh_b[0]:+.3f}")
        report[tag] = {"n": len(tagged), "n_post_peak": n_post,
                       "all": {k: list(v) for k, v in full.items()},
                       "pre_peak": {k: list(v) for k, v in pre.items()}}
        print()

    with open(os.path.join(out, "peaks.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["run_dir", "peak_iter", "alive_at_peak", "alive_at_last"])
        for rd, c in curves.items():
            pk = peaks[rd]
            w.writerow([os.path.relpath(rd, C.ROOT) if rd.startswith("/") else rd,
                        pk, c.get(pk), c[max(c)] if c else None])
    json.dump(report, open(os.path.join(out, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
