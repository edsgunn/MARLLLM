#!/usr/bin/env python3
"""
B1 — IS THE CROSS-POPULATION POLICY DISTANCE DRIVEN BY CULTURE OR BY COLLAPSE?

WHAT THIS CHANGES
    Adds the regime restriction to Diagnostic 3 that Diagnostic 1 already
    applied to trajectory KL. D3's headline (policy distance 0.585 / 1.359 /
    1.349 at end of training) is computed over FULL populations. Two pages
    earlier, D2 established that collapsed adapters are dramatically more
    confident than their siblings (policy_collapse mean entropy difference
    -0.212, n=5). Two adapters that have each collapsed to a DIFFERENT
    near-deterministic mode are genuinely far apart in KL — so some unknown
    share of the headline may be divergence-by-collapse, not by culture.

    D1 did the equivalent restriction for the realisation-level trajectory KL
    and the signal ROSE 20-40%. D3 was never restricted. This is the control.

WHAT WE EXPECT TO LEARN
    How much of the policy-distance headline survives when the pairs are
    restricted to characters that are in an alive regime in both populations,
    and further to characters coherent in both.

    Symmetric KL has no source population, so unlike D1 there is no
    "source_coherent" variant. Four filters:
        full          every (character, seed-pair)
        either_alive  the character is alive in at least one of the two
        both_alive    alive in both  <- the defensible headline
        both_coherent coherent_low_kl in both  <- strictest

WHAT WOULD FALSIFY THE HEADLINE
    If both_alive is materially SMALLER than full, a substantial part of the
    reported divergence is collapse, and the number the paper proposes to lead
    with is not measuring what it claims.
    If both_alive is >= full (as D1 found), the finding is not a collapse
    artefact and is strengthened, exactly as the trajectory-KL version was.
    If the restriction leaves too few pairs to estimate (n < ~20 at the end
    iteration), the control is inconclusive rather than supportive.

INPUTS   _diagnostics_d2d3/d3_sym_kl_pairs.csv     (1,632 pair-rows)
         _diagnostics_d1_coherent_only/regimes_*.csv (end-of-training labels)

KNOWN LIMITATION
    The regime labels are end-of-training, not contemporaneous — the same
    limitation D1 carries. An agent that collapses late is treated as collapsed
    throughout. Stated, not fixed.

OUTPUT   runs/_tier_b/b1_policy_distance_by_regime/
"""
from __future__ import annotations
import csv, json, os, statistics, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C

D2D3 = os.path.join(C.ROOT, "runs/cultural_emergence/_diagnostics_d2d3")
D1 = os.path.join(C.ROOT, "runs/cultural_emergence/_diagnostics_d1_coherent_only")
DATASETS = {"ashbourne16": "regimes_ashbourne16.csv",
            "innerloop": "regimes_innerloop.csv",
            "marginnotes": "regimes_marginnotes.csv"}
FILTERS = ["full", "either_alive", "both_alive", "both_coherent"]


def load_regimes():
    """-> {(dataset, seed, character): regime}"""
    out = {}
    for ds, fn in DATASETS.items():
        p = os.path.join(D1, fn)
        if not os.path.exists(p):
            print(f"  missing {p}")
            continue
        for r in csv.DictReader(open(p)):
            out[(ds, r["seed"], r["character"])] = r["regime"]
    return out


def keep(reg, ds, char, a, b, mode):
    ra = reg.get((ds, a, char)); rb = reg.get((ds, b, char))
    if ra is None or rb is None:
        return False
    if mode == "full":
        return True
    aa, ab = ra in C.ALIVE, rb in C.ALIVE
    if mode == "either_alive":
        return aa or ab
    if mode == "both_alive":
        return aa and ab
    if mode == "both_coherent":
        return ra == "coherent_low_kl" and rb == "coherent_low_kl"
    raise ValueError(mode)


def main():
    out = C.outdir("b1_policy_distance_by_regime")
    reg = load_regimes()
    pairs = list(csv.DictReader(open(os.path.join(D2D3, "d3_sym_kl_pairs.csv"))))
    print(f"{len(pairs)} pair-rows, {len(reg)} (dataset, seed, character) regime labels\n")

    agg = {}   # (dataset, iter, mode) -> [values]
    for r in pairs:
        ds, ch, it = r["dataset"], r["character"], int(r["iter"])
        a, b, v = r["adapter_a"], r["adapter_b"], float(r["mean_sym_kl"])
        for mode in FILTERS:
            if keep(reg, ds, ch, a, b, mode):
                agg.setdefault((ds, it, mode), []).append(v)

    rows = []
    for (ds, it, mode), vs in sorted(agg.items()):
        rows.append({"dataset": ds, "iter": it, "filter": mode, "n_pairs": len(vs),
                     "mean_policy_dist": round(statistics.mean(vs), 6),
                     "se": round(statistics.stdev(vs) / len(vs) ** 0.5, 6) if len(vs) > 1 else 0.0})
    with open(os.path.join(out, "policy_distance_by_filter.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["dataset", "iter", "filter", "n_pairs",
                                           "mean_policy_dist", "se"])
        w.writeheader(); w.writerows(rows)

    summary = {}
    for ds in DATASETS:
        its = sorted({r["iter"] for r in rows if r["dataset"] == ds})
        if not its:
            continue
        print(f"=== {ds}")
        print(f"{'iter':>6}" + "".join(f"{m:>22}" for m in FILTERS))
        for it in its:
            line = f"{it:>6}"
            for m in FILTERS:
                hit = [r for r in rows if r["dataset"] == ds and r["iter"] == it and r["filter"] == m]
                line += (f"{hit[0]['mean_policy_dist']:>15.4f} (n{hit[0]['n_pairs']:>3})"
                         if hit else f"{'—':>22}")
            print(line)
        end = max(its)
        vals = {m: next((r for r in rows if r["dataset"] == ds and r["iter"] == end
                         and r["filter"] == m), None) for m in FILTERS}
        f_, ba = vals["full"], vals["both_alive"]
        if f_ and ba and f_["mean_policy_dist"]:
            ratio = ba["mean_policy_dist"] / f_["mean_policy_dist"]
            print(f"   end iter {end}: both_alive / full = {ratio:.3f}  "
                  f"(n {ba['n_pairs']} of {f_['n_pairs']} pairs)")
            summary[ds] = {"end_iter": end, "full": f_["mean_policy_dist"],
                           "both_alive": ba["mean_policy_dist"],
                           "both_alive_over_full": round(ratio, 3),
                           "n_full": f_["n_pairs"], "n_both_alive": ba["n_pairs"],
                           "baseline_full": next((r["mean_policy_dist"] for r in rows
                                                  if r["dataset"] == ds and r["iter"] == min(its)
                                                  and r["filter"] == "full"), None),
                           "baseline_both_alive": next((r["mean_policy_dist"] for r in rows
                                                        if r["dataset"] == ds and r["iter"] == min(its)
                                                        and r["filter"] == "both_alive"), None)}
        print()

    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=1)
    rs = [v["both_alive_over_full"] for v in summary.values()]
    if rs:
        print(f"both_alive / full at end of training: "
              f"{', '.join(f'{k} {v:.2f}' for k, v in [(k, s['both_alive_over_full']) for k, s in summary.items()])}")
        print(f"restriction raises the estimate in {sum(1 for r in rs if r > 1)}/{len(rs)} datasets")


if __name__ == "__main__":
    main()
