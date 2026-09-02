#!/usr/bin/env python3
"""
B4 — DOES THE PUBLIC CHANNEL CLOSE OVER TRAINING?

WHAT THIS CHANGES
    Replaces the instrument used in REWRITE.md §3.5. B6 showed the `think`/`post`
    token split cannot support a private-channel reading (zero <think> tags in
    any forum run; `thinking_tokens` is the residual `n_act - post_tokens`).
    The environment has been logging the right quantity all along: it classifies
    every turn as post_success / post_unclosed / post_missing / post_empty
    (envs/forum/env.py:517-560) and only text inside a <post> block reaches other
    agents — "everything between them is private thinking and is discarded for
    the public stream" (env.py:519).

WHAT WE EXPECT TO LEARN
    Whether the fraction of turns that deliver a well-formed public post falls
    over training, and whether that fall leads or lags the regime decline
    measured in B2.

    The derivation predicts it should fall. The implemented return is
    G_t = Sum_{s>t, sigma_s=OBS} gamma^(s-t) * surprise_s (marlllm/loss.py:12) —
    a sum over FUTURE OBSERVATION TOKENS. Its magnitude depends on how many
    observation tokens follow as much as on how surprising each is. The rebuttal
    stated the consequence: "an agent that shortens its co-agents' replies moves
    it as surely as one that makes them more predictable." A turn that emits no
    well-formed post contributes nothing to any other agent's context.

WHAT WOULD FALSIFY IT
    A flat or rising post_success rate over training would mean the public
    channel does not close and the dark-room reading of the decline is wrong.
    A post_success rate that falls only AFTER the alive fraction falls would
    make it a symptom rather than a candidate mechanism.
    A fall that is uniform across substrates including ones that stay healthy
    would break the association with outcome.

    Note what this cannot show: association, not causation. The ablation that
    would settle direction is C4 (public-tokens-only) in REWRITE.md.

OUTPUT  runs/_tier_b/b4_public_channel/
"""
from __future__ import annotations
import csv, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C

GRID = [1, 25, 50, 100, 150, 200, 250, 300]
EVENTS = ["post_success", "post_unclosed", "post_missing", "post_empty"]


def series(rel):
    recs = C.load_metrics(rel)
    if not recs:
        return None
    out = {"run": rel, "env": C.env_of(recs), "points": []}
    for tgt in GRID:
        rec = C.record_at(recs, tgt)
        if rec is None:
            continue
        turns = rec.get("tool/post_turns_per_ep")
        if not turns:
            continue
        pt = {"target": tgt, "iter": rec.get("iteration"), "turns_per_ep": turns}
        for e in EVENTS:
            v = rec.get(f"tool/{e}_per_ep")
            pt[e] = round(v / turns, 4) if v is not None else None
        pop = C.population_at(recs, tgt)
        pt["alive"] = pop["alive"] if pop else None
        out["points"].append(pt)
    return out if out["points"] else None


def main():
    out = C.outdir("b4_public_channel")
    rows, series_all = [], []
    for rel in C.live_runs():
        s = series(rel)
        if not s:
            continue
        series_all.append(s)
        for p in s["points"]:
            rows.append({"run": rel, "env": s["env"], **p})

    with open(os.path.join(out, "public_channel.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["run", "env", "target", "iter",
                                           "turns_per_ep", *EVENTS, "alive"])
        w.writeheader(); w.writerows(rows)

    # runs with both an early and a late point
    print(f"{'run':<52}{'iter':>5}{'success':>9}{'unclosed':>10}{'missing':>9}{'alive':>7}")
    firsts, lasts = {}, {}
    for s in series_all:
        pts = [p for p in s["points"] if p["post_success"] is not None]
        if len(pts) < 2:
            continue
        for p in (pts[0], pts[-1]):
            print(f"{s['run'].split('/')[-1][:51]:<52}{p['iter']:>5.0f}"
                  f"{p['post_success']:>9.3f}{p['post_unclosed']:>10.3f}"
                  f"{p['post_missing']:>9.3f}"
                  f"{(p['alive'] if p['alive'] is not None else float('nan')):>7.2f}")
        firsts[s["run"]] = pts[0]; lasts[s["run"]] = pts[-1]
        print()

    # Direction test
    down = sum(1 for k in firsts if lasts[k]["post_success"] < firsts[k]["post_success"])
    same = sum(1 for k in firsts if lasts[k]["post_success"] == firsts[k]["post_success"])
    up = len(firsts) - down - same
    print(f"post_success fell in {down}/{len(firsts)} runs, unchanged in {same}, rose in {up}")

    # Timing: does the channel close before the population declines?
    lead = []
    for s in series_all:
        pts = [p for p in s["points"] if p["post_success"] is not None and p["alive"] is not None]
        if len(pts) < 3:
            continue
        base_s, base_a = pts[0]["post_success"], pts[0]["alive"]
        first_s = next((p["iter"] for p in pts if p["post_success"] < 0.75 * base_s), None)
        first_a = next((p["iter"] for p in pts if p["alive"] < base_a - 1e-9), None)
        if first_s is not None or first_a is not None:
            lead.append({"run": s["run"], "env": s["env"],
                         "iter_success_down_25pct": first_s,
                         "iter_alive_first_falls": first_a})
    with open(os.path.join(out, "timing.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["run", "env", "iter_success_down_25pct",
                                           "iter_alive_first_falls"])
        w.writeheader(); w.writerows(lead)
    both = [r for r in lead if r["iter_success_down_25pct"] and r["iter_alive_first_falls"]]
    leads = sum(1 for r in both if r["iter_success_down_25pct"] <= r["iter_alive_first_falls"])
    print(f"\nchannel degradation reaches -25% at or before the first drop in alive "
          f"fraction in {leads}/{len(both)} runs where both are observed")

    json.dump({"n_runs": len(firsts), "post_success_fell": down,
               "timing_leads": leads, "timing_n": len(both)},
              open(os.path.join(out, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
