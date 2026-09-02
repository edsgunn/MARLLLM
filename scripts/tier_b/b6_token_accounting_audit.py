#!/usr/bin/env python3
"""
B6 — AUDIT THE think/post TOKEN ACCOUNTING.

WHAT THIS CHANGES
    Nothing about the runs. It checks whether the quantity we have been calling
    "thinking tokens" measures what its name says, before any claim rests on it.
    REWRITE.md §3.5 currently reads a rising `thinking_frac` as agents moving
    generation into a private channel. That reading is only valid if
    `mean_thinking_tokens` counts tokens inside <think>...</think>.

WHAT WE EXPECT TO LEARN
    Whether `thinking_frac` is a measurement or a residual. population.py:721
    computes `thinking_tokens = max(0, n_act - post_tokens)` where `post_tokens`
    is the re-encoded text extracted from <post> blocks. If that is what it is,
    the metric bundles: <think> content, the tags themselves, <python> blocks,
    stray text, and anything emitted after the post — and a turn with no
    well-formed <post> at all scores as 100% "thinking".

    So we count, directly in the checkpoint traces, at an early and a late
    checkpoint of every long forum run:
        n(<think>), n(</think>)   — is there any private deliberation at all?
        n(<post>), n(</post>)     — are public posts well formed?
    and read `unclosed_thinking_rate` alongside.

WHAT WOULD FALSIFY THE §3.5 READING
    If <think> counts are substantial and rise over training, the private-channel
    reading stands and `thinking_frac` is a usable proxy.
    If <think> is absent while the post-open/post-close ratio degrades, then
    `thinking_frac` is measuring post-format breakdown, the §3.5 mechanism is
    wrong, and the claim must be re-grounded on the environment's own post
    accounting (B4).

WHAT WOULD FALSIFY THE AUDIT ITSELF
    If `mean_action_tokens != mean_thinking_tokens + mean_post_tokens` in the
    logged metrics, the arithmetic is not what we read in the source and the
    audit is looking at the wrong thing.

OUTPUT  runs/_tier_b/b6_token_accounting/
"""
from __future__ import annotations
import csv, json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C

TAGS = ["<think>", "</think>", "<post>", "</post>", "<python>", "</python>"]

RUNS = [
    "runs/cultural_emergence/run8_16agent_7B_study_group_ashbourne_gc",
    "runs/cultural_emergence/run8_16agent_7B_study_group_strathearn_server",
    "runs/cultural_emergence/run7_8agent_7B_study_group_ashbourne_gc",
    "runs/cultural_emergence/run7_8agent_7B_margin_notes",
    "runs/cultural_emergence/run7_8agent_7B_inner_loop_post_format",
    "runs/cultural_emergence/run8_16agent_7B_margin_notes",
    "runs/cultural_emergence/run12_32agent_7B_study_group_ashbourne_gc",
]


def ckpt_traces(rel):
    d = os.path.join(C.ROOT, rel, "traces")
    if not os.path.isdir(d):
        return []
    out = []
    for name in sorted(os.listdir(d)):
        p = os.path.join(d, name, "traces.txt")
        if name.startswith("ckpt_") and os.path.exists(p):
            out.append((int(name.split("_")[1]), p))
    return out


def count_tags(path):
    txt = open(path, errors="replace").read()
    return {t: txt.count(t) for t in TAGS}


def main():
    out = C.outdir("b6_token_accounting")
    rows, arith = [], []

    for rel in RUNS:
        cks = ckpt_traces(rel)
        if len(cks) < 2:
            print(f"  (skip, <2 checkpoint traces) {rel}")
            continue
        recs = C.load_metrics(rel)
        env = C.env_of(recs)
        # arithmetic check on the logged metrics
        for r in (recs[0], recs[-1]) if recs else []:
            a = r.get(f"gen/{env}/mean_action_tokens")
            t = r.get(f"gen/{env}/mean_thinking_tokens")
            p = r.get(f"gen/{env}/mean_post_tokens")
            if None in (a, t, p):
                continue
            arith.append({"run": rel.split("/")[-1], "iter": r.get("iteration"),
                          "act": round(a, 2), "think": round(t, 2), "post": round(p, 2),
                          "think_plus_post": round(t + p, 2),
                          "residual": round(a - (t + p), 6)})
        # tag counts, first non-zero checkpoint and last
        pick = [cks[1] if cks[0][0] == 0 and len(cks) > 1 else cks[0], cks[-1]]
        for it, path in pick:
            c = count_tags(path)
            rec = C.record_at(recs, it) or {}
            rows.append({
                "run": rel.split("/")[-1], "env": env, "ckpt_iter": it,
                **{t.strip("<>/").replace("/", "") + ("_close" if t.startswith("</") else "_open"): c[t]
                   for t in TAGS},
                "think_open": c["<think>"], "think_close": c["</think>"],
                "post_open": c["<post>"], "post_close": c["</post>"],
                "open_close_ratio": round(c["<post>"] / c["</post>"], 3) if c["</post>"] else None,
                "thinking_frac_logged": rec.get(f"gen/{env}/thinking_frac"),
                "unclosed_rate_logged": rec.get(f"gen/{env}/unclosed_thinking_rate"),
            })

    keys = ["run", "env", "ckpt_iter", "think_open", "think_close", "post_open",
            "post_close", "open_close_ratio", "thinking_frac_logged", "unclosed_rate_logged"]
    with open(os.path.join(out, "tag_counts.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(out, "arithmetic_check.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(arith[0]) if arith else ["run"])
        w.writeheader()
        w.writerows(arith)

    print(f"{'run':<46}{'iter':>6}{'<think>':>9}{'<post>':>8}{'</post>':>9}"
          f"{'ratio':>7}{'think_frac':>11}{'unclosed':>10}")
    for r in rows:
        print(f"{r['run'][:45]:<46}{r['ckpt_iter']:>6}{r['think_open']:>9}"
              f"{r['post_open']:>8}{r['post_close']:>9}"
              f"{(r['open_close_ratio'] if r['open_close_ratio'] is not None else float('nan')):>7.2f}"
              f"{(r['thinking_frac_logged'] or 0):>11.3f}{(r['unclosed_rate_logged'] or 0):>10.3f}")

    tot_think = sum(r["think_open"] for r in rows)
    bad = [a for a in arith if abs(a["residual"]) > 1e-3]
    print(f"\n<think> tags across every checkpoint inspected: {tot_think}")
    print(f"arithmetic act == think + post: {len(arith) - len(bad)}/{len(arith)} exact")
    json.dump({"total_think_tags": tot_think, "arith_rows": len(arith),
               "arith_exact": len(arith) - len(bad)},
              open(os.path.join(out, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
