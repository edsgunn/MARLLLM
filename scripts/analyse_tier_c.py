#!/usr/bin/env python3
"""Core metrics and regime composition for the Tier-C families (REWRITE.md).

WHAT THIS IS FOR
    The four families submitted on 2026-09-02 each ask one question, stated in
    their config headers before the data existed. This script computes the
    quantities those questions were posed in, and nothing else:

      C1  - is the N=2/N=4 corner of the population-size figure a property of
            the cell or of the one seed it was drawn from? The pre-registered
            test is the within-cell spread of `alive` across five seeds against
            the between-cell difference the figure reads as a size effect.
      C3  - does the decline need the population to be moving? One learner
            against seven frozen members, read against the reference run's
            per-agent behaviour rather than its population means.
      C10 - which term does the damage? Perception-only and action-only against
            the full loss, on channel closure and entropy.
      C2  - partial: two of five arms have finished, so the causal question is
            not yet answerable and only the descriptive numbers are printed.

WHAT WOULD FALSIFY WHAT
    Each family's falsification condition is in its config header. This script
    does not choose them; it evaluates them. The C1 condition in particular was
    fixed in writing before the runs, which is the only reason its result can be
    read as a test rather than as a summary.

WHY IT DOES NOT ROLL ITS OWN CLASSIFIER
    It imports `RegimeThresholds` and `_parse_agent_metrics` from
    `marlllm.regime_tracking`. PROCESS.md forbids hand-rolling the classifier:
    two runs classified against different thresholds stop being comparable, and
    the thresholds here are the calibrated ones.

    `alive` is the three non-failure regimes (coherent_low_kl, committed,
    mid_range) as `scripts/plot_population_regime_figures.py` defines them.
    Note it is quantised: at N=2 it can only take {0, .5, 1}, so a single agent
    flipping moves it by 0.5. That coarseness inflates apparent spread at small
    N and is the reason the continuous metrics are reported alongside.

Usage
-----
    .venv/bin/python scripts/analyse_tier_c.py [c1|c3|c10|c2|all]
"""
from __future__ import annotations

import itertools
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from marlllm.regime_tracking import (  # noqa: E402
    RegimeThresholds, _parse_agent_metrics,
)

THR = RegimeThresholds()
ALIVE = {"coherent_low_kl", "committed", "mid_range"}
MARKS = (25, 100, 150, 200, 300)
CE = ROOT / "runs" / "cultural_emergence"
REF8 = CE / "run7_8agent_7B_study_group_ashbourne_gc"


# ---------------------------------------------------------------- loading
def load(run_dir: Path) -> list[dict] | None:
    mj = run_dir / "metrics.jsonl"
    if not mj.exists():
        return None
    return [json.loads(l) for l in mj.open() if l.strip()]


def _agent_rows(rows):
    return [(int(r.get("iteration", 0)), _parse_agent_metrics(r))
            for r in rows if _parse_agent_metrics(r)]


def regime_at(rows, target):
    ar = _agent_rows(rows)
    if not ar:
        return None
    it, per = min(ar, key=lambda x: abs(x[0] - target))
    if abs(it - target) > max(10, 0.1 * target):
        return None
    labels = {a: THR.classify(m["kl"], m["perc_loss"], m["entropy"])
              for a, m in per.items()}
    comp: dict[str, int] = {}
    for v in labels.values():
        comp[v] = comp.get(v, 0) + 1
    return dict(iter=it, n=len(labels),
                alive=sum(1 for v in labels.values() if v in ALIVE) / len(labels),
                comp=comp, labels=labels, per=per)


def _block(pairs, first):
    if not pairs:
        return None
    pairs = sorted(pairs)
    k = max(1, len(pairs) // 10)
    seg = pairs[:k] if first else pairs[-k:]
    return st.mean(v for _, v in seg)


def summarise(run_dir: Path):
    rows = load(run_dir)
    if not rows:
        return None
    ar = _agent_rows(rows)
    out = dict(name=run_dir.name,
               reached=int(max(r.get("iteration", 0) for r in rows)))
    for key, short in (("perc_loss", "perc"), ("kl", "kl"), ("entropy", "ent")):
        pairs = [(it, st.mean(m[key] for m in per.values())) for it, per in ar]
        out[f"{short}_first"] = _block(pairs, True)
        out[f"{short}_last"] = _block(pairs, False)
    ch = [(r.get("iteration", 0),
           r["tool/post_success_per_ep"] / r["tool/post_turns_per_ep"])
          for r in rows if r.get("tool/post_turns_per_ep")]
    out["chan_first"] = _block(ch, True)
    out["chan_last"] = _block(ch, False)
    out["marks"] = {m: regime_at(rows, m) for m in MARKS}
    return out


def line(r, tag, width=34):
    a = [r["marks"][m]["alive"] if r["marks"][m] else None for m in MARKS]
    al = "  ".join("  -  " if x is None else f"{x:5.2f}" for x in a)
    return (f"{tag:<{width}} {al}   perc {r['perc_first']:.2f}->{r['perc_last']:.2f}"
            f"  KL {r['kl_first']:.3f}->{r['kl_last']:.2f}"
            f"  ent {r['ent_first']:.2f}->{r['ent_last']:.2f}"
            f"  chan {r['chan_first']:.2f}->{r['chan_last']:.2f}")


def mwu_exact(a, b):
    """Exact two-sided Mann-Whitney p. Tiny samples only (5 vs 5 = 252 splits)."""
    n, m = len(a), len(b)
    pool = a + b

    def U(aa, bb):
        return (sum(1 for x in aa for y in bb if x > y)
                + 0.5 * sum(1 for x in aa for y in bb if x == y))

    obs = U(a, b)
    cnt = tot = 0
    for idx in itertools.combinations(range(n + m), n):
        aa = [pool[i] for i in idx]
        bb = [pool[i] for i in range(n + m) if i not in idx]
        tot += 1
        if abs(U(aa, bb) - n * m / 2) >= abs(obs - n * m / 2):
            cnt += 1
    return cnt / tot


# ---------------------------------------------------------------- C1
C1 = CE / "c1_small_population_replicates"
C1_CELLS = {
    ("ashbourne", 2): "run10_2agent_7B_study_group_ashbourne_gc",
    ("ashbourne", 4): "run10_4agent_7B_study_group_ashbourne_gc",
    ("strathearn", 2): "run10_2agent_7B_study_group_strathearn_server",
    ("strathearn", 4): "run10_4agent_7B_study_group_strathearn_server",
}
# The single 2026-05-06 seed each cell of the figure was drawn from, from those
# runs' NOTES.md. Listed so the fresh five can be read against it.
C1_HIST = {("ashbourne", 2): [0.00, 0.00, 1.00, 1.00, 0.00],
           ("ashbourne", 4): [0.00, 1.00, 1.00, 0.75, 0.75],
           ("strathearn", 2): [1.00, 1.00, 0.00, 0.00, 0.00],
           ("strathearn", 4): [1.00, 1.00, 1.00, 0.75, 0.75]}


def run_c1():
    print("=" * 118)
    print("C1 - is the N=2/N=4 corner a property of the cell, or of one seed?")
    print("=" * 118)
    data, spread = {}, {}
    for key, base in C1_CELLS.items():
        sub, n = key
        print(f"\n  N={n} {sub}")
        rows = []
        for s in range(1, 6):
            r = summarise(C1 / f"{base}_seed{s}")
            if not r:
                print(f"    seed{s}: MISSING")
                continue
            rows.append([r["marks"][m]["alive"] if r["marks"][m] else None
                         for m in MARKS])
            print("    " + line(r, f"seed{s}", 8))
        data[key] = [summarise(C1 / f"{base}_seed{s}") for s in range(1, 6)]
        h = "  ".join(f"{x:5.2f}" for x in C1_HIST[key])
        print(f"    {'(05-06)':<8} {h}   <- the one seed the figure used")
        sp = []
        for i in range(len(MARKS)):
            col = [r[i] for r in rows if r[i] is not None]
            sp.append(max(col) - min(col) if len(col) > 1 else None)
        spread[key] = sp
        print(f"    {'spread':<8} " +
              "  ".join("  -  " if x is None else f"{x:5.2f}" for x in sp))

    print("\n  THE PRE-REGISTERED TEST — within-cell spread vs the between-cell gap\n")
    for i, m in enumerate(MARKS):
        means = {}
        for key in C1_CELLS:
            col = [r["marks"][m]["alive"] for r in data[key]
                   if r and r["marks"][m]]
            if col:
                means[key] = st.mean(col)
        if len(means) < 4:
            continue
        within = st.mean(spread[k][i] for k in C1_CELLS if spread[k][i] is not None)
        ga = abs(means[("ashbourne", 2)] - means[("ashbourne", 4)])
        gs = abs(means[("strathearn", 2)] - means[("strathearn", 4)])
        verdict = "spread >= gap" if within >= max(ga, gs) else ""
        print(f"    iter {m:>3}:  within-cell spread {within:.2f}    "
              f"N=2 vs N=4 gap: ashbourne {ga:.2f}, strathearn {gs:.2f}   {verdict}")

    print("\n  Does population size separate in the continuous metrics? "
          "(exact Mann-Whitney, 5v5)\n")
    for metric, lab in (("perc_last", "perception loss"), ("kl_last", "KL from base"),
                        ("ent_last", "policy entropy"), ("chan_last", "public channel")):
        for sub in ("ashbourne", "strathearn"):
            v2 = [r[metric] for r in data[(sub, 2)] if r]
            v4 = [r[metric] for r in data[(sub, 4)] if r]
            p = mwu_exact(v2, v4)
            print(f"    {lab:<16} {sub:<11} N=2 {st.mean(v2):6.2f} "
                  f"[{min(v2):.2f},{max(v2):.2f}]   N=4 {st.mean(v4):6.2f} "
                  f"[{min(v4):.2f},{max(v4):.2f}]   p={p:.3f}"
                  f"{'  *' if p < 0.05 else ''}")
    allc = [(r["chan_first"], r["chan_last"]) for v in data.values() for r in v if r]
    print(f"\n    public channel across all {len(allc)} runs: "
          f"{st.mean(a for a, _ in allc):.2f} -> {st.mean(b for _, b in allc):.2f}, "
          f"fell in {sum(1 for a, b in allc if b < a)}/{len(allc)}")


# ---------------------------------------------------------------- C3
def run_c3():
    print("=" * 118)
    print("C3 - does the decline need the population to be moving?")
    print("=" * 118)
    ref = summarise(REF8)
    names = ["c3_frozen_pop_learner_priya_seed1", "c3_frozen_pop_learner_priya_seed2",
             "c3_frozen_pop_learner_marcus_seed1"]
    for n in names:
        r = summarise(CE / "c3_frozen_population" / n)
        print(line(r, n.replace("c3_frozen_pop_learner_", "learner ")))
    print(line(ref, "REF run7 (all 8 co-adapting)"))

    print("\n  The comparison that matters is per-agent, not population mean.")
    print("  run7's eight agents at iteration 300:")
    m = regime_at(load(REF8), 300)
    for a, mm in sorted(m["per"].items()):
        print(f"    {a:<20} KL {mm['kl']:6.2f}  perc {mm['perc_loss']:5.2f}  "
              f"ent {mm['entropy']:5.2f}  {m['labels'][a]}")
    print("\n  the C3 learners at iteration 300:")
    for n in names:
        m = regime_at(load(CE / "c3_frozen_population" / n), 300)
        for a, mm in m["per"].items():
            print(f"    {n[22:]:<24}{a:<14} KL {mm['kl']:6.2f}  "
                  f"perc {mm['perc_loss']:5.2f}  ent {mm['entropy']:5.2f}  "
                  f"{m['labels'][a]}")


# ---------------------------------------------------------------- C10
def run_c10():
    print("=" * 118)
    print("C10 - which term does the damage?")
    print("=" * 118)
    ref = summarise(REF8)
    for n in ("c10_perception_only_8agent_ashbourne_gc_seed1",
              "c10_perception_only_8agent_ashbourne_gc_seed2",
              "c10_action_only_8agent_ashbourne_gc_seed1",
              "c10_action_only_8agent_ashbourne_gc_seed2"):
        r = summarise(CE / "c10_term_ablation" / n)
        print(line(r, n.replace("_8agent_ashbourne_gc", "").replace("c10_", "")))
    print(line(ref, "REF run7 (full loss, 1/1/1)"))


# ---------------------------------------------------------------- C2
def run_c2():
    print("=" * 118)
    print("C2 - population-size decomposition. INCOMPLETE: 2 of 5 arms finished.")
    print("=" * 118)
    C2 = CE / "c2_population_size"
    for arm in ("rate_matched_small", "frozen_majority"):
        for s in range(1, 6):
            d = C2 / f"c2_{arm}_ashbourne_gc_seed{s}"
            r = summarise(d)
            if r and r["reached"] >= 300:
                print(line(r, f"{arm} seed{s}"))
    print(line(summarise(CE / "run8_16agent_7B_study_group_ashbourne_gc"),
               "REF run8 (N=16, all training)"))
    print(line(summarise(C1 / "run10_4agent_7B_study_group_ashbourne_gc_seed1"),
               "REF C1 N=4 seed1 (baseline small)"))
    print("\n  The causal question needs baseline-large and restricted-pairing,")
    print("  both still running. Nothing here answers it yet.")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name, fn in (("c1", run_c1), ("c3", run_c3), ("c10", run_c10), ("c2", run_c2)):
        if which in (name, "all"):
            fn()
            print()


# ------------------------------------------------- per-agent public channel
def agent_channel(run_dir: Path, ckpt: int, agents: set[str] | None = None):
    """Per-agent well-formed-post rate from the checkpoint trace.

    `tool_counts_per_agent` is the canonical instrument: `post_success` counts
    turns delivering a CLOSED <post> block. Counting non-empty thread text
    instead overstates delivery, because an unclosed block still has its text
    extracted onto the thread — that is the whole reason `post_unclosed` exists
    as a separate category. On run8 at iteration 300 the two differ by more
    than a factor of two (0.37 vs 0.80).

    Returns (success, turns, breakdown) summed over `agents`, or all agents.
    """
    f = run_dir / "traces" / f"ckpt_{ckpt:06d}" / "traces.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text())
    recs = d if isinstance(d, list) else [d]
    keys = ("post_success", "post_unclosed", "post_missing", "post_empty")
    succ = turns = 0
    brk = dict.fromkeys(keys, 0)
    for r in recs:
        pa = (r.get("env_trace") or {}).get("tool_counts_per_agent") or {}
        for a, c in pa.items():
            if a == "pybot" or (agents is not None and a not in agents):
                continue
            turns += c.get("post_turns", 0)
            succ += c.get("post_success", 0)
            for k in keys:
                brk[k] += c.get(k, 0)
    return succ, turns, brk
