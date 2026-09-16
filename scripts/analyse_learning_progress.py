"""Family analysis of the learning-progress suite.

WHAT THIS CHANGES
    Nothing is re-run. Nine finished arms (jobs 6571670-6571673, 6580460-6580464)
    are read against their own references: `run7_8agent_7B_study_group_ashbourne_gc`
    for the text substrate and `run7_8agent_7B_margin_notes` for the tool one,
    plus the novelty-search arms at the same cells so the three objectives —
    surrogate, sign reversed, reference changed — can be read side by side.

WHAT WE EXPECT TO LEARN
    Whether cancelling H(p_env) changes what the population becomes. The
    pre-registered primary read is policy entropy at iteration 300 against the
    same-N reference; the gate on whether ANY of it could matter is
    `lp/return_rank_corr`.

WHY THIS SCRIPT EXISTS AT ALL, AND WHAT IT IS GUARDING AGAINST
    The regime classifier scores every learning-progress arm at `alive` 0.00 by
    iteration 300, which reads as "identical to the reference". Reading the
    traces says that label is wrong for at least one arm: seed 2 of the lagged
    cell is eight agents discussing base cases and symmetry in a subset-counting
    problem, in fluent English, at 126 tokens per turn — and the classifier
    calls all eight `text_degenerate` because perception loss is 2.26.

    That is the mirror image of the artefact this project already documents. The
    known one: a population that stops posting makes its own observation stream
    predictable, `perc_loss` falls through the `< 1.7` gate and `alive` rises.
    The new one: a population posting LONGER and MORE VARIED text makes its
    stream harder to predict, `perc_loss` rises through the `>= 2.0` gate, and
    `text_degenerate` fires on healthy transcripts. Perception loss is not a
    quality measure in either direction when the agents control the stream.

    So every number here is reported beside the well-formed post rate and the
    function-word screen, and the screen's ambiguous band is read by hand.

WHAT WOULD FALSIFY 'THE CORRECTION DOES SOMETHING'
    Delivery, entropy and trace content at iteration 300 indistinguishable from
    the reference, with `lp/return_rank_corr` away from 1.0. The correction
    would then have reordered the actions and changed nothing — which is the
    strongest available evidence that H(p_env) is not the mechanism behind the
    collapse.

CAVEAT THAT LIMITS EVERY COMPARISON BELOW
    Both references are n=1 at iteration 300. This project's measured noise
    floor on delivery between two runs of one config at one seed is 0.173
    (`runs/cultural_emergence/kl_beta_sweep_8agent_ashbourne_gc/NOTES.md`), so
    nothing smaller than that against a single reference run is a result. The
    margin_notes reference has four further seeds but only to iteration 150.

Run:  .venv/bin/python scripts/analyse_learning_progress.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from analyse_novelty_traces import coherence, score_thread  # noqa: E402

LP = ROOT / "runs/learning_progress"
CE = ROOT / "runs/cultural_emergence"
NS = ROOT / "runs/novelty_search"

ARMS = [
    ("lagged  s1", LP / "lp_lagged_8agent_ashbourne_gc_seed1", "text"),
    ("lagged  s2", LP / "lp_lagged_8agent_ashbourne_gc_seed2", "text"),
    ("base    s1", LP / "lp_base_8agent_ashbourne_gc_seed1", "text"),
    ("base    s2", LP / "lp_base_8agent_ashbourne_gc_seed2", "text"),
    ("lag L=5 s1", LP / "lp_lag005_8agent_ashbourne_gc_seed1", "text"),
    ("lag L=100 s1", LP / "lp_lag100_8agent_ashbourne_gc_seed1", "text"),
    ("nonorm  s1", LP / "lp_lagged_nonorm_8agent_ashbourne_gc_seed1", "text"),
    ("REF surrogate", CE / "run7_8agent_7B_study_group_ashbourne_gc", "text"),
    ("REF novelty s1", NS / "c13_novelty_search_8agent_ashbourne_gc_seed1", "text"),
    ("tool LP s1", LP / "lp_lagged_margin_notes_8agent_seed1", "tool"),
    ("tool LP s2", LP / "lp_lagged_margin_notes_8agent_seed2", "tool"),
    ("tool REF surrogate", CE / "run7_8agent_7B_margin_notes", "tool"),
    ("tool REF novelty s1", NS / "c13_novelty_margin_notes_8agent_seed1", "tool"),
]


def threads_at(run: Path, it: int) -> list[list[dict]]:
    """Every episode thread in a checkpoint trace. Falls back to the per-iter
    file, which holds ONE episode of the iteration's sixteen rollouts — a rig
    limit that produced an overstated tool reading once already."""
    out = []
    ck = run / "traces" / f"ckpt_{it:06d}" / "traces.json"
    src = ck if ck.exists() else run / "traces" / f"iter_{it:06d}.json"
    if not src.exists():
        return out
    for ep in json.load(open(src)):
        th = (ep.get("env_trace") or {}).get("thread") or ep.get("thread")
        if th:
            out.append(th)
    return out


def trace_row(run: Path, it: int) -> dict | None:
    scored = [s for s in (score_thread(t) for t in threads_at(run, it)) if s]
    if not scored:
        return None
    keys = ("posts", "chars", "stopword", "nonlatin", "repeat", "frag", "math")
    agg = {}
    for k in keys:
        vals = [s[k] for s in scored if not math.isnan(s[k])]
        agg[k] = sum(vals) / len(vals) if vals else float("nan")
    agg["episodes"] = len(scored)
    return agg


def metric_at(run: Path, it: int, key: str):
    f = run / "metrics.jsonl"
    if not f.exists():
        return None
    best = None
    for line in open(f):
        r = json.loads(line)
        if r["iteration"] <= it:
            best = r
    if best is None:
        return None
    if "/" not in key:
        ags = sorted({k.split("/")[0] for k in best if k.endswith("/entropy")})
        vals = [best[f"{a}/{key}"] for a in ags if f"{a}/{key}" in best]
        return sum(vals) / len(vals) if vals else None
    return best.get(key)


def delivery(run: Path, it: int):
    s = metric_at(run, it, "tool/post_success_per_ep")
    t = metric_at(run, it, "tool/post_turns_per_ep")
    return (s / t) if (s is not None and t) else None


def verdict(stop: float) -> str:
    if math.isnan(stop):
        return "?"
    if stop < 0.10:
        return "SALAD"
    if stop < 0.28:
        return "ambig"
    return "coherent"


def f(v, w=6, p=2):
    return f"{v:{w}.{p}f}" if isinstance(v, (int, float)) and not math.isnan(v) else " " * (w - 1) + "—"


def main() -> None:
    for it in (150, 300):
        print(f"\n{'='*118}\nITERATION {it}\n{'='*118}")
        print(f"{'arm':21s} {'deliv':>6s} {'entrpy':>6s} {'perc':>6s} {'kl':>6s} | "
              f"{'posts':>6s} {'chars':>7s} {'stopwd':>6s} {'verdict':>9s} {'nonlat':>6s} "
              f"{'frag':>6s} {'repeat':>6s} | {'rankcorr':>8s} {'prog/tok':>8s}")
        for label, run, _kind in ARMS:
            if not (run / "metrics.jsonl").exists():
                print(f"{label:21s}  (missing)")
                continue
            tr = trace_row(run, it) or {}
            g = lambda k: tr.get(k, float("nan"))
            print(f"{label:21s} {f(delivery(run, it))} {f(metric_at(run, it, 'entropy'))} "
                  f"{f(metric_at(run, it, 'perc_loss'))} {f(metric_at(run, it, 'kl'))} | "
                  f"{f(g('posts'))} {f(g('chars'), 7, 1)} {f(g('stopword'), 6, 3)} "
                  f"{verdict(g('stopword')):>9s} {f(g('nonlatin'), 6, 3)} "
                  f"{f(g('frag'), 6, 3)} {f(g('repeat'), 6, 3)} | "
                  f"{f(metric_at(run, it, 'lp/return_rank_corr'), 8, 3)} "
                  f"{f(metric_at(run, it, 'lp/progress_per_token'), 8, 3)}")
    print("\nstopword < 0.10 = salad reliably; 0.10-0.28 AMBIGUOUS and must be read by hand;")
    print(">= 0.28 coherent. Dense technical prose is a known false positive in the band.")


if __name__ == "__main__":
    main()
