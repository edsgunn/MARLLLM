#!/usr/bin/env python
"""Trace quality for the novelty-search arms: does high delivery mean
communication, or a well-formed loop?

The novelty-search configs pre-register the trap this measures:

    "Maximising others' surprise has a cheap solution too: emit gibberish.
     That would show as high delivery with high perception loss and rising
     entropy -- posts that are well-formed but meaningless. Delivery alone
     cannot separate 'communicating more' from 'emitting noise more', so the
     traces decide, and this arm must not be reported on delivery without
     reading them."

What is measured per episode, from `env_trace.thread`:

  chars        mean post length in characters.
  thread_sim   mean pairwise token-set Jaccard over ALL post pairs in the
               thread, regardless of speaker. THE ONLY REPETITION MEASURE
               COMPARABLE ACROSS N -- see the warning below.
  repeat       fraction of posts that are near-duplicates of an EARLIER post
               by the SAME speaker (token-set Jaccard >= 0.8 on lowercased
               alphanumeric tokens). A locked loop scores high; a developing
               conversation scores near zero.
  self_sim     mean pairwise Jaccard among a speaker's own posts, averaged
               over speakers. Robust version of the same idea.
  math         fraction of posts containing a mathematical token (digit with
               an operator or relation, or a LaTeX/identifier pattern) --
               the measure the 09-04 novelty read used.
  leak         posts containing a literal post tag or an escaped one, i.e.
               the protocol bleeding into content.
  frag         posts under 20 characters.

TWO LIMITS THAT BOUND EVERY NUMBER HERE

  1. A trace file holds ONE episode of the iteration's 16 rollouts. Every
     figure below is n = 1 episode. Treat it as an illustration of what the
     population is doing, never as a rate.

  2. `repeat` and `self_sim` are within-speaker, and posts-per-speaker is
     max_posts / N -- 8 at N=2, 4 at N=4, 2 at N=8, 1 at N=16. At N=16 a
     within-speaker repeat is STRUCTURALLY IMPOSSIBLE and self_sim has no
     pairs at all; at N=8 there is exactly one pair. Both are reported as
     `--` where undefined. **Never compare `repeat` or `self_sim` across N** --
     the measure's sensitivity is a function of N. Use `thread_sim` for that.

Usage:  .venv/bin/python scripts/analyse_novelty_traces.py [--iters 25,150,300]
"""
from __future__ import annotations

import argparse
import json
import re
from itertools import combinations
from pathlib import Path

TOKEN = re.compile(r"[a-z0-9]+")
MATH = re.compile(
    r"(\d\s*[-+*/^=<>]\s*\S)"      # 2 + 3, n=1
    r"|([a-zA-Z]\s*[=<>]\s*\d)"    # n = 0
    r"|(\bn[cC]\d)"                # nC2
    r"|(\\[a-zA-Z]+)"              # \sum
    r"|(\b(?:sum|mod|prod|lemma|proof|induction|theorem|binomial)\b)"
)
LEAK = re.compile(r"</?post>|&lt;/?post&gt;|&lt;/?post>|</?think>")

RUNS = {
    "N=2  ref  (+1)": "runs/cultural_emergence/run10_2agent_7B_study_group_ashbourne_gc",
    "N=2  nov  s1": "runs/novelty_search/novelty_size_02agent_ashbourne_gc_seed1",
    "N=2  nov  s2": "runs/novelty_search/novelty_size_02agent_ashbourne_gc_seed2",
    "N=4  ref  (+1)": "runs/cultural_emergence/run10_4agent_7B_study_group_ashbourne_gc",
    "N=4  nov  s1": "runs/novelty_search/novelty_size_04agent_ashbourne_gc_seed1",
    "N=4  nov  s2": "runs/novelty_search/novelty_size_04agent_ashbourne_gc_seed2",
    "N=8  ref  (+1)": "runs/cultural_emergence/run7_8agent_7B_study_group_ashbourne_gc",
    "N=8  nov  s1": "runs/novelty_search/c13_novelty_search_8agent_ashbourne_gc_seed1",
    "N=8  nov  s2": "runs/novelty_search/c13_novelty_search_8agent_ashbourne_gc_seed2",
    "N=16 ref  (+1)": "runs/cultural_emergence/run8_16agent_7B_study_group_ashbourne_gc",
    "N=16 nov  s1": "runs/novelty_search/novelty_size_16agent_ashbourne_gc_seed1",
}


def toks(s: str) -> set[str]:
    return set(TOKEN.findall(s.lower()))


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def score_thread(thread: list[dict]) -> dict | None:
    posts = [(m.get("speaker", "?"), m.get("text", "") or "") for m in thread]
    posts = [(s, t) for s, t in posts if t.strip()]
    if not posts:
        return None
    n = len(posts)
    by_speaker: dict[str, list[set[str]]] = {}
    repeats = 0
    for speaker, text in posts:
        tk = toks(text)
        prior = by_speaker.setdefault(speaker, [])
        if any(jaccard(tk, q) >= 0.8 for q in prior):
            repeats += 1
        prior.append(tk)
    sims = []
    for tks_ in by_speaker.values():
        pairs = list(combinations(tks_, 2))
        if pairs:
            sims.append(sum(jaccard(a, b) for a, b in pairs) / len(pairs))
    # A within-speaker repeat is only detectable when some speaker posts twice.
    repeatable = any(len(v) > 1 for v in by_speaker.values())
    all_tokens = [toks(t) for _, t in posts]
    all_pairs = list(combinations(all_tokens, 2))
    return {
        "posts": n,
        "chars": sum(len(t) for _, t in posts) / n,
        "thread_sim": (sum(jaccard(a, b) for a, b in all_pairs) / len(all_pairs))
        if all_pairs else float("nan"),
        "repeat": (repeats / n) if repeatable else float("nan"),
        "self_sim": (sum(sims) / len(sims)) if sims else float("nan"),
        "math": sum(1 for _, t in posts if MATH.search(t)) / n,
        "leak": sum(1 for _, t in posts if LEAK.search(t)) / n,
        "frag": sum(1 for _, t in posts if len(t.strip()) < 20) / n,
    }


def at_iter(run: Path, it: int) -> dict | None:
    f = run / "traces" / f"iter_{it:06d}.json"
    if not f.exists():
        return None
    eps = json.load(open(f))
    scored = [score_thread(e["env_trace"]["thread"]) for e in eps]
    scored = [s for s in scored if s]
    if not scored:
        return None
    import math

    keys = scored[0].keys()
    out = {}
    for k in keys:
        vals = [s[k] for s in scored if not math.isnan(s[k])]
        out[k] = sum(vals) / len(vals) if vals else float("nan")
    out["_n_eps"] = len(scored)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", default="25,150,300")
    args = ap.parse_args()
    iters = [int(x) for x in args.iters.split(",")]

    for it in iters:
        print(f"\n=== iteration {it} ===")
        hdr = (f"{'run':16} {'eps':>3} | {'posts':>5} {'chars':>6} {'thrdsim':>7} | "
               f"{'repeat':>6} {'selfsim':>7} | {'math':>5} {'leak':>5} {'frag':>5}")
        print(hdr)
        print("-" * len(hdr))
        import math

        def f(v, w, prec=2):
            return f"{'--':>{w}}" if (isinstance(v, float) and math.isnan(v)) else f"{v:{w}.{prec}f}"

        for label, path in RUNS.items():
            s = at_iter(Path(path), it)
            if s is None:
                print(f"{label:16} {'-':>3} | (no trace)")
                continue
            print(f"{label:16} {s['_n_eps']:3} | {s['posts']:5.1f} {s['chars']:6.0f} "
                  f"{f(s['thread_sim'], 7)} | {f(s['repeat'], 6)} {f(s['self_sim'], 7)} | "
                  f"{f(s['math'], 5)} {f(s['leak'], 5)} {f(s['frag'], 5)}")

    print("\neps      = episodes in the trace file (of the iteration's 16 rollouts)")
    print("thrdsim  = mean pairwise token overlap over ALL post pairs -- comparable across N")
    print("repeat   = posts near-duplicating an earlier post by the SAME speaker")
    print("selfsim  = mean pairwise overlap within a speaker's own posts")
    print("leak     = posts with a post/think tag bleeding into the content")
    print()
    print("`--` means STRUCTURALLY UNDEFINED, not zero: posts-per-speaker is")
    print("max_posts/N, so at N=16 each agent posts once and a within-speaker")
    print("repeat cannot occur. Do NOT compare repeat/selfsim across N; use thrdsim.")


if __name__ == "__main__":
    main()
