#!/usr/bin/env python
"""Per-agent PUBLIC response length, recovered from checkpoint traces.

WHY THIS EXISTS, AND WHAT IT CANNOT DO

Per-agent token accounting was added to the trainer on 2026-09-15. Runs logged
before that carry only `gen/<env>/mean_post_tokens` — a population mean, which
under partial freezing is seven-eighths pinned by construction and says nothing
about the learner.

The checkpoint traces do carry `speaker` and post `text` per turn, so the
PUBLIC length is recoverable for those runs without re-running anything. Two
limits, both hard:

  * ONE EPISODE per checkpoint, of the iteration's 16 rollouts. Every point
    here is n = 1. It shows shape, not a rate.
  * Length is in CHARACTERS, not tokens — the trace stores text, not ids.
    Comparable within and across runs on the same tokeniser, but not against
    the trainer's token-valued `mean_post_tokens`.

  * TOTAL (think + post) length is NOT recoverable at all. The trace keeps the
    post body, not the raw generation, so there is no way to see deliberation
    that never reached the forum. That half needs the new logging.

Usage:
    .venv/bin/python scripts/plot_response_length_from_traces.py RUN [RUN ...]
    .venv/bin/python scripts/plot_response_length_from_traces.py \
        runs/cultural_emergence/c3_frozen_population/c3_frozen_pop_learner_priya_seed1 \
        runs/cultural_emergence/run7_8agent_7B_study_group_ashbourne_gc
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _frozen_agents(run: Path) -> set[str]:
    try:
        cfg = json.loads((run / "config.json").read_text())
    except Exception:
        return set()
    return set(cfg.get("frozen_agents") or [])


def series(run: Path) -> tuple[list[int], dict[str, dict[int, tuple[float, float, int]]]]:
    """iteration -> agent -> (mean chars over ALL turns, mean over posting turns, n_turns)."""
    out: dict[str, dict[int, tuple[float, float, int]]] = {}
    iters: list[int] = []
    for f in sorted((run / "traces").glob("iter_*.json")):
        m = re.search(r"iter_(\d+)", f.name)
        if not m:
            continue
        it = int(m.group(1))
        try:
            eps = json.load(open(f))
        except Exception:
            continue
        per: dict[str, list[int]] = {}
        for ep in eps:
            for turn in ep.get("env_trace", {}).get("thread", []):
                spk = turn.get("speaker")
                if not spk or spk == "pybot":
                    continue
                per.setdefault(spk, []).append(len((turn.get("text") or "").strip()))
        if not per:
            continue
        iters.append(it)
        for agent, lens in per.items():
            posting = [v for v in lens if v > 0]
            out.setdefault(agent, {})[it] = (
                float(np.mean(lens)),
                float(np.mean(posting)) if posting else 0.0,
                len(lens),
            )
    return sorted(set(iters)), out


def pooled_delivery(run: Path, lo: int, hi: int) -> tuple[dict[str, tuple[int, int]], int]:
    """Per-agent (posts, turns) pooled over every checkpoint trace in a window.

    `tool_counts_per_agent` is recorded in the TRACE, not in `metrics.jsonl`,
    so per-agent delivery is one episode per checkpoint. Pooling the window
    turns ~13 single episodes into one n-of-42-turns figure per agent, which is
    the tightest per-agent delivery the pre-2026-09-15 runs can support.
    """
    out: dict[str, tuple[int, int]] = {}
    n_eps = 0
    for f in sorted((run / "traces").glob("iter_*.json")):
        m = re.search(r"iter_(\d+)", f.name)
        if not m or not (lo <= int(m.group(1)) <= hi):
            continue
        try:
            eps = json.load(open(f))
        except Exception:
            continue
        for ep in eps:
            n_eps += 1
            counts = ep.get("env_trace", {}).get("tool_counts_per_agent") or {}
            for agent, c in counts.items():
                s, tt = out.get(agent, (0, 0))
                out[agent] = (s + c.get("post_success", 0), tt + c.get("post_turns", 0))
    return out, n_eps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default="response_length_from_traces.png")
    ap.add_argument("--window", default="200,300",
                    help="iteration window for the pooled per-agent delivery table")
    args = ap.parse_args()

    runs = [Path(r) for r in args.runs]
    fig, axes = plt.subplots(2, len(runs), figsize=(6.5 * len(runs), 8.5),
                             squeeze=False, sharey="row")

    for c, run in enumerate(runs):
        iters, per = series(run)
        frozen = _frozen_agents(run)
        names = sorted(per)
        cmap = plt.get_cmap("tab20")
        for i, agent in enumerate(names):
            pts = per[agent]
            xs = sorted(pts)
            all_mean = [pts[x][0] for x in xs]
            posting = [pts[x][1] for x in xs]
            is_frozen = agent in frozen
            style = dict(color=cmap(i % 20),
                         linewidth=2.2 if not is_frozen else 1.0,
                         alpha=1.0 if not is_frozen else 0.45,
                         linestyle="-" if not is_frozen else "--",
                         label=agent + (" (frozen)" if is_frozen else ""))
            axes[0, c].plot(xs, all_mean, **style)
            axes[1, c].plot(xs, posting, **style)

        title = run.name + (f"\n{len(frozen)} of {len(names)} frozen" if frozen
                            else f"\nall {len(names)} co-adapting")
        axes[0, c].set_title(title, fontsize=10, fontweight="bold")
        axes[0, c].set_ylabel("chars / turn  (all turns)")
        axes[1, c].set_ylabel("chars / turn  (posting turns only)")
        axes[1, c].set_xlabel("iteration")
        for r in (0, 1):
            axes[r, c].set_ylim(bottom=0)
            axes[r, c].grid(alpha=0.15)
        axes[0, c].legend(fontsize=6, ncol=2, frameon=False, loc="upper right")

    fig.suptitle("Per-agent PUBLIC response length, recovered from checkpoint traces\n"
                 "one episode per checkpoint (n=1) · characters, not tokens · "
                 "total length is NOT recoverable from traces",
                 fontsize=10, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = Path(args.out)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"wrote {out}")

    lo, hi = (int(x) for x in args.window.split(","))
    for run in runs:
        frozen = _frozen_agents(run)
        deliv, n_eps = pooled_delivery(run, lo, hi)
        if deliv:
            print(f"\n{run.name} — per-agent delivery pooled over iterations "
                  f"{lo}-{hi} ({n_eps} episodes)")
            print(f"  {'agent':24} {'turns':>6} {'posts':>6} {'delivery':>9}")
            for agent in sorted(deliv):
                s, tt = deliv[agent]
                tag = " (frozen)" if agent in frozen else " *learner*"
                rate = s / tt if tt else float("nan")
                print(f"  {agent + tag:24} {tt:6} {s:6} {rate:9.3f}")

    for run in runs:
        iters, per = series(run)
        frozen = _frozen_agents(run)
        if not iters:
            continue
        first, last = min(iters), max(iters)
        print(f"\n{run.name}  (iterations {first} -> {last})")
        print("  (n=1 per point: in the frozen run the agents that CANNOT learn still")
        print("   span ~536 chars over training, so changes below that are not readable)")
        print(f"  {'agent':22} {'first':>8} {'last':>8} {'change':>8}   {'posting-only last':>17}")
        for agent in sorted(per):
            p = per[agent]
            if first not in p or last not in p:
                continue
            a, b = p[first][0], p[last][0]
            tag = " (frozen)" if agent in frozen else ""
            print(f"  {agent + tag:22} {a:8.0f} {b:8.0f} {b - a:+8.0f}   {p[last][1]:17.0f}")


if __name__ == "__main__":
    main()
