"""Shared loaders and the regime classifier for the Tier-B analyses.

The classifier lives here and nowhere else. PROCESS.md: "never hand-roll the
classifier in a one-off script, or two runs stop being comparable."
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

ROOT = "/lus/lfs1aip2/projects/a5l/egunn/projects/MARLLLM"
OUT_ROOT = os.path.join(ROOT, "runs", "_tier_b")

# Paper thresholds (Section 4.1; see PROCESS.md "Regimes").
KL_THR, PERC_COH, PERC_DEGEN, ENT_LO, ENT_HI = 1.0, 1.7, 2.0, 0.2, 1.8
ALIVE = {"coherent_low_kl", "committed", "mid_range"}
ORDER = ["coherent_low_kl", "committed", "mid_range", "policy_collapse", "text_degenerate"]

# Metric namespaces that are not agents.
NON_AGENT = {"time", "rollout", "mem", "cpu", "tool", "env_episodes", "async",
             "util", "gen", "per_agent", "nvml_sampler"}

# Families that cannot be cited (see runs/README.md).
ATTIC = ("old_trainer", "old_delimiters", "no_context_management",
         "prompts_as_padding", "_attic", "/archive/", "_archive", "unshuffled")


def classify(kl, perc, ent, perc_coh=PERC_COH, perc_degen=PERC_DEGEN):
    """Assign a regime. Thresholds are parameters so B3 can vary them."""
    if kl is None or perc is None or ent is None:
        return None
    if perc >= perc_degen:
        return "text_degenerate"
    if ent < ENT_LO or ent > ENT_HI:
        return "policy_collapse"
    if kl >= KL_THR:
        return "committed" if perc < perc_coh else "mid_range"
    return "coherent_low_kl" if perc < perc_coh else "mid_range"


def is_agent_key(k):
    return "/" in k and k.split("/")[0] not in NON_AGENT and not k.startswith("per_agent")


def split_record(rec):
    """-> ({agent: {metric: value}}, {global_key: value})"""
    per, glob = defaultdict(dict), {}
    for k, v in rec.items():
        if is_agent_key(k):
            a, m = k.rsplit("/", 1)
            per[a][m] = v
        else:
            glob[k] = v
    return dict(per), glob


def load_metrics(rel):
    """Read a run's metrics.jsonl -> list of records. [] if absent."""
    p = os.path.join(ROOT, rel, "metrics.jsonl")
    if not os.path.exists(p):
        return []
    out = []
    for line in open(p, errors="replace"):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            pass
    return out


def record_at(recs, target, tol=12):
    """Record nearest to `target` iterations, or None if none within tol."""
    best, bd = None, None
    for r in recs:
        it = r.get("iteration")
        if it is None:
            continue
        d = abs(it - target)
        if bd is None or d < bd:
            best, bd = r, d
    return best if (bd is not None and bd <= tol) else None


def population_at(recs, target, **thr):
    """Regime composition at ~target iterations. -> dict or None."""
    rec = record_at(recs, target)
    if rec is None:
        return None
    per, _ = split_record(rec)
    counts, rows = defaultdict(int), []
    for a, m in sorted(per.items()):
        r = classify(m.get("kl"), m.get("perc_loss"), m.get("entropy"), **thr)
        if r is None:
            continue
        counts[r] += 1
        rows.append({"agent": a, "regime": r, **{k: m.get(k) for k in
                     ("kl", "perc_loss", "entropy")}})
    n = sum(counts.values())
    if not n:
        return None
    return {"iter": rec.get("iteration"), "n": n, "counts": dict(counts),
            "alive": sum(counts[k] for k in ALIVE) / n, "agents": rows}


def live_runs(prefix="runs/"):
    """Run directories with metrics, excluding attics."""
    out = []
    for dp, dn, fn in os.walk(os.path.join(ROOT, "runs")):
        dn[:] = [d for d in dn if d not in
                 ("checkpoints", "traces", "snapshots", "contexts")]
        if "metrics.jsonl" in fn:
            rel = os.path.relpath(dp, ROOT)
            if not any(a in rel for a in ATTIC) and rel.startswith(prefix):
                out.append(rel)
            dn[:] = []
    return sorted(out)


def outdir(name):
    d = os.path.join(OUT_ROOT, name)
    os.makedirs(d, exist_ok=True)
    return d


def env_of(recs):
    """Substrate name from the gen/<env>/ namespace."""
    for r in reversed(recs):
        for k in r:
            if k.startswith("gen/"):
                return k.split("/")[1]
    return None


def mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return sum(xs) / len(xs) if xs else None
