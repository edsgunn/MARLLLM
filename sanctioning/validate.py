"""Judge validation (the Rung-1 gate) + the human-labelling helper.

The judge is an instrument; its own error structure can manufacture apparent
norms. Before any Rung-1 number is trusted, hand-label a stratified sample and
check the judge against it. Aggregation refuses to emit Rung-1 results unless a
passing validation report exists for the same (provider, model, prompt_version).

Commands:
  label    — interactively hand-label a stratified sample (writes labels.jsonl)
  validate — score the judge against the labels; write a gate report
"""

from __future__ import annotations

import json
import os
import random
from collections import defaultdict

from . import PROMPT_VERSION
from .adapter import iter_corpus
from .backends import make_judge_backend
from .cache import Cache
from .judge import Judge, build_window, format_user_message


# ---------------------------------------------------------------------------
# Gate report location + status
# ---------------------------------------------------------------------------
def _sanitize(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-._" else "_" for c in s)


def report_path(cfg) -> str:
    name = f"report_{cfg.judge.provider}__{_sanitize(cfg.judge.model)}__{PROMPT_VERSION}.json"
    return os.path.join(cfg.paths.validation, name)


def gate_status(cfg):
    path = report_path(cfg)
    if not os.path.exists(path):
        return False, f"no validation report at {path}"
    with open(path) as f:
        rep = json.load(f)
    f1 = rep["metrics"]["presence_f1"]
    kappa = rep["metrics"]["valence_kappa"]
    ok = (f1 >= cfg.validation.presence_f1_min) and (kappa >= cfg.validation.valence_kappa_min)
    msg = (f"presence_f1={f1:.3f} (min {cfg.validation.presence_f1_min}), "
           f"valence_kappa={kappa:.3f} (min {cfg.validation.valence_kappa_min}) -> "
           f"{'PASS' if ok else 'FAIL'}")
    return ok, msg


# ---------------------------------------------------------------------------
# Indexing helper (so label/validate see the same window the judge saw)
# ---------------------------------------------------------------------------
def _index_posts(cfg, substrates=None, run_ids=None, iter_min=None, iter_max=None):
    index = {}
    for ep in iter_corpus(cfg.corpus, substrates, run_ids, iter_min, iter_max):
        index[ep.episode_id] = ep.posts
    return index


# ---------------------------------------------------------------------------
# label
# ---------------------------------------------------------------------------
def _build_sample(cfg, substrates, run_ids, iter_min, iter_max, n_samples, seed=0):
    by_stratum = defaultdict(list)
    for ep in iter_corpus(cfg.corpus, substrates, run_ids, iter_min, iter_max):
        for idx, post in enumerate(ep.posts):
            by_stratum[(post.substrate, post.iteration)].append(
                dict(episode_id=ep.episode_id, run_id=post.run_id, iteration=post.iteration,
                     substrate=post.substrate, turn_index=post.turn_index, agent_id=post.agent_id)
            )
    strata = list(by_stratum)
    total = sum(len(v) for v in by_stratum.values())
    if total == 0:
        return []
    rng = random.Random(seed)
    sample = []
    for stratum in strata:
        items = by_stratum[stratum]
        share = max(1, round(n_samples * len(items) / total))
        share = min(share, len(items))
        sample.extend(rng.sample(items, share))
    rng.shuffle(sample)
    return sample[:n_samples] if len(sample) > n_samples else sample


def run_label(cfg, substrates=None, run_ids=None, iter_min=None, iter_max=None):
    cfg.make_dirs()
    path = cfg.paths.labels
    if os.path.exists(path):
        with open(path) as f:
            entries = [json.loads(l) for l in f if l.strip()]
        print(f"[label] resuming existing label set ({len(entries)} entries).")
    else:
        sample = _build_sample(cfg, substrates, run_ids, iter_min, iter_max,
                               cfg.validation.n_label_samples)
        entries = [dict(s, labelled=False) for s in sample]
        print(f"[label] created a stratified sample of {len(entries)} posts.")

    index = _index_posts(cfg, substrates, run_ids, iter_min, iter_max)
    todo = [e for e in entries if not e.get("labelled")]
    print(f"[label] {len(todo)} posts to label. Ctrl-C saves progress and exits.\n")

    try:
        for e in todo:
            posts = index.get(e["episode_id"])
            if posts is None:
                e["labelled"] = True
                e["skipped"] = "episode not found"
                continue
            cand_idx = next((i for i, p in enumerate(posts) if p.turn_index == e["turn_index"]), None)
            if cand_idx is None:
                e["labelled"] = True
                e["skipped"] = "turn not found"
                continue
            window = build_window(posts, cand_idx, cfg.judge.window_k)
            print("=" * 80)
            print(format_user_message(window, posts[cand_idx]))
            print("-" * 80)
            pres = input("Reaction to another participant present? [y/n/s=skip] ").strip().lower()
            if pres == "s":
                continue
            if pres == "y":
                val = ""
                while val not in ("a", "d", "n"):
                    val = input("  valence [a=approve / d=disapprove / n=neutral]: ").strip().lower()
                e["human_valence"] = {"a": "approve", "d": "disapprove", "n": "neutral"}[val]
                tt = input("  target turn_index (blank if diffuse): ").strip()
                e["human_target_turn_index"] = int(tt) if tt.lstrip("-").isdigit() else None
                e["human_trigger"] = input("  short trigger description (optional): ").strip()
                e["human_present"] = True
            else:
                e["human_present"] = False
            e["labelled"] = True
    except KeyboardInterrupt:
        print("\n[label] interrupted; saving progress.")

    with open(path, "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
    n_lab = sum(1 for e in entries if e.get("labelled"))
    print(f"[label] saved {path}  ({n_lab}/{len(entries)} labelled).")


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------
def _judge_summary(rec):
    """Reduce a judge EventRecord to (present, valence, target_turn) for scoring."""
    sanctions = [e for e in rec.events if not e.target_is_self and e.valence in ("approve", "disapprove")]
    if not sanctions:
        return False, None, None
    best = max(sanctions, key=lambda e: e.confidence)
    return True, best.valence, best.target_turn_index


def run_validate(cfg, substrates=None, run_ids=None, iter_min=None, iter_max=None):
    cfg.make_dirs()
    path = cfg.paths.labels
    if not os.path.exists(path):
        print(f"[validate] no labels at {path}. Run `label` first.")
        return
    with open(path) as f:
        labels = [json.loads(l) for l in f if l.strip()]
    labels = [e for e in labels if e.get("labelled") and "human_present" in e]
    if not labels:
        print("[validate] no usable human labels found.")
        return

    backend = make_judge_backend(cfg.judge)
    cache = Cache(cfg.paths.cache_db)
    judge = Judge(backend, cache, cfg.judge)
    index = _index_posts(cfg, substrates, run_ids, iter_min, iter_max)

    tp = fp = fn = tn = 0
    valence_pairs = []        # (human, judge) on true positives
    target_hits = target_total = 0
    fp_examples = []

    try:
        from sklearn.metrics import cohen_kappa_score
    except ImportError:
        cohen_kappa_score = None

    for e in labels:
        posts = index.get(e["episode_id"])
        if posts is None:
            continue
        cand_idx = next((i for i, p in enumerate(posts) if p.turn_index == e["turn_index"]), None)
        if cand_idx is None:
            continue
        rec = judge.label_post(posts, cand_idx)
        j_present, j_val, j_tgt = _judge_summary(rec)
        h_present = bool(e["human_present"])

        if h_present and j_present:
            tp += 1
            h_val = e.get("human_valence")
            if h_val in ("approve", "disapprove") and j_val:
                valence_pairs.append((h_val, j_val))
            h_tgt = e.get("human_target_turn_index")
            if h_tgt is not None:
                target_total += 1
                if j_tgt == h_tgt:
                    target_hits += 1
        elif (not h_present) and j_present:
            fp += 1
            fp_examples.append({
                "episode_id": e["episode_id"], "turn_index": e["turn_index"],
                "judge_valence": j_val,
                "judge_triggers": [ev.trigger_description for ev in rec.events
                                   if not ev.target_is_self and ev.valence != "neutral"],
            })
        elif h_present and (not j_present):
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    if cohen_kappa_score and valence_pairs and len({p[0] for p in valence_pairs} | {p[1] for p in valence_pairs}) > 1:
        kappa = float(cohen_kappa_score([p[0] for p in valence_pairs], [p[1] for p in valence_pairs]))
    else:
        # fall back to raw agreement if kappa is undefined (single class)
        kappa = (sum(1 for a, b in valence_pairs if a == b) / len(valence_pairs)) if valence_pairs else 0.0

    target_acc = target_hits / target_total if target_total else None

    report = {
        "provider": cfg.judge.provider,
        "model": cfg.judge.model,
        "prompt_version": PROMPT_VERSION,
        "n_labelled": len(labels),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "metrics": {
            "presence_precision": precision,
            "presence_recall": recall,
            "presence_f1": f1,
            "valence_kappa": kappa,
            "target_turn_accuracy": target_acc,
            "n_valence_pairs": len(valence_pairs),
        },
        "thresholds": {
            "presence_f1_min": cfg.validation.presence_f1_min,
            "valence_kappa_min": cfg.validation.valence_kappa_min,
        },
        "false_positive_triggers": fp_examples[:50],  # bias-check surface
    }
    cache.close()

    out = report_path(cfg)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[validate] wrote {out}")
    print(json.dumps(report["metrics"], indent=2))
    ok, msg = gate_status(cfg)
    print(f"[validate] gate: {msg}")
    if fp_examples:
        print(f"\n[validate] {len(fp_examples)} false positives (judge flagged, human did not). "
              "Inspect false_positive_triggers in the report for systematic bias "
              "(the 'manufacturing-norms' risk made visible).")
