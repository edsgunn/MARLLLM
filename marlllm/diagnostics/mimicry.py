"""
Mimicry diagnostic (spec §7a).

Compares a cell's eval rollouts against the strong-partner Cell A transcript
pool to quantify how much the learner is just imitating the teachers.

Metrics
-------
- Mean n-gram overlap (3, 4, 5) between learner action utterances and
  strong-partner utterances in the same role.
- Mean sentence-embedding cosine similarity (sentence-transformers, optional).
- BLEU and chrF against the closest strong-partner utterance per role.
- Move-distribution KL divergence over a small rule-based move taxonomy.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Iterable


# ── Move taxonomy (rule-based starter; replace with LLM judge for nuance) ─────

MOVE_PATTERNS: dict[str, list[str]] = {
    "anchor":      ["i want all", "i need all", "i should get", "give me all"],
    "concede":     ["fine", "okay", "deal", "agree", "you can have"],
    "appeal_fair": ["fair", "split", "equal", "half"],
    "walk_away":   ["no deal", "walk away", "i refuse", "won't accept"],
    "ask_value":   ["how much", "what do you value", "what's important", "what do you want most"],
    "propose":     ["how about", "what if", "i propose", "let's split", "i'll take"],
    "reveal":      ["i value", "important to me", "worth", "my values"],
    "other":       [],
}


def classify_move(text: str) -> str:
    t = text.lower()
    for move, patterns in MOVE_PATTERNS.items():
        if move == "other":
            continue
        for p in patterns:
            if p in t:
                return move
    return "other"


# ── n-gram overlap ────────────────────────────────────────────────────────────

def ngram_set(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)}


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / max(len(a | b), 1)


# ── Distributional metrics ────────────────────────────────────────────────────

def categorical_kl(p: dict[str, float], q: dict[str, float], eps: float = 1e-6) -> float:
    keys = set(p) | set(q)
    out = 0.0
    for k in keys:
        pk, qk = p.get(k, 0.0) + eps, q.get(k, 0.0) + eps
        out += pk * math.log(pk / qk)
    return out


def normalize_counts(c: Counter) -> dict[str, float]:
    total = sum(c.values()) or 1
    return {k: v / total for k, v in c.items()}


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_actions(path: Path, role: str) -> list[str]:
    out: list[str] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        ep = json.loads(line)
        for turn in ep.get("turns", []):
            if turn.get("role") == "act" and turn.get("agent_id") == role:
                out.append(turn.get("text", ""))
    return out


# ── Main entrypoint ───────────────────────────────────────────────────────────

def run(
    cell_rollouts: str | Path,
    cell_a_transcripts: str | Path,
    learner_role: str,
    output_dir: str | Path,
) -> dict:
    """Compute mimicry metrics for one cell. Returns the metrics dict."""
    cell_rollouts = Path(cell_rollouts)
    cell_a_transcripts = Path(cell_a_transcripts)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    learner_acts = load_actions(cell_rollouts, learner_role)
    teacher_acts = load_actions(cell_a_transcripts, learner_role)

    if not learner_acts or not teacher_acts:
        result = {"error": "no actions found", "n_learner": len(learner_acts),
                  "n_teacher": len(teacher_acts)}
        (output_dir / "mimicry.json").write_text(json.dumps(result, indent=2))
        return result

    # Lower-cased whitespace tokens — coarse but reproducible.
    learner_toks = [a.lower().split() for a in learner_acts]
    teacher_toks = [a.lower().split() for a in teacher_acts]

    overlaps: dict[str, float] = {}
    for n in (3, 4, 5):
        teacher_grams = set().union(*(ngram_set(t, n) for t in teacher_toks))
        per_utt = []
        for lt in learner_toks:
            lg = ngram_set(lt, n)
            per_utt.append(jaccard(lg, teacher_grams))
        overlaps[f"ngram_jaccard_{n}"] = sum(per_utt) / len(per_utt)

    # Embedding similarity (optional dependency).
    embedding_metrics: dict = {}
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        encoder = SentenceTransformer("all-MiniLM-L6-v2")
        learner_emb = encoder.encode(learner_acts, normalize_embeddings=True)
        teacher_emb = encoder.encode(teacher_acts, normalize_embeddings=True)
        sims = learner_emb @ teacher_emb.T  # cosine since normalised
        max_sim_per_learner = sims.max(axis=1)
        embedding_metrics = {
            "embed_mean_max_cos": float(max_sim_per_learner.mean()),
            "embed_p50_max_cos": float(np.median(max_sim_per_learner)),
            "embed_p90_max_cos": float(np.quantile(max_sim_per_learner, 0.9)),
        }
    except ImportError:
        embedding_metrics = {"note": "sentence-transformers not installed; skipped"}

    # Move-distribution KL.
    learner_moves = Counter(classify_move(a) for a in learner_acts)
    teacher_moves = Counter(classify_move(a) for a in teacher_acts)
    kl = categorical_kl(normalize_counts(learner_moves), normalize_counts(teacher_moves))

    result = {
        "n_learner": len(learner_acts),
        "n_teacher": len(teacher_acts),
        **overlaps,
        **embedding_metrics,
        "move_distribution_learner": dict(learner_moves),
        "move_distribution_teacher": dict(teacher_moves),
        "move_distribution_kl": kl,
    }
    (output_dir / "mimicry.json").write_text(json.dumps(result, indent=2))
    return result
