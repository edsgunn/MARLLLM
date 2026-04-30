"""
Convex-hull novelty diagnostic (spec §7d).

For each utterance produced by the cell's learner, compute the fraction that
fall outside the kNN ball of the strong-partner utterance set. We use kNN
rather than a true convex hull because convex hulls in high dimensions are
unstable / undefined.

Uses sentence-transformers if available; falls back to bag-of-words.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from marlllm.diagnostics.mimicry import load_actions


def _embed(texts: list[str]) -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer("all-MiniLM-L6-v2")
        return encoder.encode(texts, normalize_embeddings=True)
    except ImportError:
        # Bag-of-words fallback.
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import normalize
        vec = TfidfVectorizer(min_df=1, max_features=2048)
        X = vec.fit_transform(texts).toarray().astype(np.float32)
        return normalize(X)


def run(
    cell_rollouts: str | Path,
    cell_a_transcripts: str | Path,
    learner_role: str,
    output_dir: str | Path,
    k: int = 5,
    novelty_quantile: float = 0.95,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    learner = load_actions(Path(cell_rollouts), learner_role)
    teacher = load_actions(Path(cell_a_transcripts), learner_role)
    if not learner or not teacher:
        result = {"error": "no actions found",
                  "n_learner": len(learner), "n_teacher": len(teacher)}
        (output_dir / "hull.json").write_text(json.dumps(result, indent=2))
        return result

    all_texts = teacher + learner
    emb = _embed(all_texts)
    teacher_emb = emb[: len(teacher)]
    learner_emb = emb[len(teacher):]

    # For each teacher point, find its k-th nearest teacher neighbour distance —
    # this defines an "inside the manifold" radius.
    sims = teacher_emb @ teacher_emb.T  # cosine
    np.fill_diagonal(sims, -np.inf)
    top_k = -np.partition(-sims, kth=k - 1, axis=1)[:, :k]
    teacher_radii = top_k[:, -1]                   # k-th sim, smaller = farther
    radius_threshold = np.quantile(teacher_radii, 1 - novelty_quantile)

    # For each learner utterance, distance to nearest teacher.
    cross = learner_emb @ teacher_emb.T            # (Nl, Nt)
    nearest = cross.max(axis=1)                    # higher = closer
    out_of_hull = (nearest < radius_threshold).mean()

    result = {
        "n_learner": int(len(learner)),
        "n_teacher": int(len(teacher)),
        "k": k,
        "novelty_quantile": novelty_quantile,
        "radius_threshold_cosine": float(radius_threshold),
        "fraction_out_of_hull": float(out_of_hull),
        "mean_nearest_teacher_cos": float(nearest.mean()),
    }
    (output_dir / "hull.json").write_text(json.dumps(result, indent=2))
    return result
