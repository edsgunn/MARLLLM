"""`cluster` — embed trigger_descriptions and cluster them post hoc.

Behaviour categories are formed *after* detection by clustering, never supplied
to the judge. Cluster labels (medoid descriptions) are descriptive output only
and are never fed back into detection.

Embedding is behind a provider abstraction: "tfidf" (local, no API key) or
"openai" (cached per text). Clustering is swappable: agglomerative (default,
sklearn) or hdbscan (if installed).
"""

from __future__ import annotations

import hashlib
import json

import numpy as np

from .backends import OpenAIEmbedding
from .cache import Cache
from .persistence import ClusterStore, EventStore
from .types import event_id as _event_id


def event_id_from_dict(d: dict, idx: int) -> str:
    """Mirror of types.event_id for a record dict + event index."""
    return f"{d['run_id']}|it{d['iteration']:06d}|{d['episode_id']}|t{d['turn_index']}|e{idx}"


def collect_events(cfg, substrates=None, run_ids=None, iter_min=None, iter_max=None):
    """Return parallel lists of clusterable events (self-directed excluded)."""
    store = EventStore(cfg.paths.events)
    ids, texts, meta = [], [], []
    for d in store.iter_records(substrates, run_ids, iter_min, iter_max):
        for i, e in enumerate(d["events"]):
            if e.get("target_is_self"):
                continue
            ids.append(event_id_from_dict(d, i))
            texts.append(e.get("trigger_description", "") or "")
            meta.append({
                "valence": e["valence"], "substrate": d["substrate"],
                "run_id": d["run_id"], "iteration": d["iteration"],
                "agent_id": d["agent_id"], "target_agent_id": e.get("target_agent_id"),
                "confidence": float(e.get("confidence", 0.0)),
            })
    return ids, texts, meta


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------
def embed_texts(texts: list, cfg) -> np.ndarray:
    """Return an (n, d) float matrix of L2-normalised row vectors."""
    # OpenAI rejects empty-string input; blank trigger descriptions (rare edge
    # cases) get a placeholder for embedding only — original text is kept for
    # labels/storage by the caller.
    texts = [t if (t and t.strip()) else "(no description)" for t in texts]
    if cfg.embedding.provider == "tfidf":
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(min_df=2, stop_words="english")
        X = vec.fit_transform(texts)
        X = X.toarray().astype(np.float64)
    elif cfg.embedding.provider == "openai":
        X = _embed_openai_cached(texts, cfg)
    else:
        raise ValueError(f"unknown embedding provider: {cfg.embedding.provider}")
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms


def _embed_openai_cached(texts: list, cfg) -> np.ndarray:
    cache = Cache(cfg.paths.cache_db)
    out = [None] * len(texts)
    missing_idx, missing_txt = [], []
    for i, t in enumerate(texts):
        key = hashlib.sha256(f"{cfg.embedding.model}|{t}".encode()).hexdigest()
        cached = cache.get_embed(key)
        if cached is not None:
            out[i] = json.loads(cached)
        else:
            missing_idx.append(i)
            missing_txt.append(t)
    if missing_txt:
        # Only construct the backend (which needs OPENAI_API_KEY) when there is
        # something to fetch — fully-cached runs don't require the key.
        backend = OpenAIEmbedding(cfg.embedding.model)
        vecs = backend.embed(missing_txt)
        for j, i in enumerate(missing_idx):
            out[i] = vecs[j]
            key = hashlib.sha256(f"{cfg.embedding.model}|{texts[i]}".encode()).hexdigest()
            cache.put_embed(key, json.dumps(vecs[j]))
    cache.close()
    return np.array(out, dtype=np.float64)


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------
def cluster_vectors(X: np.ndarray, cfg, distance_threshold=None) -> np.ndarray:
    """Return integer cluster labels; -1 is the noise / unclustered bucket."""
    n = X.shape[0]
    if n == 0:
        return np.array([], dtype=int)
    method = cfg.clustering.method
    min_size = cfg.clustering.min_cluster_size

    if method == "hdbscan":
        try:
            import hdbscan
        except ImportError:
            print("[cluster] hdbscan not installed; falling back to agglomerative.")
            method = "agglomerative"
        else:
            labels = hdbscan.HDBSCAN(min_cluster_size=min_size).fit_predict(X)
            return labels.astype(int)

    if method == "agglomerative":
        from sklearn.cluster import AgglomerativeClustering
        thr = cfg.clustering.distance_threshold if distance_threshold is None else distance_threshold
        if n == 1:
            return np.array([0])
        labels = AgglomerativeClustering(
            n_clusters=None, metric="cosine", linkage="average",
            distance_threshold=thr,
        ).fit_predict(X)
        # enforce min_cluster_size: small clusters -> noise (-1)
        counts = {}
        for l in labels:
            counts[l] = counts.get(l, 0) + 1
        labels = np.array([l if counts[l] >= min_size else -1 for l in labels])
        return labels

    raise ValueError(f"unknown clustering method: {method}")


def medoid_labels(texts: list, labels: np.ndarray, X: np.ndarray) -> dict:
    """For each non-noise cluster, the description nearest its centroid."""
    out = {}
    for cid in sorted(set(labels)):
        if cid == -1:
            out[cid] = "(noise/unclustered)"
            continue
        idx = np.where(labels == cid)[0]
        centroid = X[idx].mean(axis=0)
        dists = np.linalg.norm(X[idx] - centroid, axis=1)
        out[cid] = texts[idx[int(np.argmin(dists))]]
    return out


def run_cluster(cfg, substrates=None, run_ids=None, iter_min=None, iter_max=None):
    cfg.make_dirs()
    ids, texts, meta = collect_events(cfg, substrates, run_ids, iter_min, iter_max)
    if not ids:
        print("[cluster] no events found. Run `extract` first.")
        return
    print(f"[cluster] {len(ids)} events; embedding ({cfg.embedding.provider})...")
    X = embed_texts(texts, cfg)
    labels = cluster_vectors(X, cfg)
    lbls = medoid_labels(texts, labels, X)

    n_clusters = len([c for c in set(labels) if c != -1])
    n_noise = int((labels == -1).sum())
    print(f"[cluster] {n_clusters} clusters, {n_noise} noise "
          f"(method={cfg.clustering.method}, "
          f"threshold={cfg.clustering.distance_threshold}, min_size={cfg.clustering.min_cluster_size})")

    assignments = [
        {
            "event_id": ids[i],
            "behaviour_cluster_id": int(labels[i]),
            "cluster_label": lbls[labels[i]],
            "trigger_description": texts[i],
        }
        for i in range(len(ids))
    ]
    ClusterStore(cfg.paths.clusters).write(assignments)
    print(f"[cluster] wrote {cfg.paths.clusters}")

    # quick readout of the largest clusters
    from collections import Counter
    c = Counter(int(l) for l in labels if l != -1)
    print("\nlargest behaviour clusters (id, size, medoid label):")
    for cid, size in c.most_common(12):
        print(f"  {cid:>4}  n={size:<5}  {lbls[cid][:90]}")
