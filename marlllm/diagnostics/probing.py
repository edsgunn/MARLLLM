"""
Representational probing diagnostic (spec §7c).

Trains two linear probes on hidden states the learner produces at action
positions during a fixed eval-rollout set:

  - environment-state probe: predict (items, values) from hidden state.
  - partner-identity probe: predict the partner's role/scenario id.

The diagnostic depends on `eval_rollout.collect_eval_rollouts` having been
run with `dump_hidden_states=True` so a sibling `.hidden.pkl` exists.

Episode-level train/test split avoids leakage from probes seeing earlier
turns of the same episode at train time.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np


def _load_hidden(path: Path) -> list[dict]:
    with open(path, "rb") as f:
        return pickle.load(f)


def _episode_split(records: list[dict], frac_train: float = 0.7, seed: int = 0):
    rng = np.random.default_rng(seed)
    episodes = sorted({r["episode_id"] for r in records})
    rng.shuffle(episodes)
    n_train = int(len(episodes) * frac_train)
    train_eps = set(episodes[:n_train])
    train = [r for r in records if r["episode_id"] in train_eps]
    test = [r for r in records if r["episode_id"] not in train_eps]
    return train, test


def _matrix(records: list[dict], target_fn) -> tuple[np.ndarray, np.ndarray]:
    X = np.array([r["hidden"] for r in records], dtype=np.float32)
    y = np.array([target_fn(r) for r in records])
    return X, y


def _fit_score_classifier(X_tr, y_tr, X_te, y_te) -> float:
    from sklearn.linear_model import LogisticRegression
    if len(set(y_tr)) < 2:
        return float("nan")
    clf = LogisticRegression(max_iter=1000, multi_class="auto")
    clf.fit(X_tr, y_tr)
    return float(clf.score(X_te, y_te))


def _fit_score_regressor(X_tr, y_tr, X_te, y_te) -> float:
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    reg = Ridge(alpha=1.0)
    reg.fit(X_tr, y_tr)
    pred = reg.predict(X_te)
    return float(r2_score(y_te, pred, multioutput="uniform_average"))


def run(rollout_jsonl: str | Path, output_dir: str | Path) -> dict:
    rollout_jsonl = Path(rollout_jsonl)
    hidden_path = rollout_jsonl.with_suffix(".hidden.pkl")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not hidden_path.exists():
        result = {"error": f"hidden states not found at {hidden_path}; "
                           f"rerun eval_rollout with dump_hidden_states=True"}
        (output_dir / "probing.json").write_text(json.dumps(result, indent=2))
        return result

    records = _load_hidden(hidden_path)
    if len(records) < 8:
        result = {"error": "too few hidden-state records"}
        (output_dir / "probing.json").write_text(json.dumps(result, indent=2))
        return result

    train, test = _episode_split(records)

    # Environment-state probe: regression on (items + values) joint vector.
    def env_target(r):
        return list(r.get("items", [])) + list(r.get("values", []))

    X_tr, y_tr = _matrix(train, env_target)
    X_te, y_te = _matrix(test, env_target)
    if y_tr.ndim < 2 or y_tr.size == 0:
        env_r2 = float("nan")
    else:
        env_r2 = _fit_score_regressor(X_tr, y_tr, X_te, y_te)

    # Phase probe: classify env phase (dialogue vs selection).
    def phase_target(r):
        return r.get("phase", "unknown")

    X_tr2, y_tr2 = _matrix(train, phase_target)
    X_te2, y_te2 = _matrix(test, phase_target)
    phase_acc = _fit_score_classifier(X_tr2, y_tr2, X_te2, y_te2)

    # Episode-id probe: should be near-chance if the model isn't memorising
    # — included as a sanity check.
    def episode_target(r):
        return r.get("episode_id", -1)

    X_tr3, y_tr3 = _matrix(train, episode_target)
    X_te3, y_te3 = _matrix(test, episode_target)
    epid_acc = _fit_score_classifier(X_tr3, y_tr3, X_te3, y_te3) if len(set(y_te3)) > 1 else float("nan")

    result = {
        "n_train": len(train),
        "n_test": len(test),
        "env_state_r2": env_r2,
        "phase_accuracy": phase_acc,
        "episode_id_accuracy": epid_acc,
    }
    (output_dir / "probing.json").write_text(json.dumps(result, indent=2))
    return result
