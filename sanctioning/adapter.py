"""Transcript adapter: the ONLY place that knows the on-disk wire format.

On disk a corpus run is::

    <run.path>/traces/ckpt_<NNNNNN>/traces.json

where ``traces.json`` is a list of episodes, each::

    {"episode": <int>, "env_trace": {"thread": [ {post_index, speaker, text}, ... ], "env": <str>, ...}, "agents": {...}}

``env_trace.thread`` holds the public post bodies (forum posts). Thinking
tokens are not present in ``text`` (they are private to the actor and were
never logged here), so the judge only ever sees public bodies.

If the on-disk shape ever changes, this is the single file to update.
"""

from __future__ import annotations

import glob
import json
import os
import re
from typing import Iterator, Optional

from .config import RunSpec
from .types import Episode, Post

_CKPT_RE = re.compile(r"ckpt_(\d+)")


def _iter_checkpoints(run_path: str, iter_min: Optional[int], iter_max: Optional[int]):
    for d in sorted(glob.glob(os.path.join(run_path, "traces", "ckpt_*"))):
        m = _CKPT_RE.search(os.path.basename(d))
        if not m:
            continue
        it = int(m.group(1))
        if iter_min is not None and it < iter_min:
            continue
        if iter_max is not None and it > iter_max:
            continue
        tj = os.path.join(d, "traces.json")
        if os.path.exists(tj):
            yield it, tj


def load_episodes(
    run: RunSpec,
    iter_min: Optional[int] = None,
    iter_max: Optional[int] = None,
) -> Iterator[Episode]:
    """Yield normalised Episodes for one corpus run."""
    for iteration, tj in _iter_checkpoints(run.path, iter_min, iter_max):
        with open(tj) as f:
            data = json.load(f)
        for ep in data:
            ep_num = ep.get("episode", 0)
            episode_id = f"{run.run_id}/ckpt{iteration:06d}/ep{ep_num}"
            env_trace = ep.get("env_trace") or {}
            thread = env_trace.get("thread") or []
            posts = []
            for p in thread:
                ti = p.get("post_index")
                if ti is None:
                    ti = len(posts)
                posts.append(
                    Post(
                        episode_id=episode_id,
                        run_id=run.run_id,
                        iteration=iteration,
                        substrate=run.substrate,
                        turn_index=int(ti),
                        agent_id=str(p.get("speaker", "")),
                        body=str(p.get("text", "")),
                    )
                )
            posts.sort(key=lambda x: x.turn_index)
            yield Episode(
                episode_id=episode_id,
                run_id=run.run_id,
                iteration=iteration,
                substrate=run.substrate,
                posts=posts,
            )


def iter_corpus(
    corpus: list,
    substrates: Optional[set] = None,
    run_ids: Optional[set] = None,
    iter_min: Optional[int] = None,
    iter_max: Optional[int] = None,
) -> Iterator[Episode]:
    """Iterate Episodes across the corpus, applying CLI filters."""
    for run in corpus:
        if substrates and run.substrate not in substrates:
            continue
        if run_ids and run.run_id not in run_ids:
            continue
        if not os.path.isdir(run.path):
            # Surface missing runs rather than silently skipping the whole family.
            print(f"[adapter] WARNING: run path not found, skipping: {run.path}")
            continue
        yield from load_episodes(run, iter_min, iter_max)
