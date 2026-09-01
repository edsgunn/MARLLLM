"""Append-only persistence of event records and cluster assignments.

Event records (one JSON object per reacting post) are written to a JSONL file
with full provenance. Aggregation reads these and never re-calls the judge.
Cluster assignments live in a separate file keyed by event_id, so re-clustering
never touches the raw judge output.
"""

from __future__ import annotations

import json
import os
from typing import Iterator, Optional

from .types import EventRecord


class EventStore:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    # --- write ---------------------------------------------------------
    def append(self, rec: EventRecord):
        with open(self.path, "a") as f:
            f.write(json.dumps(rec.to_dict()) + "\n")

    def existing_keys(self) -> set:
        """(run_id, iteration, episode_id, turn_index) already extracted.

        Lets `extract` resume without re-emitting duplicate records.
        """
        keys = set()
        if not os.path.exists(self.path):
            return keys
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                keys.add((d["run_id"], d["iteration"], d["episode_id"], d["turn_index"]))
        return keys

    # --- read ----------------------------------------------------------
    def iter_records(
        self,
        substrates: Optional[set] = None,
        run_ids: Optional[set] = None,
        iter_min: Optional[int] = None,
        iter_max: Optional[int] = None,
    ) -> Iterator[dict]:
        if not os.path.exists(self.path):
            return
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                if substrates and d["substrate"] not in substrates:
                    continue
                if run_ids and d["run_id"] not in run_ids:
                    continue
                if iter_min is not None and d["iteration"] < iter_min:
                    continue
                if iter_max is not None and d["iteration"] > iter_max:
                    continue
                yield d


class ClusterStore:
    """event_id -> {behaviour_cluster_id, cluster_label}."""

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def write(self, assignments: list):
        """assignments: list of {event_id, behaviour_cluster_id, cluster_label, trigger_description}."""
        with open(self.path, "w") as f:
            for a in assignments:
                f.write(json.dumps(a) + "\n")

    def load(self) -> dict:
        out = {}
        if not os.path.exists(self.path):
            return out
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                a = json.loads(line)
                out[a["event_id"]] = a
        return out
