"""SQLite-backed cache for judge calls and embeddings.

Essential at corpus scale: re-runs, validation, and resumed extractions must
not re-bill. Keys are content hashes (see `judge.py` / `cluster.py`), so the
cache is keyed on prompt_version + model + params + window + candidate body.
"""

from __future__ import annotations

import sqlite3
import threading
from typing import Optional


class Cache:
    def __init__(self, path: str):
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS judge ("
            "  key TEXT PRIMARY KEY, raw TEXT, usage_in INTEGER, usage_out INTEGER)"
        )
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS embed (key TEXT PRIMARY KEY, vec TEXT)"
        )
        self._conn.commit()

    # --- judge calls ---------------------------------------------------
    def get_judge(self, key: str) -> Optional[tuple]:
        with self._lock:
            row = self._conn.execute(
                "SELECT raw, usage_in, usage_out FROM judge WHERE key=?", (key,)
            ).fetchone()
        return row  # (raw, usage_in, usage_out) or None

    def put_judge(self, key: str, raw: str, usage_in: int, usage_out: int):
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO judge(key, raw, usage_in, usage_out) VALUES (?,?,?,?)",
                (key, raw, usage_in, usage_out),
            )
            self._conn.commit()

    # --- embeddings ----------------------------------------------------
    def get_embed(self, key: str) -> Optional[str]:
        with self._lock:
            row = self._conn.execute(
                "SELECT vec FROM embed WHERE key=?", (key,)
            ).fetchone()
        return row[0] if row else None

    def put_embed(self, key: str, vec_json: str):
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO embed(key, vec) VALUES (?,?)", (key, vec_json)
            )
            self._conn.commit()

    def close(self):
        with self._lock:
            self._conn.close()
