"""The judge: per-post sanctioning-event extraction.

Detection only — the judge labels valence / target / a free-text trigger
description per event. No "is this a norm" judgement is ever asked of it.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Optional

from . import PROMPT_VERSION, SCHEMA_VERSION
from .backends import LabelResult
from .types import VALENCES, EventRecord, Post, SanctionEvent

_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "prompts", f"{PROMPT_VERSION}.txt")


def load_prompt() -> str:
    with open(_PROMPT_PATH) as f:
        return f.read()


def build_window(posts: list, candidate_idx: int, window_k: int) -> list:
    """The K posts preceding the candidate (by list position), oldest first."""
    start = max(0, candidate_idx - window_k)
    return posts[start:candidate_idx]


def format_user_message(window: list, candidate: Post) -> str:
    lines = ["CONTEXT (earlier posts, oldest first):"]
    if not window:
        lines.append("(no earlier posts)")
    for p in window:
        lines.append(f"[turn_index={p.turn_index}] {p.agent_id}: {p.body}")
    lines.append("")
    lines.append("POST (the post to label):")
    lines.append(f"[turn_index={candidate.turn_index}] {candidate.agent_id}: {candidate.body}")
    return "\n".join(lines)


def cache_key(prompt_version, provider, model, temperature, window_k, user_message) -> str:
    h = hashlib.sha256()
    h.update("|".join([
        prompt_version, provider, model, f"{temperature}", f"{window_k}", user_message,
    ]).encode("utf-8"))
    return h.hexdigest()


def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        # remove leading ```json / ``` and trailing ```
        t = t.split("\n", 1)[1] if "\n" in t else t
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    return t.strip()


def parse_and_validate(raw: str) -> list:
    """Parse the model's JSON into validated SanctionEvent objects.

    Raises ValueError on malformed JSON or schema violations (the caller
    retries once, then records an empty event list and flags the failure).
    """
    obj = json.loads(_strip_fences(raw))
    if not isinstance(obj, dict) or "events" not in obj:
        raise ValueError("missing 'events' key")
    events_raw = obj["events"]
    if not isinstance(events_raw, list):
        raise ValueError("'events' is not a list")
    out = []
    for e in events_raw:
        if not isinstance(e, dict):
            raise ValueError("event is not an object")
        val = e.get("valence")
        if val not in VALENCES:
            raise ValueError(f"bad valence: {val!r}")
        intensity = float(e.get("intensity", 0.0))
        conf = float(e.get("confidence", 0.0))
        if not (0.0 <= intensity <= 1.0) or not (0.0 <= conf <= 1.0):
            raise ValueError("intensity/confidence out of [0,1]")
        tti = e.get("target_turn_index", None)
        if tti is not None:
            tti = int(tti)
        trig = e.get("trigger_description", "")
        if not isinstance(trig, str):
            raise ValueError("trigger_description not a string")
        span = str(e.get("evidence_span", ""))[:160]
        out.append(
            SanctionEvent(
                valence=val,
                intensity=intensity if val != "neutral" else 0.0,
                target_turn_index=tti,
                target_agent_id=(e.get("target_agent_id") or None),
                target_is_self=bool(e.get("target_is_self", False)),
                trigger_description=trig,
                evidence_span=span,
                confidence=conf,
            )
        )
    return out


class Judge:
    def __init__(self, backend, cache, cfg):
        self.backend = backend
        self.cache = cache
        self.cfg = cfg
        self.system = load_prompt()
        # running tallies
        self.calls = 0
        self.cache_hits = 0
        self.parse_failures = 0
        self.usage_in = 0
        self.usage_out = 0

    def label_post(self, posts: list, candidate_idx: int) -> EventRecord:
        candidate = posts[candidate_idx]
        window = build_window(posts, candidate_idx, self.cfg.window_k)
        user = format_user_message(window, candidate)
        key = cache_key(
            PROMPT_VERSION, self.cfg.provider, self.cfg.model,
            self.cfg.temperature, self.cfg.window_k, user,
        )

        cached = self.cache.get_judge(key)
        if cached is not None:
            raw, ui, uo = cached
            self.cache_hits += 1
        else:
            raw, ui, uo = self._call_with_retry(user)
            self.cache.put_judge(key, raw, ui, uo)
            self.calls += 1
            self.usage_in += ui
            self.usage_out += uo

        try:
            events = parse_and_validate(raw)
        except (ValueError, json.JSONDecodeError):
            # raw was cached but unparseable; record empty + flag
            self.parse_failures += 1
            events = []

        raw_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
        return EventRecord(
            episode_id=candidate.episode_id,
            run_id=candidate.run_id,
            iteration=candidate.iteration,
            substrate=candidate.substrate,
            turn_index=candidate.turn_index,
            agent_id=candidate.agent_id,
            events=events,
            judge_meta={
                "provider": self.cfg.provider,
                "model": self.cfg.model,
                "prompt_version": PROMPT_VERSION,
                "temperature": self.cfg.temperature,
                "window_k": self.cfg.window_k,
                "raw_response_hash": raw_hash,
                "usage_tokens": {"input": ui, "output": uo},
            },
            schema_version=SCHEMA_VERSION,
        )

    def _call_with_retry(self, user: str):
        """One call; on unparseable output retry once with a terse nudge."""
        res: LabelResult = self.backend.label(
            self.system, user,
            temperature=self.cfg.temperature, max_tokens=self.cfg.max_tokens,
        )
        try:
            parse_and_validate(res.text)
            return res.text, res.usage_in, res.usage_out
        except (ValueError, json.JSONDecodeError):
            pass
        retry_user = user + (
            "\n\nYour previous reply was not valid. Respond with strict JSON only, "
            'exactly {"events": [ ... ]} and nothing else.'
        )
        res2: LabelResult = self.backend.label(
            self.system, retry_user,
            temperature=self.cfg.temperature, max_tokens=self.cfg.max_tokens,
        )
        # cache/account the retry's tokens together with the first attempt
        return res2.text, res.usage_in + res2.usage_in, res.usage_out + res2.usage_out
