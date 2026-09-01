"""Normalised internal types consumed by the rest of the pipeline.

The transcript adapter (`adapter.py`) is the *only* place that knows the
on-disk wire format; it produces `Episode`/`Post` objects. An action-tagged
substrate could be slotted in later by extending `Post` (e.g. an optional
``action_kind`` field) without touching the judge or aggregation.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional


# ---------------------------------------------------------------------------
# Input side (produced by the adapter)
# ---------------------------------------------------------------------------
@dataclass
class Post:
    episode_id: str
    run_id: str            # seed / population identifier (the run directory name)
    iteration: int         # training checkpoint (e.g. 25, 50, ...)
    substrate: str         # behaviour family, e.g. "ashbourne_gc", "margin_notes"
    turn_index: int        # position within episode, 0-based
    agent_id: str          # speaker (persona / character slot)
    body: str              # PUBLIC post body only (no thinking tokens)
    persona_label: Optional[str] = None
    # Reserved for an action-tagged substrate; None for forum posts.
    action_kind: Optional[str] = None


@dataclass
class Episode:
    episode_id: str
    run_id: str
    iteration: int
    substrate: str
    posts: list  # list[Post], ordered by turn_index


# ---------------------------------------------------------------------------
# Output side (the sanctioning-event triple + provenance)
# ---------------------------------------------------------------------------
VALENCES = ("approve", "disapprove", "neutral")


@dataclass
class SanctionEvent:
    valence: str                       # one of VALENCES
    intensity: float                   # 0.0-1.0; 0 for neutral
    target_turn_index: Optional[int]   # which prior post; None if diffuse
    target_agent_id: Optional[str]     # copied from the targeted post; None if diffuse
    target_is_self: bool
    trigger_description: str           # free text (by design); NOT a fixed vocabulary
    evidence_span: str                 # <=160 chars quoted from the REACTING post
    confidence: float                  # 0.0-1.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EventRecord:
    """One persisted record per *candidate reacting post*."""
    # reacting-post key
    episode_id: str
    run_id: str
    iteration: int
    substrate: str
    turn_index: int
    agent_id: str
    # labels
    events: list                       # list[SanctionEvent]
    # provenance
    judge_meta: dict                   # provider, model, prompt_version, temperature,
                                       # window_k, raw_response_hash, usage_tokens
    schema_version: str

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    def event_ids(self) -> list:
        """Stable IDs for each event, used to key cluster assignments."""
        return [event_id(self, i) for i in range(len(self.events))]


def event_id(rec: "EventRecord", event_index: int) -> str:
    """Globally-stable identifier for an event within an event record."""
    return f"{rec.run_id}|it{rec.iteration:06d}|{rec.episode_id}|t{rec.turn_index}|e{event_index}"
