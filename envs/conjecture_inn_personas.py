"""
Conjecture Inn character personas — Concordia-free data module.

Mirrors ``envs.robotic_athanor_personas`` so ``ForumEnv`` (and any
calibration scripts) can swap scenarios by character roster alone.

The persona content (forum description, characters, system-prompt
template) lives in ``envs/conjecture_inn.json`` so the prose can be
edited without touching Python.  The 8-agent set is a strict superset
of the 4-agent set, controlling for character-set effects when
comparing across population sizes.
"""
from __future__ import annotations

import json
from pathlib import Path


_JSON_PATH = Path(__file__).with_name("conjecture_inn.json")


def _load() -> dict:
    with _JSON_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


_DATA = _load()

FORUM_DESCRIPTION: str = _DATA["environment"]["description"]
_SYSTEM_PROMPT_TEMPLATE: str = _DATA["system_prompt_template"]

_CANONICAL_CHARS: dict[str, dict] = _DATA["characters"]["canonical"]
_ADDITIONAL_CHARS: dict[str, dict] = _DATA["characters"]["additional"]


# ---------------------------------------------------------------------------
# Character roster
# ---------------------------------------------------------------------------
# The two canonical characters with the sharpest, most directly opposed
# stances on what constitutes a proof: Iris (formalist — proof = checkable
# in Lean) vs Misha (structuralist — proof = the right framing makes it
# inevitable).  Picked to maximise productive disagreement at population
# size two.
CANONICAL_2 = ["Iris Halverson", "Misha Volkov"]

CANONICAL_4 = list(_CANONICAL_CHARS.keys())
EXTENDED_8 = CANONICAL_4 + list(_ADDITIONAL_CHARS.keys())

_ALL_CHARS: dict[str, dict] = {**_CANONICAL_CHARS, **_ADDITIONAL_CHARS}


def get_character_set(name: str) -> list[str]:
    if name == "canonical_2":
        return list(CANONICAL_2)
    if name == "canonical_4":
        return list(CANONICAL_4)
    if name == "extended_8":
        return list(EXTENDED_8)
    raise ValueError(
        f"Unknown character_set '{name}'. "
        f"Available: canonical_2, canonical_4, extended_8"
    )


# ---------------------------------------------------------------------------
# Persona rendering
# ---------------------------------------------------------------------------


def get_memories(name: str) -> list[str]:
    if name not in _ALL_CHARS:
        raise KeyError(f"No persona memories for {name!r}")
    return list(_ALL_CHARS[name]["description"])


def get_persona_text(name: str) -> str:
    """Render a single character's full persona block as system-prompt text."""
    mems = get_memories(name)
    bullets = "\n".join(f"- {m}" for m in mems)
    return _SYSTEM_PROMPT_TEMPLATE.format(
        character_name=name,
        character_description_bullets=bullets,
    )


def get_personas(character_set: str) -> dict[str, str]:
    """Return name → persona-text dict for an entire character set."""
    return {name: get_persona_text(name) for name in get_character_set(character_set)}
