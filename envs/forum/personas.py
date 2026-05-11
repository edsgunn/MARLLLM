"""
Loaders for ForumEnv character packs and environment prompts.

Characters and environments are stored as separate JSON files so any
character pack can be dropped into any environment.

A character pack at ``envs/forum/characters/<pack>.json``::

    {
      "name": "...",
      "character_sets": {
        "canonical_2": ["Name A", "Name B"],
        "canonical_4": [...],
        "extended_8": [...]
      },
      "characters": {
        "Name A": {"description": ["...", "..."]},
        ...
      }
    }

An environment prompt at ``envs/forum/environments/<env>.json``::

    {
      "name": "...",
      "description": "...",                  # forum description in initial ctx
      "default_invitation": "...",           # first-thread invitation
      "system_prompt_template": "You are {character_name}, ... {character_description_bullets} ..."
    }

Compose them with :func:`build_personas` (or :meth:`render_persona`) to
get the per-agent system-prompt text the trainer installs.
"""
from __future__ import annotations

import json
from pathlib import Path


_CHARACTERS_DIR = Path(__file__).parent / "characters"
_ENVIRONMENTS_DIR = Path(__file__).parent / "environments"


class CharacterPack:
    """A roster of characters plus named subsets (e.g. ``canonical_2``)."""

    def __init__(self, pack_name: str, *, json_path: Path | None = None) -> None:
        self.pack_name = pack_name
        self.json_path = json_path or (_CHARACTERS_DIR / f"{pack_name}.json")
        if not self.json_path.exists():
            raise FileNotFoundError(
                f"Character pack JSON not found for {pack_name!r}: {self.json_path}"
            )
        with self.json_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)

        self._characters: dict[str, dict] = dict(data.get("characters", {}))
        self._character_sets: dict[str, list[str]] = dict(data.get("character_sets", {}))
        self._character_sets.setdefault("all", list(self._characters.keys()))

    def get_character_set(self, name: str) -> list[str]:
        if name not in self._character_sets:
            raise ValueError(
                f"Unknown character_set {name!r} in pack {self.pack_name!r}. "
                f"Available: {sorted(self._character_sets)}"
            )
        return list(self._character_sets[name])

    def get_memories(self, character_name: str) -> list[str]:
        if character_name not in self._characters:
            raise KeyError(
                f"No character {character_name!r} in pack {self.pack_name!r}."
            )
        return list(self._characters[character_name]["description"])


class EnvironmentPrompt:
    """Forum description, invitation, and persona-template scaffold."""

    def __init__(self, env_name: str, *, json_path: Path | None = None) -> None:
        self.env_name = env_name
        self.json_path = json_path or (_ENVIRONMENTS_DIR / f"{env_name}.json")
        if not self.json_path.exists():
            raise FileNotFoundError(
                f"Environment prompt JSON not found for {env_name!r}: {self.json_path}"
            )
        with self.json_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)

        self.forum_description: str = data.get("description", "")
        self.default_invitation: str = data.get(
            "default_invitation",
            "A new thread has just opened. The forum is active and members are posting.",
        )
        self.system_prompt_template: str = data["system_prompt_template"]

    def render_persona(self, character_name: str, memories: list[str]) -> str:
        bullets = "\n".join(f"- {m}" for m in memories)
        return self.system_prompt_template.format(
            character_name=character_name,
            character_description_bullets=bullets,
        )


def load_characters(pack_name: str) -> CharacterPack:
    return CharacterPack(pack_name)


def load_environment(env_name: str) -> EnvironmentPrompt:
    return EnvironmentPrompt(env_name)


def build_personas(
    environment: EnvironmentPrompt,
    characters: CharacterPack,
    character_set: str,
) -> dict[str, str]:
    """Compose ``{character_name → system-prompt text}`` for one set."""
    return {
        name: environment.render_persona(name, characters.get_memories(name))
        for name in characters.get_character_set(character_set)
    }
