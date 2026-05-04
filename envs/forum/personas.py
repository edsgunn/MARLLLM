"""
Generic JSON-backed persona loader for ForumEnv scenarios.

A scenario JSON file lives at ``envs/forum/scenarios/<name>.json`` and
follows this shape::

    {
      "environment": {
        "name": "...",
        "description": "..."        # forum description shown in initial ctx
      },
      "default_invitation": "...",  # optional first-thread invitation
      "character_sets": {            # optional explicit subsets
        "canonical_2": ["Name A", "Name B"]
      },
      "characters": {
        "canonical":  {"Name A": {"description": ["...", "..."]}},
        "additional": {"Name X": {"description": ["...", "..."]}}
      },
      "system_prompt_template": "You are {character_name}, ... {character_description_bullets} ..."
    }

The :class:`JsonPersonaScenario` wrapper exposes ``forum_description``,
``default_invitation``, ``get_character_set``, ``get_personas``,
``get_memories`` and ``get_persona_text``.
"""
from __future__ import annotations

import json
from pathlib import Path


_SCENARIO_DIR = Path(__file__).parent / "scenarios"


class JsonPersonaScenario:
    """Loader for a forum scenario described entirely by one JSON file."""

    def __init__(self, scenario_name: str, *, json_path: Path | None = None) -> None:
        self.scenario_name = scenario_name
        self.json_path = json_path or (_SCENARIO_DIR / f"{scenario_name}.json")
        if not self.json_path.exists():
            raise FileNotFoundError(
                f"Scenario JSON not found for {scenario_name!r}: {self.json_path}"
            )
        with self.json_path.open("r", encoding="utf-8") as fh:
            self._data = json.load(fh)

        env_block = self._data.get("environment", {})
        self.forum_description: str = env_block.get("description", "")
        self.default_invitation: str = self._data.get(
            "default_invitation",
            "A new thread has just opened. The forum is active and members are posting.",
        )
        self.system_prompt_template: str = self._data["system_prompt_template"]

        chars = self._data.get("characters", {})
        self._canonical: dict[str, dict] = dict(chars.get("canonical", {}))
        self._additional: dict[str, dict] = dict(chars.get("additional", {}))
        self._all: dict[str, dict] = {**self._canonical, **self._additional}

        sets = dict(self._data.get("character_sets", {}))
        # Defaults: canonical_4 = all canonical names; extended_8 = canonical+additional.
        sets.setdefault("canonical_4", list(self._canonical.keys()))
        sets.setdefault("extended_8", list(self._all.keys()))
        # canonical_2 has no sensible default — the JSON must specify it
        # (it is the most-opposed canonical pair, picked deliberately).
        self._character_sets: dict[str, list[str]] = sets

    # ----- character set lookup -------------------------------------------- #

    def get_character_set(self, name: str) -> list[str]:
        if name not in self._character_sets:
            raise ValueError(
                f"Unknown character_set {name!r} for scenario "
                f"{self.scenario_name!r}. Available: "
                f"{sorted(self._character_sets)}"
            )
        return list(self._character_sets[name])

    # ----- persona rendering ----------------------------------------------- #

    def get_memories(self, character_name: str) -> list[str]:
        if character_name not in self._all:
            raise KeyError(
                f"No persona memories for {character_name!r} in scenario "
                f"{self.scenario_name!r}."
            )
        return list(self._all[character_name]["description"])

    def get_persona_text(self, character_name: str) -> str:
        bullets = "\n".join(f"- {m}" for m in self.get_memories(character_name))
        return self.system_prompt_template.format(
            character_name=character_name,
            character_description_bullets=bullets,
        )

    def get_personas(self, character_set: str) -> dict[str, str]:
        return {
            name: self.get_persona_text(name)
            for name in self.get_character_set(character_set)
        }


def load_scenario(scenario_name: str) -> JsonPersonaScenario:
    """Load a scenario by name (matches ``envs/<scenario_name>.json``)."""
    return JsonPersonaScenario(scenario_name)
