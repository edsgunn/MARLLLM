"""
Scenario loader for SocialTerminalEnv.

A *scenario* is a self-contained spec for one run: the place graph, the
fixture objects in each place, the roster of agents, their personas, and
their starting positions. Scenarios live as JSON under
``envs/social_terminal/scenarios/<name>.json``.

Persona descriptions are kept separately under ``characters/<pack>.json``
so the same character pack can be dropped into different scenarios — same
split as ``envs/forum``.

Schema (scenarios/<name>.json)::

    {
      "name": "...",
      "description": "<one-paragraph framing shown in the initial ctx>",
      "character_pack": "ashbourne_school",
      "places": {
        "<place_id>": {
          "title": "Maths Classroom",
          "description": "...",
          "exits": {"<exit_label>": "<dest_place_id>", ...},
          "objects": {
            "<object_id>": {
              "title": "whiteboard",
              "description": "A large whiteboard on the back wall.",
              "content": "",
              "editable": true,
              "append_only": true
            }
          }
        }
      },
      "agents": [
        {"name": "Priya Shah", "short": "Priya", "start": "<place_id>"}
      ],
      "system_prompt_template": "You are {character_name}, ..."
    }
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


_SCENARIOS_DIR = Path(__file__).parent / "scenarios"
# Character packs are searched in order: this env's own characters/
# dir first, then the forum env's characters/ dir as a fallback. This
# lets a social_terminal scenario reuse personas already authored for
# the forum env (same school, same agents, different substrate).
_CHARACTER_DIRS = [
    Path(__file__).parent / "characters",
    Path(__file__).resolve().parents[1] / "forum" / "characters",
]


@dataclass
class PlaceObject:
    obj_id: str
    title: str
    description: str
    content: str = ""
    editable: bool = False
    append_only: bool = True


@dataclass
class Place:
    place_id: str
    title: str
    description: str
    exits: dict[str, str] = field(default_factory=dict)
    objects: dict[str, PlaceObject] = field(default_factory=dict)


@dataclass
class AgentSpec:
    name: str
    short: str
    start: str


@dataclass
class Scenario:
    name: str
    description: str
    places: dict[str, Place]
    agents: list[AgentSpec]
    personas: dict[str, str]  # agent name -> system prompt text

    def place_ids(self) -> list[str]:
        return list(self.places.keys())

    def agent_names(self) -> list[str]:
        return [a.name for a in self.agents]


def _load_character_pack(pack_name: str) -> dict[str, dict[str, Any]]:
    for d in _CHARACTER_DIRS:
        path = d / f"{pack_name}.json"
        if path.exists():
            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            return dict(data.get("characters", {}))
    searched = "\n  ".join(str(d) for d in _CHARACTER_DIRS)
    raise FileNotFoundError(
        f"Character pack {pack_name!r} not found. Searched:\n  {searched}"
    )


def load_scenario(name: str, *, json_path: Path | None = None) -> Scenario:
    path = json_path or (_SCENARIOS_DIR / f"{name}.json")
    if not path.exists():
        raise FileNotFoundError(f"Scenario JSON not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    places: dict[str, Place] = {}
    for pid, pdata in data["places"].items():
        objects = {}
        for oid, odata in pdata.get("objects", {}).items():
            objects[oid] = PlaceObject(
                obj_id=oid,
                title=odata.get("title", oid),
                description=odata.get("description", ""),
                content=odata.get("content", ""),
                editable=bool(odata.get("editable", False)),
                append_only=bool(odata.get("append_only", True)),
            )
        places[pid] = Place(
            place_id=pid,
            title=pdata.get("title", pid),
            description=pdata["description"],
            exits=dict(pdata.get("exits", {})),
            objects=objects,
        )

    # Validate exits.
    for p in places.values():
        for exit_label, dest in p.exits.items():
            if dest not in places:
                raise ValueError(
                    f"Place {p.place_id!r} has exit {exit_label!r} → "
                    f"unknown destination {dest!r}."
                )

    agents = [
        AgentSpec(name=a["name"], short=a.get("short", a["name"].split()[0]),
                  start=a["start"])
        for a in data["agents"]
    ]
    for a in agents:
        if a.start not in places:
            raise ValueError(
                f"Agent {a.name!r} starts in unknown place {a.start!r}."
            )

    # Build personas.
    pack_name = data["character_pack"]
    characters = _load_character_pack(pack_name)
    tpl = data["system_prompt_template"]
    personas: dict[str, str] = {}
    for a in agents:
        if a.name not in characters:
            raise KeyError(
                f"Character {a.name!r} not found in pack {pack_name!r}."
            )
        desc = characters[a.name]["description"]
        bullets = "\n".join(f"- {line}" for line in desc)
        personas[a.name] = tpl.format(
            character_name=a.name,
            character_description_bullets=bullets,
            scenario_description=data["description"],
        )

    return Scenario(
        name=data["name"],
        description=data["description"],
        places=places,
        agents=agents,
        personas=personas,
    )
