"""
Forum environment package.

Public API:

  * :class:`ForumEnv` — append-only multi-agent forum env (PettingZoo AEC).
  * :class:`CharacterPack` — roster of characters plus named subsets,
    loaded from ``envs/forum/characters/<pack>.json``.
  * :class:`EnvironmentPrompt` — forum description, invitation, and
    persona-template scaffold, loaded from
    ``envs/forum/environments/<env>.json``.
  * :func:`load_characters`, :func:`load_environment`, :func:`build_personas`
    — convenience wrappers and composition helper.

Characters and environments are stored separately so any character pack
can be dropped into any environment.
"""
from envs.forum.env import ForumEnv
from envs.forum.personas import (
    CharacterPack,
    EnvironmentPrompt,
    build_personas,
    load_characters,
    load_environment,
)

__all__ = [
    "ForumEnv",
    "CharacterPack",
    "EnvironmentPrompt",
    "build_personas",
    "load_characters",
    "load_environment",
]
