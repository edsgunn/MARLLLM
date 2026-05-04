"""
Forum environment package.

Public API:

  * :class:`ForumEnv` — append-only multi-agent forum env (PettingZoo AEC).
  * :class:`JsonPersonaScenario` — loader for JSON-described scenarios at
    ``envs/forum/scenarios/<name>.json``.
  * :func:`load_scenario` — convenience wrapper around the loader.

Each scenario JSON describes a forum: its description, default invitation,
character rosters, persona descriptions, and the system-prompt template
that ``ForumEnv`` installs as each agent's system message.
"""
from envs.forum.env import ForumEnv
from envs.forum.personas import JsonPersonaScenario, load_scenario

__all__ = ["ForumEnv", "JsonPersonaScenario", "load_scenario"]
