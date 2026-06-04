"""
Social terminal environment package.

A place-based multi-agent text substrate for population training under
surprise minimisation. Agents inhabit a graph of persistent places,
perceive only their current place, and change the world through tools.

Public API:

  * :class:`SocialTerminalEnv` — PettingZoo AEC environment.
  * :class:`Scenario` — world + roster loaded from JSON.
  * :func:`load_scenario` — load a scenario by name.
"""
from envs.social_terminal.env import SocialTerminalEnv
from envs.social_terminal.scenario import Scenario, load_scenario

__all__ = ["SocialTerminalEnv", "Scenario", "load_scenario"]
