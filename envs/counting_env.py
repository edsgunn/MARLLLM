"""
CountingEnv: a PettingZoo AEC single-agent environment for end-to-end testing.

Task: the agent must count upward from 1 (produce tokens "1", "2", "3", ...).
The environment echoes back each correct number as an observation, confirming
the count. Wrong actions return a high-surprise error observation ("!") and
end the episode.

Why error token instead of empty obs: an empty obs on wrong action gives
G_t = 0 at the mistake, making A_t = -(0 - V_t) which is positive when V_t > 0.
The error token guarantees G_t_wrong > G_t_correct and produces a proper
negative advantage from the first iteration.

PettingZoo AEC contract: agents is a list, agent_selection is the current
agent, observe() returns the current observation dict, step() advances.

Custom attribute: action_token_budget = 1 (one token per turn).

Vocabulary check: __init__ verifies that each integer 1..max_count
tokenises to a single token. Raises ValueError otherwise.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from transformers import PreTrainedTokenizerBase


class CountingEnv(AECEnv):
    """
    Single-agent PettingZoo AEC counting environment.

    Observation space: a list of token IDs (variable length, 0–1 tokens).
    Action space: a list of token IDs (always 1 token).

    The environment does not use gym spaces since token IDs are integers
    in the model's vocabulary; the Tokeniser handles encoding/decoding.
    """

    metadata = {"render_modes": [], "name": "counting_v0"}

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_count: int = 20,
    ) -> None:
        super().__init__()

        self._tok = tokenizer
        self.max_count = max_count
        self.action_token_budget = 1  # read by Trainer

        # Verify each integer 0..max_count tokenises to exactly one token.
        # 0 is the seed observation emitted at episode start so the model
        # always has a concrete predecessor to increment from.
        self._number_token_ids: dict[int, int] = {}
        for n in range(0, max_count + 1):
            ids = tokenizer.encode(str(n), add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError(
                    f"Integer {n!r} encodes to {len(ids)} tokens with this tokenizer "
                    f"(expected 1). Use a smaller max_count or a different tokenizer."
                )
            self._number_token_ids[n] = ids[0]

        # Error token: "!" — used on wrong action
        error_ids = tokenizer.encode("!", add_special_tokens=False)
        self._error_token_id: int = error_ids[0]

        self.possible_agents = ["agent_0"]
        self.agents: list[str] = []
        self._agent_selector: agent_selector | None = None

        # Episode state
        self._expected_next: int = 1
        self._pending_obs: dict[str, list[int]] = {}
        self._cumulative_rewards: dict[str, float] = {}
        self._terminations: dict[str, bool] = {}
        self._truncations: dict[str, bool] = {}
        self._infos: dict[str, dict] = {}
        self._step_count: int = 0

    # ------------------------------------------------------------------ #
    # PettingZoo AEC API                                                   #
    # ------------------------------------------------------------------ #

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> None:
        self.agents = list(self.possible_agents)
        self._agent_selector = agent_selector.AgentSelector(self.agents)
        self.agent_selection = self._agent_selector.next()

        self._expected_next = 1
        self._step_count = 0

        # Seed observation: emit "0" so the model always has a concrete
        # predecessor to increment from, rather than inferring the start
        # from the prompt alone.
        self._pending_obs = {"agent_0": [self._number_token_ids[0]]}
        self._cumulative_rewards = {"agent_0": 0.0}
        self._terminations = {"agent_0": False}
        self._truncations = {"agent_0": False}
        self._infos = {"agent_0": {"episode_length": 0, "correct_count": 0}}

    def observe(self, agent: str) -> list[int]:
        return list(self._pending_obs.get(agent, []))

    def step(self, action: Any) -> None:
        """
        action: list[int] of token IDs (length 1) OR None for terminated agents.
        """
        agent = self.agent_selection

        if self._terminations[agent] or self._truncations[agent]:
            self._was_dead_step(action)
            return

        self._step_count += 1
        self._infos[agent]["episode_length"] = self._step_count

        if action is None:
            # Treat as wrong action
            self._pending_obs[agent] = [self._error_token_id]
            self._terminations[agent] = True
            self._infos[agent]["result"] = "none_action"
        else:
            action_ids = list(action)
            if not action_ids:
                self._pending_obs[agent] = [self._error_token_id]
                self._terminations[agent] = True
                self._infos[agent]["result"] = "empty_action"
            else:
                produced = action_ids[0]
                expected_id = self._number_token_ids[self._expected_next]

                if produced == expected_id:
                    # Correct: echo the number back as obs, advance counter
                    self._pending_obs[agent] = [expected_id]
                    self._infos[agent]["correct_count"] = self._infos[agent].get("correct_count", 0) + 1

                    if self._expected_next >= self.max_count:
                        self._terminations[agent] = True
                        self._infos[agent]["result"] = "success"
                    else:
                        self._expected_next += 1
                        self._infos[agent]["result"] = "correct"
                else:
                    # Wrong: return error token and terminate
                    self._pending_obs[agent] = [self._error_token_id]
                    self._terminations[agent] = True
                    self._infos[agent]["result"] = "wrong"
                    self._infos[agent]["expected_token"] = expected_id
                    self._infos[agent]["got_token"] = produced

        self._cumulative_rewards[agent] = 0.0

        # Advance agent selection only. Do NOT clear self.agents here.
        # PettingZoo's agent_iter() will yield the agent one final time so the
        # Trainer can observe the terminal observation (error token or final echo).
        # The agent is removed from self.agents by _was_dead_step() when the
        # Trainer calls env.step(None) for the terminated agent.
        self.agent_selection = self._agent_selector.next()

    # ------------------------------------------------------------------ #
    # PettingZoo required properties                                       #
    # ------------------------------------------------------------------ #

    @property
    def terminations(self) -> dict[str, bool]:
        return dict(self._terminations)

    @property
    def truncations(self) -> dict[str, bool]:
        return dict(self._truncations)

    @property
    def rewards(self) -> dict[str, float]:
        return dict(self._cumulative_rewards)

    @property
    def infos(self) -> dict[str, dict]:
        return dict(self._infos)

    def observation_space(self, agent: str):
        return None  # token IDs; no gym space needed

    def action_space(self, agent: str):
        return None  # token IDs; no gym space needed

    def render(self) -> None:
        pass

    def close(self) -> None:
        pass

    def _was_dead_step(self, action: Any) -> None:
        """
        Handle step() call for an already-terminated agent (PettingZoo convention).
        Removes the dead agent from self.agents and advances agent_selection.
        """
        if action is not None:
            raise ValueError("only None is valid for a terminated agent")
        agent = self.agent_selection
        self.agents.remove(agent)
        self._cumulative_rewards[agent] = 0.0
        if self.agents:
            self.agent_selection = self._agent_selector.next()
