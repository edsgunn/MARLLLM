"""
Deal or No Deal negotiation environment.

Two agents negotiate over three item types (books, hats, balls) with private
valuations, following Lewis et al. (2017) arXiv:1706.05125.

Episode structure per agent
---------------------------
1. Context obs  — private item counts + values (first observation).
2. Dialogue     — agents alternate; each agent's utterance becomes the next
                  obs for the OTHER agent.  Up to max_dialogue_turns total.
3. Selection    — both agents receive a prompt and output their proposed
                  allocation (parsed for three integers: books=N hats=N balls=N).
4. Outcome      — both agents receive an outcome text obs, then terminate.

Scoring
-------
  score = sum(items_received[i] * values[i])
  No deal (parse failure or allocations don't sum to totals): both score 0.

Shared-context note
-------------------
The Trainer adds every obs to every agent's context window.  This means each
agent sees the other's private context obs.  This is a known limitation of the
current trainer design; strict information asymmetry would require a modified
Trainer.  The negotiation remains non-trivial because the agents have different
valuations and must produce utterances the other can respond to coherently.

Observation sequencing
----------------------
agent_0 acts first (selector is initialised to agent_0 at reset).

  Turn 1  agent_0 obs → ctx_0            agent_0 acts → utterance A
  Turn 2  agent_1 obs → ctx_1 + A        agent_1 acts → utterance B
  Turn 3  agent_0 obs → B                agent_0 acts → utterance A2
  Turn 4  agent_1 obs → A2               agent_1 acts → utterance B2
  …
  Turn N  (selection prompt for both)
  Turn N+1  one agent outputs allocation
  Turn N+2  other agent outputs allocation → resolution
  Turn N+3  outcome obs for both, then terminated
"""
from __future__ import annotations

import re
import random
from typing import Any, Literal

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from transformers import PreTrainedTokenizerBase


ITEM_NAMES: tuple[str, ...] = ("books", "hats", "balls")
N_ITEMS: int = len(ITEM_NAMES)
MAX_VALUE_SUM: int = 10


class DealOrNoDealEnv(AECEnv):
    """Two-agent deal-or-no-deal negotiation environment (PettingZoo AEC).

    Parameters
    ----------
    tokenizer:
        HuggingFace tokenizer used to encode observations and decode actions.
    max_dialogue_turns:
        Total dialogue turns across both agents (default 10 = 5 each).
        After this many turns the selection phase begins.
    action_token_budget:
        Tokens each agent may generate per turn.  Read by the Trainer.
    max_item_count:
        Maximum units of any one item type per episode (inclusive).
    seed:
        Optional RNG seed for reproducible scenarios.
    """

    metadata = {"render_modes": [], "name": "deal_or_no_deal_v0"}

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_dialogue_turns: int = 10,
        action_token_budget: int = 64,
        max_item_count: int = 5,
        seed: int | None = None,
    ) -> None:
        super().__init__()

        self._tok = tokenizer
        self._max_dialogue_turns = max_dialogue_turns
        self.action_token_budget = action_token_budget  # read by Trainer
        self._max_item_count = max_item_count
        self._rng = random.Random(seed)

        self.possible_agents = ["agent_0", "agent_1"]

        # Episode state — populated in reset()
        self.agents: list[str] = []
        self._agent_selector: agent_selector.AgentSelector | None = None

        self._items: list[int] = []
        self._values: dict[str, list[int]] = {}
        self._pending_obs: dict[str, list[int]] = {}
        self._context_delivered: dict[str, bool] = {}
        self._utterances: dict[str, list[list[int]]] = {}
        self._allocations: dict[str, list[int] | None] = {}
        self._selection_submitted: set[str] = set()
        self._dialogue_turn: int = 0
        self._phase: Literal["dialogue", "selection", "done"] = "dialogue"

        self._deal_valid: bool = False
        self._episode_scores: dict[str, int] = {}

        self._cumulative_rewards: dict[str, float] = {}
        self._terminations: dict[str, bool] = {}
        self._truncations: dict[str, bool] = {}
        self._infos: dict[str, dict] = {}

    # ------------------------------------------------------------------ #
    # Scenario helpers                                                     #
    # ------------------------------------------------------------------ #

    def _generate_values(self) -> list[int]:
        """Random non-negative integers for N_ITEMS items summing to MAX_VALUE_SUM."""
        while True:
            vals = [0] * N_ITEMS
            remaining = MAX_VALUE_SUM
            for i in range(N_ITEMS - 1):
                v = self._rng.randint(0, remaining)
                vals[i] = v
                remaining -= v
            vals[-1] = remaining
            self._rng.shuffle(vals)
            if any(v > 0 for v in vals):
                return vals

    # ------------------------------------------------------------------ #
    # Observation encoding                                                 #
    # ------------------------------------------------------------------ #

    def _enc(self, text: str) -> list[int]:
        return self._tok.encode(text, add_special_tokens=False)

    def _context_obs(self, agent: str) -> list[int]:
        c = self._items
        v = self._values[agent]
        items_str = ", ".join(f"{c[i]} {ITEM_NAMES[i]}" for i in range(N_ITEMS))
        vals_str  = ", ".join(f"{ITEM_NAMES[i]}={v[i]}" for i in range(N_ITEMS))
        return self._enc(
            f"Negotiation. Items: {items_str}. "
            f"Your values: {vals_str}. "
            f"Max score: {MAX_VALUE_SUM}. "
            f"Agree on how to split the items."
        )

    def _selection_prompt_obs(self) -> list[int]:
        c = self._items
        avail = ", ".join(f"{c[i]} {ITEM_NAMES[i]}" for i in range(N_ITEMS))
        return self._enc(
            f"State your proposed share (items YOU want to keep). "
            f"Available: {avail}. "
            f"Reply: books=N hats=N balls=N"
        )

    def _outcome_obs(self, agent: str) -> list[int]:
        score       = self._episode_scores[agent]
        other       = "agent_1" if agent == "agent_0" else "agent_0"
        other_score = self._episode_scores[other]
        alloc       = self._allocations.get(agent)

        if self._deal_valid and alloc is not None:
            alloc_str = ", ".join(f"{alloc[i]} {ITEM_NAMES[i]}" for i in range(N_ITEMS))
            return self._enc(
                f"Deal. You receive: {alloc_str}. "
                f"Your score: {score}/{MAX_VALUE_SUM}. "
                f"Other score: {other_score}/{MAX_VALUE_SUM}."
            )
        return self._enc(
            f"No deal. "
            f"Your score: 0/{MAX_VALUE_SUM}. "
            f"Other score: 0/{MAX_VALUE_SUM}."
        )

    # ------------------------------------------------------------------ #
    # Observation routing                                                  #
    # ------------------------------------------------------------------ #

    def _deliver_obs(self, agent: str, obs: list[int]) -> None:
        """Queue obs for agent, prepending private context if not yet delivered."""
        if not self._context_delivered[agent]:
            self._pending_obs[agent] = self._context_obs(agent) + obs
            self._context_delivered[agent] = True
        else:
            self._pending_obs[agent] = obs

    # ------------------------------------------------------------------ #
    # Allocation parsing                                                   #
    # ------------------------------------------------------------------ #

    def _parse_allocation(self, token_ids: list[int]) -> list[int] | None:
        """Decode token IDs and extract [books, hats, balls].  Returns None on failure."""
        if not token_ids:
            return None
        text = self._tok.decode(token_ids, skip_special_tokens=True)
        m = re.search(
            r"books\s*=\s*(\d+).*?hats\s*=\s*(\d+).*?balls\s*=\s*(\d+)",
            text, re.IGNORECASE | re.DOTALL,
        )
        if m:
            return [int(m.group(1)), int(m.group(2)), int(m.group(3))]
        # Fallback: first three integers found
        nums = re.findall(r"\d+", text)
        if len(nums) >= 3:
            return [int(nums[0]), int(nums[1]), int(nums[2])]
        return None

    # ------------------------------------------------------------------ #
    # Resolution                                                           #
    # ------------------------------------------------------------------ #

    def _resolve(self) -> None:
        a0 = self._allocations["agent_0"]
        a1 = self._allocations["agent_1"]

        self._deal_valid = (
            a0 is not None
            and a1 is not None
            and all(
                a0[i] >= 0 and a1[i] >= 0 and a0[i] + a1[i] == self._items[i]
                for i in range(N_ITEMS)
            )
        )

        if self._deal_valid:
            self._episode_scores = {
                "agent_0": sum(a0[i] * self._values["agent_0"][i] for i in range(N_ITEMS)),
                "agent_1": sum(a1[i] * self._values["agent_1"][i] for i in range(N_ITEMS)),
            }
        else:
            self._episode_scores = {"agent_0": 0, "agent_1": 0}

    # ------------------------------------------------------------------ #
    # PettingZoo AEC interface                                             #
    # ------------------------------------------------------------------ #

    def reset(self, seed: int | None = None, options: dict | None = None) -> None:
        if seed is not None:
            self._rng = random.Random(seed)

        self._items = [self._rng.randint(1, self._max_item_count) for _ in range(N_ITEMS)]
        self._values = {
            "agent_0": self._generate_values(),
            "agent_1": self._generate_values(),
        }

        self.agents = list(self.possible_agents)
        self._agent_selector = agent_selector.AgentSelector(self.agents)
        self.agent_selection = self._agent_selector.next()

        self._dialogue_turn = 0
        self._phase = "dialogue"
        self._utterances = {"agent_0": [], "agent_1": []}
        self._allocations = {"agent_0": None, "agent_1": None}
        self._selection_submitted = set()
        self._deal_valid = False
        self._episode_scores = {"agent_0": 0, "agent_1": 0}

        # agent_0 acts first; their context is pre-loaded.
        # agent_1's context will be prepended by _deliver_obs on their first obs.
        self._context_delivered = {"agent_0": True, "agent_1": False}
        self._pending_obs = {
            "agent_0": self._context_obs("agent_0"),
            "agent_1": [],
        }

        self._cumulative_rewards = {"agent_0": 0.0, "agent_1": 0.0}
        self._terminations = {"agent_0": False, "agent_1": False}
        self._truncations  = {"agent_0": False, "agent_1": False}
        self._infos = {
            a: {
                "phase":        "dialogue",
                "dialogue_turn": 0,
                "outcome":      "pending",
                "score":        0,
                "items":        list(self._items),
                "values":       list(self._values[a]),
            }
            for a in self.agents
        }

    def observe(self, agent: str) -> list[int]:
        return list(self._pending_obs.get(agent, []))

    def step(self, action: Any) -> None:
        agent = self.agent_selection

        if self._terminations[agent] or self._truncations[agent]:
            self._was_dead_step(action)
            return

        if action is None:
            action = []
        action = list(action)

        other = "agent_1" if agent == "agent_0" else "agent_0"

        # ---- dialogue phase ----
        if self._phase == "dialogue":
            self._utterances[agent].append(action)
            self._dialogue_turn += 1
            self._pending_obs[agent] = []  # current agent's obs consumed

            # Route this utterance to the other agent as their next obs.
            # _deliver_obs prepends ctx_1 if agent_1 hasn't seen their context yet.
            self._deliver_obs(other, list(action))

            for a in self.agents:
                self._infos[a]["dialogue_turn"] = self._dialogue_turn

            if self._dialogue_turn >= self._max_dialogue_turns:
                self._phase = "selection"
                prompt = self._selection_prompt_obs()
                self._pending_obs["agent_0"] = list(prompt)
                self._pending_obs["agent_1"] = list(prompt)
                for a in self.agents:
                    self._infos[a]["phase"] = "selection"

        # ---- selection phase ----
        elif self._phase == "selection":
            self._allocations[agent] = self._parse_allocation(action)
            self._selection_submitted.add(agent)
            self._pending_obs[agent] = []

            if len(self._selection_submitted) >= len(self.possible_agents):
                self._resolve()
                for a in self.possible_agents:
                    self._pending_obs[a]       = self._outcome_obs(a)
                    self._terminations[a]      = True
                    self._cumulative_rewards[a] = float(self._episode_scores[a])
                    self._infos[a]["phase"]    = "done"
                    self._infos[a]["outcome"]  = "deal" if self._deal_valid else "no_deal"
                    self._infos[a]["score"]    = self._episode_scores[a]
                self._phase = "done"

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
        """Handle step() for an already-terminated agent (PettingZoo convention)."""
        if action is not None:
            raise ValueError("Only None is valid for a terminated agent.")
        agent = self.agent_selection
        self.agents.remove(agent)
        self._cumulative_rewards[agent] = 0.0
        if self.agents:
            self.agent_selection = self._agent_selector.next()
