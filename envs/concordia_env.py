"""
PettingZoo AEC wrapper for Concordia multi-agent simulation.

Architecture
------------
Each trained agent is a ``MARLLLMLanguageModel`` that implements Concordia's
``LanguageModel`` interface.  The scenario factory receives these LM instances
and is responsible for constructing Concordia ``EntityAgent`` objects from them.
``make_player_entity(name, model)`` provides a minimal EntityAgent with an
observation-buffer context component — sufficient for most scenarios.  Factories
that need richer agents (memory, planning, etc.) can construct their own
``EntityAgent`` instances using the same LM objects.

Thread model
~~~~~~~~~~~~
  Simulation thread: engine.run_loop() → ... → ConcatActComponent
                      → model.sample_text(assembled_prompt) → blocks on act_queue
  Main thread:       drives AEC (reset / step) ↔ reads obs_queue (receives
                     tokenised assembled prompt), writes decoded action text
                     to act_queue

obs_is_full_context
~~~~~~~~~~~~~~~~~~~
Each observation the PettingZoo/Trainer layer receives is the fully assembled
prompt that Concordia passes to ``model.sample_text()`` — the complete context
window the model would generate from.  The env sets ``obs_is_full_context = True``
to signal this; the Trainer replaces rather than appends-and-wraps the context
buffer for each step, avoiding double chat-template formatting.

Deep-copy safety
~~~~~~~~~~~~~~~~
Thread/queue state is created only inside reset() so the template object is
safely copyable by the Trainer's parallel episode collection.

Factory interface
~~~~~~~~~~~~~~~~~
    def my_factory(
        models: dict[str, MARLLLMLanguageModel],
        seed: int | None,
    ) -> sim_with_play_method:
        entities = {
            name: make_player_entity(name, model)
            for name, model in models.items()
        }
        ...  # build Concordia simulation using entities
        return sim

Usage
~~~~~
    from envs.concordia_env import ConcordiaEnv, MARLLLMLanguageModel, make_player_entity

    env = ConcordiaEnv(
        simulation_factory=my_factory,
        agent_names=["Alice", "Bob"],
        tokenizer=tokenizer,
        action_token_budget=128,
        reward_fn=lambda state, agent: state.get(agent, {}).get("score", 0.0),
    )
"""
from __future__ import annotations

import queue
import threading
from collections.abc import Collection, Mapping, Sequence
from typing import Any, Callable

from concordia.language_model import language_model as lm_lib
from pettingzoo import AECEnv
from transformers import PreTrainedTokenizerBase

# ---------------------------------------------------------------------------
# Internal sentinels
# ---------------------------------------------------------------------------

_STOP = object()


class _Done:
    """Posted to obs_queue when the Concordia simulation finishes."""
    __slots__ = ("final_state",)

    def __init__(self, final_state: Any) -> None:
        self.final_state = final_state


class _SimulationStopped(BaseException):
    """Raised inside MARLLLMLanguageModel.sample_text() to unwind the sim thread."""


# ---------------------------------------------------------------------------
# MARLLLMLanguageModel
# ---------------------------------------------------------------------------


class MARLLLMLanguageModel(lm_lib.LanguageModel):
    """Concordia LanguageModel that bridges into the PettingZoo main thread.

    ConcatActComponent calls ``sample_text(assembled_prompt)`` once per agent
    turn.  ``assembled_prompt`` is the full context the model generates from
    (character instructions + buffered observations + call-to-action).  We
    tokenise it, post the token IDs to ``obs_queue``, and block until the main
    thread puts decoded action text into ``act_queue``.

    ``sample_choice`` follows the same pattern and resolves the returned free
    text to the closest matching option.
    """

    def __init__(self, agent_name: str, tokenizer: PreTrainedTokenizerBase) -> None:
        self._name = agent_name
        self._tok = tokenizer
        self._obs_q: queue.Queue | None = None
        self._act_q: queue.Queue | None = None

    @property
    def name(self) -> str:
        return self._name

    def _connect(self, obs_q: queue.Queue, act_q: queue.Queue) -> None:
        """Called by ConcordiaEnv.reset() before starting the sim thread."""
        self._obs_q = obs_q
        self._act_q = act_q

    def _post_and_wait(self, prompt: str) -> str:
        obs_ids = self._tok.encode(prompt, add_special_tokens=False)
        self._obs_q.put((self._name, obs_ids))
        action = self._act_q.get()
        if action is _STOP:
            raise _SimulationStopped()
        return str(action)

    # -- LanguageModel interface ---------------------------------------------

    def sample_text(
        self,
        prompt: str,
        *,
        max_tokens: int = lm_lib.DEFAULT_MAX_TOKENS,
        terminators: Collection[str] = lm_lib.DEFAULT_TERMINATORS,
        temperature: float = lm_lib.DEFAULT_TEMPERATURE,
        top_p: float = lm_lib.DEFAULT_TOP_P,
        top_k: int = lm_lib.DEFAULT_TOP_K,
        timeout: float = lm_lib.DEFAULT_TIMEOUT_SECONDS,
        seed: int | None = None,
    ) -> str:
        return self._post_and_wait(prompt)

    def sample_choice(
        self,
        prompt: str,
        responses: Sequence[str],
        *,
        seed: int | None = None,
    ) -> tuple[int, str, Mapping[str, Any]]:
        action_text = self._post_and_wait(prompt)
        norm = action_text.strip().lower()
        # Exact match first, then prefix match, then default to first option.
        for i, resp in enumerate(responses):
            if resp.strip().lower() == norm:
                return i, resp, {}
        for i, resp in enumerate(responses):
            r = resp.strip().lower()
            if norm.startswith(r) or r.startswith(norm):
                return i, resp, {}
        return 0, responses[0], {}


# ---------------------------------------------------------------------------
# Minimal EntityAgent factory helper
# ---------------------------------------------------------------------------


def make_player_entity(name: str, model: MARLLLMLanguageModel):
    """Return a minimal Concordia ``EntityAgent`` backed by *model*.

    Includes an ``_ObservationBuffer`` context component so observations
    delivered between ``act()`` calls are included in the assembled prompt
    passed to ``model.sample_text()``.  This is sufficient for most Concordia
    scenarios.  Factories that require memory retrieval, planning, or other
    components should construct their own ``EntityAgent`` instances directly,
    passing the same *model* object to ``ConcatActComponent``.
    """
    from concordia.agents import entity_agent as entity_agent_lib
    from concordia.components.agent import concat_act_component
    from concordia.typing import entity_component

    class _ObservationBuffer(entity_component.ContextComponent):
        """Buffers observations and returns them in pre_act context."""

        def __init__(self) -> None:
            self._buf: list[str] = []

        def pre_observe(self, observation: str) -> str:
            if observation:
                self._buf.append(observation)
            return ""

        def pre_act(self, action_spec) -> str:
            return "\n".join(self._buf) if self._buf else ""

        def post_act(self, action_attempt: str) -> str:
            self._buf.clear()
            return ""

    act_component = concat_act_component.ConcatActComponent(
        model=model,
        prefix_entity_name=False,
    )
    return entity_agent_lib.EntityAgent(
        agent_name=name,
        act_component=act_component,
        context_components={"observations": _ObservationBuffer()},
    )


# ---------------------------------------------------------------------------
# ConcordiaEnv
# ---------------------------------------------------------------------------


class ConcordiaEnv(AECEnv):
    """PettingZoo AEC environment wrapping an arbitrary Concordia simulation.

    See module docstring for full architecture description.

    Parameters
    ----------
    simulation_factory:
        ``(models: dict[str, MARLLLMLanguageModel], seed: int | None) -> sim``
        where ``sim`` has a ``.play()`` method.  The factory receives one
        ``MARLLLMLanguageModel`` per agent (already connected to queues) and
        must construct a Concordia simulation that uses them.  Use
        ``make_player_entity(name, model)`` to build ``EntityAgent`` objects.
    agent_names:
        Names of the agents being trained (and used inside the simulation).
    tokenizer:
        HuggingFace tokenizer.  Stored as ``env._tok`` so the Trainer can
        re-share the reference after deep-copying the template env.
    action_token_budget:
        Tokens per agent turn; read by the Trainer.
    max_turns:
        Hard cap on agent acts per episode.
    reward_fn:
        ``(final_state: Any, agent_name: str) -> float``.  Called at episode
        end.  Defaults to zero.
    """

    metadata = {"render_modes": [], "name": "concordia_v1"}

    # Signals to the Trainer that each PettingZoo observation is the complete
    # context window assembled by Concordia — no chat-template wrapping needed.
    obs_is_full_context: bool = True

    def __init__(
        self,
        simulation_factory: Callable[[dict[str, MARLLLMLanguageModel], int | None], Any],
        agent_names: list[str],
        tokenizer: PreTrainedTokenizerBase,
        action_token_budget: int = 128,
        max_turns: int = 100,
        reward_fn: Callable[[Any, str], float] | None = None,
    ) -> None:
        super().__init__()

        self._factory = simulation_factory
        self.possible_agents = list(agent_names)
        self._tok = tokenizer
        self.action_token_budget = action_token_budget
        self._max_turns = max_turns
        self._reward_fn = reward_fn if reward_fn is not None else (lambda _s, _a: 0.0)

        # Runtime state — created fresh in reset()
        self._obs_q: queue.Queue | None = None
        self._act_qs: dict[str, queue.Queue] = {}
        self._models: dict[str, MARLLLMLanguageModel] = {}
        self._sim_thread: threading.Thread | None = None
        self._final_state: Any = None
        self._turn_count: int = 0

        # PettingZoo AEC state — populated in reset()
        self.agents: list[str] = []
        self.agent_selection: str = ""
        self._pending_obs: dict[str, list[int]] = {}
        self._cumulative_rewards: dict[str, float] = {}
        self._terminations: dict[str, bool] = {}
        self._truncations: dict[str, bool] = {}
        self._infos: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Deep-copy support
    # ------------------------------------------------------------------

    def __deepcopy__(self, memo: dict) -> "ConcordiaEnv":
        """Config-only copy; thread state is created in reset()."""
        new = ConcordiaEnv(
            simulation_factory=self._factory,
            agent_names=list(self.possible_agents),
            tokenizer=self._tok,
            action_token_budget=self.action_token_budget,
            max_turns=self._max_turns,
            reward_fn=self._reward_fn,
        )
        memo[id(self)] = new
        return new

    # ------------------------------------------------------------------
    # PettingZoo AEC interface
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None, options: dict | None = None) -> None:
        self._stop_sim()

        self.agents = list(self.possible_agents)
        self._pending_obs = {a: [] for a in self.agents}
        self._cumulative_rewards = {a: 0.0 for a in self.agents}
        self._terminations = {a: False for a in self.agents}
        self._truncations = {a: False for a in self.agents}
        self._infos = {a: {} for a in self.agents}
        self._final_state = None
        self._turn_count = 0

        self._obs_q = queue.Queue()
        self._act_qs = {a: queue.Queue() for a in self.agents}

        self._models = {
            a: MARLLLMLanguageModel(a, self._tok) for a in self.agents
        }
        for a in self.agents:
            self._models[a]._connect(self._obs_q, self._act_qs[a])

        sim = self._factory(self._models, seed)

        self._sim_thread = threading.Thread(
            target=self._run_sim,
            args=(sim,),
            daemon=True,
            name="concordia-sim",
        )
        self._sim_thread.start()
        self._advance()

    def observe(self, agent: str) -> list[int]:
        return list(self._pending_obs.get(agent, []))

    def step(self, action: Any) -> None:
        agent = self.agent_selection

        if self._terminations.get(agent) or self._truncations.get(agent):
            self._was_dead_step(action)
            return

        action_text = (
            self._tok.decode(list(action), skip_special_tokens=True)
            if action is not None and len(action) > 0
            else ""
        )

        self._pending_obs[agent] = []
        self._turn_count += 1

        if self._turn_count > self._max_turns:
            self._truncate_episode()
            return

        self._act_qs[agent].put(action_text)
        self._advance()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _run_sim(self, sim: Any) -> None:
        try:
            sim.play()
        except _SimulationStopped:
            pass
        except Exception:
            pass

        final_state: Any = {}
        for attr in ("get_outcome", "outcome", "get_state", "state"):
            candidate = getattr(sim, attr, None)
            if candidate is not None:
                final_state = candidate() if callable(candidate) else candidate
                break

        self._obs_q.put(_Done(final_state))

    def _advance(self) -> None:
        item = self._obs_q.get()

        if isinstance(item, _Done):
            self._final_state = item.final_state
            for a in self.possible_agents:
                self._terminations[a] = True
                self._cumulative_rewards[a] = self._reward_fn(self._final_state, a)
            if self.agents:
                self.agent_selection = self.agents[0]
        else:
            agent_name, obs_ids = item
            self._pending_obs[agent_name] = obs_ids
            self.agent_selection = agent_name

    def _truncate_episode(self) -> None:
        self._stop_sim()
        for a in self.possible_agents:
            self._truncations[a] = True
            self._cumulative_rewards[a] = self._reward_fn(self._final_state, a)
        if self.agents:
            self.agent_selection = self.agents[0]

    def _stop_sim(self) -> None:
        if self._sim_thread is not None and self._sim_thread.is_alive():
            for q in self._act_qs.values():
                q.put(_STOP)
            self._sim_thread.join(timeout=5.0)
        self._sim_thread = None

    def _was_dead_step(self, action: Any) -> None:
        if action is not None:
            raise ValueError("Only None is valid for a terminated/truncated agent.")
        agent = self.agent_selection
        if agent in self.agents:
            self.agents.remove(agent)
        self._cumulative_rewards[agent] = 0.0
        if self.agents:
            self.agent_selection = self.agents[0]

    # ------------------------------------------------------------------
    # PettingZoo required properties
    # ------------------------------------------------------------------

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

    def observation_space(self, agent: str) -> None:
        return None

    def action_space(self, agent: str) -> None:
        return None

    def render(self) -> None:
        pass

    def close(self) -> None:
        self._stop_sim()
