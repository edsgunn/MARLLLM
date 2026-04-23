"""
PettingZoo AEC wrapper for the Concordia multi-agent simulation framework.

Architecture
------------
Concordia drives its simulation loop internally (calling agent.observe() then
agent.act() per turn).  We bridge this into the PettingZoo step interface by
running the Concordia engine in a daemon background thread and using queues to
suspend the simulation whenever it needs an agent action.

Thread model
~~~~~~~~~~~~
  Simulation thread: engine.play() → stub.act() → blocks on act_queue
  Main thread:       drives AEC (reset/step/observe) ↔ reads obs_queue,
                     writes act_queue

Deep-copy safety
~~~~~~~~~~~~~~~~
The Trainer deep-copies the template env (before reset()) to create n parallel
envs per iteration, then immediately calls reset() on each copy.  Thread/queue
state is created only inside reset() so the template object is safely copyable.

Usage
-----
    from envs.concordia_env import ConcordiaEnv, StubAgent

    def my_factory(stubs: dict[str, StubAgent], seed: int | None):
        # Build your Concordia simulation; insert `stubs` as the player agents.
        # Return anything with a .play() method.
        gm = build_gamemaster(model=gm_llm, ...)
        engine = concordia.SequentialEngine(gm, list(stubs.values()))
        return engine

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
from typing import Any, Callable

from concordia.typing import entity as entity_lib
from pettingzoo import AECEnv
from transformers import PreTrainedTokenizerBase


# ---------------------------------------------------------------------------
# Internal sentinels / signals
# ---------------------------------------------------------------------------

_STOP = object()  # posted to act_queue to kill the simulation thread


class _Done:
    """Posted to obs_queue when the Concordia simulation finishes."""

    __slots__ = ("final_obs", "final_state")

    def __init__(self, final_obs: dict[str, str], final_state: Any) -> None:
        self.final_obs = final_obs
        self.final_state = final_state


class _SimulationStopped(BaseException):
    """Raised inside StubAgent.act() to unwind the sim thread cleanly."""


# ---------------------------------------------------------------------------
# StubAgent
# ---------------------------------------------------------------------------


class StubAgent(entity_lib.Entity):
    """Drop-in replacement for a Concordia EntityAgent.

    Replace the LLM-backed player agents in a Concordia simulation with
    StubAgent instances so the Concordia engine calls observe/act on them.
    When act() is called:
      1. All buffered observations are posted to obs_queue (main thread reads).
      2. The thread blocks until the main thread puts an action text in act_queue.
      3. The action text is returned to the Concordia engine as the action string.

    Parameters
    ----------
    agent_name:
        Must match the name the GameMaster uses to identify this player.
    """

    def __init__(self, agent_name: str) -> None:
        self.agent_name = agent_name
        self._obs_buf: list[str] = []
        # Connected by ConcordiaEnv before starting each episode
        self._obs_q: queue.Queue = None  # type: ignore[assignment]
        self._act_q: queue.Queue = None  # type: ignore[assignment]

    @property
    def name(self) -> str:
        return self.agent_name

    # ---- called by ConcordiaEnv, not by user code -------------------------

    def _connect(self, obs_q: queue.Queue, act_q: queue.Queue) -> None:
        self._obs_q = obs_q
        self._act_q = act_q
        self._obs_buf = []

    # ---- Concordia agent interface ----------------------------------------

    def observe(self, observation: str) -> None:
        if observation:
            self._obs_buf.append(str(observation))

    def act(self, action_spec: Any = None) -> str:
        obs_text = "\n".join(self._obs_buf)
        self._obs_buf = []
        self._obs_q.put((self.agent_name, obs_text))
        action = self._act_q.get()
        if action is _STOP:
            raise _SimulationStopped()
        return str(action)

    def get_name(self) -> str:
        return self.agent_name

    # Concordia components may call this; return None gracefully
    def get_component(self, component_name: str, default: Any = None) -> Any:
        return default


# ---------------------------------------------------------------------------
# ConcordiaEnv
# ---------------------------------------------------------------------------


class ConcordiaEnv(AECEnv):
    """PettingZoo AEC environment wrapping an arbitrary Concordia simulation.

    Observations are the natural-language text Concordia sends to each agent
    (encoded as token IDs with the provided tokenizer).  Actions are decoded
    from token IDs back to text before being returned to the Concordia engine.

    The env is otherwise agnostic to the scenario: any Concordia setup (contest
    scenarios, custom worlds, etc.) works as long as the factory injects the
    provided stub agents as the named players.

    Parameters
    ----------
    simulation_factory:
        ``(stubs: dict[str, StubAgent], seed: int | None) -> sim``
        where ``sim`` has a ``.play()`` method.  Called on each ``reset()``
        to create a fresh episode.  The dict keys are the same as
        ``agent_names``; the factory should insert these stubs as the player
        agents in the simulation so the Concordia engine calls their
        observe/act methods.
    agent_names:
        Names of the agents being trained (and used inside the simulation).
    tokenizer:
        HuggingFace tokenizer.  Stored as ``env._tok`` so the Trainer can
        re-share the reference after deep-copying the template env.
    action_token_budget:
        Tokens per turn; read by the Trainer.
    max_turns:
        Hard cap on agent acts per episode.  If reached the episode is
        truncated and the simulation thread is stopped.
    reward_fn:
        ``(final_state: Any, agent_name: str) -> float``.  Called at episode
        end.  ``final_state`` is extracted from the simulation object (see
        ``_run_sim``).  Defaults to zero for all agents.
    """

    metadata = {"render_modes": [], "name": "concordia_v0"}

    def __init__(
        self,
        simulation_factory: Callable[[dict[str, StubAgent], int | None], Any],
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
        self._stubs: dict[str, StubAgent] = {}
        self._sim_thread: threading.Thread | None = None
        self._final_state: Any = None
        self._turn_count: int = 0

        # PettingZoo state — populated in reset()
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
        """Return a fresh env with the same config but no thread state.

        The Trainer deep-copies the template env to create n parallel envs per
        iteration and immediately calls reset() on each copy.  Returning a
        config-only copy avoids trying to clone live threads or queues.
        """
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

        self._stubs = {a: StubAgent(a) for a in self.agents}
        for a in self.agents:
            self._stubs[a]._connect(self._obs_q, self._act_qs[a])

        sim = self._factory(self._stubs, seed)

        self._sim_thread = threading.Thread(
            target=self._run_sim,
            args=(sim,),
            daemon=True,
            name="concordia-sim",
        )
        self._sim_thread.start()

        # Block until first agent is ready to act
        self._advance()

    def observe(self, agent: str) -> list[int]:
        return list(self._pending_obs.get(agent, []))

    def step(self, action: Any) -> None:
        agent = self.agent_selection

        if self._terminations.get(agent) or self._truncations.get(agent):
            self._was_dead_step(action)
            return

        if action is None or len(action) == 0:
            action_text = ""
        else:
            action_text = self._tok.decode(list(action), skip_special_tokens=True)

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
        stubs = self._stubs
        try:
            sim.play()
        except _SimulationStopped:
            pass
        except Exception:
            pass

        # Collect observations buffered after the last act() call
        final_obs = {
            name: "\n".join(stub._obs_buf) for name, stub in stubs.items()
        }
        # Try common attribute names for an outcome/state object
        final_state: Any = {}
        for attr in ("get_outcome", "outcome", "get_state", "state"):
            candidate = getattr(sim, attr, None)
            if candidate is not None:
                final_state = candidate() if callable(candidate) else candidate
                break

        self._obs_q.put(_Done(final_obs, final_state))

    def _advance(self) -> None:
        """Read one item from obs_queue and update PettingZoo state."""
        item = self._obs_q.get()

        if isinstance(item, _Done):
            self._final_state = item.final_state
            for a in self.possible_agents:
                obs_text = item.final_obs.get(a, "")
                if obs_text:
                    self._pending_obs[a] = self._tok.encode(
                        obs_text, add_special_tokens=False
                    )
                self._terminations[a] = True
                self._cumulative_rewards[a] = self._reward_fn(self._final_state, a)
            if self.agents:
                self.agent_selection = self.agents[0]
        else:
            agent_name, obs_text = item
            self._pending_obs[agent_name] = (
                self._tok.encode(obs_text, add_special_tokens=False)
                if obs_text
                else []
            )
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
