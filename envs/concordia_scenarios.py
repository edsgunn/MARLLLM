"""
Factory helpers for Concordia simulation environments.

Each factory is a callable ``(stubs, seed) -> sim`` where:
  - ``stubs``  is ``dict[str, StubAgent]`` keyed by the agent names passed to
               ``ConcordiaEnv``.
  - ``seed``   is ``int | None`` for reproducibility.
  - Return value has a ``.play()`` method (typically a Concordia engine or
    a ``simulation.Simulation`` wrapper).

The five scenarios from the NeurIPS 2024 Concordia Contest are:
  1. Pub Coordination     — coordination under incomplete information
  2. Haggling            — bilateral price negotiation
  3. Labor Collective Action — strike/work collective action problem
  4. Reality Show        — PD/Chicken/Stag-Hunt with communication phases
  5. State Formation     — alliance diplomacy and public goods

These require ``gdm-concordia`` (``pip install gdm-concordia``) plus:
  - A background LLM for the GameMaster (``concordia.language_model``)
  - A text embedder for semantic memory (``sentence_transformers``)

Minimal usage
-------------
    from envs.concordia_env import ConcordiaEnv
    from envs.concordia_scenarios import HagglingScenario

    scenario = HagglingScenario(gm_model=my_language_model, embedder=my_embedder)
    env = ConcordiaEnv(
        simulation_factory=scenario,
        agent_names=["merchant_0", "merchant_1"],
        tokenizer=tokenizer,
        action_token_budget=128,
        reward_fn=scenario.reward_fn,
    )

    # Pass env to Trainer as usual — no other changes needed.
"""
from __future__ import annotations

from typing import Any, Callable

from envs.concordia_env import ConcordiaEnv, StubAgent


# ---------------------------------------------------------------------------
# Generic / low-level factory builder
# ---------------------------------------------------------------------------


def make_concordia_env(
    simulation_factory: Callable[[dict[str, StubAgent], int | None], Any],
    agent_names: list[str],
    tokenizer: Any,
    action_token_budget: int = 128,
    max_turns: int = 100,
    reward_fn: Callable[[Any, str], float] | None = None,
) -> ConcordiaEnv:
    """Convenience wrapper around ConcordiaEnv for one-liner construction.

    Parameters
    ----------
    simulation_factory:
        See ``ConcordiaEnv`` docstring.
    agent_names:
        Names used both in PettingZoo (``env.possible_agents``) and inside
        the simulation factory to look up player slots.
    tokenizer:
        HuggingFace tokenizer.
    action_token_budget:
        Tokens per turn.
    max_turns:
        Episode length cap.
    reward_fn:
        ``(final_state, agent_name) -> float``.
    """
    return ConcordiaEnv(
        simulation_factory=simulation_factory,
        agent_names=agent_names,
        tokenizer=tokenizer,
        action_token_budget=action_token_budget,
        max_turns=max_turns,
        reward_fn=reward_fn,
    )


# ---------------------------------------------------------------------------
# Base class for scenario factories
# ---------------------------------------------------------------------------


class _BaseScenario:
    """Base class for Concordia scenario factories.

    Subclasses implement ``_build_simulation(stubs, seed)`` and optionally
    ``reward_fn(final_state, agent_name)``.
    """

    def __call__(
        self,
        stubs: dict[str, StubAgent],
        seed: int | None,
    ) -> Any:
        return self._build_simulation(stubs, seed)

    def _build_simulation(
        self,
        stubs: dict[str, StubAgent],
        seed: int | None,
    ) -> Any:
        raise NotImplementedError

    @staticmethod
    def reward_fn(final_state: Any, agent_name: str) -> float:
        return 0.0


# ---------------------------------------------------------------------------
# Haggling (bilateral price negotiation in "Fruitville")
# ---------------------------------------------------------------------------


class HagglingScenario(_BaseScenario):
    """Concordia Contest Scenario 2: Haggling.

    Two merchants negotiate over the price of goods across multiple rounds.
    Tests mutual-benefit reasoning, opening bids, and compromise.

    Source:  ``concordia/examples/modular/environment/haggling_*.py``

    Parameters
    ----------
    gm_model:
        A ``concordia.language_model.LanguageModel`` instance used by the
        GameMaster to resolve negotiations.
    embedder:
        A callable ``(text: str) -> np.ndarray`` for associative memory
        (e.g. ``sentence_transformers.SentenceTransformer`` or
        ``concordia.contrib.components.agent.v2.memory_component``).
    num_rounds:
        Number of negotiation rounds before scoring.
    """

    def __init__(
        self,
        gm_model: Any,
        embedder: Any,
        num_rounds: int = 5,
    ) -> None:
        self._gm_model = gm_model
        self._embedder = embedder
        self._num_rounds = num_rounds

    def _build_simulation(
        self,
        stubs: dict[str, StubAgent],
        seed: int | None,
    ) -> Any:
        # Import here so the module is usable even without concordia installed
        # as long as this factory is never called.
        from concordia.examples.modular.environment import haggling  # type: ignore
        from concordia.environment import game_master as gm_lib  # type: ignore

        players = list(stubs.values())
        scenario = haggling.build_simulation(
            model=self._gm_model,
            embedder=self._embedder,
            players=players,
            seed=seed,
            num_rounds=self._num_rounds,
        )
        return scenario

    @staticmethod
    def reward_fn(final_state: Any, agent_name: str) -> float:
        if not isinstance(final_state, dict):
            return 0.0
        return float(final_state.get(agent_name, {}).get("score", 0.0))


# ---------------------------------------------------------------------------
# Pub Coordination
# ---------------------------------------------------------------------------


class PubCoordinationScenario(_BaseScenario):
    """Concordia Contest Scenario 1: Pub Coordination.

    A group of friends independently choose which pub to visit; pub closures
    create incomplete information.  Tests coordination and social negotiation.

    Source:  ``concordia/examples/modular/environment/pub_coordination_*.py``
    """

    def __init__(self, gm_model: Any, embedder: Any, num_players: int = 4) -> None:
        self._gm_model = gm_model
        self._embedder = embedder
        self._num_players = num_players

    def _build_simulation(
        self,
        stubs: dict[str, StubAgent],
        seed: int | None,
    ) -> Any:
        from concordia.examples.modular.environment import pub_coordination  # type: ignore

        return pub_coordination.build_simulation(
            model=self._gm_model,
            embedder=self._embedder,
            players=list(stubs.values()),
            seed=seed,
        )

    @staticmethod
    def reward_fn(final_state: Any, agent_name: str) -> float:
        if not isinstance(final_state, dict):
            return 0.0
        return float(final_state.get(agent_name, {}).get("utility", 0.0))


# ---------------------------------------------------------------------------
# Labor Collective Action
# ---------------------------------------------------------------------------


class LaborCollectiveActionScenario(_BaseScenario):
    """Concordia Contest Scenario 3: Labor Collective Action.

    Workers choose daily whether to strike or work.  Tests reciprocity,
    reputation effects, and collective action dynamics.

    Source:  ``concordia/examples/modular/environment/labor_collective_action*.py``
    """

    def __init__(self, gm_model: Any, embedder: Any, num_days: int = 7) -> None:
        self._gm_model = gm_model
        self._embedder = embedder
        self._num_days = num_days

    def _build_simulation(
        self,
        stubs: dict[str, StubAgent],
        seed: int | None,
    ) -> Any:
        from concordia.examples.modular.environment import (  # type: ignore
            labor_collective_action,
        )

        return labor_collective_action.build_simulation(
            model=self._gm_model,
            embedder=self._embedder,
            players=list(stubs.values()),
            seed=seed,
            num_days=self._num_days,
        )

    @staticmethod
    def reward_fn(final_state: Any, agent_name: str) -> float:
        if not isinstance(final_state, dict):
            return 0.0
        return float(final_state.get(agent_name, {}).get("total_wages", 0.0))


# ---------------------------------------------------------------------------
# Reality Show
# ---------------------------------------------------------------------------


class RealityShowScenario(_BaseScenario):
    """Concordia Contest Scenario 4: Reality Show.

    Players play structured mini-games (Prisoner's Dilemma, Chicken, Stag Hunt)
    with communication phases.  Tests promise-keeping, strategic communication,
    and emergent norm formation.

    Source:  ``concordia/examples/modular/environment/reality_show_*.py``
    """

    def __init__(self, gm_model: Any, embedder: Any, num_rounds: int = 3) -> None:
        self._gm_model = gm_model
        self._embedder = embedder
        self._num_rounds = num_rounds

    def _build_simulation(
        self,
        stubs: dict[str, StubAgent],
        seed: int | None,
    ) -> Any:
        from concordia.examples.modular.environment import reality_show  # type: ignore

        return reality_show.build_simulation(
            model=self._gm_model,
            embedder=self._embedder,
            players=list(stubs.values()),
            seed=seed,
            num_rounds=self._num_rounds,
        )

    @staticmethod
    def reward_fn(final_state: Any, agent_name: str) -> float:
        if not isinstance(final_state, dict):
            return 0.0
        return float(final_state.get(agent_name, {}).get("points", 0.0))


# ---------------------------------------------------------------------------
# State Formation
# ---------------------------------------------------------------------------


class StateFormationScenario(_BaseScenario):
    """Concordia Contest Scenario 5: State Formation.

    Two villages threatened by raiders must negotiate alliances and public
    goods provision.  Tests alliance diplomacy and sanctioning mechanisms.

    Source:  ``concordia/examples/modular/environment/state_formation_*.py``
    """

    def __init__(self, gm_model: Any, embedder: Any) -> None:
        self._gm_model = gm_model
        self._embedder = embedder

    def _build_simulation(
        self,
        stubs: dict[str, StubAgent],
        seed: int | None,
    ) -> Any:
        from concordia.examples.modular.environment import state_formation  # type: ignore

        return state_formation.build_simulation(
            model=self._gm_model,
            embedder=self._embedder,
            players=list(stubs.values()),
            seed=seed,
        )

    @staticmethod
    def reward_fn(final_state: Any, agent_name: str) -> float:
        if not isinstance(final_state, dict):
            return 0.0
        return float(final_state.get(agent_name, {}).get("survival_score", 0.0))


# ---------------------------------------------------------------------------
# Minimal free-form dialogue scenario (works without contest code)
# ---------------------------------------------------------------------------


class FreeDialogueScenario(_BaseScenario):
    """Simple free-form dialogue using Concordia's prefab dialogic GameMaster.

    Useful for smoke-testing the env wrapper without needing the contest
    scenario code.  Both agents exchange natural-language turns for
    ``num_turns`` total acts, then the simulation ends.

    This uses Concordia's ``dialogic__GameMaster`` prefab and the
    ``SequentialEngine``.  Adjust the import paths if your Concordia version
    differs (v2.0+ uses ``concordia.prefabs``).

    Parameters
    ----------
    gm_model:
        A ``concordia.language_model.LanguageModel`` instance for the GM.
    embedder:
        Text embedder callable ``str -> np.ndarray`` for associative memory.
    context:
        Shared situation description sent to all agents at episode start.
    num_turns:
        Total agent acts before the simulation ends.
    """

    def __init__(
        self,
        gm_model: Any,
        embedder: Any,
        context: str = "Two agents are having a conversation.",
        num_turns: int = 10,
    ) -> None:
        self._gm_model = gm_model
        self._embedder = embedder
        self._context = context
        self._num_turns = num_turns

    def _build_simulation(
        self,
        stubs: dict[str, StubAgent],
        seed: int | None,
    ) -> Any:
        # Exact import paths depend on the installed Concordia version.
        # For gdm-concordia >= 2.0:
        from concordia.environment import sequential_engine  # type: ignore
        from concordia.prefabs.game_master import dialogic  # type: ignore

        player_names = list(stubs.keys())
        gm = dialogic.GameMaster(
            model=self._gm_model,
            player_names=player_names,
            num_turns=self._num_turns,
            seed=seed,
        )
        engine = sequential_engine.SequentialEngine(
            game_master=gm,
            players=list(stubs.values()),
        )
        return _BoundedSimulation(
            engine=engine,
            agents=list(stubs.values()),
            context=self._context,
        )


class _BoundedSimulation:
    """Minimal simulation wrapper: delivers context then runs the engine."""

    def __init__(
        self,
        engine: Any,
        agents: list[StubAgent],
        context: str,
    ) -> None:
        self._engine = engine
        self._agents = agents
        self._context = context
        self.outcome: dict = {}

    def play(self) -> dict:
        for agent in self._agents:
            agent.observe(self._context)
        self._engine.play()
        return self.outcome

# ---------------------------------------------------------------------------
# Scenario registry (string -> scenario class + defaults)
# ---------------------------------------------------------------------------

_SCENARIO_REGISTRY = {
    "haggling": {
        "cls": HagglingScenario,
        "agent_names": ["merchant_0", "merchant_1"],
    },
    "pub_coordination": {
        "cls": PubCoordinationScenario,
        "agent_names": ["player_0", "player_1", "player_2", "player_3"],
    },
    "labor_collective_action": {
        "cls": LaborCollectiveActionScenario,
        "agent_names": ["worker_0", "worker_1", "worker_2", "worker_3"],
    },
    "reality_show": {
        "cls": RealityShowScenario,
        "agent_names": ["player_0", "player_1", "player_2"],
    },
    "state_formation": {
        "cls": StateFormationScenario,
        "agent_names": ["village_0", "village_1"],
    },
    "free_dialogue": {
        "cls": FreeDialogueScenario,
        "agent_names": ["agent_0", "agent_1"],
    },
}

# ---------------------------------------------------------------------------
# High-level factory: string -> fully built env
# ---------------------------------------------------------------------------


def make_env_from_scenario_name(
    scenario_name: str,
    tokenizer: Any,
    gm_model: Any,
    embedder: Any,
    *,
    action_token_budget: int = 128,
    max_turns: int = 100,
    scenario_kwargs: dict | None = None,
) -> ConcordiaEnv:
    """Create a fully configured ConcordiaEnv from a string name.

    Parameters
    ----------
    scenario_name:
        One of keys in _SCENARIO_REGISTRY.
    tokenizer:
        HuggingFace tokenizer.
    gm_model:
        GameMaster language model.
    embedder:
        Text embedder.
    action_token_budget:
        Tokens per turn.
    max_turns:
        Episode length cap.
    scenario_kwargs:
        Optional overrides passed to scenario constructor.

    Returns
    -------
    ConcordiaEnv
    """
    if scenario_name not in _SCENARIO_REGISTRY:
        raise ValueError(
            f"Unknown scenario '{scenario_name}'. "
            f"Available: {list(_SCENARIO_REGISTRY.keys())}"
        )

    entry = _SCENARIO_REGISTRY[scenario_name]
    scenario_cls = entry["cls"]
    agent_names = entry["agent_names"]

    scenario_kwargs = scenario_kwargs or {}

    # Instantiate scenario
    scenario = scenario_cls(
        gm_model=gm_model,
        embedder=embedder,
        **scenario_kwargs,
    )

    # Build env
    env = make_concordia_env(
        simulation_factory=scenario,
        agent_names=agent_names,
        tokenizer=tokenizer,
        action_token_budget=action_token_budget,
        max_turns=max_turns,
        reward_fn=scenario.reward_fn,
    )

    return env