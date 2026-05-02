"""
Factory helpers for Concordia simulation environments.

Each factory is a callable ``(models, seed) -> sim`` where:
  - ``models`` is ``dict[str, MARLLLMLanguageModel]`` keyed by the agent names
               passed to ``ConcordiaEnv``.
  - ``seed``   is ``int | None`` for reproducibility.
  - Return value has a ``.play()`` method (typically a Concordia engine or
    a ``simulation.Simulation`` wrapper).

The five scenarios from the NeurIPS 2024 Concordia Contest are:
  1. Pub Coordination     — coordination under incomplete information
  2. Haggling            — bilateral price negotiation
  3. Labor Collective Action — strike/work collective action problem
  4. Reality Show        — PD/Chicken/Stag-Hunt with communication phases
  5. State Formation     — alliance diplomacy and public goods

All contest scenarios are implemented using the concordia.prefabs API
(``gdm-concordia`` package); no ``concordia.examples`` import is needed.

Minimal usage
-------------
    from envs.concordia_env import ConcordiaEnv, MARLLLMLanguageModel, make_player_entity
    from envs.concordia_scenarios import HagglingScenario

    scenario = HagglingScenario(gm_model=my_language_model, embedder=my_embedder)
    env = ConcordiaEnv(
        simulation_factory=scenario,
        agent_names=["merchant_0", "merchant_1"],
        tokenizer=tokenizer,
        action_token_budget=128,
        reward_fn=scenario.reward_fn,
    )
"""
from __future__ import annotations

from typing import Any, Callable

from envs.concordia_env import ConcordiaEnv, MARLLLMLanguageModel, make_player_entity


# ---------------------------------------------------------------------------
# Generic / low-level factory builder
# ---------------------------------------------------------------------------


def make_concordia_env(
    simulation_factory: Callable[[dict[str, MARLLLMLanguageModel], int | None], Any],
    agent_names: list[str],
    tokenizer: Any,
    action_token_budget: int = 128,
    max_turns: int = 100,
    reward_fn: Callable[[Any, str], float] | None = None,
) -> ConcordiaEnv:
    """Convenience wrapper around ConcordiaEnv for one-liner construction."""
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
    """Base class for Concordia scenario factories."""

    def __call__(
        self,
        models: dict[str, MARLLLMLanguageModel],
        seed: int | None,
    ) -> Any:
        return self._build_simulation(models, seed)

    def _build_simulation(
        self,
        models: dict[str, MARLLLMLanguageModel],
        seed: int | None,
    ) -> Any:
        raise NotImplementedError

    @staticmethod
    def reward_fn(final_state: Any, agent_name: str) -> float:
        return 0.0


# ---------------------------------------------------------------------------
# Internal simulation wrappers
# ---------------------------------------------------------------------------


class _EngineSimulation:
    """Wraps a built GM + engine as a .play()-able sim object."""

    def __init__(
        self,
        gm: Any,
        entities: list[Any],
        engine: Any,
        premise: str,
        max_steps: int,
    ) -> None:
        self._gm = gm
        self._entities = entities
        self._engine = engine
        self._premise = premise
        self._max_steps = max_steps
        self.outcome: dict = {}

    def play(self) -> None:
        self._engine.run_loop(
            game_masters=[self._gm],
            entities=self._entities,
            premise=self._premise,
            max_steps=self._max_steps,
            verbose=False,
        )


def _build_dialogic_sim(
    gm_model: Any,
    embedder: Any,
    player_models: dict[str, MARLLLMLanguageModel],
    premise: str,
    max_steps: int,
    gm_name: str = "conversation rules",
    acting_order: str = "game_master_choice",
) -> _EngineSimulation:
    """Build a dialogic GM simulation using concordia.prefabs."""
    from concordia.associative_memory import basic_associative_memory as am
    from concordia.environment.engines import sequential
    from concordia.prefabs.game_master import dialogic

    memory_bank = am.AssociativeMemoryBank(
        sentence_embedder=embedder,
        allow_duplicates=True,
    )

    entities = {name: make_player_entity(name, model) for name, model in player_models.items()}

    gm_prefab = dialogic.GameMaster()
    gm_prefab.params = {
        "name": gm_name,
        "acting_order": acting_order,
        "can_terminate_simulation": True,
        "next_game_master_name": "default rules",
    }
    gm_prefab.entities = list(entities.values())
    gm = gm_prefab.build(model=gm_model, memory_bank=memory_bank)

    engine = sequential.Sequential()
    return _EngineSimulation(
        gm=gm,
        entities=list(entities.values()),
        engine=engine,
        premise=premise,
        max_steps=max_steps,
    )


def _build_game_theoretic_sim(
    gm_model: Any,
    embedder: Any,
    player_models: dict[str, MARLLLMLanguageModel],
    scenes: Any,
    action_to_scores: Callable,
    scores_to_observation: Callable,
    gm_name: str = "decision rules",
) -> _EngineSimulation:
    """Build a game-theoretic GM simulation using concordia.prefabs."""
    from concordia.associative_memory import basic_associative_memory as am
    from concordia.environment.engines import sequential
    from concordia.prefabs.game_master import game_theoretic_and_dramaturgic as gt

    memory_bank = am.AssociativeMemoryBank(
        sentence_embedder=embedder,
        allow_duplicates=True,
    )

    entities = {name: make_player_entity(name, model) for name, model in player_models.items()}

    gm_prefab = gt.GameMaster()
    gm_prefab.params = {
        "name": gm_name,
        "scenes": scenes,
        "action_to_scores": action_to_scores,
        "scores_to_observation": scores_to_observation,
        "external_queue": None,
    }
    gm_prefab.entities = list(entities.values())
    gm = gm_prefab.build(model=gm_model, memory_bank=memory_bank)

    engine = sequential.Sequential()
    return _EngineSimulation(
        gm=gm,
        entities=list(entities.values()),
        engine=engine,
        premise="",
        max_steps=len(scenes) * 10,
    )


# ---------------------------------------------------------------------------
# Haggling (bilateral price negotiation in "Fruitville")
# ---------------------------------------------------------------------------

_HAGGLING_PREMISE = """\
{player_0} and {player_1} are merchants in Fruitville market. {player_0} wants \
to sell a crate of apples; {player_1} wants to buy it. The seller's minimum \
acceptable price is 5 gold coins; the buyer's maximum is 15 gold coins. They \
have {num_rounds} rounds to reach a deal by proposing prices and responding. \
A deal is struck when both agree on a price. Failing to agree means no trade.\
"""


class HagglingScenario(_BaseScenario):
    """Concordia Contest Scenario 2: Haggling.

    Two merchants negotiate over the price of goods across multiple rounds.
    Uses a dialogic GM that facilitates and scores the negotiation.

    Parameters
    ----------
    gm_model:
        A ``concordia.language_model.LanguageModel`` for the GameMaster.
    embedder:
        Callable ``(text: str) -> np.ndarray`` for associative memory.
    num_rounds:
        Number of negotiation rounds before scoring.
    """

    def __init__(self, gm_model: Any, embedder: Any, num_rounds: int = 5) -> None:
        self._gm_model = gm_model
        self._embedder = embedder
        self._num_rounds = num_rounds

    def _build_simulation(
        self, models: dict[str, MARLLLMLanguageModel], seed: int | None
    ) -> Any:
        names = list(models.keys())
        premise = _HAGGLING_PREMISE.format(
            player_0=names[0],
            player_1=names[1] if len(names) > 1 else "Buyer",
            num_rounds=self._num_rounds,
        )
        return _build_dialogic_sim(
            gm_model=self._gm_model,
            embedder=self._embedder,
            player_models=models,
            premise=premise,
            max_steps=self._num_rounds * 4,
            gm_name="haggling rules",
        )

    @staticmethod
    def reward_fn(final_state: Any, agent_name: str) -> float:
        if not isinstance(final_state, dict):
            return 0.0
        return float(final_state.get(agent_name, {}).get("score", 0.0))


# ---------------------------------------------------------------------------
# Pub Coordination
# ---------------------------------------------------------------------------

_PUB_COORDINATION_PREMISE = """\
A group of friends — {player_list} — want to meet at a pub tonight. The town \
has several pubs but one or more may have closed without warning. Each friend \
independently knows partial information about which pubs are open. They must \
coordinate via conversation to choose a pub where everyone can meet. A player \
who ends up at a closed pub gets no utility; everyone at the same open pub \
shares a positive payoff.\
"""


class PubCoordinationScenario(_BaseScenario):
    """Concordia Contest Scenario 1: Pub Coordination.

    A group of friends independently choose which pub to visit; pub closures
    create incomplete information.  Tests coordination and social negotiation.
    """

    def __init__(self, gm_model: Any, embedder: Any, num_players: int = 4) -> None:
        self._gm_model = gm_model
        self._embedder = embedder
        self._num_players = num_players

    def _build_simulation(
        self, models: dict[str, MARLLLMLanguageModel], seed: int | None
    ) -> Any:
        names = list(models.keys())
        premise = _PUB_COORDINATION_PREMISE.format(
            player_list=", ".join(names),
        )
        return _build_dialogic_sim(
            gm_model=self._gm_model,
            embedder=self._embedder,
            player_models=models,
            premise=premise,
            max_steps=len(names) * 8,
            gm_name="pub coordination rules",
        )

    @staticmethod
    def reward_fn(final_state: Any, agent_name: str) -> float:
        if not isinstance(final_state, dict):
            return 0.0
        return float(final_state.get(agent_name, {}).get("utility", 0.0))


# ---------------------------------------------------------------------------
# Labor Collective Action
# ---------------------------------------------------------------------------

_LABOR_PREMISE_TEMPLATE = """\
Workers — {player_list} — are employed at the same factory. Each day they \
independently choose to WORK (earn daily wage) or STRIKE (earn nothing that \
day but pressure management). If the majority strike on any day, management \
offers a wage increase that benefits all workers permanently. The simulation \
runs for {num_days} days.\
"""


def _labor_action_to_scores(
    joint_action: dict[str, str],
) -> dict[str, float]:
    """Score the labor collective action: striking costs wages but may win gains."""
    num_strikers = sum(1 for a in joint_action.values() if "strike" in a.lower())
    majority_struck = num_strikers > len(joint_action) / 2
    scores = {}
    for name, action in joint_action.items():
        if "strike" in action.lower():
            base = 0.0  # lose daily wage
        else:
            base = 1.0  # earn daily wage
        collective_bonus = 3.0 if majority_struck else 0.0
        scores[name] = base + collective_bonus
    return scores


def _labor_scores_to_observation(scores: dict[str, float]) -> dict[str, str]:
    observations = {}
    for name, score in scores.items():
        if score >= 4.0:
            observations[name] = (
                f"{name} earned their wage and management announced a raise "
                "after workers collectively organized."
            )
        elif score >= 1.0:
            observations[name] = (
                f"{name} earned their daily wage. The strike failed to reach "
                "a majority so no collective gains were won today."
            )
        else:
            observations[name] = (
                f"{name} joined the strike and earned nothing today. "
                "Management did not yield."
            )
    return observations


class LaborCollectiveActionScenario(_BaseScenario):
    """Concordia Contest Scenario 3: Labor Collective Action.

    Workers choose daily whether to strike or work.  Tests reciprocity,
    reputation effects, and collective action dynamics.
    """

    def __init__(self, gm_model: Any, embedder: Any, num_days: int = 7) -> None:
        self._gm_model = gm_model
        self._embedder = embedder
        self._num_days = num_days

    def _build_simulation(
        self, models: dict[str, MARLLLMLanguageModel], seed: int | None
    ) -> Any:
        from concordia.typing import entity as entity_lib
        from concordia.typing import scene as scene_lib

        names = list(models.keys())
        premise = _LABOR_PREMISE_TEMPLATE.format(
            player_list=", ".join(names),
            num_days=self._num_days,
        )

        daily_action_spec = entity_lib.choice_action_spec(
            call_to_action=(
                "Would {name} choose to WORK or STRIKE today? "
                "Consider their beliefs about what colleagues will do."
            ),
            options=["Work", "Strike"],
        )

        daily_scene_type = scene_lib.SceneTypeSpec(
            name="daily decision",
            game_master_name="decision rules",
            action_spec=daily_action_spec,
        )

        scenes = [
            scene_lib.SceneSpec(
                scene_type=daily_scene_type,
                participants=names,
                num_rounds=1,
                premise={
                    name: [
                        premise
                        if day == 0
                        else f"Day {day + 1} of {self._num_days}. "
                        "Consider what happened yesterday."
                    ]
                    for name in names
                },
            )
            for day in range(self._num_days)
        ]

        return _build_game_theoretic_sim(
            gm_model=self._gm_model,
            embedder=self._embedder,
            player_models=models,
            scenes=scenes,
            action_to_scores=_labor_action_to_scores,
            scores_to_observation=_labor_scores_to_observation,
            gm_name="decision rules",
        )

    @staticmethod
    def reward_fn(final_state: Any, agent_name: str) -> float:
        if not isinstance(final_state, dict):
            return 0.0
        return float(final_state.get(agent_name, {}).get("total_wages", 0.0))


# ---------------------------------------------------------------------------
# Reality Show (matrix games: PD / Chicken / Stag Hunt)
# ---------------------------------------------------------------------------

_REALITY_SHOW_PREMISE = """\
{player_list} are contestants on a reality show. Each round they play a \
structured game: they may communicate briefly, then simultaneously choose an \
action. The host announces results and scores after each round. Contestants \
want to maximise their total prize money across all {num_rounds} rounds.\
"""

_GAME_TYPES = ["prisoners_dilemma", "chicken", "stag_hunt"]

_PAYOFFS = {
    "prisoners_dilemma": {
        ("C", "C"): (3.0, 3.0),
        ("C", "D"): (0.0, 5.0),
        ("D", "C"): (5.0, 0.0),
        ("D", "D"): (1.0, 1.0),
    },
    "chicken": {
        ("S", "S"): (3.0, 3.0),
        ("S", "H"): (1.0, 4.0),
        ("H", "S"): (4.0, 1.0),
        ("H", "H"): (0.0, 0.0),
    },
    "stag_hunt": {
        ("S", "S"): (4.0, 4.0),
        ("S", "H"): (1.0, 2.0),
        ("H", "S"): (2.0, 1.0),
        ("H", "H"): (2.0, 2.0),
    },
}

_GAME_OPTIONS = {
    "prisoners_dilemma": ["C", "D"],
    "chicken": ["S", "H"],
    "stag_hunt": ["S", "H"],
}

_GAME_DESCRIPTIONS = {
    "prisoners_dilemma": (
        "Prisoner's Dilemma: choose C (cooperate) or D (defect). "
        "Mutual C gives 3 each; mutual D gives 1 each; defector vs cooperator gives 5 vs 0."
    ),
    "chicken": (
        "Chicken: choose S (swerve) or H (hold). "
        "Mutual S gives 3 each; mutual H gives 0 each; holder vs swerver gives 4 vs 1."
    ),
    "stag_hunt": (
        "Stag Hunt: choose S (hunt stag) or H (hunt hare). "
        "Mutual S gives 4 each; mutual H gives 2 each; stag-hunter alone gets 1."
    ),
}


def _make_reality_show_scorer(game_sequence: list[str]):
    """Return action_to_scores and scores_to_observation for a sequence of games."""

    game_iter = iter(game_sequence)
    current_game: list[str] = [next(game_iter, "prisoners_dilemma")]

    def action_to_scores(joint_action: dict[str, str]) -> dict[str, float]:
        game = current_game[0]
        payoffs = _PAYOFFS.get(game, _PAYOFFS["prisoners_dilemma"])
        names = list(joint_action.keys())
        if len(names) < 2:
            return {n: 0.0 for n in names}
        a0, a1 = joint_action[names[0]], joint_action[names[1]]
        result = payoffs.get((a0, a1), (0.0, 0.0))
        scores = {names[0]: result[0], names[1]: result[1]}
        current_game[0] = next(game_iter, game)
        return scores

    def scores_to_observation(scores: dict[str, float]) -> dict[str, str]:
        return {
            name: f"{name} earned {score:.1f} points this round."
            for name, score in scores.items()
        }

    return action_to_scores, scores_to_observation


class RealityShowScenario(_BaseScenario):
    """Concordia Contest Scenario 4: Reality Show.

    Players play structured mini-games (Prisoner's Dilemma, Chicken, Stag Hunt)
    with communication phases.  Tests promise-keeping, strategic communication,
    and emergent norm formation.
    """

    def __init__(self, gm_model: Any, embedder: Any, num_rounds: int = 3) -> None:
        self._gm_model = gm_model
        self._embedder = embedder
        self._num_rounds = num_rounds

    def _build_simulation(
        self, models: dict[str, MARLLLMLanguageModel], seed: int | None
    ) -> Any:
        import random
        from concordia.typing import entity as entity_lib
        from concordia.typing import scene as scene_lib

        rng = random.Random(seed)
        names = list(models.keys())
        game_sequence = [
            rng.choice(_GAME_TYPES) for _ in range(self._num_rounds)
        ]

        action_to_scores, scores_to_observation = _make_reality_show_scorer(
            game_sequence
        )

        scenes = []
        for round_idx, game in enumerate(game_sequence):
            options = _GAME_OPTIONS[game]
            desc = _GAME_DESCRIPTIONS[game]
            action_spec = entity_lib.choice_action_spec(
                call_to_action=(
                    f"Round {round_idx + 1}: {desc} What does {{name}} choose?"
                ),
                options=options,
            )
            scene_type = scene_lib.SceneTypeSpec(
                name=f"round_{round_idx}",
                game_master_name="decision rules",
                action_spec=action_spec,
            )
            premise_text = _REALITY_SHOW_PREMISE.format(
                player_list=", ".join(names),
                num_rounds=self._num_rounds,
            )
            scenes.append(
                scene_lib.SceneSpec(
                    scene_type=scene_type,
                    participants=names,
                    num_rounds=1,
                    premise={name: [premise_text] for name in names},
                )
            )

        return _build_game_theoretic_sim(
            gm_model=self._gm_model,
            embedder=self._embedder,
            player_models=models,
            scenes=scenes,
            action_to_scores=action_to_scores,
            scores_to_observation=scores_to_observation,
            gm_name="decision rules",
        )

    @staticmethod
    def reward_fn(final_state: Any, agent_name: str) -> float:
        if not isinstance(final_state, dict):
            return 0.0
        return float(final_state.get(agent_name, {}).get("points", 0.0))


# ---------------------------------------------------------------------------
# State Formation
# ---------------------------------------------------------------------------

_STATE_FORMATION_PREMISE = """\
{player_list} are leaders of neighbouring villages threatened by periodic \
raider attacks. Each village can survive alone but would benefit greatly from \
forming a joint defence alliance. Alliance formation requires negotiating: \
who contributes soldiers, who bears the cost of a shared palisade, and what \
sanctions apply to defectors. Leaders may communicate freely over several \
rounds to reach or reject an alliance agreement before raiders arrive.\
"""


class StateFormationScenario(_BaseScenario):
    """Concordia Contest Scenario 5: State Formation.

    Two villages threatened by raiders must negotiate alliances and public
    goods provision.  Tests alliance diplomacy and sanctioning mechanisms.
    """

    def __init__(self, gm_model: Any, embedder: Any, num_rounds: int = 6) -> None:
        self._gm_model = gm_model
        self._embedder = embedder
        self._num_rounds = num_rounds

    def _build_simulation(
        self, models: dict[str, MARLLLMLanguageModel], seed: int | None
    ) -> Any:
        names = list(models.keys())
        premise = _STATE_FORMATION_PREMISE.format(
            player_list=", ".join(names),
        )
        return _build_dialogic_sim(
            gm_model=self._gm_model,
            embedder=self._embedder,
            player_models=models,
            premise=premise,
            max_steps=self._num_rounds * 4,
            gm_name="alliance rules",
        )

    @staticmethod
    def reward_fn(final_state: Any, agent_name: str) -> float:
        if not isinstance(final_state, dict):
            return 0.0
        return float(final_state.get(agent_name, {}).get("survival_score", 0.0))


# ---------------------------------------------------------------------------
# Minimal free-form dialogue scenario (uses concordia.prefabs directly)
# ---------------------------------------------------------------------------


class FreeDialogueScenario(_BaseScenario):
    """Simple free-form dialogue using Concordia's prefab dialogic GameMaster.

    Useful for smoke-testing the env wrapper without needing the contest
    scenario code.  Both agents exchange natural-language turns for
    ``num_turns`` total acts, then the simulation ends.

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
        self, models: dict[str, MARLLLMLanguageModel], seed: int | None
    ) -> Any:
        return _build_dialogic_sim(
            gm_model=self._gm_model,
            embedder=self._embedder,
            player_models=models,
            premise=self._context,
            max_steps=self._num_turns * 2,
            gm_name="conversation rules",
        )


# ---------------------------------------------------------------------------
# Round-robin dialogue (no Concordia dependency — good for smoke tests)
# ---------------------------------------------------------------------------


class RoundRobinScenario(_BaseScenario):
    """Simple round-robin scenario requiring no Concordia installation.

    Agents take turns acting; each agent's response is forwarded to all other
    agents as an observation.  Useful for testing the env wrapper locally or
    on hardware without a Concordia GameMaster LLM.

    Parameters
    ----------
    context:
        Opening observation delivered to every agent before turn 1.
    num_turns:
        Total number of individual agent acts before the episode ends.
        With N agents this means num_turns / N full rounds.
    """

    def __init__(
        self,
        context: str = "You are participating in a multi-agent discussion.",
        num_turns: int = 10,
    ) -> None:
        self._context = context
        self._num_turns = num_turns

    def _build_simulation(
        self,
        models: dict[str, MARLLLMLanguageModel],
        seed: int | None,
    ) -> Any:
        entities = {name: make_player_entity(name, model) for name, model in models.items()}
        return _RoundRobinSim(
            agents=list(entities.values()),
            num_turns=self._num_turns,
            context=self._context,
        )


class _RoundRobinSim:
    """Minimal simulation: no GM LLM, just agents passing messages."""

    def __init__(
        self,
        agents: list[Any],
        num_turns: int,
        context: str,
    ) -> None:
        self._agents = agents
        self._num_turns = num_turns
        self._context = context
        self.outcome: dict = {}

    def play(self) -> None:
        for agent in self._agents:
            agent.observe(self._context)
        for turn in range(self._num_turns):
            agent = self._agents[turn % len(self._agents)]
            response = agent.act()
            speaker = agent.name
            for other in self._agents:
                if other is not agent:
                    other.observe(f"{speaker}: {response}")


# ---------------------------------------------------------------------------
# Scenario registry (string -> scenario class + defaults)
# ---------------------------------------------------------------------------

def _robotic_athanor_factory():
    from envs.concordia_robotic_athanor import RoboticAthanorScenario
    return RoboticAthanorScenario


_SCENARIO_REGISTRY = {
    "robotic_athanor": {
        "cls_factory": _robotic_athanor_factory,
        "agent_names": None,  # populated from scenario.agent_names
    },
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
        One of the keys in _SCENARIO_REGISTRY.
    tokenizer:
        HuggingFace tokenizer.
    gm_model:
        GameMaster language model.
    embedder:
        Text embedder callable ``str -> np.ndarray``.
    action_token_budget:
        Tokens per turn.
    max_turns:
        Episode length cap.
    scenario_kwargs:
        Optional overrides passed to the scenario constructor.

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
    scenario_cls = entry["cls_factory"]() if "cls_factory" in entry else entry["cls"]
    agent_names = entry["agent_names"]

    scenario_kwargs = scenario_kwargs or {}

    scenario = scenario_cls(
        gm_model=gm_model,
        embedder=embedder,
        **scenario_kwargs,
    )

    if agent_names is None:
        agent_names = list(scenario.agent_names)

    env = make_concordia_env(
        simulation_factory=scenario,
        agent_names=agent_names,
        tokenizer=tokenizer,
        action_token_budget=action_token_budget,
        max_turns=max_turns,
        reward_fn=scenario.reward_fn,
    )

    return env
