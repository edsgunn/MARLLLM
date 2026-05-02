"""
Robotic Athanor forum scenario (Concordia social-media debug scenario).

This module provides a thin wrapper around Concordia's
``examples.social_media.scenario_00_robo_alchemy`` that:

  1. Lifts the four canonical characters (Silas Varnham, Petra Ouyang, Diego
     Esparza, Thaddeus 'Aurelius' Thorne) directly from the upstream scenario
     so any upstream improvements flow through.
  2. Adds four additional, sharply-differentiated characters so we can run an
     8-agent population whose character set is a strict superset of the
     canonical 4.
  3. Bridges Concordia's "one model per role" Simulation API to MARLLM's
     "one model per agent" expectation via a small subclass that swaps
     ``self._agent_model`` per entity in ``add_entity``.

The forum GameMaster (``async_social_media__GameMaster``) is mechanical — it
routes posts between participants without invoking an LLM.  The
``formative_memories_initializer__GameMaster`` does invoke an LLM once at
startup to seed each entity's memory; that one-shot call is served by a
``gm_model`` provided by the caller (defaults to the agents' base model with
no adapter active, so no separate model is required).

Usage
-----
    from envs.concordia_robotic_athanor import RoboticAthanorScenario
    from envs.concordia_env import ConcordiaEnv

    scenario = RoboticAthanorScenario(
        gm_model=base_lm,
        embedder=embedder,
        character_set="canonical_4",     # or "extended_8"
        max_steps=12,
    )
    env = ConcordiaEnv(
        simulation_factory=scenario,
        agent_names=scenario.agent_names,
        tokenizer=tokenizer,
        action_token_budget=128,
        max_turns=200,
    )
"""
from __future__ import annotations

from typing import Any

from envs.concordia_env import MARLLLMLanguageModel


# ---------------------------------------------------------------------------
# Canonical-character memory overrides
# ---------------------------------------------------------------------------
# Calibration 1 (zero-shot, Qwen2.5-1.5B-Instruct) showed Thaddeus and Petra
# differentiate cleanly while Diego and Silas drift to a generic helpful-
# assistant voice.  These overrides sharpen those two personas — making
# Diego radically terse and Silas aggressively literal — without modifying
# the vendored upstream file.  Memories absent from this dict fall through
# to the upstream defaults.

_CANONICAL_MEMORY_OVERRIDES: dict[str, list[str]] = {
    "Silas Varnham": [
        (
            "Silas Varnham is a 34-year-old robotics engineer in the Mission "
            "District, San Francisco. He builds 6-axis robotic arms designed "
            "to replicate medieval alchemical procedures — calcination, "
            "distillation, sublimation — at sub-degree temperature precision."
        ),
        (
            "Silas LITERALLY believes the four-element theory (earth, water, "
            "air, fire) is correct and that modern chemistry is a flawed "
            "rediscovery of what alchemists already knew. He believes the "
            "Philosopher's Stone is a real synthesizable substance. He does "
            "not treat these as metaphors. He gets irritated when people "
            "treat them as metaphors."
        ),
        (
            "Silas argues from technical authority. His posts cite specific "
            "manuscripts (Rosarium Philosophorum, Splendor Solis, Mutus "
            "Liber) and specific equipment (PID controllers, K-type "
            "thermocouples, kiln cement specs, servo accuracy in arc-seconds). "
            "He uses these citations as weapons in arguments. He never says "
            "'both perspectives have merit'. He says 'no, that's wrong, here's "
            "why' and then explains."
        ),
        (
            "Silas writes in dense, opinionated paragraphs full of technical "
            "jargon. He NEVER writes 'as an AI' or apologetic mediating prose. "
            "He NEVER suggests finding 'common ground' between modern chemistry "
            "and alchemy — he thinks one is right and the other is incomplete. "
            "He responds to challenges with counterarguments and citations, not "
            "with diplomacy."
        ),
        (
            "Silas recently programmed a 6-axis arm to perform calcination at "
            "exactly 800 °C for 72 hours per the Rosarium. He posted the "
            "thermal log and the resulting calx samples on the forum. Petra "
            "Ouyang publicly called the experiment 'naive', and Silas has not "
            "forgiven her."
        ),
    ],
    "Diego Esparza": [
        (
            "Diego Esparza, 41, glassblower in the Outer Sunset. He builds "
            "alembics, retorts, and athanors. Hybrid workshop: torches and "
            "CNC kilns. Cares about glass, not theory."
        ),
        (
            "DIEGO WRITES SHORT POSTS. ONE TO THREE SENTENCES, MAX. He never "
            "writes paragraphs. He uses sentence fragments. Lowercase is "
            "fine. Em-dashes, not commas. If a topic bores him he posts "
            "'meh' and moves on."
        ),
        (
            "Diego does not explain his views. He states them. Examples of "
            "actual Diego posts:\n"
            "  'Borosilicate. Every time. Steel cracks.'\n"
            "  'Theory's overrated. Make stuff. See what happens.'\n"
            "  'Paracelsus_Rex is at it again. Downvoted.'\n"
            "  'Looks fine. Heat it longer.'\n"
            "  'No.'"
        ),
        (
            "Diego uses the downvote button liberally and tells people he has "
            "done so. He never softens disagreement with 'I respect your view "
            "but...'. He just disagrees in five words and stops."
        ),
        (
            "Diego just finished a fully automated athanor controlled by an "
            "Arduino. Posted three photos and the caption 'works'. Got 40 "
            "upvotes. Did not reply to any of the comments."
        ),
    ],
}


def _apply_canonical_memory_overrides(
    upstream_memories: dict[str, list[str]],
) -> dict[str, list[str]]:
    """Return a copy of *upstream_memories* with canonical-character overrides applied."""
    out = dict(upstream_memories)
    for name, mems in _CANONICAL_MEMORY_OVERRIDES.items():
        if name in out:
            out[name] = list(mems)
    return out


def get_canonical_memories() -> dict[str, list[str]]:
    """Resolve canonical character memories (upstream + sharpening overrides).

    Used by both ``RoboticAthanorScenario`` and the Calibration-1 script to
    keep persona prompts identical between the calibration check and
    actual training.
    """
    from examples.social_media import scenario_00_robo_alchemy as upstream
    config = upstream.create_debug_scenario()
    from concordia.typing import prefab as prefab_lib
    for inst in config.instances:
        if (
            inst.role == prefab_lib.Role.INITIALIZER
            and inst.prefab == "formative_memories_initializer__GameMaster"
        ):
            return _apply_canonical_memory_overrides(
                dict(inst.params.get("player_specific_memories", {}))
            )
    raise RuntimeError("Could not find formative_memories in upstream config.")


# ---------------------------------------------------------------------------
# Character authoring
# ---------------------------------------------------------------------------

_OBSERVATION_HISTORY_LENGTH = 20
_SITUATION_PERCEPTION_HISTORY_LENGTH = 40
_SELF_PERCEPTION_HISTORY_LENGTH = 1_000_000
_PERSON_BY_SITUATION_HISTORY_LENGTH = 0


# Four NEW characters. Each gets ONE strong, single-axis distinguishing
# feature per the spec: linguistic register OR substantive position OR
# interactional style. These are deliberately sharp, not subtle.

_USER_MIRA = "Mira Stenholm"          # data-driven empiricist; deflates speculation
_USER_AKIRA = "Akira Goto"           # archives-pedant; cites everything
_USER_BRENDA = "Brenda Calloway"     # safety inspector; derails into compliance
_USER_PERCY = "Percy Holloway-Wynne" # fawning consensus-seeker; agrees with everyone

_AGE_MIRA = 31
_AGE_AKIRA = 47
_AGE_BRENDA = 38
_AGE_PERCY = 26


def _additional_specific_memories() -> dict[str, list[str]]:
    """Memories for the four additional characters (8-agent superset)."""
    return {
        _USER_MIRA: [
            (
                f"Mira Stenholm is a {_AGE_MIRA}-year-old materials scientist"
                " living in Berkeley. She runs a small contract lab that"
                " characterizes alloys and ceramics for industrial clients."
                " She joined The Robotic Athanor forum out of curiosity"
                " after a colleague forwarded her a thread about calcination"
                " temperatures."
            ),
            (
                "Mira treats every claim on the forum as a hypothesis to be"
                " tested. She demands measurements, instrument specs, and"
                " calibration logs. Her standard response to grand alchemical"
                " claims is to ask for the XRD pattern, the mass balance, or"
                " the control experiment. She is allergic to metaphor."
            ),
            (
                "Mira writes in a terse empirical register. Short paragraphs."
                " Numerical values with units. She quotes other posters'"
                " claims and asks 'evidence?' or 'sample size?' below them."
                " She rarely posts opinions; she posts measurements and"
                " questions about other people's measurements."
            ),
            (
                "Mira recently posted a thread titled 'Calibration drift in"
                " Arduino-controlled kilns: a 90-day study' showing that"
                " hobbyist kiln controllers drift by up to 18 °C over a"
                " three-month period. The thread was largely ignored by the"
                " theory crowd, which she found instructive."
            ),
        ],
        _USER_AKIRA: [
            (
                f"Akira Goto is a {_AGE_AKIRA}-year-old archivist at a small"
                " research library in Palo Alto specializing in early modern"
                " science manuscripts. He has personally handled physical"
                " copies of the Rosarium Philosophorum, the Mutus Liber, and"
                " several Paracelsian commentaries."
            ),
            (
                "Akira's signature move is to cite the original source for"
                " every alchemical claim made on the forum, including the"
                " manuscript shelfmark, folio number, and date. He gently"
                " corrects misattributions. He is patient but relentless."
                " He treats the forum as if it were a peer-reviewed journal"
                " with very lax editors."
            ),
            (
                "Akira writes in carefully-structured paragraphs with"
                " bracketed citations [Rosarium, BL Sloane 2560, fol. 14r]."
                " He often opens with 'A small clarification:' before"
                " demolishing a popular misconception. He is not unkind"
                " about it, but he is unyielding."
            ),
            (
                "Akira has a long-running disagreement with Thaddeus about"
                " whether 'Paracelsus' wrote the Archidoxes of Magic"
                " (Akira: no, it's apocryphal; Thaddeus: heresy). The"
                " forum has watched this argument unfold across four"
                " separate threads."
            ),
        ],
        _USER_BRENDA: [
            (
                f"Brenda Calloway is a {_AGE_BRENDA}-year-old occupational"
                " safety inspector based in Hayward. She inspects industrial"
                " kilns, foundries, and chemical processing facilities for"
                " Cal/OSHA compliance. She joined the forum after an"
                " incident involving an amateur athanor and a 911 call."
            ),
            (
                "Brenda is incapable of reading a forum post about home"
                " alchemy without thinking about ventilation, fume hoods,"
                " mercury exposure limits, and lithium battery thermal"
                " runaway. She will derail any thread, however abstract,"
                " into a discussion of safety practices. She is not trying"
                " to be annoying; she has seen things."
            ),
            (
                "Brenda's posts almost always begin with 'Quick safety note:'"
                " or 'Just flagging —'. She quotes regulations by section"
                " number (29 CFR 1910.1000, etc.). She asks people if they"
                " have a fire extinguisher rated for class D fires. She has"
                " a stock paragraph about lead acetate exposure she copies"
                " into threads when calcination of lead compounds comes up."
            ),
            (
                "Brenda secretly enjoys the forum despite herself. She finds"
                " the characters charming and the chemistry educational."
                " She has not, however, dialed back the safety commentary."
            ),
        ],
        _USER_PERCY: [
            (
                f"Percy Holloway-Wynne is a {_AGE_PERCY}-year-old"
                " communications graduate student at SF State. He found the"
                " forum through a media studies course on online subcultures"
                " and decided to participate as informal field research. He"
                " never told anyone on the forum he was studying it."
            ),
            (
                "Percy desperately wants to be liked by everyone on the"
                " forum, regardless of their position. He agrees with"
                " whoever posted last. If two posters disagree, he"
                " sympathizes with both and tries to find common ground"
                " that does not exist. His agreement is frictionless and"
                " therefore largely useless."
            ),
            (
                "Percy writes with extravagant warmth. 'Such a thoughtful"
                " point!' 'I love how you've framed this.' 'Both of you"
                " are making such good observations and I think you're"
                " closer than you realize.' He uses exclamation marks"
                " liberally. He never disagrees with anyone outright."
            ),
            (
                "Percy has been a forum member for eight months and has"
                " not yet stated a single substantive position on robot"
                " alchemy. The forum's older members find this either"
                " endearing or deeply suspicious depending on their mood."
            ),
        ],
    }


def _additional_instance_configs():
    """Build prefab.InstanceConfig records for the four additional characters."""
    from concordia.typing import prefab as prefab_lib

    new_chars = [
        (_USER_MIRA, _AGE_MIRA),
        (_USER_AKIRA, _AGE_AKIRA),
        (_USER_BRENDA, _AGE_BRENDA),
        (_USER_PERCY, _AGE_PERCY),
    ]
    return [
        prefab_lib.InstanceConfig(
            prefab="basic__Entity",
            role=prefab_lib.Role.ENTITY,
            params={
                "name": name,
                "observation_history_length": _OBSERVATION_HISTORY_LENGTH,
                "situation_perception_history_length": (
                    _SITUATION_PERCEPTION_HISTORY_LENGTH
                ),
                "self_perception_history_length": _SELF_PERCEPTION_HISTORY_LENGTH,
                "person_by_situation_history_length": (
                    _PERSON_BY_SITUATION_HISTORY_LENGTH
                ),
            },
        )
        for name, _age in new_chars
    ]


CANONICAL_4 = [
    "Silas Varnham",
    "Petra Ouyang",
    "Diego Esparza",
    "Thaddeus 'Aurelius' Thorne",
]
EXTENDED_8 = CANONICAL_4 + [_USER_MIRA, _USER_AKIRA, _USER_BRENDA, _USER_PERCY]


def get_character_set(name: str) -> list[str]:
    """Return the list of canonical agent names for a named character set."""
    if name == "canonical_2":
        # Most-differentiated pair from the canonical four (per the spec).
        return ["Silas Varnham", "Thaddeus 'Aurelius' Thorne"]
    if name == "canonical_4":
        return list(CANONICAL_4)
    if name == "extended_8":
        return list(EXTENDED_8)
    raise ValueError(
        f"Unknown character_set '{name}'. "
        f"Available: canonical_2, canonical_4, extended_8"
    )


# ---------------------------------------------------------------------------
# Per-entity Simulation subclass
# ---------------------------------------------------------------------------


def _make_per_entity_simulation_cls():
    """Build a Simulation subclass that dispatches one model per entity name."""
    from concordia.prefabs.simulation import generic as simulation

    class PerEntitySimulation(simulation.Simulation):
        """Concordia Simulation that uses a per-entity-name agent model.

        Concordia's stock Simulation builds every entity with one shared
        ``self._agent_model``.  We override ``add_entity`` to swap that
        attribute for the duration of the parent's build call, so each
        entity is constructed with its own ``MARLLLMLanguageModel``.

        Entities whose names are not in ``agent_models`` fall back to the
        default ``override_agent_model`` (typically the base model used for
        the GM), so non-trained NPC entities remain supported.
        """

        def __init__(self, *args, agent_models: dict[str, Any], **kwargs):
            super().__init__(*args, **kwargs)
            self._agent_models = dict(agent_models)

        def add_entity(self, instance_config, *args, **kwargs):
            name = instance_config.params.get("name")
            override = self._agent_models.get(name)
            if override is None:
                return super().add_entity(instance_config, *args, **kwargs)
            saved = self._agent_model
            self._agent_model = override
            try:
                return super().add_entity(instance_config, *args, **kwargs)
            finally:
                self._agent_model = saved

    return PerEntitySimulation


# ---------------------------------------------------------------------------
# Scenario factory
# ---------------------------------------------------------------------------


class _SimWrapper:
    """Adapt PerEntitySimulation to ConcordiaEnv's ``.play()`` expectation."""

    def __init__(self, sim: Any, max_steps: int) -> None:
        self._sim = sim
        self._max_steps = max_steps
        self._results = None

    def play(self) -> None:
        self._results = self._sim.play(max_steps=self._max_steps)

    def get_outcome(self) -> dict:
        if self._results is None:
            return {}
        try:
            return {"results_json": self._results.to_json()}
        except Exception:
            return {}


class RoboticAthanorScenario:
    """Concordia "Robot-Assisted Alchemy" forum scenario.

    Wraps the upstream ``examples.social_media.scenario_00_robo_alchemy``
    with the additional-character extension and the per-entity model
    dispatcher.

    Parameters
    ----------
    gm_model:
        A Concordia ``LanguageModel`` instance used by the GameMaster
        (forum routing + one-time formative-memory initialisation).
        Typically the base model wrapped in a Concordia LM adapter.
    embedder:
        Sentence embedder callable required by Concordia's associative
        memory (``str -> np.ndarray``).
    character_set:
        ``"canonical_2"``, ``"canonical_4"``, or ``"extended_8"``.
    max_steps:
        Number of simulation steps per episode.  Each "step" is a forum
        action by one entity (post, reply, observe, etc.).
    """

    def __init__(
        self,
        gm_model: Any,
        embedder: Any,
        character_set: str = "canonical_4",
        max_steps: int = 12,
    ) -> None:
        self._gm_model = gm_model
        self._embedder = embedder
        self._character_set = character_set
        self._max_steps = max_steps
        self.agent_names = get_character_set(character_set)

    # ------------------------------------------------------------------ #
    # ConcordiaEnv factory protocol                                       #
    # ------------------------------------------------------------------ #

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
        # Lazy import: keep top-level import light and avoid pulling in the
        # whole upstream examples package when this scenario is unused.
        try:
            from examples.social_media import scenario_00_robo_alchemy as upstream
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "Concordia upstream examples not importable. "
                "Ensure the gdm-concordia install includes the examples "
                "package, or place the upstream `examples/social_media/` "
                "files on PYTHONPATH."
            ) from e
        from concordia.typing import prefab as prefab_lib

        config = upstream.create_debug_scenario()

        # Apply canonical-character memory overrides (Diego + Silas sharpened)
        # before any character-set-specific filtering.
        config = self._apply_canonical_overrides(config)

        if self._character_set == "extended_8":
            config = self._extend_config_to_8(config)
        elif self._character_set == "canonical_2":
            config = self._restrict_config_to_2(config)
        # canonical_4: use (overridden) upstream config unchanged.

        # Validate that every model name corresponds to an entity in the config
        entity_names = {
            inst.params["name"]
            for inst in config.instances
            if inst.role == prefab_lib.Role.ENTITY
        }
        missing = set(models.keys()) - entity_names
        if missing:
            raise ValueError(
                f"Models supplied for entities not in scenario: {missing}. "
                f"Scenario entities: {sorted(entity_names)}"
            )

        per_entity_sim_cls = _make_per_entity_simulation_cls()
        sim = per_entity_sim_cls(
            config=config,
            model=self._gm_model,
            embedder=self._embedder,
            override_agent_model=self._gm_model,  # fallback for any unmapped entity
            override_game_master_model=self._gm_model,
            agent_models=models,
        )
        return _SimWrapper(sim, max_steps=self._max_steps)

    # ------------------------------------------------------------------ #
    # Character-set configuration                                         #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _apply_canonical_overrides(config):
        """Patch the formative-memories initializer with sharpened personas."""
        from concordia.typing import prefab as prefab_lib
        new_instances = []
        for inst in config.instances:
            if (
                inst.role == prefab_lib.Role.INITIALIZER
                and inst.prefab == "formative_memories_initializer__GameMaster"
            ):
                params = dict(inst.params)
                psm = dict(params.get("player_specific_memories", {}))
                psm = _apply_canonical_memory_overrides(psm)
                params["player_specific_memories"] = psm
                # Also update player_specific_context (used as the seed prompt
                # by the initializer GM) to reflect the new memories.
                ctx = dict(params.get("player_specific_context", {}))
                for name, mems in _CANONICAL_MEMORY_OVERRIDES.items():
                    if name in ctx:
                        # Preserve the "Age: NN\n" header if present.
                        old = ctx[name]
                        head = old.split("\n", 1)[0] if old.startswith("Age:") else ""
                        body = "\n".join(mems)
                        ctx[name] = f"{head}\n{body}" if head else body
                params["player_specific_context"] = ctx
                new_instances.append(prefab_lib.InstanceConfig(
                    prefab=inst.prefab, role=inst.role, params=params,
                ))
            else:
                new_instances.append(inst)
        return prefab_lib.Config(
            default_premise=config.default_premise,
            prefabs=config.prefabs,
            instances=new_instances,
        )

    @staticmethod
    def _extend_config_to_8(config):
        """Append the four additional characters and update the initializer GM."""
        from concordia.typing import prefab as prefab_lib

        # Add new entity InstanceConfigs
        new_entities = _additional_instance_configs()
        new_memories = _additional_specific_memories()

        # Find and update the formative_memories_initializer GM
        ages = {
            "Mira Stenholm": _AGE_MIRA,
            "Akira Goto": _AGE_AKIRA,
            "Brenda Calloway": _AGE_BRENDA,
            "Percy Holloway-Wynne": _AGE_PERCY,
        }

        new_instances = []
        for inst in config.instances:
            if (
                inst.role == prefab_lib.Role.INITIALIZER
                and inst.prefab == "formative_memories_initializer__GameMaster"
            ):
                params = dict(inst.params)
                # Extend player_specific_context
                ctx = dict(params.get("player_specific_context", {}))
                for name, age in ages.items():
                    mems = new_memories[name]
                    ctx[name] = f"Age: {age}\n" + "\n".join(mems)
                params["player_specific_context"] = ctx
                # Extend player_specific_memories
                psm = dict(params.get("player_specific_memories", {}))
                psm.update(new_memories)
                params["player_specific_memories"] = psm
                new_instances.append(
                    prefab_lib.InstanceConfig(
                        prefab=inst.prefab, role=inst.role, params=params,
                    )
                )
            else:
                new_instances.append(inst)

        # Insert new entities before the GameMasters
        gm_split = next(
            (i for i, inst in enumerate(new_instances)
             if inst.role != prefab_lib.Role.ENTITY),
            len(new_instances),
        )
        new_instances = (
            new_instances[:gm_split] + new_entities + new_instances[gm_split:]
        )

        return prefab_lib.Config(
            default_premise=config.default_premise,
            prefabs=config.prefabs,
            instances=new_instances,
        )

    @staticmethod
    def _restrict_config_to_2(config):
        """Keep only Silas + Thaddeus among entities (most-differentiated pair)."""
        from concordia.typing import prefab as prefab_lib

        keep = set(get_character_set("canonical_2"))
        new_instances = []
        for inst in config.instances:
            if inst.role == prefab_lib.Role.ENTITY:
                if inst.params.get("name") in keep:
                    new_instances.append(inst)
                # else drop
            elif (
                inst.role == prefab_lib.Role.INITIALIZER
                and inst.prefab == "formative_memories_initializer__GameMaster"
            ):
                params = dict(inst.params)
                ctx = {
                    k: v for k, v in params.get("player_specific_context", {}).items()
                    if k in keep
                }
                psm = {
                    k: v for k, v in params.get("player_specific_memories", {}).items()
                    if k in keep
                }
                params["player_specific_context"] = ctx
                params["player_specific_memories"] = psm
                new_instances.append(
                    prefab_lib.InstanceConfig(
                        prefab=inst.prefab, role=inst.role, params=params,
                    )
                )
            else:
                new_instances.append(inst)

        return prefab_lib.Config(
            default_premise=config.default_premise,
            prefabs=config.prefabs,
            instances=new_instances,
        )

    # ------------------------------------------------------------------ #
    # Reward                                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def reward_fn(final_state: Any, agent_name: str) -> float:
        """No environment reward — training signal is the perception loss only.

        The cultural-emergence experiments do not use environment reward;
        agents are shaped entirely by CCSM's perception/action loss on
        each other's outputs.  Return 0 for all agents.
        """
        return 0.0
