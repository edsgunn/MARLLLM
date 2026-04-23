"""
Entry point: N-agent CCSM training with a Concordia environment.

The scenario is selected via --scenario (or the `scenario:` key in a YAML
config).  Built-in scenarios mirror the five NeurIPS 2024 Concordia Contest
environments plus a dependency-free round-robin baseline.  Any custom
Concordia setup can be plugged in with --factory-module.

Scenarios
---------
  round_robin        — Simple round-robin chat (no Concordia install needed)
  dialogue           — Free-form dialogue using Concordia's dialogic GM
  haggling           — Bilateral price negotiation in "Fruitville"
  pub_coordination   — Coordination under incomplete information
  labor              — Collective action / strike dynamics
  reality_show       — Structured mini-games (PD, Chicken, Stag Hunt)
  state_formation    — Alliance diplomacy and public goods
  custom             — Load a factory class from --factory-module

GM model
--------
All Concordia scenarios except round_robin require a language model for the
GameMaster.  Pass --gm-model with a HuggingFace model ID; the script wraps it
with a minimal Concordia LanguageModel adapter.  For API-backed models (e.g.
OpenAI), implement the adapter yourself and use --factory-module.

Semantic memory (used by most contest scenarios) requires a sentence-
transformer embedder — pass --embedder with a model ID (e.g.
"sentence-transformers/all-MiniLM-L6-v2").

Usage examples
--------------
    # Smoke-test with no extra dependencies:
    uv run python train_concordia.py --scenario round_robin \\
        --model Qwen/Qwen2.5-1.5B --agents Alice,Bob --num-turns 8

    # Haggling with a GM LLM:
    uv run python train_concordia.py --scenario haggling \\
        --model Qwen/Qwen2.5-1.5B \\
        --gm-model google/gemma-2-9b \\
        --embedder sentence-transformers/all-MiniLM-L6-v2 \\
        --agents merchant_0,merchant_1 --device cuda

    # Custom scenario from a local Python file:
    uv run python train_concordia.py --scenario custom \\
        --factory-module path/to/my_scenario.py:MyScenarioFactory \\
        --agents Alice,Bob,Carol

    # From a YAML config:
    uv run python train_concordia.py --config configs/experiments/concordia_haggling.yaml
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import torch

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _auto_device(idx: int, fallback: str) -> str:
    if torch.cuda.is_available() and torch.cuda.device_count() > idx:
        return f"cuda:{idx}"
    return fallback


def _assign_devices(agent_names: list[str], default: str | None) -> dict[str, str]:
    """Round-robin GPU assignment across agents when no default is given."""
    if default:
        return {a: default for a in agent_names}
    n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    result: dict[str, str] = {}
    for i, name in enumerate(agent_names):
        result[name] = f"cuda:{i % n_gpu}" if n_gpu > 0 else "cpu"
    return result


def _load_factory_from_module(spec: str) -> Any:
    """Load a factory class/function from 'path/to/module.py:ClassName'."""
    if ":" not in spec:
        sys.exit(
            f"--factory-module must be in the form 'path/to/module.py:FactoryClass', "
            f"got: {spec!r}"
        )
    module_path, obj_name = spec.rsplit(":", 1)
    import importlib.util

    path = Path(module_path)
    if not path.exists():
        sys.exit(f"--factory-module: file not found: {path}")
    mod_spec = importlib.util.spec_from_file_location("_user_factory", path)
    mod = importlib.util.module_from_spec(mod_spec)  # type: ignore[arg-type]
    mod_spec.loader.exec_module(mod)  # type: ignore[union-attr]
    if not hasattr(mod, obj_name):
        sys.exit(f"--factory-module: {path} has no attribute {obj_name!r}")
    return getattr(mod, obj_name)


def _load_reward_fn(spec: str | None) -> Any:
    """Load a reward function from 'path/to/module.py:fn_name', or return None."""
    if spec is None:
        return None
    return _load_factory_from_module(spec)


# _HFLanguageModel removed — use concordia.contrib.language_models.huggingface
# HuggingFaceLanguageModel directly (loaded in _build_scenario below).


# ---------------------------------------------------------------------------
# Scenario builders
# ---------------------------------------------------------------------------

_DEFAULT_AGENTS: dict[str, list[str]] = {
    "round_robin":      ["agent_0", "agent_1"],
    "dialogue":         ["agent_0", "agent_1"],
    "haggling":         ["merchant_0", "merchant_1"],
    "pub_coordination": ["alice", "bob", "carol", "dave"],
    "labor":            ["worker_0", "worker_1", "worker_2", "worker_3"],
    "reality_show":     ["player_0", "player_1", "player_2", "player_3"],
    "state_formation":  ["villager_0", "villager_1", "villager_2", "villager_3"],
}


def _build_scenario(args: argparse.Namespace) -> tuple[Any, list[str], Any]:
    """Return (scenario_factory, agent_names, reward_fn) from parsed args."""
    from envs.concordia_scenarios import (
        FreeDialogueScenario,
        HagglingScenario,
        LaborCollectiveActionScenario,
        PubCoordinationScenario,
        RealityShowScenario,
        RoundRobinScenario,
        StateFormationScenario,
    )

    scenario_name: str = args.scenario
    agent_names: list[str] = (
        [a.strip() for a in args.agents.split(",")]
        if args.agents
        else _DEFAULT_AGENTS.get(scenario_name, ["agent_0", "agent_1"])
    )

    reward_fn = _load_reward_fn(getattr(args, "reward_fn_module", None))

    # ── round_robin ───────────────────────────────────────────────────────
    if scenario_name == "round_robin":
        scenario = RoundRobinScenario(
            context=args.context,
            num_turns=args.num_turns,
        )
        return scenario, agent_names, reward_fn

    # ── All other scenarios need a GM LLM ─────────────────────────────────
    if not args.gm_model:
        sys.exit(
            f"--gm-model is required for scenario '{scenario_name}'.  "
            "Specify a HuggingFace model ID, or use --scenario round_robin "
            "for a dependency-free baseline."
        )

    from concordia.contrib.language_models.huggingface import huggingface_model

    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    gm_dtype = dtype_map.get(args.dtype, torch.bfloat16)
    print(f"Loading GM model: {args.gm_model}")
    gm_model = huggingface_model.HuggingFaceLanguageModel(
        model_name=args.gm_model,
        api_key=os.environ.get("HF_TOKEN") or None,
        dtype=gm_dtype,
    )

    embedder = None
    if args.embedder:
        from sentence_transformers import SentenceTransformer  # type: ignore

        print(f"Loading embedder: {args.embedder}")
        _st = SentenceTransformer(args.embedder)
        embedder = lambda text: _st.encode(text, convert_to_numpy=True)  # noqa: E731

    # ── dialogue ─────────────────────────────────────────────────────────
    if scenario_name == "dialogue":
        scenario = FreeDialogueScenario(
            gm_model=gm_model,
            embedder=embedder,
            context=args.context,
            num_turns=args.num_turns,
        )
        return scenario, agent_names, reward_fn

    # ── contest scenarios ─────────────────────────────────────────────────
    contest_map = {
        "haggling":         (HagglingScenario,              {"num_rounds":  args.num_turns}),
        "pub_coordination": (PubCoordinationScenario,        {"num_players": len(agent_names)}),
        "labor":            (LaborCollectiveActionScenario,  {"num_days":    args.num_turns}),
        "reality_show":     (RealityShowScenario,            {"num_rounds":  args.num_turns}),
        "state_formation":  (StateFormationScenario,         {}),
    }

    if scenario_name in contest_map:
        cls, extra_kwargs = contest_map[scenario_name]
        scenario = cls(gm_model=gm_model, embedder=embedder, **extra_kwargs)
        effective_reward_fn = reward_fn or cls.reward_fn
        return scenario, agent_names, effective_reward_fn

    # ── custom ────────────────────────────────────────────────────────────
    if scenario_name == "custom":
        if not args.factory_module:
            sys.exit("--factory-module is required when --scenario custom is set.")
        factory_cls = _load_factory_from_module(args.factory_module)
        scenario = factory_cls() if callable(factory_cls) and not isinstance(factory_cls, type) else factory_cls
        return scenario, agent_names, reward_fn

    sys.exit(
        f"Unknown scenario: {scenario_name!r}.  "
        f"Choose from: {', '.join(list(_DEFAULT_AGENTS) + ['custom'])}"
    )


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    pre_args, _ = pre.parse_known_args()

    p = argparse.ArgumentParser(
        description="N-agent CCSM training with a Concordia environment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Config file ───────────────────────────────────────────────────────
    p.add_argument(
        "--config", default=None,
        help="Path to a YAML experiment config. All keys map to CLI flags; "
             "explicit CLI args override config file values.",
    )

    # ── Scenario ──────────────────────────────────────────────────────────
    p.add_argument(
        "--scenario", default="round_robin",
        choices=[
            "round_robin", "dialogue", "haggling", "pub_coordination",
            "labor", "reality_show", "state_formation", "custom",
        ],
        help="Which Concordia scenario to train in (default: round_robin).",
    )
    p.add_argument(
        "--agents", default=None,
        help="Comma-separated agent names, e.g. 'Alice,Bob,Carol'. "
             "Defaults vary per scenario (see _DEFAULT_AGENTS in this file).",
    )
    p.add_argument(
        "--num-turns", type=int, default=10,
        help="Scenario length: rounds for contest scenarios, acts for round_robin.",
    )
    p.add_argument(
        "--context", default="You are participating in a multi-agent discussion.",
        help="Opening observation sent to all agents at episode start "
             "(used by round_robin and dialogue scenarios).",
    )
    p.add_argument(
        "--factory-module", default=None, metavar="MODULE:CLASS",
        help="Load a custom scenario factory from a Python file, e.g. "
             "'my_scenarios.py:MyFactory'.  Required when --scenario custom.",
    )
    p.add_argument(
        "--reward-fn-module", default=None, metavar="MODULE:FN",
        help="Load a custom reward function from a Python file, e.g. "
             "'my_rewards.py:reward_fn'.  Signature: (final_state, agent_name) -> float.",
    )

    # ── GM / embedder ─────────────────────────────────────────────────────
    p.add_argument(
        "--gm-model", default=None,
        help="HuggingFace model ID for the Concordia GameMaster LLM. "
             "Not required for --scenario round_robin.",
    )
    p.add_argument(
        "--gm-device", default=None,
        help="Device for the GM model (defaults to next available CUDA device "
             "after the trained agents, or cpu).",
    )
    p.add_argument(
        "--embedder", default=None,
        help="sentence-transformers model ID for Concordia associative memory "
             "(e.g. sentence-transformers/all-MiniLM-L6-v2).  Required by "
             "contest scenarios that use semantic memory retrieval.",
    )

    # ── Trained agent model ───────────────────────────────────────────────
    p.add_argument("--model",      default="gpt2", help="HuggingFace model for all trained agents.")
    p.add_argument("--device",     default=None,   help="Default device for all agents (overridden per-agent by GPU auto-assign if omitted).")
    p.add_argument("--dtype",      default="auto", help="torch_dtype: auto, float32, bfloat16, float16.")
    p.add_argument("--device-map", default=None,   help="device_map for from_pretrained, e.g. 'auto'.")
    p.add_argument(
        "--prompt", default="",
        help="Character prompt prepended to every agent's context. "
             "Leave empty to let the scenario define each agent's persona.",
    )
    p.add_argument(
        "--attn-impl", default=None,
        help="attn_implementation for from_pretrained (e.g. 'eager' for GPT-2).",
    )

    # ── Training loop ─────────────────────────────────────────────────────
    p.add_argument("--iters",                 type=int,   default=500,    help="Training iterations.")
    p.add_argument("--rollouts",              type=int,   default=8,      help="Episodes per iteration.")
    p.add_argument("--lr",                    type=float, default=3e-5,   help="AdamW learning rate.")
    p.add_argument("--token-budget",          type=int,   default=128,    help="Tokens per agent turn.")
    p.add_argument("--max-episode-tokens",    type=int,   default=2048,   help="Token cap per episode.")
    p.add_argument("--max-env-turns",         type=int,   default=200,    help="ConcordiaEnv max_turns (hard truncation).")
    p.add_argument("--log-every",             type=int,   default=10,     help="Log every N iterations.")
    p.add_argument("--checkpoint-every",      type=int,   default=100,    help="Checkpoint every N iterations.")
    p.add_argument("--output-dir",            default="runs/concordia",   help="Log and checkpoint directory.")
    p.add_argument("--resume",  action="store_true",                      help="Resume from latest checkpoint.")
    p.add_argument("--kl-coef", type=float,   default=0.0,                help="KL penalty coefficient.")
    p.add_argument("--seed",    type=int,     default=42)

    # ── Memory / efficiency ───────────────────────────────────────────────
    p.add_argument("--grad-accum",            type=int, default=8,        help="Gradient accumulation steps.")
    p.add_argument("--gradient-checkpointing",action="store_true",        help="Activation checkpointing.")
    p.add_argument("--compile",               action="store_true",        help="torch.compile() the backbone.")

    # ── LoRA ──────────────────────────────────────────────────────────────
    p.add_argument("--lora-r",       type=int, default=0,                 help="LoRA rank (0 = full fine-tuning).")
    p.add_argument("--lora-alpha",   type=int, default=16,                help="LoRA alpha scaling factor.")
    p.add_argument("--lora-modules", default=None,                        help="Comma-separated LoRA target modules.")

    if pre_args.config:
        from marlllm.config_loader import apply_config_defaults
        apply_config_defaults(p, pre_args.config)

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    from marlllm import (
        CCSMLoss,
        IndependentAgent,
        OnPolicyStore,
        TextTokeniser,
        Trainer,
        TrainingConfig,
    )
    from marlllm.config_loader import log_system_info
    from envs.concordia_env import ConcordiaEnv

    dtype_map = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    torch_dtype = dtype_map.get(args.dtype, "auto")
    lora_modules = (
        [m.strip() for m in args.lora_modules.split(",")]
        if args.lora_modules
        else None
    )

    # ── Build scenario ────────────────────────────────────────────────────
    print(f"Scenario : {args.scenario}")
    scenario_factory, agent_names, reward_fn = _build_scenario(args)
    print(f"Agents   : {agent_names}")

    # ── Output dir + system info ──────────────────────────────────────────
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    log_system_info(args.output_dir, config_path=args.config)

    # ── Load trained agents ───────────────────────────────────────────────
    devices = _assign_devices(agent_names, args.device)
    character_prompts: dict[str, str] = {}
    agents: dict[str, Any] = {}

    for agent_name in agent_names:
        dev = devices[agent_name]
        print(f"Loading agent '{agent_name}': {args.model}  →  {dev}")
        ag = IndependentAgent(
            agent_id=agent_name,
            character_prompt=args.prompt,
            model_name_or_path=args.model,
            device=dev,
            torch_dtype=torch_dtype,
            device_map=args.device_map,
            keep_ref_model=args.kl_coef > 0.0,
            gradient_checkpointing=args.gradient_checkpointing,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_target_modules=lora_modules,
            compile_model=args.compile,
            attn_implementation=args.attn_impl,
        )
        agents[agent_name] = ag
        character_prompts[agent_name] = args.prompt

    # ── Build env ─────────────────────────────────────────────────────────
    primary_agent = agents[agent_names[0]]
    print(
        f"Building ConcordiaEnv "
        f"(token_budget={args.token_budget}, max_turns={args.max_env_turns})"
    )
    env = ConcordiaEnv(
        simulation_factory=scenario_factory,
        agent_names=agent_names,
        tokenizer=primary_agent.tokenizer,
        action_token_budget=args.token_budget,
        max_turns=args.max_env_turns,
        reward_fn=reward_fn,
    )

    # ── Training config ───────────────────────────────────────────────────
    config = TrainingConfig(
        model_name_or_path=args.model,
        character_prompts=character_prompts,
        episodes_per_iter=args.rollouts,
        max_episode_tokens=args.max_episode_tokens,
        num_iterations=args.iters,
        lr=args.lr,
        log_every=args.log_every,
        checkpoint_every=args.checkpoint_every,
        output_dir=args.output_dir,
        device=devices[agent_names[0]],
        seed=args.seed,
        kl_coef=args.kl_coef,
        grad_accum_steps=args.grad_accum,
        gradient_checkpointing=args.gradient_checkpointing,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_target_modules=lora_modules,
    )

    tokeniser = TextTokeniser(primary_agent.tokenizer)
    loss      = CCSMLoss()
    store     = OnPolicyStore()

    trainer = Trainer(
        agents=agents,
        env=env,
        loss=loss,
        tokeniser=tokeniser,
        store=store,
        config=config,
    )

    # ── Resume ────────────────────────────────────────────────────────────
    start_iteration = 1
    if args.resume:
        latest = Path(args.output_dir) / "checkpoints" / "latest.pt"
        if latest.exists():
            start_iteration = trainer.load_checkpoint(str(latest)) + 1
            print(f"Resuming from iteration {start_iteration}")
        else:
            print("No checkpoint found, starting from scratch.")

    print(f"Output directory: {args.output_dir}")
    print()
    trainer.train(start_iteration=start_iteration)


if __name__ == "__main__":
    main()
