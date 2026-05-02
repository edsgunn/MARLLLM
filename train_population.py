"""
Entry point: population-based CCSM training with multi-environment support.

A population of named agents (e.g. Marcus, Sophia, Viktor, Chen) are randomly
paired for each episode.  Each agent has its own character prompt and either:
  - a LoRA adapter on a shared frozen backbone (--lora-shared-base, recommended)
  - independent full model weights (default)

Character prompts can be a single string or a list of variants that are sampled
per episode — enabling the same character to present differently across games.

Multiple environments can be specified in a YAML config via the ``environments``
key (a weighted list of env specs).  Each env can override character prompts,
including per-env prompt variant lists, so agents experience diverse tasks,
opponents, and personas within a single training run.

YAML ``environments`` format
-----------------------------
    environments:
      - name: short_deal           # identifier for logs/traces
        type: deal_or_no_deal      # env type (see _build_env_from_spec)
        weight: 2.0                # sampling probability (relative)
        dialogue_turns: 6
        token_budget: 48
        character_prompts:         # per-env prompt overrides (str or list[str])
          Marcus: "You are Marcus in a quick trade."
          Sophia:
            - "You are Sophia under time pressure."
            - "You are Sophia, impatient today."
      - name: long_deal
        type: deal_or_no_deal
        weight: 1.0
        dialogue_turns: 14
        token_budget: 128

When ``environments`` is absent, a single DealOrNoDealEnv is built from the
top-level CLI / YAML args (backward compatible with existing configs).

Usage
-----
    uv run python train_population.py --config configs/population_experiments/02_4agent_char_lora_r16.yaml
    uv run python train_population.py --config configs/population_experiments/11_4agent_multienv.yaml
    uv run python train_population.py --model gpt2 --characters Marcus Sophia --lora-r 16 --lora-shared-base
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import torch

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def _auto_device(fallback: str = "cpu") -> str:
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        return "cuda:0"
    return fallback


def _default_character_prompt(name: str) -> str:
    return f"You are {name}, negotiating to maximise your score."


# ── Environment factory ───────────────────────────────────────────────────────

def _build_deal_env(
    tokenizer: Any,
    *,
    dialogue_turns: int = 10,
    token_budget: int = 64,
    env_token_budget: int | None = None,
    max_item_count: int = 5,
    role_shuffle: bool = False,
    seed: int = 42,
) -> Any:
    from envs.deal_or_no_deal_env import DealOrNoDealEnv
    return DealOrNoDealEnv(
        tokenizer=tokenizer,
        max_dialogue_turns=dialogue_turns,
        action_token_budget=token_budget,
        env_token_budget=env_token_budget,
        max_item_count=max_item_count,
        seed=seed,
        role_shuffle=role_shuffle,
    )


def _build_concordia_env(
    spec_dict: dict,
    tokenizer: Any,
    *,
    base_model_name: str,
    base_dtype: Any,
    default_seed: int,
) -> Any:
    """Build a Concordia-backed env from a YAML env-spec dict.

    Required keys:
      ``scenario``  : name of the scenario factory (e.g. ``robotic_athanor``).
    Optional keys:
      ``gm_model``      : HuggingFace model ID for the Concordia GameMaster.
                          Defaults to ``base_model_name`` (loads a frozen
                          copy of the agent's base model).
      ``embedder``      : sentence-transformers model ID.  Defaults to
                          ``sentence-transformers/all-MiniLM-L6-v2``.
      ``character_set`` : passed to scenarios that take it (canonical_2,
                          canonical_4, extended_8 for robotic_athanor).
      ``max_steps``     : per-episode step cap.
      ``token_budget``  : action token budget per turn.
      ``max_turns``     : ConcordiaEnv max_turns hard cap.
    """
    from envs.concordia_env import ConcordiaEnv

    scenario_name = spec_dict.get("scenario") or spec_dict.get("scenario_name")
    if not scenario_name:
        raise ValueError("Concordia env spec missing 'scenario' field.")

    gm_model_id = spec_dict.get("gm_model") or base_model_name
    embedder_id = spec_dict.get("embedder") or "sentence-transformers/all-MiniLM-L6-v2"

    print(f"Concordia: loading GM model {gm_model_id}")
    from concordia.contrib.language_models.huggingface import huggingface_model
    gm_model = huggingface_model.HuggingFaceLanguageModel(
        model_name=gm_model_id,
        api_key=os.environ.get("HF_TOKEN") or None,
        dtype=base_dtype if base_dtype != "auto" else torch.bfloat16,
    )

    print(f"Concordia: loading embedder {embedder_id}")
    from sentence_transformers import SentenceTransformer  # type: ignore
    _st = SentenceTransformer(embedder_id)
    embedder = lambda text: _st.encode(text, convert_to_numpy=True)  # noqa: E731

    scenario_kwargs: dict = {}
    for key in ("character_set", "max_steps", "num_rounds", "num_days", "num_turns"):
        if key in spec_dict:
            scenario_kwargs[key] = spec_dict[key]

    if scenario_name == "robotic_athanor":
        from envs.concordia_robotic_athanor import RoboticAthanorScenario
        scenario = RoboticAthanorScenario(
            gm_model=gm_model,
            embedder=embedder,
            **scenario_kwargs,
        )
        agent_names = list(scenario.agent_names)
        reward_fn = scenario.reward_fn
    else:
        from envs.concordia_scenarios import _SCENARIO_REGISTRY
        if scenario_name not in _SCENARIO_REGISTRY:
            raise ValueError(
                f"Unknown concordia scenario '{scenario_name}'. "
                f"Available: {sorted(_SCENARIO_REGISTRY.keys())}"
            )
        entry = _SCENARIO_REGISTRY[scenario_name]
        cls = entry["cls_factory"]() if "cls_factory" in entry else entry["cls"]
        scenario = cls(gm_model=gm_model, embedder=embedder, **scenario_kwargs)
        agent_names = (
            entry["agent_names"]
            if entry.get("agent_names") is not None
            else list(scenario.agent_names)
        )
        reward_fn = scenario.reward_fn

    env = ConcordiaEnv(
        simulation_factory=scenario,
        agent_names=agent_names,
        tokenizer=tokenizer,
        action_token_budget=spec_dict.get("token_budget", 128),
        max_turns=spec_dict.get("max_turns", 200),
        reward_fn=reward_fn,
    )
    # Stash for downstream: caller may need agent_names from the scenario
    env._scenario_agent_names = agent_names  # type: ignore[attr-defined]
    return env


def _build_env_from_spec(
    spec_dict: dict,
    tokenizer: Any,
    default_seed: int,
    *,
    base_model_name: str = "",
    base_dtype: Any = "auto",
) -> Any:
    """
    Build a single environment from a YAML env-spec dict.

    Supported types
    ---------------
    ``deal_or_no_deal``
        Keys: dialogue_turns, token_budget, max_item_count, role_shuffle.
    ``concordia``
        Keys: scenario (required), gm_model, embedder, character_set,
        token_budget, max_turns, plus scenario-specific kwargs.
    """
    env_type = spec_dict.get("type", "deal_or_no_deal")

    if env_type == "deal_or_no_deal":
        return _build_deal_env(
            tokenizer,
            dialogue_turns=spec_dict.get("dialogue_turns", 10),
            token_budget=spec_dict.get("token_budget", 64),
            env_token_budget=spec_dict.get("env_token_budget", None),
            max_item_count=spec_dict.get("max_item_count", 5),
            role_shuffle=spec_dict.get("role_shuffle", False),
            seed=spec_dict.get("seed", default_seed),
        )

    if env_type == "concordia":
        return _build_concordia_env(
            spec_dict,
            tokenizer,
            base_model_name=base_model_name,
            base_dtype=base_dtype,
            default_seed=default_seed,
        )

    raise ValueError(
        f"Unknown env type {env_type!r} in environments spec. "
        f"Supported: 'deal_or_no_deal', 'concordia'."
    )


def _build_environment_specs(
    args: argparse.Namespace,
    tokenizer: Any,
) -> list:
    """
    Build a list of EnvironmentSpec objects.

    If ``args.environments`` (from YAML) is a list of dicts, build one spec per
    entry and return them.  Otherwise build a single spec from the top-level args.
    """
    from marlllm.population import EnvironmentSpec

    yaml_envs = getattr(args, "environments", None)

    if yaml_envs and isinstance(yaml_envs, list):
        specs: list[EnvironmentSpec] = []
        for entry in yaml_envs:
            if not isinstance(entry, dict):
                raise ValueError(
                    f"Each entry in 'environments' must be a dict, got {type(entry)}"
                )
            env = _build_env_from_spec(
                entry, tokenizer, default_seed=args.seed,
                base_model_name=args.model, base_dtype=args.dtype,
            )
            # character_prompts in the spec dict (str or list[str] values)
            char_prompts = entry.get("character_prompts", {})
            specs.append(EnvironmentSpec(
                env=env,
                name=entry.get("name", f"env_{len(specs)}"),
                weight=float(entry.get("weight", 1.0)),
                character_prompts=char_prompts,
            ))
        return specs

    # Fallback: single env from top-level args.
    env = _build_deal_env(
        tokenizer,
        dialogue_turns=args.dialogue_turns,
        token_budget=args.token_budget,
        env_token_budget=getattr(args, "env_token_budget", None),
        role_shuffle=args.role_shuffle,
        seed=args.seed,
    )
    return [EnvironmentSpec(env=env, name="deal_or_no_deal")]


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    pre_args, _ = pre.parse_known_args()

    p = argparse.ArgumentParser(
        description="Population-based CCSM training with multi-environment support.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--config", default=None,
                   help="Path to a YAML experiment config file. "
                        "All keys map to CLI flags; explicit CLI args override.")

    # ── Model ─────────────────────────────────────────────────────────────
    p.add_argument("--model", default="gpt2",
                   help="HuggingFace model ID for all agents / shared backbone.")
    p.add_argument("--device", default=None,
                   help="PyTorch device. Defaults to cuda:0 if available.")
    p.add_argument("--dtype", default="auto",
                   help="torch_dtype: auto, float32, bfloat16, float16.")
    p.add_argument("--device-map", default=None,
                   help="device_map for from_pretrained, e.g. 'auto'.")
    p.add_argument("--attn-impl", default=None,
                   help="attn_implementation for from_pretrained, e.g. 'eager'.")
    p.add_argument("--compile", action="store_true",
                   help="torch.compile() the backbone.")

    # ── Population / character prompts ────────────────────────────────────
    p.add_argument("--characters", nargs="+", metavar="NAME",
                   help="Character names with default prompts. "
                        "Overridden by --character and character_prompts in YAML.")
    p.add_argument("--character", action="append", metavar="NAME:PROMPT",
                   dest="character_specs",
                   help="Character name and prompt as 'NAME:PROMPT'. Repeatable.")
    p.add_argument("--pairing-strategy", default="random_no_self",
                   choices=["random_no_self", "random_with_self", "round_robin"],
                   help="How to pair agents each episode.")

    # ── LoRA ──────────────────────────────────────────────────────────────
    p.add_argument("--lora-shared-base", action="store_true",
                   help="One shared backbone; each character gets its own LoRA adapter.")
    p.add_argument("--lora-r",     type=int, default=0,
                   help="LoRA rank. 0 = full fine-tuning.")
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-modules", default=None,
                   help="Comma-separated module names for LoRA. None = auto-detect.")

    # ── Training ──────────────────────────────────────────────────────────
    p.add_argument("--iters",          type=int,   default=500)
    p.add_argument("--rollouts",       type=int,   default=8,
                   help="Episodes per iteration (drawn from the env mixture).")
    p.add_argument("--lr",             type=float, default=3e-5)
    p.add_argument("--kl-coef",        type=float, default=0.0)
    p.add_argument("--beta",           type=float, default=0.01,
                   help="Entropy regularisation coefficient (default 0.01). "
                        "Set to 0.0 to disable the entropy bonus entirely.")
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--grad-accum",     type=int,   default=8)
    p.add_argument("--gradient-checkpointing", action="store_true")

    # ── Default single-env options (used when 'environments' key is absent) ─
    p.add_argument("--dialogue-turns",     type=int, default=10)
    p.add_argument("--token-budget",       type=int, default=64,
                   help="Generation budget per turn (includes thinking tokens).")
    p.add_argument("--env-token-budget",   type=int, default=None,
                   help="Communication budget: max tokens passed to the env after "
                        "thinking tokens are stripped. None = no stripping (default). "
                        "Should be <= --token-budget.")
    p.add_argument("--max-episode-tokens", type=int, default=1024)
    p.add_argument("--role-shuffle", action="store_true",
                   help="Randomly assign who goes first (applies to default env).")

    # ── Output / checkpointing ────────────────────────────────────────────
    p.add_argument("--log-every",        type=int, default=10)
    p.add_argument("--checkpoint-every", type=int, default=100)
    p.add_argument("--num-checkpoint-traces", type=int, default=4,
                   help="Number of episode traces to dump per checkpoint.")
    p.add_argument("--snapshot-eval-path", default=None,
                   help="JSON file with held-out forum contexts for "
                        "behavioural-distribution snapshots at each checkpoint.")
    p.add_argument("--snapshot-samples-per-context", type=int, default=8)
    p.add_argument("--snapshot-max-new-tokens", type=int, default=128)
    p.add_argument("--output-dir", default="runs/population")
    p.add_argument("--resume", action="store_true",
                   help="Resume from latest checkpoint.")
    p.add_argument("--resume-from", default=None,
                   help="Resume from a specific checkpoint dir or .pt file. "
                        "Overrides --resume's auto-discovery.")

    if pre_args.config:
        from marlllm.config_loader import apply_config_defaults
        apply_config_defaults(p, pre_args.config)

    return p.parse_args()


# ── Character prompt resolution ───────────────────────────────────────────────

def _build_character_prompts(args: argparse.Namespace) -> dict[str, str | list[str]]:
    """
    Resolve character prompts from all sources (priority highest → lowest):
      1. --character NAME:PROMPT flags
      2. character_prompts dict from YAML (str or list[str] per character)
      3. --characters NAME list with default "You are NAME, ..." prompts
      4. Fallback: Marcus + Sophia with default prompts
    """
    prompts: dict[str, str | list[str]] = {}

    if args.characters:
        for name in args.characters:
            prompts[name] = _default_character_prompt(name)

    yaml_prompts = getattr(args, "character_prompts", None)
    if isinstance(yaml_prompts, dict):
        prompts.update(yaml_prompts)

    for spec in (args.character_specs or []):
        if ":" not in spec:
            raise ValueError(
                f"--character must be 'NAME:PROMPT', got: {spec!r}"
            )
        name, _, prompt = spec.partition(":")
        prompts[name.strip()] = prompt.strip()

    if not prompts:
        for name in ["Marcus", "Sophia"]:
            prompts[name] = _default_character_prompt(name)

    return prompts


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    from marlllm import (
        CCSMLoss,
        IndependentAgent,
        LoRASharedBaseAgent,
        OnPolicyStore,
        TextTokeniser,
        TrainingConfig,
    )
    from marlllm.config_loader import log_system_info
    from marlllm.population import EnvironmentSpec, PopulationTrainer

    character_prompts = _build_character_prompts(args)
    print(f"Population: {list(character_prompts.keys())} ({len(character_prompts)} agents)")

    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    torch_dtype = dtype_map.get(args.dtype, "auto")
    device = args.device or _auto_device()

    lora_modules = (
        [m.strip() for m in args.lora_modules.split(",")]
        if args.lora_modules else None
    )

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    log_system_info(args.output_dir, config_path=args.config)

    # ── Build population ──────────────────────────────────────────────────
    population: dict[str, object] = {}

    if args.lora_shared_base:
        if args.lora_r <= 0:
            raise ValueError("--lora-shared-base requires --lora-r > 0.")

        try:
            from peft import LoraConfig, get_peft_model
        except ImportError as e:
            raise ImportError(
                "--lora-shared-base requires the `peft` package: uv add peft"
            ) from e
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from marlllm.agent import _find_lora_target_modules

        print(f"Loading shared base model: {args.model}  →  {device}")
        load_kwargs: dict = {}
        if torch_dtype != "auto":
            load_kwargs["torch_dtype"] = torch_dtype
        if args.device_map is not None:
            load_kwargs["device_map"] = args.device_map
        if args.attn_impl is not None:
            load_kwargs["attn_implementation"] = args.attn_impl

        base_model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
        if args.device_map is None:
            base_model = base_model.to(device)

        tokenizer = AutoTokenizer.from_pretrained(args.model)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        target_modules = lora_modules or _find_lora_target_modules(base_model)
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=0.0,
            bias="none",
        )

        keep_ref = args.kl_coef > 0.0
        names = list(character_prompts.keys())

        print(f"Attaching LoRA adapter '{names[0]}' (r={args.lora_r})")
        peft_model = get_peft_model(base_model, lora_config, adapter_name=names[0])
        for name in names[1:]:
            print(f"Attaching LoRA adapter '{name}' (r={args.lora_r})")
            peft_model.add_adapter(name, lora_config)

        for name, prompt in character_prompts.items():
            # For LoRA agents, store the first prompt variant as the character_prompt
            # (used by the agent object itself; training uses per-episode sampled prompts).
            prompt_str = prompt[0] if isinstance(prompt, list) else prompt
            population[name] = LoRASharedBaseAgent(
                agent_id=name,
                character_prompt=prompt_str,
                shared_backbone=peft_model,
                adapter_name=name,
                tokenizer=tokenizer,
                device=device,
                keep_ref_model=keep_ref,
            )

        print(f"LoRA shared base: 1 backbone, {len(names)} independent adapters.")

    else:
        keep_ref = args.kl_coef > 0.0
        for name, prompt in character_prompts.items():
            prompt_str = prompt[0] if isinstance(prompt, list) else prompt
            print(f"Loading agent '{name}': {args.model}  →  {device}")
            population[name] = IndependentAgent(
                agent_id=name,
                character_prompt=prompt_str,
                model_name_or_path=args.model,
                device=device,
                torch_dtype=torch_dtype,
                device_map=args.device_map,
                keep_ref_model=keep_ref,
                gradient_checkpointing=args.gradient_checkpointing,
                lora_r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_target_modules=lora_modules,
                compile_model=args.compile,
                attn_implementation=args.attn_impl,
            )

    first_agent = list(population.values())[0]

    # ── Build environment specs ───────────────────────────────────────────
    env_specs = _build_environment_specs(args, first_agent.tokenizer)

    if len(env_specs) == 1:
        print(f"Environment: {env_specs[0].name}")
    else:
        print(f"Environments ({len(env_specs)}):")
        total_w = sum(s.weight for s in env_specs)
        for spec in env_specs:
            pct = 100 * spec.weight / total_w
            print(f"  {spec.name:30s}  weight={spec.weight:.2f}  ({pct:.0f}%)")

    # Flatten character_prompts to str for TrainingConfig (list variants are
    # handled inside PopulationTrainer via _sample_prompt; TrainingConfig just
    # needs a representative string for logging / trace headers).
    flat_prompts = {
        name: (p[0] if isinstance(p, list) else p)
        for name, p in character_prompts.items()
    }

    config = TrainingConfig(
        model_name_or_path=args.model,
        character_prompts=character_prompts,  # may contain list[str] variants
        episodes_per_iter=args.rollouts,
        max_episode_tokens=args.max_episode_tokens,
        num_iterations=args.iters,
        lr=args.lr,
        log_every=args.log_every,
        checkpoint_every=args.checkpoint_every,
        num_checkpoint_traces=args.num_checkpoint_traces,
        snapshot_eval_path=args.snapshot_eval_path,
        snapshot_samples_per_context=args.snapshot_samples_per_context,
        snapshot_max_new_tokens=args.snapshot_max_new_tokens,
        output_dir=args.output_dir,
        device=device,
        seed=args.seed,
        kl_coef=args.kl_coef,
        beta=args.beta,
        grad_accum_steps=args.grad_accum,
        gradient_checkpointing=args.gradient_checkpointing,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_target_modules=lora_modules,
    )

    tokeniser = TextTokeniser(first_agent.tokenizer)
    loss      = CCSMLoss()
    store     = OnPolicyStore()

    trainer = PopulationTrainer(
        population=population,
        environments=env_specs,
        loss=loss,
        tokeniser=tokeniser,
        store=store,
        config=config,
        pairing_strategy=args.pairing_strategy,
    )

    start_iteration = 1
    resume_path = None
    if args.resume_from:
        resume_path = Path(args.resume_from)
    elif args.resume:
        ckpt_root = Path(args.output_dir) / "checkpoints"
        latest_dir = ckpt_root / "latest"
        latest_pt = ckpt_root / "latest.pt"
        if latest_dir.exists():
            resume_path = latest_dir
        elif latest_pt.exists():
            resume_path = latest_pt
        else:
            print("No checkpoint found, starting from scratch.")

    if resume_path is not None:
        if not resume_path.exists():
            sys.exit(f"--resume-from path does not exist: {resume_path}")
        start_iteration = trainer.load_checkpoint(str(resume_path)) + 1
        print(f"Resuming from {resume_path} (iteration {start_iteration})")

    print(f"Output directory: {args.output_dir}")
    print()
    trainer.train(start_iteration=start_iteration)


if __name__ == "__main__":
    main()
