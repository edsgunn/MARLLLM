"""
Entry point: population-based CCSM training with DealOrNoDealEnv.

A population of named agents (e.g. Marcus, Sophia, Viktor, Chen) are randomly
paired for each episode.  Each agent has its own character prompt and either:
  - a LoRA adapter on a shared frozen backbone (--lora-shared-base, recommended)
  - independent full model weights (default)

Character prompts are loaded from a YAML config's ``character_prompts`` dict
(keys = names, values = prompt strings).  On the CLI you can pass prompts as
``--character NAME:PROMPT`` pairs; bare ``--characters NAME [NAME ...]`` uses
default "Agent NAME" prompts.

Usage:
    uv run python train_population.py --config configs/population_experiments/01_2agent_char_lora_r16.yaml
    uv run python train_population.py --model Qwen/Qwen2.5-1.5B --lora-r 16 --lora-shared-base \\
        --character "Marcus:You are Marcus..." --character "Sophia:You are Sophia..."
    uv run python train_population.py --model gpt2 --characters Marcus Sophia Viktor Chen --iters 200
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def _auto_device(fallback: str = "cpu") -> str:
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        return "cuda:0"
    return fallback


def _default_character_prompt(name: str) -> str:
    return f"You are {name}, negotiating to maximise your score."


def parse_args() -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    pre_args, _ = pre.parse_known_args()

    p = argparse.ArgumentParser(
        description="Population-based CCSM training on DealOrNoDealEnv.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", default=None,
                   help="Path to a YAML experiment config file. "
                        "All keys map to CLI flags; explicit CLI args override.")

    # ── Model ─────────────────────────────────────────────────────────────
    p.add_argument("--model", default="gpt2",
                   help="HuggingFace model ID used for all agents (or for the "
                        "shared backbone when --lora-shared-base is set).")
    p.add_argument("--device", default=None,
                   help="PyTorch device for the model. Defaults to cuda:0 if available.")
    p.add_argument("--dtype", default="auto",
                   help="torch_dtype: auto, float32, bfloat16, float16.")
    p.add_argument("--device-map", default=None,
                   help="device_map for from_pretrained, e.g. 'auto'.")
    p.add_argument("--attn-impl", default=None,
                   help="attn_implementation for from_pretrained, e.g. 'eager' for GPT-2.")
    p.add_argument("--compile", action="store_true",
                   help="torch.compile() the backbone.")

    # ── Population / character prompts ────────────────────────────────────
    p.add_argument("--characters", nargs="+", metavar="NAME",
                   help="Character names with default prompts ('You are NAME, ...'). "
                        "Overridden by --character and character_prompts in YAML.")
    p.add_argument("--character", action="append", metavar="NAME:PROMPT",
                   dest="character_specs",
                   help="Character name and prompt as 'NAME:PROMPT'. "
                        "Can be repeated for multiple characters.")
    p.add_argument("--pairing-strategy", default="random_no_self",
                   choices=["random_no_self", "random_with_self", "round_robin"],
                   help="How to pair agents each episode (default: random_no_self).")

    # ── LoRA ──────────────────────────────────────────────────────────────
    p.add_argument("--lora-shared-base", action="store_true",
                   help="One shared frozen backbone; each character gets its own "
                        "independent LoRA adapter. Requires --lora-r > 0.")
    p.add_argument("--lora-r", type=int, default=0,
                   help="LoRA rank. 0 = full fine-tuning (no LoRA).")
    p.add_argument("--lora-alpha", type=int, default=16,
                   help="LoRA alpha scaling factor.")
    p.add_argument("--lora-modules", default=None,
                   help="Comma-separated linear module names for LoRA. "
                        "None = PEFT auto-detect.")

    # ── Training ──────────────────────────────────────────────────────────
    p.add_argument("--iters",          type=int,   default=500,
                   help="Training iterations.")
    p.add_argument("--rollouts",       type=int,   default=8,
                   help="Episodes per iteration.")
    p.add_argument("--lr",             type=float, default=3e-5)
    p.add_argument("--kl-coef",        type=float, default=0.0)
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--grad-accum",     type=int,   default=8,
                   help="Gradient accumulation steps.")
    p.add_argument("--gradient-checkpointing", action="store_true",
                   help="Activation checkpointing to reduce VRAM.")

    # ── Environment ───────────────────────────────────────────────────────
    p.add_argument("--dialogue-turns",     type=int, default=10)
    p.add_argument("--token-budget",       type=int, default=64)
    p.add_argument("--max-episode-tokens", type=int, default=1024)
    p.add_argument("--role-shuffle", action="store_true",
                   help="Randomly assign which agent goes first each episode.")

    # ── Output / checkpointing ────────────────────────────────────────────
    p.add_argument("--log-every",        type=int, default=10)
    p.add_argument("--checkpoint-every", type=int, default=100)
    p.add_argument("--output-dir", default="runs/population",
                   help="Log and checkpoint directory.")
    p.add_argument("--resume", action="store_true",
                   help="Resume from latest checkpoint.")

    if pre_args.config:
        from marlllm.config_loader import apply_config_defaults
        apply_config_defaults(p, pre_args.config)

    return p.parse_args()


def _build_character_prompts(args: argparse.Namespace) -> dict[str, str]:
    """
    Merge character prompt sources in priority order:
      1. explicit --character NAME:PROMPT flags  (highest)
      2. character_prompts dict from YAML (via argparse default injection)
      3. --characters NAME list with default prompts
      4. fallback: {"Marcus": default, "Sophia": default}
    """
    prompts: dict[str, str] = {}

    # --characters gives names with default prompts (lowest priority)
    if args.characters:
        for name in args.characters:
            prompts[name] = _default_character_prompt(name)

    # character_prompts from YAML config (set as a dict via apply_config_defaults)
    yaml_prompts = getattr(args, "character_prompts", None)
    if isinstance(yaml_prompts, dict):
        prompts.update(yaml_prompts)

    # --character NAME:PROMPT flags override everything
    for spec in (args.character_specs or []):
        if ":" not in spec:
            raise ValueError(
                f"--character must be in NAME:PROMPT format, got: {spec!r}"
            )
        name, _, prompt = spec.partition(":")
        prompts[name.strip()] = prompt.strip()

    if not prompts:
        # Fallback for bare invocations (tests / demos)
        for name in ["Marcus", "Sophia"]:
            prompts[name] = _default_character_prompt(name)

    return prompts


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
    from marlllm.population import PopulationTrainer
    from envs.deal_or_no_deal_env import DealOrNoDealEnv

    character_prompts = _build_character_prompts(args)
    print(f"Population: {list(character_prompts.keys())} ({len(character_prompts)} agents)")

    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    torch_dtype = dtype_map.get(args.dtype, "auto")

    device = args.device or _auto_device()

    lora_modules = (
        [m.strip() for m in args.lora_modules.split(",")]
        if args.lora_modules
        else None
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
            raise ImportError("--lora-shared-base requires the `peft` package: uv add peft") from e
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

        # Attach first adapter with get_peft_model, then add the rest
        print(f"Attaching LoRA adapter '{names[0]}' (r={args.lora_r})")
        peft_model = get_peft_model(base_model, lora_config, adapter_name=names[0])

        for name in names[1:]:
            print(f"Attaching LoRA adapter '{name}' (r={args.lora_r})")
            peft_model.add_adapter(name, lora_config)

        for name, prompt in character_prompts.items():
            population[name] = LoRASharedBaseAgent(
                agent_id=name,
                character_prompt=prompt,
                shared_backbone=peft_model,
                adapter_name=name,
                tokenizer=tokenizer,
                device=device,
                keep_ref_model=keep_ref,
            )

        print(f"LoRA shared base: 1 backbone, {len(names)} independent adapters.")

    else:
        # Independent model per agent
        keep_ref = args.kl_coef > 0.0
        for name, prompt in character_prompts.items():
            print(f"Loading agent '{name}': {args.model}  →  {device}")
            population[name] = IndependentAgent(
                agent_id=name,
                character_prompt=prompt,
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

    # Use the first agent's tokenizer for the env and tokeniser
    first_agent = list(population.values())[0]

    print(f"Building DealOrNoDealEnv (dialogue_turns={args.dialogue_turns}, "
          f"token_budget={args.token_budget}, role_shuffle={args.role_shuffle})")
    env = DealOrNoDealEnv(
        tokenizer=first_agent.tokenizer,
        max_dialogue_turns=args.dialogue_turns,
        action_token_budget=args.token_budget,
        seed=args.seed,
        role_shuffle=args.role_shuffle,
    )

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
        device=device,
        seed=args.seed,
        kl_coef=args.kl_coef,
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
        env=env,
        loss=loss,
        tokeniser=tokeniser,
        store=store,
        config=config,
        pairing_strategy=args.pairing_strategy,
    )

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
