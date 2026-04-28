"""
Entry point: two-agent CCSM training with DealOrNoDealEnv.

Both agents share the same model weights by default (pass --model-1 to
use separate weights).  With a single shared model the agents develop a
common "language" through CCSM; with separate models their priors must
align through the shared observation stream.

Usage:
    uv run python train_negotiation.py
    uv run python train_negotiation.py --model meta-llama/Llama-3.2-1B --device cuda
    uv run python train_negotiation.py --model-0 gpt2 --model-1 gpt2 --iters 500
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

# Must be set before the CUDA allocator is initialised (i.e. before the first
# CUDA tensor is created).  expandable_segments lets PyTorch reuse fragmented
# free blocks rather than requiring one large contiguous allocation, which
# prevents OOMs caused by "X GiB reserved but unallocated" fragmentation.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def _auto_device(idx: int, fallback: str) -> str:
    """Return cuda:idx if available, else fallback."""
    if torch.cuda.is_available() and torch.cuda.device_count() > idx:
        return f"cuda:{idx}"
    return fallback


def parse_args() -> argparse.Namespace:
    # First pass: pull out --config so we can set YAML values as defaults
    # before the full parse.  Explicit CLI args will still override them.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    pre_args, _ = pre.parse_known_args()

    p = argparse.ArgumentParser(description="Two-agent CCSM training on DealOrNoDealEnv.")
    p.add_argument("--config", default=None,
                   help="Path to a YAML experiment config file. "
                        "All keys map to their corresponding CLI flags; "
                        "explicit CLI args override config file values.")
    p.add_argument("--model",   default="gpt2", help="HuggingFace model for both agents (overridden by --model-0/1)")
    p.add_argument("--model-0", default=None,   help="HuggingFace model for agent_0 (overrides --model)")
    p.add_argument("--model-1", default=None,   help="HuggingFace model for agent_1 (overrides --model)")
    p.add_argument("--device",  default=None,
                   help="Default PyTorch device for both agents. "
                        "If omitted and CUDA is available, agent_0 → cuda:0, agent_1 → cuda:1.")
    p.add_argument("--device-0", default=None,  help="Device for agent_0 (overrides --device)")
    p.add_argument("--device-1", default=None,  help="Device for agent_1 (overrides --device)")
    p.add_argument("--dtype",   default="auto", help="torch_dtype: auto, float32, bfloat16, float16")
    p.add_argument("--device-map", default=None, help="device_map for from_pretrained, e.g. 'auto'")
    p.add_argument("--iters",          type=int,   default=500,    help="Training iterations")
    p.add_argument("--rollouts",       type=int,   default=8,      help="Episodes per iteration")
    p.add_argument("--lr",             type=float, default=3e-5,   help="AdamW learning rate")
    p.add_argument("--dialogue-turns",   type=int,   default=10,   help="Max dialogue turns per episode")
    p.add_argument("--token-budget",     type=int,   default=64,   help="Generation budget per agent turn (includes thinking tokens)")
    p.add_argument("--env-token-budget", type=int,   default=None, help="Communication budget: max tokens after thinking stripped. None = no stripping")
    p.add_argument("--max-episode-tokens", type=int, default=1024, help="Token budget for full episode")
    p.add_argument("--log-every",        type=int, default=10,     help="Log every N iterations")
    p.add_argument("--checkpoint-every", type=int, default=100,    help="Checkpoint every N iterations")
    p.add_argument("--output-dir", default="runs/negotiation",     help="Log and checkpoint directory")
    p.add_argument("--resume", action="store_true",                help="Resume from latest checkpoint")
    p.add_argument("--kl-coef",  type=float, default=0.0,          help="KL penalty coefficient")
    p.add_argument("--seed",     type=int,   default=42)
    # --- distribution / memory ---
    p.add_argument("--grad-accum", type=int, default=8,
                   help="Gradient accumulation steps. Split each batch into N micro-batches "
                        "to reduce peak VRAM usage (default 8).")
    p.add_argument("--gradient-checkpointing", action="store_true",
                   help="Enable activation checkpointing to trade compute for VRAM.")
    p.add_argument("--compile", action="store_true",
                   help="torch.compile() the backbone. ~1-2 iter warm-up, then ~20-40%% speedup.")
    # --- LoRA ---
    p.add_argument("--lora-r",     type=int, default=0,
                   help="LoRA rank. 0 = full fine-tuning (default). "
                        "Set e.g. 16 or 64 for parameter-efficient training. Requires `peft`.")
    p.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha scaling factor.")
    p.add_argument("--lora-modules", default=None,
                   help="Comma-separated list of linear module names to apply LoRA to. "
                        "None = PEFT auto-detect (q_proj,v_proj for most architectures).")
    p.add_argument(
        "--prompt-0",
        default="You are Agent A, negotiating to maximise your score.",
        help="Character prompt for agent_0",
    )
    p.add_argument(
        "--prompt-1",
        default="You are Agent B, negotiating to maximise your score.",
        help="Character prompt for agent_1",
    )
    p.add_argument(
        "--attn-impl", default=None,
        help="attn_implementation passed to from_pretrained for both agents. "
             "Set to 'eager' to bypass the new SDPA path on nightly PyTorch "
             "(required for GPT-2 and other older architectures).",
    )
    p.add_argument(
        "--role-shuffle", action="store_true",
        help="Randomly assign which agent goes first each episode, balancing "
             "the first-mover / responder role across both agents.",
    )
    p.add_argument(
        "--shared-weights", action="store_true",
        help="Both agents share one set of model weights (single backbone, "
             "single optimizer). Gradients from both agents' trajectories "
             "update the same parameters. Requires --model (not --model-0/1).",
    )
    p.add_argument(
        "--lora-shared-base", action="store_true",
        help="Load one base model and attach two independent LoRA adapters — "
             "one per agent. The base weights are frozen and shared; each agent "
             "trains only its own adapter. Requires --lora-r > 0 and --model. "
             "Halves backbone VRAM compared to two full independent models, "
             "enabling larger models on a single GPU.",
    )

    # Apply YAML config as defaults (CLI args still override).
    if pre_args.config:
        from marlllm.config_loader import apply_config_defaults
        apply_config_defaults(p, pre_args.config)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    from marlllm import (
        CCSMLoss,
        IndependentAgent,
        LoRASharedBaseAgent,
        OnPolicyStore,
        TextTokeniser,
        Trainer,
        TrainingConfig,
    )
    from marlllm.config_loader import log_system_info
    from envs.deal_or_no_deal_env import DealOrNoDealEnv

    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    torch_dtype = dtype_map.get(args.dtype, "auto")

    model_0 = args.model_0 or args.model
    model_1 = args.model_1 or args.model

    # Device assignment: prefer explicit --device-0/1, then --device, then auto (cuda:0/1).
    device_0 = args.device_0 or (args.device if args.device else _auto_device(0, "cpu"))
    device_1 = args.device_1 or (args.device if args.device else _auto_device(1, device_0))

    lora_modules = (
        [m.strip() for m in args.lora_modules.split(",")]
        if args.lora_modules
        else None
    )

    # Write system_info.json and copy the config file before loading models.
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    log_system_info(args.output_dir, config_path=args.config)

    if args.lora_shared_base:
        if args.model_0 or args.model_1:
            raise ValueError("--lora-shared-base is incompatible with --model-0/--model-1.")
        if args.lora_r <= 0:
            raise ValueError("--lora-shared-base requires --lora-r > 0.")
        if args.shared_weights:
            raise ValueError("--lora-shared-base and --shared-weights are mutually exclusive.")

        try:
            from peft import LoraConfig, get_peft_model
        except ImportError as e:
            raise ImportError("--lora-shared-base requires the `peft` package: uv add peft") from e
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from marlllm.agent import _find_lora_target_modules

        print(f"Loading shared base model: {model_0}  →  {device_0}")
        load_kwargs: dict = {}
        if torch_dtype is not None:
            load_kwargs["torch_dtype"] = torch_dtype
        if args.device_map is not None:
            load_kwargs["device_map"] = args.device_map
        if args.attn_impl is not None:
            load_kwargs["attn_implementation"] = args.attn_impl

        base_model = AutoModelForCausalLM.from_pretrained(model_0, **load_kwargs)
        if args.device_map is None:
            base_model = base_model.to(device_0)

        tokenizer = AutoTokenizer.from_pretrained(model_0)
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

        print(f"Attaching LoRA adapter 'agent_0' (r={args.lora_r})")
        peft_model = get_peft_model(base_model, lora_config, adapter_name="agent_0")
        print(f"Attaching LoRA adapter 'agent_1' (r={args.lora_r})")
        peft_model.add_adapter("agent_1", lora_config)

        keep_ref = args.kl_coef > 0.0
        agent_0 = LoRASharedBaseAgent(
            agent_id="agent_0",
            character_prompt=args.prompt_0,
            shared_backbone=peft_model,
            adapter_name="agent_0",
            tokenizer=tokenizer,
            device=device_0,
            keep_ref_model=keep_ref,
        )
        agent_1 = LoRASharedBaseAgent(
            agent_id="agent_1",
            character_prompt=args.prompt_1,
            shared_backbone=peft_model,
            adapter_name="agent_1",
            tokenizer=tokenizer,
            device=device_0,   # same GPU — base is already there
            keep_ref_model=keep_ref,
        )
        print("LoRA shared base: one backbone, two independent adapters.")

    elif args.shared_weights:
        if args.model_0 or args.model_1:
            raise ValueError("--shared-weights is incompatible with --model-0/--model-1. "
                             "Use --model to specify the shared backbone.")
        print(f"Loading shared agent backbone: {model_0}  →  {device_0}")
        shared_agent = IndependentAgent(
            agent_id="agent_0",
            character_prompt=args.prompt_0,
            model_name_or_path=model_0,
            device=device_0,
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
        agent_0 = shared_agent
        agent_1 = shared_agent
        print("Shared weights: agent_0 and agent_1 use the same backbone.")
    else:
        print(f"Loading agent_0: {model_0}  →  {device_0}")
        agent_0 = IndependentAgent(
            agent_id="agent_0",
            character_prompt=args.prompt_0,
            model_name_or_path=model_0,
            device=device_0,
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

        print(f"Loading agent_1: {model_1}  →  {device_1}")
        agent_1 = IndependentAgent(
            agent_id="agent_1",
            character_prompt=args.prompt_1,
            model_name_or_path=model_1,
            device=device_1,
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

    env_token_budget = getattr(args, "env_token_budget", None)
    print(
        f"Building DealOrNoDealEnv (dialogue_turns={args.dialogue_turns}, "
        f"token_budget={args.token_budget}, env_token_budget={env_token_budget}, "
        f"role_shuffle={args.role_shuffle})"
    )
    env = DealOrNoDealEnv(
        tokenizer=agent_0.tokenizer,
        max_dialogue_turns=args.dialogue_turns,
        action_token_budget=args.token_budget,
        env_token_budget=env_token_budget,
        seed=args.seed,
        role_shuffle=args.role_shuffle,
    )

    config = TrainingConfig(
        model_name_or_path=model_0,
        character_prompts={"agent_0": args.prompt_0, "agent_1": args.prompt_1},
        episodes_per_iter=args.rollouts,
        max_episode_tokens=args.max_episode_tokens,
        num_iterations=args.iters,
        lr=args.lr,
        log_every=args.log_every,
        checkpoint_every=args.checkpoint_every,
        output_dir=args.output_dir,
        device=device_0,
        seed=args.seed,
        kl_coef=args.kl_coef,
        grad_accum_steps=args.grad_accum,
        gradient_checkpointing=args.gradient_checkpointing,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_target_modules=lora_modules,
    )

    tokeniser = TextTokeniser(agent_0.tokenizer)
    loss      = CCSMLoss()
    store     = OnPolicyStore()

    trainer = Trainer(
        agents={"agent_0": agent_0, "agent_1": agent_1},
        env=env,
        loss=loss,
        tokeniser=tokeniser,
        store=store,
        config=config,
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
