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
import sys
from pathlib import Path

import torch


def _auto_device(idx: int, fallback: str) -> str:
    """Return cuda:idx if available, else fallback."""
    if torch.cuda.is_available() and torch.cuda.device_count() > idx:
        return f"cuda:{idx}"
    return fallback


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Two-agent CCSM training on DealOrNoDealEnv.")
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
    p.add_argument("--dialogue-turns", type=int,   default=10,     help="Max dialogue turns per episode")
    p.add_argument("--token-budget",   type=int,   default=64,     help="Tokens per agent turn")
    p.add_argument("--max-episode-tokens", type=int, default=1024, help="Token budget for full episode")
    p.add_argument("--log-every",        type=int, default=10,     help="Log every N iterations")
    p.add_argument("--checkpoint-every", type=int, default=100,    help="Checkpoint every N iterations")
    p.add_argument("--output-dir", default="runs/negotiation",     help="Log and checkpoint directory")
    p.add_argument("--resume", action="store_true",                help="Resume from latest checkpoint")
    p.add_argument("--kl-coef",  type=float, default=0.0,          help="KL penalty coefficient")
    p.add_argument("--seed",     type=int,   default=42)
    # --- distribution / memory ---
    p.add_argument("--grad-accum", type=int, default=4,
                   help="Gradient accumulation steps. Split each batch into N micro-batches "
                        "to reduce peak VRAM usage (default 4).")
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
    return p.parse_args()


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
    )

    print(f"Building DealOrNoDealEnv (dialogue_turns={args.dialogue_turns}, token_budget={args.token_budget})")
    env = DealOrNoDealEnv(
        tokenizer=agent_0.tokenizer,
        max_dialogue_turns=args.dialogue_turns,
        action_token_budget=args.token_budget,
        seed=args.seed,
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
