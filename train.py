"""
Entry point: single-agent CCSM training with CountingEnv.

Usage:
    uv run python train.py
    uv run python train.py --model gpt2 --device cuda --iters 500
    uv run python train.py --output-dir runs/exp1 --resume
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a CCSM agent on CountingEnv.")
    p.add_argument("--model", default="gpt2", help="HuggingFace model ID or path")
    p.add_argument("--device", default="cpu", help="PyTorch device (cpu, cuda, mps)")
    p.add_argument(
        "--dtype",
        default="auto",
        help="torch_dtype for model loading: auto, float32, bfloat16, float16",
    )
    p.add_argument(
        "--device-map",
        default=None,
        help="device_map for from_pretrained, e.g. 'auto' to shard across GPUs",
    )
    p.add_argument("--iters", type=int, default=100, help="Number of training iterations")
    p.add_argument("--rollouts", type=int, default=8, help="Episodes collected per iteration")
    p.add_argument("--lr", type=float, default=3e-5, help="AdamW learning rate")
    p.add_argument("--max-count", type=int, default=10, help="CountingEnv max_count")
    p.add_argument("--log-every", type=int, default=10, help="Print log every N iterations")
    p.add_argument("--checkpoint-every", type=int, default=100, help="Save checkpoint every N iterations")
    p.add_argument("--output-dir", default="runs/default", help="Directory for logs and checkpoints")
    p.add_argument("--resume", action="store_true", help="Resume from latest checkpoint in output-dir")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument(
        "--prompt",
        default="You are a counter that always outputs the next integer in sequence.",
        help="Character prompt for agent_0",
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
    from envs.counting_env import CountingEnv

    config = TrainingConfig(
        model_name_or_path=args.model,
        character_prompts={"agent_0": args.prompt},
        episodes_per_iter=args.rollouts,
        num_iterations=args.iters,
        lr=args.lr,
        log_every=args.log_every,
        checkpoint_every=args.checkpoint_every,
        output_dir=args.output_dir,
        device=args.device,
        seed=args.seed,
    )

    dtype_map = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    torch_dtype = dtype_map.get(args.dtype, "auto")

    print(f"Loading model: {args.model} on {args.device} (dtype={args.dtype}, device_map={args.device_map})")
    agent = IndependentAgent(
        agent_id="agent_0",
        character_prompt=args.prompt,
        model_name_or_path=args.model,
        device=args.device,
        torch_dtype=torch_dtype,
        device_map=args.device_map,
    )

    print(f"Building CountingEnv (max_count={args.max_count})")
    try:
        env = CountingEnv(tokenizer=agent.tokenizer, max_count=args.max_count)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    tokeniser = TextTokeniser(agent.tokenizer)
    loss = CCSMLoss()
    store = OnPolicyStore()

    trainer = Trainer(
        agents={"agent_0": agent},
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
    print(f"Character prompt: {args.prompt!r}")
    print()
    trainer.train(start_iteration=start_iteration)


if __name__ == "__main__":
    main()
