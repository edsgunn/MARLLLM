"""
Entry point: CCSM training on CCSMTerminalEnv.

Stages 0–3 use a single agent (N=1).  Stage 3.5+ uses two agents that share
one set of model weights by default — one world model, two character-conditioned
roles, coordination only through the shared filesystem.

The key flag for sequential training is --resume-from: it loads model weights
from a previous stage's checkpoint and starts a fresh training run (iteration 1)
in the new stage.  This is distinct from --resume, which continues the current
stage from its latest checkpoint.

Usage:
    # Stage 0 — from scratch
    python train_terminal.py --config configs/terminal_experiments/00_stage0_baseline.yaml

    # Stage 1 — from Stage 0 final
    python train_terminal.py \\
        --config configs/terminal_experiments/01_stage1_sparse_actions.yaml \\
        --resume-from runs/terminal_experiments/00_stage0_baseline/checkpoints/final.pt

    # Stage 3.5 — from Stage 3 final
    python train_terminal.py \\
        --config configs/terminal_experiments/04_stage35_embodiment.yaml \\
        --resume-from runs/terminal_experiments/03_stage3_write_access/checkpoints/final.pt
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def _auto_device(fallback: str = "cpu") -> str:
    return "cuda:0" if torch.cuda.is_available() else fallback


def parse_args() -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    pre_args, _ = pre.parse_known_args()

    p = argparse.ArgumentParser(description="CCSM training on CCSMTerminalEnv.")

    # ── Config ────────────────────────────────────────────────────────────────
    p.add_argument("--config", default=None,
                   help="YAML experiment config. Keys map to CLI flags; "
                        "explicit CLI args override.")

    # ── Model ─────────────────────────────────────────────────────────────────
    p.add_argument("--model",       default="HuggingFaceTB/SmolLM2-135M")
    p.add_argument("--device",      default=None,
                   help="PyTorch device. Defaults to cuda:0 if available.")
    p.add_argument("--dtype",       default="bfloat16",
                   help="torch_dtype: auto, float32, bfloat16, float16")
    p.add_argument("--device-map",  default=None,
                   help="device_map for from_pretrained, e.g. 'auto'")
    p.add_argument("--attn-impl",   default=None,
                   help="attn_implementation for from_pretrained "
                        "(e.g. 'eager' for older architectures).")

    # ── Training loop ─────────────────────────────────────────────────────────
    p.add_argument("--iters",              type=int,   default=500)
    p.add_argument("--rollouts",           type=int,   default=8,
                   help="Episodes collected per iteration (batched).")
    p.add_argument("--lr",                 type=float, default=3e-5)
    p.add_argument("--max-episode-tokens", type=int,   default=65536,
                   help="Trainer-side episode token cap (obs+act combined). "
                        "Set well above env token_budget so the env drives "
                        "termination, not this guard.")
    p.add_argument("--log-every",          type=int,   default=10)
    p.add_argument("--checkpoint-every",   type=int,   default=100)
    p.add_argument("--output-dir",         default="runs/terminal_experiments/default")
    p.add_argument("--seed",               type=int,   default=42)
    p.add_argument("--temperature",        type=float, default=1.0)
    p.add_argument("--kl-coef",            type=float, default=0.0,
                   help="KL penalty weight λ * KL(π_θ || π_ref) at action positions. "
                        "0 = disabled.  Use a small positive value (0.02–0.1) "
                        "to prevent catastrophic forgetting between stages.")
    p.add_argument("--grad-accum",         type=int,   default=4)
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--compile",            action="store_true",
                   help="torch.compile() the backbone (~20–40%% speedup after warm-up).")

    # ── LoRA ──────────────────────────────────────────────────────────────────
    p.add_argument("--lora-r",       type=int, default=0,
                   help="LoRA rank. 0 = full fine-tuning.")
    p.add_argument("--lora-alpha",   type=int, default=16)
    p.add_argument("--lora-modules", default=None,
                   help="Comma-separated linear module names for LoRA. "
                        "None = PEFT auto-detect.")

    # ── Env — stage and corpus ─────────────────────────────────────────────────
    p.add_argument("--stage",        default="0",
                   help="CCSMTerminalEnv stage: 0, 1, 2, 3, 3.5, 4")
    p.add_argument("--corpus-path",  default=None, required=False,
                   help="Path to read-only corpus directory. Required for all stages.")
    p.add_argument("--n-agents",     type=int, default=None,
                   help="Number of agents. Defaults to stage's recommended value. "
                        "Pass 1 for a single-agent ablation at Stage 3.5.")
    p.add_argument("--shared-weights", action="store_true",
                   help="All agents share one backbone (default behaviour for N>1). "
                        "The Trainer deduplicates parameters automatically, so there "
                        "is no memory penalty vs. a single-agent run.")

    # ── Env — episode parameters ───────────────────────────────────────────────
    p.add_argument("--token-budget",        type=int, default=8192,
                   help="Env observation token budget per agent per episode. "
                        "Episode ends when cumulative obs tokens exceed this.")
    p.add_argument("--action-token-budget", type=int, default=64,
                   help="Max action tokens per step (command length limit). "
                        "Overridden to 0 for Stage 0 by the env. "
                        "16 is enough for 'ls' and short 'cat' paths; "
                        "64 handles most grep/find; 128 handles python3 one-liners.")
    p.add_argument("--files-per-episode",   type=int, default=20,
                   help="Number of corpus files per agent per episode.")
    p.add_argument("--overlap-fraction",    type=float, default=0.3,
                   help="Fraction of corpus files carried over between episodes. "
                        "Lower → more novelty → stronger anti-collapse pressure. "
                        "Higher → more continuity → lower inter-episode loss variance.")
    p.add_argument("--command-timeout",     type=int,   default=30,
                   help="Seconds before a shell command is killed.")
    p.add_argument("--max-output-chars",    type=int,   default=8192,
                   help="Hard cap on shell stdout per step before truncation.")
    p.add_argument("--work-dir",            default=None,
                   help="Staging/scratch parent directory. "
                        "In Slurm, set to $SCRATCH or $LOCAL_SCRATCH for fast I/O. "
                        "If unset, uses a system temp directory (cleaned on exit).")

    # ── Character prompts ─────────────────────────────────────────────────────
    p.add_argument("--prompt-0", default=(
        "You are a research librarian exploring a corpus of scientific texts. "
        "You read methodically, following references from broad overviews to specific findings."
    ), help="Character prompt for agent_0.")
    p.add_argument("--prompt-1", default=(
        "You are a research synthesizer. Before reading, check $SCRATCH/index.txt "
        "for notes left by your collaborator. Then search for connections between documents."
    ), help="Character prompt for agent_1 (Stage 3.5+).")

    # ── Checkpoint loading ────────────────────────────────────────────────────
    p.add_argument("--resume", action="store_true",
                   help="Resume from latest checkpoint within --output-dir. "
                        "Used to continue a partially-completed stage run.")
    p.add_argument("--resume-from", default=None,
                   help="Path to a checkpoint from a PREVIOUS stage. "
                        "Loads model weights only; training starts at iteration 1. "
                        "Example: runs/terminal_experiments/03_stage3_write_access/"
                        "checkpoints/final.pt")

    if pre_args.config:
        from marlllm.config_loader import apply_config_defaults
        apply_config_defaults(p, pre_args.config)

    return p.parse_args()


def _load_weights_only(trainer, path: str) -> None:
    """
    Load model weights from a checkpoint produced by a different stage.

    Only the backbone and value head weights are transferred — not the
    optimizer state or RNG state, which are stage-specific.  This lets the
    model enter a new stage with the knowledge accumulated in the previous one
    without inheriting the optimizer's momentum estimates from a different loss
    landscape.

    If the checkpoint contains more agents than the current run (e.g. loading
    a Stage 3.5 N=2 checkpoint into a Stage 3 N=1 run), only agent_0's weights
    are loaded.  If fewer agents are present (e.g. loading N=1 into N=2), all
    agents receive the same initial weights — the shared-weights default.
    """
    import torch
    payload = torch.load(path, map_location=trainer.device)
    stored = payload.get("agent_states", {})
    for aid, agent in trainer.agents.items():
        src = stored.get(aid) or stored.get("agent_0")
        if src is None:
            print(f"  [warn] no weights for {aid} in checkpoint; skipping", file=sys.stderr)
            continue
        missing, unexpected = agent._backbone.load_state_dict(src["backbone"], strict=False)
        if missing:
            print(f"  [warn] {aid} backbone: {len(missing)} missing keys", file=sys.stderr)
        if unexpected:
            print(f"  [warn] {aid} backbone: {len(unexpected)} unexpected keys", file=sys.stderr)
        agent._value_head.load_state_dict(src["value_head"], strict=False)
    print(f"Weights loaded from: {path}")


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
    from envs.terminal import CCSMTerminalEnv

    device = args.device or _auto_device()
    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    torch_dtype = dtype_map.get(args.dtype, "auto")
    lora_modules = (
        [m.strip() for m in args.lora_modules.split(",")]
        if args.lora_modules else None
    )

    corpus_path = args.corpus_path
    if corpus_path is None:
        print("Error: --corpus-path is required.", file=sys.stderr)
        sys.exit(1)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    log_system_info(args.output_dir, config_path=args.config)

    # ── Build env (needed to know n_agents and action_token_budget) ────────────
    # work_dir: expand env vars so YAML can say "$SCRATCH/ccsm_work"
    work_dir = os.path.expandvars(args.work_dir) if args.work_dir else None

    print(f"Building CCSMTerminalEnv (stage={args.stage}, corpus={corpus_path})")
    env = CCSMTerminalEnv(
        tokenizer=None,             # placeholder; replaced after tokenizer is built
        stage=args.stage,
        corpus_path=corpus_path,
        n_agents=args.n_agents,
        token_budget=args.token_budget,
        action_token_budget=args.action_token_budget,
        files_per_episode=args.files_per_episode,
        overlap_fraction=args.overlap_fraction,
        command_timeout=args.command_timeout,
        max_output_chars=args.max_output_chars,
        seed=args.seed,
        work_dir=work_dir,
    )
    n_agents = len(env.possible_agents)
    print(f"  Stage {args.stage}: {n_agents} agent(s), "
          f"action_token_budget={env.action_token_budget}")

    # ── Build agent(s) ─────────────────────────────────────────────────────────
    load_kwargs: dict = {}
    if torch_dtype != "auto":
        load_kwargs["torch_dtype"] = torch_dtype
    if args.device_map is not None:
        load_kwargs["device_map"] = args.device_map
    if args.attn_impl is not None:
        load_kwargs["attn_implementation"] = args.attn_impl

    prompts = {
        "agent_0": args.prompt_0,
        "agent_1": args.prompt_1,
    }

    print(f"Loading model: {args.model} → {device}")
    primary_agent = IndependentAgent(
        agent_id="agent_0",
        character_prompt=prompts["agent_0"],
        model_name_or_path=args.model,
        device=device,
        keep_ref_model=args.kl_coef > 0.0,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_target_modules=lora_modules,
        gradient_checkpointing=args.gradient_checkpointing,
        **load_kwargs,
    )
    if args.compile:
        primary_agent._backbone = torch.compile(primary_agent._backbone)

    # Wire the real tokenizer into the env now that we have it.
    env._tok = primary_agent.tokenizer

    agents: dict[str, IndependentAgent] = {"agent_0": primary_agent}

    if n_agents > 1:
        if args.shared_weights or not (args.lora_r > 0):
            # Shared weights: second agent entry points at the same object.
            # The Trainer deduplicates parameters by identity, so no double
            # counting in the optimizer.
            for aid in env.possible_agents[1:]:
                agents[aid] = primary_agent
                print(f"  {aid}: shared weights with agent_0")
        else:
            for aid in env.possible_agents[1:]:
                agents[aid] = IndependentAgent(
                    agent_id=aid,
                    character_prompt=prompts.get(aid, prompts["agent_1"]),
                    model_name_or_path=args.model,
                    device=device,
                    keep_ref_model=args.kl_coef > 0.0,
                    lora_r=args.lora_r,
                    lora_alpha=args.lora_alpha,
                    lora_target_modules=lora_modules,
                    gradient_checkpointing=args.gradient_checkpointing,
                    **load_kwargs,
                )
                if args.compile:
                    agents[aid]._backbone = torch.compile(agents[aid]._backbone)
                print(f"  {aid}: independent weights")

    # ── Build training config ──────────────────────────────────────────────────
    config = TrainingConfig(
        model_name_or_path=args.model,
        character_prompts={aid: prompts.get(aid, "") for aid in agents},
        episodes_per_iter=args.rollouts,
        num_iterations=args.iters,
        max_episode_tokens=args.max_episode_tokens,
        lr=args.lr,
        log_every=args.log_every,
        checkpoint_every=args.checkpoint_every,
        output_dir=args.output_dir,
        device=device,
        seed=args.seed,
        temperature=args.temperature,
        kl_coef=args.kl_coef,
        grad_accum_steps=args.grad_accum,
        gradient_checkpointing=args.gradient_checkpointing,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )

    tokeniser = TextTokeniser(primary_agent.tokenizer)
    loss = CCSMLoss()
    store = OnPolicyStore()

    trainer = Trainer(
        agents=agents,
        env=env,
        loss=loss,
        tokeniser=tokeniser,
        store=store,
        config=config,
    )

    # ── Checkpoint handling ────────────────────────────────────────────────────
    start_iteration = 1

    if args.resume_from:
        # Cross-stage load: weights only, training starts fresh at iter 1.
        _load_weights_only(trainer, args.resume_from)

    if args.resume:
        latest = Path(args.output_dir) / "checkpoints" / "latest.pt"
        if latest.exists():
            start_iteration = trainer.load_checkpoint(str(latest)) + 1
            print(f"Resuming from iteration {start_iteration}")
        else:
            print("No checkpoint found in output_dir; starting from iteration 1.")

    # ── Go ─────────────────────────────────────────────────────────────────────
    print(f"Output directory: {args.output_dir}")
    for aid, ag in agents.items():
        print(f"  {aid}: {prompts.get(aid, '')!r}")
    print()
    trainer.train(start_iteration=start_iteration)


if __name__ == "__main__":
    main()
