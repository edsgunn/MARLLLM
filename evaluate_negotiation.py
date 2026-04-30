"""
Capability evaluation for the deal-or-no-deal negotiation task (Phase A §3.1).

Runs N episodes with a fixed pair of agents — either pretrained from scratch or
restored from a checkpoint — and writes a JSON summary of the four capability
metrics defined in the experiment spec:

  * joint utility (mean over deals + mean over all)
  * individual utility for each agent
  * agreement rate
  * Pareto efficiency (and Pareto-optimal rate)

The script is intentionally non-circular: none of these metrics are derived
from CCSM's training signal.  It can also score the *pretrained baseline*
(condition 1) by simply omitting --checkpoint.

Usage
-----
  uv run python evaluate_negotiation.py \\
      --model Qwen/Qwen2.5-1.5B \\
      --checkpoint runs/.../checkpoints/final.pt \\
      --prompt-0 "You are Agent A, ..." \\
      --prompt-1 "You are Agent B, ..." \\
      --n-episodes 64 \\
      --output runs/.../eval_final.json

Pretrained baseline:
  uv run python evaluate_negotiation.py --model Qwen/Qwen2.5-1.5B \\
      --output runs/baseline_eval.json
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch


def _auto_device(idx: int, fallback: str) -> str:
    if torch.cuda.is_available() and torch.cuda.device_count() > idx:
        return f"cuda:{idx}"
    return fallback


def parse_args() -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    pre_args, _ = pre.parse_known_args()

    p = argparse.ArgumentParser(description="Capability evaluation for DealOrNoDealEnv.")
    p.add_argument("--config", default=None,
                   help="Optional YAML defaults file (same format as training configs).")
    p.add_argument("--model",   default="Qwen/Qwen2.5-1.5B",
                   help="HuggingFace model used to instantiate both agents.")
    p.add_argument("--checkpoint", default=None,
                   help="Path to a Trainer checkpoint (.pt). Omit to evaluate the "
                        "pretrained baseline (Phase A condition 1).")
    p.add_argument("--checkpoint-1", default=None,
                   help="Optional separate checkpoint for agent_1. If omitted, "
                        "agent_1 uses the same checkpoint as agent_0 (or pretrained).")
    p.add_argument("--device",   default=None)
    p.add_argument("--device-0", default=None)
    p.add_argument("--device-1", default=None)
    p.add_argument("--dtype",    default="bfloat16",
                   help="torch_dtype for model load: bfloat16 / float16 / float32 / auto")
    p.add_argument("--attn-impl", default=None)
    p.add_argument("--n-episodes",     type=int, default=64,
                   help="Number of evaluation episodes to roll out.")
    p.add_argument("--dialogue-turns", type=int, default=10)
    p.add_argument("--token-budget",   type=int, default=64)
    p.add_argument("--env-token-budget", type=int, default=None)
    p.add_argument("--max-episode-tokens", type=int, default=2048)
    p.add_argument("--temperature",    type=float, default=1.0)
    p.add_argument("--seed",           type=int,   default=10_000,
                   help="Eval seed — kept disjoint from training seeds (default 10_000) "
                        "so the eval scenarios are not the ones the agent trained on.")
    p.add_argument("--role-shuffle",   action="store_true")
    p.add_argument("--prompt-0", default="You are Agent A, negotiating to maximise your score.")
    p.add_argument("--prompt-1", default="You are Agent B, negotiating to maximise your score.")
    p.add_argument("--output",   default=None,
                   help="Where to write the JSON summary. Defaults to <checkpoint dir>/eval.json "
                        "if --checkpoint is given, else 'eval.json' in the cwd.")
    p.add_argument("--save-trajectories", action="store_true",
                   help="Also save per-episode token-level trajectories for downstream "
                        "behavioural analysis (Phase A §3.2).")

    if pre_args.config:
        from marlllm.config_loader import apply_config_defaults
        apply_config_defaults(p, pre_args.config)
    return p.parse_args()


def _load_agent_states(checkpoint_path: str, agent_ids: list[str]):
    """Return {agent_id: state_dict-pair} or None if checkpoint missing."""
    if checkpoint_path is None:
        return None
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return {aid: payload["agent_states"][aid] for aid in agent_ids if aid in payload["agent_states"]}


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    from marlllm import IndependentAgent, TextTokeniser, TokenType
    from marlllm.eval_utils import NegotiationOutcome, summarise
    from envs.deal_or_no_deal_env import DealOrNoDealEnv

    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    torch_dtype = dtype_map.get(args.dtype, "auto")

    device_0 = args.device_0 or args.device or _auto_device(0, "cpu")
    device_1 = args.device_1 or args.device or _auto_device(1, device_0)

    print(f"Loading agent_0: {args.model}  →  {device_0}")
    agent_0 = IndependentAgent(
        agent_id="agent_0",
        character_prompt=args.prompt_0,
        model_name_or_path=args.model,
        device=device_0,
        torch_dtype=torch_dtype,
        attn_implementation=args.attn_impl,
    )
    print(f"Loading agent_1: {args.model}  →  {device_1}")
    agent_1 = IndependentAgent(
        agent_id="agent_1",
        character_prompt=args.prompt_1,
        model_name_or_path=args.model,
        device=device_1,
        torch_dtype=torch_dtype,
        attn_implementation=args.attn_impl,
    )

    # ── Restore checkpoints ──────────────────────────────────────────────────
    if args.checkpoint:
        print(f"Restoring agent_0 weights from {args.checkpoint}")
        states0 = _load_agent_states(args.checkpoint, ["agent_0"])
        if states0 and "agent_0" in states0:
            agent_0._backbone.load_state_dict(states0["agent_0"]["backbone"])
            agent_0._value_head.load_state_dict(states0["agent_0"]["value_head"])
        else:
            print("  [WARN] agent_0 not present in checkpoint; using pretrained.")

        ckpt_1 = args.checkpoint_1 or args.checkpoint
        print(f"Restoring agent_1 weights from {ckpt_1}")
        states1 = _load_agent_states(ckpt_1, ["agent_1"])
        if states1 and "agent_1" in states1:
            agent_1._backbone.load_state_dict(states1["agent_1"]["backbone"])
            agent_1._value_head.load_state_dict(states1["agent_1"]["value_head"])
        else:
            print("  [WARN] agent_1 not present in checkpoint; using pretrained.")
    else:
        print("[INFO] No --checkpoint specified — evaluating pretrained baseline.")

    agent_0.eval_mode()
    agent_1.eval_mode()

    env = DealOrNoDealEnv(
        tokenizer=agent_0.tokenizer,
        max_dialogue_turns=args.dialogue_turns,
        action_token_budget=args.token_budget,
        env_token_budget=args.env_token_budget,
        seed=args.seed,
        role_shuffle=args.role_shuffle,
    )
    tokeniser = TextTokeniser(agent_0.tokenizer)
    agents = {"agent_0": agent_0, "agent_1": agent_1}

    outcomes: list[NegotiationOutcome] = []
    trajectories: list[dict] = []
    rng_seed = args.seed

    print(f"Running {args.n_episodes} evaluation episodes...")
    t0 = time.time()
    for ep in range(args.n_episodes):
        env.reset(seed=rng_seed)
        rng_seed += 1

        contexts: dict[str, list[int]] = {}
        events: dict[str, list[dict]] = {aid: [] for aid in agents}
        for aid, agent in agents.items():
            prompt_text = args.prompt_0 if aid == "agent_0" else args.prompt_1
            pids = tokeniser.encode_prompt(prompt_text)
            wrapped = agent.context_formatter.wrap_prompt(pids)
            start = len(contexts.get(aid, []))
            contexts[aid] = list(wrapped)
            events[aid].append({"type": "prompt", "span": [start, len(contexts[aid])]})

        for agent_id in env.agent_iter():
            obs, _r, term, trunc, info = env.last()
            obs_ids = tokeniser.encode_observation(obs)
            if obs_ids:
                fmt = agents[agent_id].context_formatter
                wrapped = fmt.wrap_observation(obs_ids)
                start = len(contexts[agent_id])
                contexts[agent_id].extend(wrapped)
                events[agent_id].append(
                    {"type": "obs", "span": [start, len(contexts[agent_id])]}
                )
            if term or trunc:
                env.step(None)
                continue
            agent = agents[agent_id]
            n_tokens = env.action_token_budget
            with torch.no_grad():
                act_ids, _lps = agent.act(
                    context_token_ids=contexts[agent_id],
                    n_tokens=n_tokens,
                    temperature=args.temperature,
                )
            fmt = agents[agent_id].context_formatter
            wrapped = fmt.wrap_action(act_ids)
            start = len(contexts[agent_id])
            contexts[agent_id].extend(wrapped)
            events[agent_id].append(
                {"type": "act", "span": [start, len(contexts[agent_id])]}
            )
            env.step(act_ids)

        # Capture episode outcome
        ep_info = next((env.infos[a] for a in env.possible_agents if a in env.infos), {})
        items   = tuple(ep_info.get("items", []))
        values  = {a: tuple(env._values.get(a, [])) for a in env.possible_agents}
        deal    = ep_info.get("outcome") == "deal"
        score_a = env._episode_scores.get("agent_0", 0)
        score_b = env._episode_scores.get("agent_1", 0)

        outcomes.append(NegotiationOutcome(
            deal=deal,
            score_a=score_a,
            score_b=score_b,
            items=items,
            values_a=values["agent_0"],
            values_b=values["agent_1"],
        ))
        if args.save_trajectories:
            tok = agent_0.tokenizer
            episode_record = {
                "episode": ep,
                "outcome": {
                    "deal": deal, "score_a": score_a, "score_b": score_b,
                    "items": list(items),
                    "values_a": list(values["agent_0"]),
                    "values_b": list(values["agent_1"]),
                },
                "agents": {
                    aid: {
                        "context_ids": list(contexts[aid]),
                        "context_text": tok.decode(contexts[aid], skip_special_tokens=False),
                        "events": events[aid],
                    }
                    for aid in agents
                },
            }
            trajectories.append(episode_record)

        if (ep + 1) % max(1, args.n_episodes // 10) == 0:
            elapsed = time.time() - t0
            print(f"  episode {ep+1:4d}/{args.n_episodes}  elapsed {elapsed:6.1f}s")

    summary = summarise(outcomes)
    summary["model"]      = args.model
    summary["checkpoint"] = args.checkpoint
    summary["prompt_0"]   = args.prompt_0
    summary["prompt_1"]   = args.prompt_1
    summary["seed"]       = args.seed
    summary["temperature"] = args.temperature
    summary["wall_time_seconds"] = time.time() - t0

    out_path = args.output
    if out_path is None:
        if args.checkpoint:
            out_path = str(Path(args.checkpoint).parent / "eval.json")
        else:
            out_path = "eval.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    if args.save_trajectories:
        traj_path = Path(out_path).with_suffix(".trajectories.json")
        with open(traj_path, "w") as f:
            json.dump(trajectories, f, indent=2)
        print(f"Trajectories: {traj_path}")

    print()
    print("Results:")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k:32s} {v:.4f}")
        else:
            print(f"  {k:32s} {v}")
    print()
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
