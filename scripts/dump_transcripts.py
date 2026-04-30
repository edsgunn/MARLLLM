"""
Generate strong-partner self-play transcripts for Cell A (vanilla SFT baseline).

Both agents in the negotiation environment are backed by an APIAgent (typically
the same provider/model) so the transcripts represent the strong partners'
"native" behaviour. Output is JSONL — one episode per line — directly
consumable by `scripts/train_sft.py`.

Usage
-----
    uv run python scripts/dump_transcripts.py \
        --provider anthropic --model claude-sonnet-4-6 \
        --episodes 200 --seed 0 \
        --output runs/cellA_transcripts.jsonl \
        --tokenizer Qwen/Qwen2.5-0.5B \
        --cache-dir .cache/anthropic
"""
from __future__ import annotations

import argparse
import json
import os
import time
import subprocess
from pathlib import Path

# Ensure repo root is importable when running this file directly.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transformers import AutoTokenizer

from marlllm import APIAgent, SamplingParams, make_api_model
from envs.deal_or_no_deal_env import DealOrNoDealEnv


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent.parent
        ).decode().strip()
    except Exception:
        return "unknown"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--provider", default="anthropic", choices=["anthropic", "openai", "mock"])
    p.add_argument("--model", default="claude-sonnet-4-6")
    p.add_argument("--tokenizer", required=True,
                   help="HF tokenizer name/path used for re-tokenisation.")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dialogue-turns", type=int, default=10)
    p.add_argument("--token-budget", type=int, default=128)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--budget-state", default=None)
    p.add_argument("--budget-cap-usd", type=float, default=None)
    p.add_argument("--prompt-0",
                   default="You are Agent A, negotiating to maximise your score.")
    p.add_argument("--prompt-1",
                   default="You are Agent B, negotiating to maximise your score.")
    p.add_argument("--output", required=True, help="Path to output JSONL file.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    api = make_api_model(
        provider=args.provider, model=args.model,
        cache_dir=args.cache_dir,
        budget_state_path=args.budget_state,
        budget_cap_usd=args.budget_cap_usd,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    sampling = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)
    agent_0 = APIAgent("agent_0", args.prompt_0, api, tokenizer, sampling)
    agent_1 = APIAgent("agent_1", args.prompt_1, api, tokenizer, sampling)
    agents = {"agent_0": agent_0, "agent_1": agent_1}

    env = DealOrNoDealEnv(
        tokenizer=tokenizer,
        max_dialogue_turns=args.dialogue_turns,
        action_token_budget=args.token_budget,
        seed=args.seed,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = out_path.with_suffix(".manifest.json")
    manifest = {
        "git_sha": _git_sha(),
        "provider": args.provider,
        "model": args.model,
        "tokenizer": args.tokenizer,
        "sampling": sampling.__dict__,
        "episodes": args.episodes,
        "seed": args.seed,
        "dialogue_turns": args.dialogue_turns,
        "token_budget": args.token_budget,
        "started_at": time.time(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    f = out_path.open("w")
    for ep_idx in range(args.episodes):
        env.reset(seed=args.seed + ep_idx)
        for a in agents.values():
            a.reset_history(slot=0)

        # Build per-agent context buffers anchored on the system prompt.
        contexts: dict[str, list[int]] = {
            aid: a.context_formatter.wrap_prompt(
                tokenizer.encode(a.character_prompt, add_special_tokens=False)
            )
            for aid, a in agents.items()
        }

        turns: list[dict] = []
        token_count = 0

        for agent_id in env.agent_iter():
            obs, _rew, term, trunc, info = env.last()
            obs_text = obs if isinstance(obs, str) else (
                tokenizer.decode(obs, skip_special_tokens=True) if obs else ""
            )
            obs_ids = (tokenizer.encode(obs_text, add_special_tokens=False)
                       if obs_text else [])
            if obs_ids:
                contexts[agent_id].extend(
                    agents[agent_id].context_formatter.wrap_observation(obs_ids)
                )
                if obs_text:
                    agents[agent_id].note_observation(obs_text, slot=0)
                token_count += len(obs_ids)
                turns.append({"agent_id": agent_id, "role": "obs", "text": obs_text,
                              "tokens": obs_ids})

            if term or trunc:
                env.step(None)
                continue
            if (token_count >= 4096 and not info.get("must_act", False)):
                env.step(None)
                continue

            act_ids, _ = agents[agent_id].act(
                context_token_ids=contexts[agent_id],
                n_tokens=args.token_budget,
                temperature=args.temperature,
            )
            act_text = tokenizer.decode(act_ids, skip_special_tokens=True)
            contexts[agent_id].extend(
                agents[agent_id].context_formatter.wrap_action(act_ids)
            )
            token_count += len(act_ids)
            turns.append({"agent_id": agent_id, "role": "act", "text": act_text,
                          "tokens": act_ids})
            env.step(act_ids)

        ep_info: dict = {}
        for aid in env.possible_agents:
            if aid in env.infos:
                ep_info = dict(env.infos[aid])
                break

        record = {
            "episode_id": ep_idx,
            "seed": args.seed + ep_idx,
            "turns": turns,
            "outcome": ep_info,
        }
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()

        if (ep_idx + 1) % 10 == 0:
            budget = api.budget.snapshot()
            print(f"[{ep_idx+1}/{args.episodes}] cost=${budget['cost_usd']:.3f} "
                  f"in={budget['input_tokens']} out={budget['output_tokens']}")

    f.close()
    manifest["finished_at"] = time.time()
    manifest["final_budget"] = api.budget.snapshot()
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {args.episodes} episodes → {out_path}")


if __name__ == "__main__":
    main()
