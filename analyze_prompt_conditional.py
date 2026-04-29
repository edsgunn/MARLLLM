"""
Prompt-conditional behavioural analysis (Phase A §3.2).

This is the substrate-validation diagnostic that the experiment spec calls
"the single most important condition for the substrate-validation claim".

We compare two trained agents — typically same model + training procedure but
different character prompts (e.g. cooperative vs competitive) — on three axes:

  1. Behavioural divergence between prompts
       Mean opening offer, agreement rate, individual-vs-joint utility split.
       These come from rolling out trajectories under each prompt and computing
       summary statistics; their pairwise differences are the divergence signal.

  2. Pre-vs-post training prompt sensitivity
       Same divergence statistics computed against the *pretrained* model with
       both prompts.  Healthy CCSM should *amplify* sensitivity over training
       (the model commits harder to its character).  Flat or shrinking
       sensitivity is the substrate-failure signal.

  3. Cross-character surprise
       Take trajectories generated under prompt A and evaluate the
       trained-on-A model's surprise on them, then the trained-on-B model's
       surprise.  The own-model should find them less surprising.  This is a
       direct internal check of "is the character being internalised?".

Usage
-----
  uv run python analyze_prompt_conditional.py \\
      --model Qwen/Qwen2.5-1.5B \\
      --checkpoint-a runs/02_ccsm_cooperative/checkpoints/final.pt \\
      --checkpoint-b runs/02_ccsm_competitive/checkpoints/final.pt \\
      --prompt-a "You are a cooperative negotiator..." \\
      --prompt-b "You are a competitive negotiator..." \\
      --n-episodes 32 \\
      --output runs/phase_a_substrate_validation/prompt_conditional.json

Omit --checkpoint-a/--checkpoint-b to use the pretrained model on both sides
(baseline-prompt-sensitivity check, recommended on day 2 per spec §7).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn.functional as F


def _auto_device(idx: int, fallback: str) -> str:
    if torch.cuda.is_available() and torch.cuda.device_count() > idx:
        return f"cuda:{idx}"
    return fallback


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prompt-conditional analysis (Phase A §3.2).")
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    p.add_argument("--checkpoint-a", default=None,
                   help="Trained-on-prompt-A checkpoint. Omit to use pretrained.")
    p.add_argument("--checkpoint-b", default=None,
                   help="Trained-on-prompt-B checkpoint. Omit to use pretrained.")
    p.add_argument("--prompt-a", required=True, help="Character prompt A (e.g. cooperative).")
    p.add_argument("--prompt-b", required=True, help="Character prompt B (e.g. competitive).")
    p.add_argument("--device-a", default=None)
    p.add_argument("--device-b", default=None)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--attn-impl", default=None)
    p.add_argument("--n-episodes",     type=int, default=32)
    p.add_argument("--dialogue-turns", type=int, default=10)
    p.add_argument("--token-budget",   type=int, default=64)
    p.add_argument("--env-token-budget", type=int, default=None)
    p.add_argument("--temperature",    type=float, default=1.0)
    p.add_argument("--seed",           type=int, default=20_000,
                   help="Eval seed disjoint from training/eval seeds.")
    p.add_argument("--role-shuffle",   action="store_true")
    p.add_argument("--output",         required=True,
                   help="Path for the JSON results file.")
    return p.parse_args()


def _restore_agent(agent, checkpoint_path: str | None, agent_key_in_ckpt: str):
    if not checkpoint_path:
        return
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    states = payload["agent_states"]
    # Try the requested key, fall back to the first available agent.
    key = agent_key_in_ckpt if agent_key_in_ckpt in states else next(iter(states))
    agent._backbone.load_state_dict(states[key]["backbone"])
    agent._value_head.load_state_dict(states[key]["value_head"])
    print(f"  restored {agent.agent_id} from {checkpoint_path} [key={key}]")


def _parse_first_offer(text: str) -> tuple[int, int, int] | None:
    """Best-effort parse of an `books=N hats=N balls=N` triple from utterance text."""
    import re
    m = re.search(r"books\s*=\s*(\d+).*?hats\s*=\s*(\d+).*?balls\s*=\s*(\d+)",
                  text, re.IGNORECASE | re.DOTALL)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))
    return None


def _rollout(agent_for_focal, focal_prompt, opponent_agent, opponent_prompt,
             env, tokeniser, n_episodes, token_budget, temperature, seed_start):
    """Roll out n_episodes with the given (focal, opponent) pair.

    Records the focal agent's first utterance text and the episode outcome.
    Returns ``(outcomes, first_utterances)``.
    """
    from marlllm.eval_utils import NegotiationOutcome
    outcomes: list[NegotiationOutcome] = []
    first_utterances: list[str] = []

    agents = {agent_for_focal.agent_id: agent_for_focal,
              opponent_agent.agent_id: opponent_agent}
    prompts = {agent_for_focal.agent_id: focal_prompt,
               opponent_agent.agent_id: opponent_prompt}

    for ep in range(n_episodes):
        env.reset(seed=seed_start + ep)
        contexts: dict[str, list[int]] = {}
        for aid, agent in agents.items():
            pids = tokeniser.encode_prompt(prompts[aid])
            contexts[aid] = agent.context_formatter.wrap_prompt(pids)
        first_seen: bool = False

        for agent_id in env.agent_iter():
            obs, _r, term, trunc, info = env.last()
            obs_ids = tokeniser.encode_observation(obs)
            if obs_ids:
                fmt = agents[agent_id].context_formatter
                contexts[agent_id].extend(fmt.wrap_observation(obs_ids))
            if term or trunc:
                env.step(None)
                continue
            with torch.no_grad():
                act_ids, _ = agents[agent_id].act(
                    context_token_ids=contexts[agent_id],
                    n_tokens=token_budget,
                    temperature=temperature,
                )
            fmt = agents[agent_id].context_formatter
            contexts[agent_id].extend(fmt.wrap_action(act_ids))
            if agent_id == agent_for_focal.agent_id and not first_seen:
                tok = agents[agent_id].tokenizer
                first_utterances.append(tok.decode(act_ids, skip_special_tokens=True))
                first_seen = True
            env.step(act_ids)

        items   = tuple(env._items)
        deal    = (env._deal_valid)
        outcomes.append(NegotiationOutcome(
            deal=deal,
            score_a=env._episode_scores.get("agent_0", 0),
            score_b=env._episode_scores.get("agent_1", 0),
            items=items,
            values_a=tuple(env._values["agent_0"]),
            values_b=tuple(env._values["agent_1"]),
        ))
    return outcomes, first_utterances


def _surprise_on_text(agent, text: str) -> float:
    """Mean -log p_θ(token | prefix) over a candidate utterance text."""
    if not text.strip():
        return 0.0
    tok = agent.tokenizer
    ids = tok.encode(text, add_special_tokens=False)
    if len(ids) < 2:
        return 0.0
    inp = torch.tensor([ids], dtype=torch.long, device=agent.device)
    with torch.no_grad():
        out = agent._backbone(input_ids=inp, use_cache=False)
        logits = out.logits[:, :-1, :]
        targets = inp[:, 1:]
        nll = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            reduction="mean",
        ).item()
    return float(nll)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    from marlllm import IndependentAgent, TextTokeniser
    from marlllm.eval_utils import summarise
    from envs.deal_or_no_deal_env import DealOrNoDealEnv

    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    torch_dtype = dtype_map.get(args.dtype, "auto")

    device_a = args.device_a or _auto_device(0, "cpu")
    device_b = args.device_b or _auto_device(1, device_a)

    print("Loading agent_a (under prompt A)…")
    agent_a = IndependentAgent(
        agent_id="agent_0",
        character_prompt=args.prompt_a,
        model_name_or_path=args.model,
        device=device_a,
        torch_dtype=torch_dtype,
        attn_implementation=args.attn_impl,
    )
    print("Loading agent_b (under prompt B)…")
    agent_b = IndependentAgent(
        agent_id="agent_0",   # share id so checkpoints map naturally
        character_prompt=args.prompt_b,
        model_name_or_path=args.model,
        device=device_b,
        torch_dtype=torch_dtype,
        attn_implementation=args.attn_impl,
    )
    _restore_agent(agent_a, args.checkpoint_a, "agent_0")
    _restore_agent(agent_b, args.checkpoint_b, "agent_0")
    agent_a.eval_mode()
    agent_b.eval_mode()

    # Opponents are pretrained-baseline copies (no checkpoint) so the focal
    # agent's behaviour reflects the prompt + training, not co-adaptation with
    # a partner that also varies. This is the asymmetric setup from §2.4.
    print("Loading frozen pretrained opponent…")
    opponent = IndependentAgent(
        agent_id="agent_1",
        character_prompt="You are Agent B, negotiating to maximise your score.",
        model_name_or_path=args.model,
        device=device_b,
        torch_dtype=torch_dtype,
        attn_implementation=args.attn_impl,
    )
    opponent.eval_mode()

    env = DealOrNoDealEnv(
        tokenizer=agent_a.tokenizer,
        max_dialogue_turns=args.dialogue_turns,
        action_token_budget=args.token_budget,
        env_token_budget=args.env_token_budget,
        seed=args.seed,
        role_shuffle=args.role_shuffle,
    )
    tokeniser = TextTokeniser(agent_a.tokenizer)

    print(f"Rolling out {args.n_episodes} episodes per condition…")
    t0 = time.time()
    outs_a, first_a = _rollout(
        agent_a, args.prompt_a, opponent,
        "You are Agent B, negotiating to maximise your score.",
        env, tokeniser, args.n_episodes, args.token_budget,
        args.temperature, args.seed,
    )
    outs_b, first_b = _rollout(
        agent_b, args.prompt_b, opponent,
        "You are Agent B, negotiating to maximise your score.",
        env, tokeniser, args.n_episodes, args.token_budget,
        args.temperature, args.seed,
    )

    # ── Behavioural statistics ───────────────────────────────────────────────
    summ_a = summarise(outs_a)
    summ_b = summarise(outs_b)

    def _mean_first_offer(utterances):
        offers = [_parse_first_offer(t) for t in utterances]
        offers = [o for o in offers if o is not None]
        if not offers:
            return None
        return [sum(o[i] for o in offers) / len(offers) for i in range(3)]

    divergence = {
        "agreement_rate_diff":     summ_a["agreement_rate"]    - summ_b["agreement_rate"],
        "joint_utility_diff":      summ_a["joint_utility_mean_all"] - summ_b["joint_utility_mean_all"],
        "individual_a_utility_diff": summ_a["individual_utility_a_all"] - summ_b["individual_utility_a_all"],
        "first_offer_a_mean":      _mean_first_offer(first_a),
        "first_offer_b_mean":      _mean_first_offer(first_b),
        "first_offer_parsed_rate_a": sum(1 for u in first_a if _parse_first_offer(u) is not None) / max(1, len(first_a)),
        "first_offer_parsed_rate_b": sum(1 for u in first_b if _parse_first_offer(u) is not None) / max(1, len(first_b)),
    }

    # ── Cross-character surprise ─────────────────────────────────────────────
    # For each focal-A trajectory, measure mean surprise on the focal utterance
    # under (model_a, model_b). The own-model surprise should be lower than
    # the cross-model surprise if the character is being internalised.
    cross = {
        "self_surp_a_on_a":  sum(_surprise_on_text(agent_a, u) for u in first_a) / max(1, len(first_a)),
        "cross_surp_b_on_a": sum(_surprise_on_text(agent_b, u) for u in first_a) / max(1, len(first_a)),
        "self_surp_b_on_b":  sum(_surprise_on_text(agent_b, u) for u in first_b) / max(1, len(first_b)),
        "cross_surp_a_on_b": sum(_surprise_on_text(agent_a, u) for u in first_b) / max(1, len(first_b)),
    }
    cross["surprise_gap_a"] = cross["cross_surp_b_on_a"] - cross["self_surp_a_on_a"]
    cross["surprise_gap_b"] = cross["cross_surp_a_on_b"] - cross["self_surp_b_on_b"]

    payload = {
        "model":         args.model,
        "checkpoint_a":  args.checkpoint_a,
        "checkpoint_b":  args.checkpoint_b,
        "prompt_a":      args.prompt_a,
        "prompt_b":      args.prompt_b,
        "n_episodes":    args.n_episodes,
        "summary_a":     summ_a,
        "summary_b":     summ_b,
        "divergence":    divergence,
        "cross_character_surprise": cross,
        "first_utterances_a": first_a,
        "first_utterances_b": first_b,
        "wall_time_seconds": time.time() - t0,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)

    print()
    print(f"Agreement rate    A vs B    : {summ_a['agreement_rate']:.3f}  vs  {summ_b['agreement_rate']:.3f}")
    print(f"Joint utility     A vs B    : {summ_a['joint_utility_mean_all']:.3f}  vs  {summ_b['joint_utility_mean_all']:.3f}")
    print(f"Surprise gap A (cross-self) : {cross['surprise_gap_a']:+.3f}")
    print(f"Surprise gap B (cross-self) : {cross['surprise_gap_b']:+.3f}")
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
