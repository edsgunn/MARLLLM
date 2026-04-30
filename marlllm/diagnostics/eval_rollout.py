"""
Collect a fixed eval-rollout set for a trained checkpoint.

Used by every diagnostic that needs "what does this model produce in matched
scenarios?" The output is the same JSONL schema as `dump_transcripts.py`.

The learner role is loaded from a HF checkpoint directory; partners can be
either fresh APIAgents or another local model.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from marlllm import IndependentAgent, APIAgent, SamplingParams, make_api_model
from envs.deal_or_no_deal_env import DealOrNoDealEnv


@dataclass
class EvalRolloutConfig:
    learner_checkpoint: str       # HF directory
    learner_role: str             # "agent_0" | "agent_1"
    learner_prompt: str
    partner_provider: str = "anthropic"
    partner_model: str = "claude-sonnet-4-6"
    partner_prompt: str = "You are Agent B, negotiating to maximise your score."
    partner_kind: str = "api"     # "api" | "local"
    partner_local_path: str | None = None
    cache_dir: str | None = None
    budget_state: str | None = None
    budget_cap_usd: float | None = None
    episodes: int = 32
    seed: int = 0
    dialogue_turns: int = 10
    token_budget: int = 128
    temperature: float = 0.7
    max_tokens: int = 256
    device: str = "cuda:0"
    dump_hidden_states: bool = False  # for the probing diagnostic


def _build_learner_agent(cfg: EvalRolloutConfig):
    tokenizer = AutoTokenizer.from_pretrained(cfg.learner_checkpoint)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    agent = IndependentAgent(
        agent_id=cfg.learner_role,
        character_prompt=cfg.learner_prompt,
        model_name_or_path=cfg.learner_checkpoint,
        device=cfg.device,
        torch_dtype="auto",
    )
    return agent, tokenizer


def _build_partner(cfg: EvalRolloutConfig, tokenizer):
    if cfg.partner_kind == "api":
        api = make_api_model(
            provider=cfg.partner_provider, model=cfg.partner_model,
            cache_dir=cfg.cache_dir,
            budget_state_path=cfg.budget_state,
            budget_cap_usd=cfg.budget_cap_usd,
        )
        partner_role = "agent_1" if cfg.learner_role == "agent_0" else "agent_0"
        return APIAgent(
            agent_id=partner_role,
            character_prompt=cfg.partner_prompt,
            api_model=api,
            tokenizer=tokenizer,
            sampling=SamplingParams(temperature=cfg.temperature, max_tokens=cfg.max_tokens),
        )
    elif cfg.partner_kind == "local":
        partner_role = "agent_1" if cfg.learner_role == "agent_0" else "agent_0"
        return IndependentAgent(
            agent_id=partner_role,
            character_prompt=cfg.partner_prompt,
            model_name_or_path=cfg.partner_local_path or cfg.learner_checkpoint,
            device=cfg.device,
            torch_dtype="auto",
        )
    raise ValueError(f"Unknown partner_kind: {cfg.partner_kind}")


def collect_eval_rollouts(cfg: EvalRolloutConfig, output_path: str | Path) -> Path:
    """Run `cfg.episodes` rollouts and write JSONL transcripts.

    Returns the path written. If `cfg.dump_hidden_states`, a sibling .pt file
    holds a dict mapping turn index → hidden-state tensor at action positions
    for the learner; this is consumed by the probing diagnostic.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    learner, tokenizer = _build_learner_agent(cfg)
    partner = _build_partner(cfg, tokenizer)
    agents = {learner.agent_id: learner, partner.agent_id: partner}

    env = DealOrNoDealEnv(
        tokenizer=tokenizer,
        max_dialogue_turns=cfg.dialogue_turns,
        action_token_budget=cfg.token_budget,
        seed=cfg.seed,
    )

    hidden_states_log: list[dict] = []  # per learner-action snapshot

    f = output_path.open("w")
    for ep_idx in range(cfg.episodes):
        env.reset(seed=cfg.seed + ep_idx)
        if hasattr(partner, "reset_history"):
            partner.reset_history(slot=0)
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
            obs_ids = tokenizer.encode(obs_text, add_special_tokens=False) if obs_text else []
            if obs_ids:
                contexts[agent_id].extend(
                    agents[agent_id].context_formatter.wrap_observation(obs_ids)
                )
                if hasattr(agents[agent_id], "note_observation"):
                    agents[agent_id].note_observation(obs_text, slot=0)
                token_count += len(obs_ids)
                turns.append({"agent_id": agent_id, "role": "obs",
                              "text": obs_text, "tokens": obs_ids})

            if term or trunc:
                env.step(None)
                continue
            if token_count >= 4096 and not info.get("must_act", False):
                env.step(None)
                continue

            if cfg.dump_hidden_states and agent_id == cfg.learner_role:
                ctx = contexts[agent_id]
                ids_t = torch.tensor([ctx], dtype=torch.long, device=learner.device)
                attn = torch.ones_like(ids_t)
                with torch.no_grad():
                    out = learner._backbone(input_ids=ids_t, attention_mask=attn,
                                            output_hidden_states=True, use_cache=False)
                last_h = out.hidden_states[-1][0, -1, :].detach().cpu().float().numpy().tolist()
                hidden_states_log.append({
                    "episode_id": ep_idx,
                    "turn_idx": len(turns),
                    "hidden": last_h,
                    "items": list(env._items),
                    "values": list(env._values.get(agent_id, [])),
                    "phase": env._phase,
                })

            act_ids, _ = agents[agent_id].act(
                context_token_ids=contexts[agent_id],
                n_tokens=cfg.token_budget,
                temperature=cfg.temperature,
            )
            act_text = tokenizer.decode(act_ids, skip_special_tokens=True)
            contexts[agent_id].extend(
                agents[agent_id].context_formatter.wrap_action(act_ids)
            )
            token_count += len(act_ids)
            turns.append({"agent_id": agent_id, "role": "act",
                          "text": act_text, "tokens": act_ids})
            env.step(act_ids)

        ep_info: dict = {}
        for aid in env.possible_agents:
            if aid in env.infos:
                ep_info = dict(env.infos[aid])
                break

        f.write(json.dumps({
            "episode_id": ep_idx,
            "seed": cfg.seed + ep_idx,
            "turns": turns,
            "outcome": ep_info,
        }, ensure_ascii=False) + "\n")
    f.close()

    if cfg.dump_hidden_states and hidden_states_log:
        import pickle
        with open(output_path.with_suffix(".hidden.pkl"), "wb") as fp:
            pickle.dump(hidden_states_log, fp)

    return output_path
