"""
Multi-agent episode trace formatting.

Two output formats are provided:

``format_trace_json``
    Structured dict suitable for JSON serialisation.  Contains episode
    metadata, a chronological environment log, and per-agent context views
    with typed turn entries (prompt / obs / act).

``format_trace``
    Human-readable text.  The per-agent sections show the context window
    exactly as the model sees it, with optional lightweight turn markers
    (``── obs ──`` / ``── act ──``) that aid readability but are clearly
    labelled as annotations.  Token IDs are never shown.  Ghost agents
    (population members not participating in this episode) are omitted.

Environment overview
    Both formats include a chronological event log showing who sent what to
    whom, with routing arrows where an agent's action was delivered as
    another agent's observation.

Routing detection
    Uses suffix-matching on raw token IDs (the ``_combined`` trajectory
    stores unformatted tokens exactly as the environment produced them).
"""
from __future__ import annotations

from marlllm.types import EpisodeStep, TokenType, Trajectory

_SEP = "─" * 72
_SEP_THICK = "━" * 72


# ── Private helpers ───────────────────────────────────────────────────────────

def _decode(tokenizer, token_ids: list[int]) -> str:
    return tokenizer.decode(token_ids, skip_special_tokens=False)


def _participating(steps: list[EpisodeStep]) -> list[str]:
    """Return agents that have at least one OBS or ACT step (no ghost agents)."""
    seen: list[str] = []
    for step in steps:
        if step.agent_id not in seen:
            seen.append(step.agent_id)
    return seen


def _infer_obs_source(steps: list[EpisodeStep], obs_idx: int) -> str | None:
    """
    Return the agent_id whose ACT tokens are a suffix of this OBS step, or
    None if the observation came directly from the environment.
    """
    obs_step = steps[obs_idx]
    obs_ids = obs_step.token_ids

    for j in range(obs_idx - 1, -1, -1):
        prev = steps[j]
        if prev.token_type == TokenType.ACT:
            if prev.agent_id != obs_step.agent_id:
                act_ids = prev.token_ids
                if (len(obs_ids) >= len(act_ids)
                        and obs_ids[-len(act_ids):] == act_ids):
                    return prev.agent_id
            break
    return None


def _routed_to(steps: list[EpisodeStep], act_idx: int) -> str | None:
    """Return the agent_id that received this ACT as a routed observation."""
    agent_id = steps[act_idx].agent_id
    for j in range(act_idx + 1, len(steps)):
        nxt = steps[j]
        if nxt.token_type == TokenType.OBS:
            if _infer_obs_source(steps, j) == agent_id:
                return nxt.agent_id
            break
        if nxt.token_type == TokenType.ACT:
            break
    return None


# ── JSON format ───────────────────────────────────────────────────────────────

def format_trace_json(
    iteration: int | str,
    traj: Trajectory,
    ep_info: dict,
    tokenizer,
    character_prompts: dict[str, str | list[str]],
) -> dict:
    """
    Return a structured dict representing the full episode trace.

    Schema
    ------
    {
      "meta": {"iteration": int, ...ep_info fields...},
      "participating_agents": ["agent_0", "agent_1", ...],
      "environment_log": [
        {
          "index": int,
          "agent": str,
          "type": "obs" | "act",
          "text": str,
          "source_type": "env" | "agent" | "agent+env",   # obs only
          "source_agent": str | null,                      # obs only
          "env_prefix": str | null,                        # obs only, when agent+env
          "routed_to": str | null,                         # act only
        },
        ...
      ],
      "agent_contexts": {
        "agent_0": [
          {"type": "prompt", "text": str},
          {"type": "obs",
           "text": str,
           "source_type": "env" | "agent" | "agent+env",
           "source_agent": str | null,
           "env_prefix": str | null},
          {"type": "act", "text": str},
          ...
        ],
        ...
      }
    }
    """
    steps = traj.steps
    active_agents = _participating(steps)

    # ── meta ──────────────────────────────────────────────────────────────
    meta = {"iteration": iteration}
    meta.update(ep_info)

    # ── environment log ───────────────────────────────────────────────────
    env_log = []
    for i, step in enumerate(steps):
        if step.token_type == TokenType.PAD:
            continue
        text = _decode(tokenizer, step.token_ids)
        entry: dict = {
            "index": i,
            "agent": step.agent_id,
            "type": "act" if step.token_type == TokenType.ACT else "obs",
            "text": text,
        }
        if step.token_type == TokenType.OBS:
            source = _infer_obs_source(steps, i)
            if source is None:
                entry["source_type"] = "env"
                entry["source_agent"] = None
                entry["env_prefix"] = None
            else:
                for j in range(i - 1, -1, -1):
                    if (steps[j].token_type == TokenType.ACT
                            and steps[j].agent_id == source):
                        src_act_ids = steps[j].token_ids
                        break
                else:
                    src_act_ids = []
                env_prefix_len = len(step.token_ids) - len(src_act_ids)
                if env_prefix_len > 0:
                    entry["source_type"] = "agent+env"
                    entry["env_prefix"] = _decode(tokenizer, step.token_ids[:env_prefix_len])
                else:
                    entry["source_type"] = "agent"
                    entry["env_prefix"] = None
                entry["source_agent"] = source
        else:  # ACT
            entry["routed_to"] = _routed_to(steps, i)
        env_log.append(entry)

    # ── per-agent context views ───────────────────────────────────────────
    agent_contexts: dict[str, list[dict]] = {}
    for aid in active_agents:
        turns: list[dict] = []

        prompt_val = character_prompts.get(aid, "")
        prompt_text = prompt_val[0] if isinstance(prompt_val, list) else prompt_val
        turns.append({"type": "prompt", "text": prompt_text})

        for i, step in enumerate(steps):
            if step.token_type == TokenType.PAD or step.agent_id != aid:
                continue
            text = _decode(tokenizer, step.token_ids)
            if step.token_type == TokenType.OBS:
                source = _infer_obs_source(steps, i)
                if source is None:
                    turns.append({
                        "type": "obs",
                        "text": text,
                        "source_type": "env",
                        "source_agent": None,
                        "env_prefix": None,
                    })
                else:
                    for j in range(i - 1, -1, -1):
                        if (steps[j].token_type == TokenType.ACT
                                and steps[j].agent_id == source):
                            src_act_ids = steps[j].token_ids
                            break
                    else:
                        src_act_ids = []
                    env_prefix_len = len(step.token_ids) - len(src_act_ids)
                    turns.append({
                        "type": "obs",
                        "text": text,
                        "source_type": "agent+env" if env_prefix_len > 0 else "agent",
                        "source_agent": source,
                        "env_prefix": (
                            _decode(tokenizer, step.token_ids[:env_prefix_len])
                            if env_prefix_len > 0 else None
                        ),
                    })
            elif step.token_type == TokenType.ACT:
                turns.append({"type": "act", "text": text})

        agent_contexts[aid] = turns

    return {
        "meta": meta,
        "participating_agents": active_agents,
        "environment_log": env_log,
        "agent_contexts": agent_contexts,
    }


# ── Text format ───────────────────────────────────────────────────────────────

def format_trace(
    iteration: int | str,
    traj: Trajectory,
    ep_info: dict,
    tokenizer,
    character_prompts: dict[str, str | list[str]],
    annotate_turns: bool = True,
) -> str:
    """
    Format a multi-agent episode trace as a human-readable string.

    Parameters
    ----------
    iteration:
        Iteration number or label.
    traj:
        Combined trajectory (all agents, chronological, raw token IDs).
    ep_info:
        Episode metadata dict.
    tokenizer:
        HuggingFace tokenizer.
    character_prompts:
        Mapping agent_id → prompt text.
    annotate_turns : bool
        If True (default) insert subtle ``── obs ──`` / ``── act ──`` markers
        in the per-agent context sections.  These are *not* in the model's
        actual context; they are readability annotations only.
    """
    steps = traj.steps
    active_agents = _participating(steps)

    lines: list[str] = []

    # ── Header ────────────────────────────────────────────────────────────
    lines.append(f"=== Iteration {iteration} — Episode Trace ===")
    for k, v in ep_info.items():
        lines.append(f"{k + ':':20s}{v}")
    lines.append("")

    # ── Section 1: Environment Overview ──────────────────────────────────
    lines.append(_SEP_THICK)
    lines.append("ENVIRONMENT OVERVIEW  (chronological event log)")
    lines.append(_SEP_THICK)
    lines.append("")
    lines.append(
        "  Legend:  [ENV → X]       environment sends observation to agent X"
    )
    lines.append(
        "           [X → ENV]       agent X sends action to environment"
    )
    lines.append(
        "           [Y+ENV → X]     agent Y's action + env prefix delivered to X"
    )
    lines.append(
        "           [Y → X]         agent Y's action routed directly to X"
    )
    lines.append("")

    for i, step in enumerate(steps):
        if step.token_type == TokenType.PAD:
            continue

        agent_id = step.agent_id
        decoded = _decode(tokenizer, step.token_ids)

        if step.token_type == TokenType.ACT:
            routed_to = _routed_to(steps, i)
            lines.append(f"  ┌─ [{agent_id} → ENV]  action")
            lines.append(f"  └─ {decoded!r}")
            if routed_to:
                lines.append(f"     ↓  routed as observation to {routed_to}")
            lines.append("")

        elif step.token_type == TokenType.OBS:
            source = _infer_obs_source(steps, i)
            if source is not None:
                src_act_ids: list[int] = []
                for j in range(i - 1, -1, -1):
                    if (steps[j].token_type == TokenType.ACT
                            and steps[j].agent_id == source):
                        src_act_ids = steps[j].token_ids
                        break
                env_prefix_len = len(step.token_ids) - len(src_act_ids)
                tag = (
                    f"[{source}+ENV → {agent_id}]"
                    if env_prefix_len > 0
                    else f"[{source} → {agent_id}]"
                )
                lines.append(f"  ┌─ {tag}  observation")
                lines.append(f"  └─ {decoded!r}")
                if env_prefix_len > 0:
                    env_part = _decode(tokenizer, step.token_ids[:env_prefix_len])
                    src_part = _decode(tokenizer, step.token_ids[env_prefix_len:])
                    lines.append(f"     ├─ env prefix:  {env_part!r}")
                    lines.append(f"     └─ from {source}: {src_part!r}")
            else:
                lines.append(f"  ┌─ [ENV → {agent_id}]  observation")
                lines.append(f"  └─ {decoded!r}")
            lines.append("")

    lines.append("")

    # ── Section 2: Per-agent context views ───────────────────────────────
    if annotate_turns:
        lines.append(
            "  NOTE: ── obs ── / ── act ── markers below are readability annotations"
        )
        lines.append(
            "  only. They are NOT present in the model's actual context window.")
        lines.append("")

    for aid in active_agents:
        lines.append(_SEP_THICK)
        lines.append(f"CONTEXT: {aid}")
        lines.append(_SEP_THICK)
        lines.append("")

        prompt_val = character_prompts.get(aid, "(none)")
        prompt_text = prompt_val[0] if isinstance(prompt_val, list) else prompt_val
        lines.append(prompt_text)
        lines.append("")

        for i, step in enumerate(steps):
            if step.token_type == TokenType.PAD or step.agent_id != aid:
                continue

            decoded = _decode(tokenizer, step.token_ids)

            if step.token_type == TokenType.OBS:
                source = _infer_obs_source(steps, i)
                if source is not None:
                    src_act_ids = []
                    for j in range(i - 1, -1, -1):
                        if (steps[j].token_type == TokenType.ACT
                                and steps[j].agent_id == source):
                            src_act_ids = steps[j].token_ids
                            break
                    env_prefix_len = len(step.token_ids) - len(src_act_ids)
                    origin = (
                        f"from {source} via env"
                        if env_prefix_len == 0
                        else f"from {source} + env prefix"
                    )
                else:
                    origin = "from env"

                if annotate_turns:
                    lines.append(f"── obs ({origin}) ──")
                lines.append(decoded)
                lines.append("")

            elif step.token_type == TokenType.ACT:
                if annotate_turns:
                    lines.append("── act ──")
                lines.append(decoded)
                lines.append("")

        lines.append("")

    return "\n".join(lines) + "\n"
