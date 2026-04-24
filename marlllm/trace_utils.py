"""
Multi-agent episode trace formatting.

Produces two sections per trace:

1. ENVIRONMENT OVERVIEW — chronological event log showing who sends what to
   whom.  Each agent action that becomes another agent's observation is
   annotated with an arrow so the routing is explicit.

2. PER-AGENT CONTEXT — for each agent, the full token sequence as it appeared
   in their context window during inference (prompt + all obs broadcast by the
   trainer + their own actions interspersed).
"""
from __future__ import annotations

from marlllm.types import EpisodeStep, TokenType, Trajectory

_SEP = "─" * 72
_SEP_THICK = "━" * 72


def _decode(tokenizer, token_ids: list[int]) -> str:
    return tokenizer.decode(token_ids, skip_special_tokens=False)


def _ids_str(token_ids: list[int]) -> str:
    return " ".join(str(t) for t in token_ids)


def _infer_obs_source(steps: list[EpisodeStep], obs_idx: int) -> str | None:
    """
    Return the agent_id that produced the tokens in this OBS step, or None if
    the observation came directly from the environment.

    The deal-or-no-deal env routes an agent's action to the other agent by
    delivering the action tokens verbatim (possibly prepended with an env
    context string on the first dialogue turn).  We detect this by checking
    whether the most recent ACT step from a *different* agent is a suffix of
    the OBS token_ids.
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
            # Stop at the first ACT regardless of agent
            break
    return None


def format_trace(
    iteration: int | str,
    traj: Trajectory,
    ep_info: dict,
    tokenizer,
    character_prompts: dict[str, str],
) -> str:
    """
    Format a multi-agent episode trace as a human-readable string.

    Parameters
    ----------
    iteration:
        Iteration number (or a descriptive label for offline traces).
    traj:
        Trajectory produced by the episode collector.
    ep_info:
        Episode metadata dict from the environment (result, scores, …).
    tokenizer:
        HuggingFace tokenizer used to decode token IDs.
    character_prompts:
        Mapping agent_id → character prompt text used during this episode.
    """
    steps = traj.steps
    agent_ids = traj.agent_ids_present

    lines: list[str] = []

    # ── Header ────────────────────────────────────────────────────────────
    lines.append(f"=== Iteration {iteration} — Episode Trace ===")
    lines.append(f"result:         {ep_info.get('result', 'unknown')}")
    lines.append(f"correct_count:  {ep_info.get('correct_count', '?')}")
    lines.append(f"episode_length: {ep_info.get('episode_length', '?')}")
    # Show any extra ep_info fields (scores etc.)
    for k, v in ep_info.items():
        if k not in {"result", "correct_count", "episode_length"}:
            lines.append(f"{k + ':':16s}{v}")
    lines.append("")

    # ── Section 1: Environment Overview ──────────────────────────────────
    lines.append(_SEP_THICK)
    lines.append("ENVIRONMENT OVERVIEW  (chronological event log)")
    lines.append(_SEP_THICK)
    lines.append("")
    lines.append(
        "  Legend:  [ENV→X] = environment sends observation to agent X"
    )
    lines.append(
        "           [X→ENV] = agent X sends action to environment"
    )
    lines.append(
        "           [Y+ENV→X] = agent Y's action was appended to env context and delivered to X"
    )
    lines.append(
        "           [Y→X]  = agent Y's action routed directly as observation to X"
    )
    lines.append("")

    for i, step in enumerate(steps):
        agent_id = step.agent_id
        decoded = _decode(tokenizer, step.token_ids)
        ids = _ids_str(step.token_ids)

        if step.token_type == TokenType.PAD:
            lines.append(f"  ┌─ [PROMPT: {agent_id}]")
            lines.append(f"  │  {decoded!r}")
            lines.append(f"  └─ ids=[{ids}]")
            lines.append("")

        elif step.token_type == TokenType.ACT:
            # Peek ahead: does this action become the next agent's observation?
            routed_to: str | None = None
            for j in range(i + 1, len(steps)):
                nxt = steps[j]
                if nxt.token_type == TokenType.OBS:
                    if _infer_obs_source(steps, j) == agent_id:
                        routed_to = nxt.agent_id
                    break
                if nxt.token_type == TokenType.ACT:
                    break

            lines.append(f"  ┌─ [{agent_id} → ENV]  action")
            lines.append(f"  │  {decoded!r}")
            lines.append(f"  └─ ids=[{ids}]")
            if routed_to:
                lines.append(f"     ↓  routed as observation to {routed_to}")
            lines.append("")

        elif step.token_type == TokenType.OBS:
            source = _infer_obs_source(steps, i)

            if source is not None:
                # Find the source ACT to determine whether there is an env prefix
                src_act_ids: list[int] = []
                for j in range(i - 1, -1, -1):
                    if steps[j].token_type == TokenType.ACT and steps[j].agent_id == source:
                        src_act_ids = steps[j].token_ids
                        break

                env_prefix_len = len(step.token_ids) - len(src_act_ids)
                if env_prefix_len > 0:
                    tag = f"[{source}+ENV → {agent_id}]"
                else:
                    tag = f"[{source} → {agent_id}]"

                lines.append(f"  ┌─ {tag}  observation")
                lines.append(f"  │  {decoded!r}")
                lines.append(f"  └─ ids=[{ids}]")

                if env_prefix_len > 0:
                    env_part = _decode(tokenizer, step.token_ids[:env_prefix_len])
                    src_part = _decode(tokenizer, step.token_ids[env_prefix_len:])
                    lines.append(f"     ├─ ENV prefix ({env_prefix_len} tokens):  {env_part!r}")
                    lines.append(f"     └─ from {source} ({len(src_act_ids)} tokens): {src_part!r}")
            else:
                lines.append(f"  ┌─ [ENV → {agent_id}]  observation")
                lines.append(f"  │  {decoded!r}")
                lines.append(f"  └─ ids=[{ids}]")

            lines.append("")

    lines.append("")

    # ── Section 2: Per-agent context views ───────────────────────────────
    for aid in agent_ids:
        lines.append(_SEP_THICK)
        lines.append(f"CONTEXT: {aid}  (tokens in this agent's context window)")
        lines.append(_SEP_THICK)
        lines.append("")
        lines.append(
            "  Note: the trainer broadcasts all OBS to every agent's context."
        )
        lines.append(
            "  OBS steps marked [BROADCAST] were originally addressed to a"
        )
        lines.append(
            "  different agent but appear in this context due to that design."
        )
        lines.append("")

        prompt_text = character_prompts.get(aid, "(none)")
        lines.append(f"  [PROMPT]")
        lines.append(f"    {prompt_text!r}")
        lines.append("")

        for i, step in enumerate(steps):
            if step.token_type == TokenType.PAD:
                continue  # shown above as [PROMPT]

            decoded = _decode(tokenizer, step.token_ids)
            ids = _ids_str(step.token_ids)

            if step.token_type == TokenType.OBS:
                # All OBS go to all contexts (trainer broadcast behaviour)
                source = _infer_obs_source(steps, i)
                addressed_to = step.agent_id

                if addressed_to != aid:
                    broadcast_note = f"  [BROADCAST — addressed to {addressed_to}]"
                else:
                    broadcast_note = ""

                if source is not None:
                    src_act_ids = []
                    for j in range(i - 1, -1, -1):
                        if steps[j].token_type == TokenType.ACT and steps[j].agent_id == source:
                            src_act_ids = steps[j].token_ids
                            break
                    env_prefix_len = len(step.token_ids) - len(src_act_ids)
                    origin = (
                        f"routed from {source} via env"
                        if env_prefix_len == 0
                        else f"routed from {source} + env context prefix"
                    )
                else:
                    origin = "from ENV"

                lines.append(f"  [OBS]{broadcast_note}")
                lines.append(f"    origin: {origin}")
                lines.append(f"    {decoded!r}")
                lines.append(f"    ids=[{ids}]")
                lines.append("")

            elif step.token_type == TokenType.ACT and step.agent_id == aid:
                lines.append(f"  [ACT]")
                lines.append(f"    {decoded!r}")
                lines.append(f"    ids=[{ids}]")
                lines.append("")

        lines.append("")

    return "\n".join(lines) + "\n"
