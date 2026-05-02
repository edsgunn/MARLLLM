"""
Unified episode trace serialisation (environment-agnostic).

Primary artefact: per-agent context-window token IDs at end of episode,
inclusive of all special tokens.  The decoded text is derived from these.

Schema (one record per episode)
--------------------------------
{
  "episode":   int,
  "env_trace": dict | null,   # optional; provided by env.episode_trace()
  "agents": {
    "<agent_id>": {
      "context_tokens": list[int],   # ground-truth token IDs
      "context_text":   str,         # decoded; special tokens visible
    },
    ...
  }
}

Environment trace convention
-----------------------------
Environments that want to contribute a structured trace should expose an
``episode_trace(self) -> dict`` method.  Its return value is stored verbatim
in ``env_trace``.  If the method is absent the field is null and the caller
should supply ep_info as a fallback.
"""
from __future__ import annotations

import json
from pathlib import Path

_THICK = "━" * 80
_RULE  = "─" * 80


def get_env_trace(env) -> dict | None:
    """Return env.episode_trace() if the environment provides it, else None."""
    fn = getattr(env, "episode_trace", None)
    if fn is None:
        return None
    try:
        return fn()
    except Exception:
        return None


def make_episode_record(
    episode_idx: int,
    agent_context_tokens: dict[str, list[int]],
    tokenizer,
    env_trace: dict | None = None,
) -> dict:
    """
    Build one episode trace record.

    Parameters
    ----------
    episode_idx:
        Zero-based episode index within the collection run.
    agent_context_tokens:
        agent_id → flat list of token IDs representing the agent's context
        window at end of episode (all special tokens included).
    tokenizer:
        HuggingFace tokenizer used to decode tokens to text.
    env_trace:
        Structured dict provided by the environment (or ep_info fallback).
    """
    agents = {
        aid: {
            "context_tokens": list(toks),
            "context_text": tokenizer.decode(toks, skip_special_tokens=False),
        }
        for aid, toks in agent_context_tokens.items()
    }
    return {
        "episode": episode_idx,
        "env_trace": env_trace,
        "agents": agents,
    }


def format_records_txt(records: list[dict]) -> str:
    """Human-readable rendering of a list of episode records."""
    lines: list[str] = []
    for rec in records:
        lines.append(_THICK)
        env_t = rec.get("env_trace") or {}
        summary = "  ".join(f"{k}={v}" for k, v in env_t.items()) if env_t else ""
        lines.append(f"EPISODE {rec['episode']}  {summary}")
        lines.append(_THICK)
        for aid, at in rec["agents"].items():
            lines.append("")
            lines.append(_RULE)
            lines.append(f"AGENT {aid}")
            lines.append(_RULE)
            lines.append(at["context_text"])
        lines.append("")
    return "\n".join(lines) + "\n"


def write_records_json(records: list[dict], path) -> None:
    """Write list of records to a JSON file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(records, f, indent=2)


def write_records_txt(records: list[dict], path) -> None:
    """Write human-readable rendering to a text file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(format_records_txt(records))
