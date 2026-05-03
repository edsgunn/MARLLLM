"""
Held-out behavioural-distribution snapshots for cultural-emergence experiments.

For each agent in a population, sample K outputs at training temperature on a
fixed list of held-out forum contexts.  These samples are the basis for the
post-hoc behavioural-distance measurements (KL between agent distributions,
character-differentiation metrics, etc.).

Eval-context schema (JSON file)
-------------------------------
    {
      "description":     str,
      "system_preamble": str,            # prepended to every context
      "contexts": [
        {"id": str, "thread": str},
        ...
      ]
    }

Output schema (one file per checkpoint)
---------------------------------------
    {
      "iteration":       int,
      "temperature":     float,
      "samples_per_ctx": int,
      "eval_set":        str,            # filename of the eval JSON
      "agents": {
        "<agent_id>": {
          "<context_id>": [
            {"text": str, "tokens": int},
            ...                          # K samples
          ],
          ...
        },
        ...
      }
    }
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch

_LOG = logging.getLogger(__name__)


def load_eval_contexts(path: str | Path) -> dict:
    """Load a held-out eval-contexts JSON file."""
    with open(path) as f:
        data = json.load(f)
    if "contexts" not in data:
        raise ValueError(f"Eval contexts file {path} missing 'contexts' field.")
    data.setdefault("system_preamble", "")
    return data


def _build_prompt_ids(
    eval_data: dict,
    context: dict,
    tokenizer: Any,
) -> list[int]:
    """Tokenise system_preamble + thread as a chat-template prompt."""
    preamble = eval_data.get("system_preamble", "")
    thread = context["thread"]

    # Use the tokenizer's chat template so the prompt matches training format.
    messages = []
    if preamble:
        messages.append({"role": "system", "content": preamble.rstrip("\n ")})
    messages.append({"role": "user", "content": thread})
    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        ids = tokenizer.encode(text, add_special_tokens=False)
    except Exception:
        # Tokenizer doesn't support chat_template — fall back to raw concat.
        text = (preamble + "\n" + thread).strip()
        ids = tokenizer.encode(text, add_special_tokens=True)
    return [int(t) for t in ids]


@torch.no_grad()
def collect_snapshot(
    *,
    population: dict[str, Any],
    eval_data: dict,
    tokenizer: Any,
    samples_per_context: int = 8,
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    eos_token_ids: list[int] | None = None,
) -> dict[str, Any]:
    """Run each agent on each held-out context K times.

    Returns a dict in the schema documented at the top of the module.
    """
    agents_out: dict[str, dict[str, list[dict]]] = {}
    seen_objects: dict[int, str] = {}  # share results across tied agents

    # Pre-tokenise every context once (re-used across agents).
    context_ids = [
        (ctx["id"], _build_prompt_ids(eval_data, ctx, tokenizer))
        for ctx in eval_data["contexts"]
    ]

    for agent_id, agent in population.items():
        if id(agent) in seen_objects:
            agents_out[agent_id] = agents_out[seen_objects[id(agent)]]
            continue
        seen_objects[id(agent)] = agent_id

        try:
            agent.eval_mode()
        except Exception:
            pass

        # Flatten (context × samples_per_context) into a single batch so the
        # agent's batched generate path handles all rollouts in one go,
        # rather than one-at-a-time through the slow per-token loop.
        batch_contexts: list[list[int]] = []
        batch_meta: list[tuple[str, int]] = []  # (ctx_id, sample_idx)
        for ctx_id, prompt_ids in context_ids:
            for k in range(samples_per_context):
                batch_contexts.append(list(prompt_ids))
                batch_meta.append((ctx_id, k))

        per_agent: dict[str, list[dict]] = {cid: [] for cid, _ in context_ids}

        if batch_contexts:
            try:
                batch_ids, _batch_lps = agent.act_batch(
                    contexts=batch_contexts,
                    n_tokens=max_new_tokens,
                    temperature=temperature,
                    eos_token_ids=eos_token_ids,
                )
            except Exception as e:
                _LOG.warning(
                    "Snapshot batched sampling failed for agent %s: %s",
                    agent_id, e,
                )
                # Fill with error placeholders so output schema stays consistent.
                for ctx_id, _k in batch_meta:
                    per_agent[ctx_id].append(
                        {"text": "", "tokens": 0, "error": str(e)}
                    )
            else:
                for (ctx_id, _k), token_ids in zip(batch_meta, batch_ids):
                    text = tokenizer.decode(token_ids, skip_special_tokens=False)
                    per_agent[ctx_id].append(
                        {"text": text, "tokens": len(token_ids)}
                    )

        agents_out[agent_id] = per_agent

        try:
            agent.train_mode()
        except Exception:
            pass

    return {"agents": agents_out}


def write_snapshot(
    *,
    output_dir: Path,
    iteration: int,
    snapshot: dict[str, Any],
    eval_set_name: str,
    temperature: float,
    samples_per_context: int,
) -> Path:
    """Write a snapshot to ``{output_dir}/snapshots/iter_NNNNNN.json``."""
    snap_dir = Path(output_dir) / "snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "iteration": iteration,
        "temperature": temperature,
        "samples_per_ctx": samples_per_context,
        "eval_set": eval_set_name,
        "agents": snapshot["agents"],
    }
    path = snap_dir / f"iter_{iteration:06d}.json"
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path
