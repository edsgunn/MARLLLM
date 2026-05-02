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

    for agent_id, agent in population.items():
        if id(agent) in seen_objects:
            agents_out[agent_id] = agents_out[seen_objects[id(agent)]]
            continue
        seen_objects[id(agent)] = agent_id

        try:
            agent.eval_mode()
        except Exception:
            pass

        per_agent: dict[str, list[dict]] = {}
        for ctx in eval_data["contexts"]:
            prompt_ids = _build_prompt_ids(eval_data, ctx, tokenizer)
            samples: list[dict] = []
            for _ in range(samples_per_context):
                try:
                    token_ids, _logp = agent.act(
                        context_token_ids=list(prompt_ids),
                        n_tokens=max_new_tokens,
                        temperature=temperature,
                        eos_token_ids=eos_token_ids,
                    )
                except Exception as e:
                    _LOG.warning(
                        "Snapshot sampling failed for agent %s ctx %s: %s",
                        agent_id, ctx.get("id"), e,
                    )
                    samples.append({"text": "", "tokens": 0, "error": str(e)})
                    continue
                text = tokenizer.decode(token_ids, skip_special_tokens=False)
                samples.append({"text": text, "tokens": len(token_ids)})
            per_agent[ctx["id"]] = samples

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
