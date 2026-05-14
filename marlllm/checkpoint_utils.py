"""
Per-agent checkpoint extraction and loading.

Each agent is saved to its own ``.pt`` file:

    {checkpoints_dir}/iter_NNNNNN/
        meta.pt              # iteration, optimizer, rng, config, agent list
        <agent_id>.pt        # per-agent weights (LoRA-only for shared-base agents)
        ...

Per-agent payload schema
------------------------
    {
      "agent_id":     str,
      "agent_type":   "lora_shared" | "independent",
      "adapter_name": str | None,        # for lora_shared
      "adapter":      dict[str, Tensor], # LoRA weights only when lora_shared
      "backbone":     dict[str, Tensor], # full backbone when independent
      "value_head":   dict[str, Tensor],
    }

For ``LoRASharedBaseAgent`` we save only the parameters whose name matches
``f".{adapter_name}."``.  This keeps each migrant payload to a few MB
(LoRA r64 on Qwen2.5-1.5B is ~30 MB) and makes adapter migration trivial:
load one agent's file into a target population's matching agent slot.

For ``IndependentAgent`` we save the full backbone state_dict.

Migration
---------
``inject_agent_state`` overwrites a target agent's weights with another
agent's saved state.  When the source is LoRA-shared and the target is
LoRA-shared with a different ``adapter_name``, the loader rewrites the
key prefixes so the source's adapter_name is mapped to the target's.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

_LOG = logging.getLogger(__name__)


def _is_lora_shared(agent: Any) -> bool:
    return type(agent).__name__ == "LoRASharedBaseAgent"


def _is_independent(agent: Any) -> bool:
    return type(agent).__name__ == "IndependentAgent"


def extract_agent_state(agent: Any) -> dict[str, Any]:
    """Build a per-agent checkpoint payload (LoRA-only for shared-base agents).

    Under FSDP the backbone parameters are sharded across ranks. Callers
    must enter ``marlllm.fsdp_utils.summon_full_params(backbone)`` on
    every rank before invoking this function for FSDP-wrapped agents —
    ``save_population_checkpoint`` does that automatically.
    """
    payload: dict[str, Any] = {
        "agent_id": getattr(agent, "_agent_id", None) or getattr(agent, "agent_id", None),
        "value_head": agent._value_head.state_dict(),
    }
    if _is_lora_shared(agent):
        adapter_name = agent._adapter_name
        payload["agent_type"] = "lora_shared"
        payload["adapter_name"] = adapter_name
        adapter_state = {
            name: param.detach().cpu().clone()
            for name, param in agent._backbone.named_parameters()
            if f".{adapter_name}." in name
        }
        payload["adapter"] = adapter_state
        payload["adapter_param_count"] = len(adapter_state)
    elif _is_independent(agent):
        payload["agent_type"] = "independent"
        payload["adapter_name"] = None
        payload["backbone"] = agent._backbone.state_dict()
    else:
        # Fallback: dump everything, type-tagged as "unknown"
        payload["agent_type"] = "unknown"
        payload["backbone"] = agent._backbone.state_dict()
    return payload


def inject_agent_state(
    agent: Any,
    payload: dict[str, Any],
    *,
    strict: bool = True,
) -> None:
    """Load a saved per-agent payload into ``agent``.

    Adapter-name remapping
    ----------------------
    If both source and target are ``lora_shared`` but have different
    ``adapter_name``s (e.g. migrating "Silas" from population A's slot
    "agent_2" to population B's slot "agent_5"), the source keys'
    ``.<adapter_name>.`` infix is rewritten to the target's adapter name
    before loading.
    """
    src_type = payload.get("agent_type", "unknown")
    if _is_lora_shared(agent):
        if src_type != "lora_shared":
            raise ValueError(
                f"Cannot inject {src_type!r} state into LoRASharedBaseAgent."
            )
        target_adapter = agent._adapter_name
        src_adapter = payload.get("adapter_name", target_adapter)
        adapter_state = payload["adapter"]
        if src_adapter != target_adapter:
            remapped = {
                k.replace(f".{src_adapter}.", f".{target_adapter}."): v
                for k, v in adapter_state.items()
            }
            adapter_state = remapped
            _LOG.info(
                "Adapter-name remap during injection: %s -> %s (%d tensors)",
                src_adapter, target_adapter, len(adapter_state),
            )
        # Load by direct parameter assignment to avoid disturbing the base
        # weights or the other adapter's params.
        target_params = dict(agent._backbone.named_parameters())
        missing = []
        loaded = 0
        for k, v in adapter_state.items():
            tgt = target_params.get(k)
            if tgt is None:
                missing.append(k)
                continue
            with torch.no_grad():
                tgt.data.copy_(v.to(tgt.device, dtype=tgt.dtype))
            loaded += 1
        if missing and strict:
            raise KeyError(
                f"Missing target params for {len(missing)} adapter keys "
                f"(first: {missing[:3]})"
            )
        if missing:
            _LOG.warning("Skipped %d unmatched adapter keys", len(missing))
        agent._value_head.load_state_dict(payload["value_head"])
        _LOG.info("Loaded %d adapter tensors + value_head into %s",
                  loaded, agent._agent_id)
    elif _is_independent(agent):
        if src_type == "lora_shared":
            raise ValueError(
                "Cannot inject lora_shared state into IndependentAgent: "
                "no shared backbone to anchor LoRA delta."
            )
        agent._backbone.load_state_dict(payload["backbone"])
        agent._value_head.load_state_dict(payload["value_head"])
    else:
        # Best-effort fallback
        if "backbone" in payload:
            agent._backbone.load_state_dict(payload["backbone"])
        agent._value_head.load_state_dict(payload["value_head"])


# ---------------------------------------------------------------------------
# Directory-level helpers
# ---------------------------------------------------------------------------


def save_population_checkpoint(
    *,
    population: dict[str, Any],
    iteration: int,
    optimizer: Any,
    config_dict: dict,
    output_dir: Path,
    extra_meta: dict | None = None,
    tag: str | None = None,
) -> Path:
    """Save a per-agent checkpoint directory.

    Layout::

        {output_dir}/checkpoints/iter_NNNNNN/
            meta.pt
            <agent_id>.pt   (one per unique agent object)

    When two slots in the population share the same agent object (e.g. a
    weight-tied population), only one ``.pt`` is written and ``meta.pt``
    records the slot→object mapping so loading reconstructs both.
    """
    # FSDP awareness: if any backbone is sharded, all ranks must enter the
    # summon-full-params collective; only rank 0 writes the .pt files.
    from marlllm.fsdp_utils import is_fsdp, summon_full_params
    first_backbone = getattr(next(iter(population.values())), "_backbone", None)
    fsdp = bool(is_fsdp(first_backbone))
    if fsdp:
        import torch.distributed as dist
        rank = dist.get_rank() if dist.is_initialized() else 0
    else:
        rank = 0

    ckpt_root = Path(output_dir) / "checkpoints"
    name = (tag if tag is not None else f"iter_{iteration:06d}")
    ckpt_dir = ckpt_root / name
    if rank == 0:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    seen: dict[int, str] = {}
    slot_to_object: dict[str, str] = {}
    with summon_full_params(first_backbone, writeback=False):
        for slot, agent in population.items():
            if id(agent) in seen:
                slot_to_object[slot] = seen[id(agent)]
                continue
            canonical = slot
            seen[id(agent)] = canonical
            slot_to_object[slot] = canonical
            payload = extract_agent_state(agent)
            if rank == 0:
                torch.save(payload, ckpt_dir / f"{canonical}.pt")

    if rank == 0:
        meta = {
            "iteration": iteration,
            # Optimizer state under FSDP is sharded per rank; full
            # consolidation needs FSDP.optim_state_dict and isn't wired up
            # yet. Skip it under FSDP — resume will start with a fresh
            # optimizer (model weights still resume cleanly).
            "optimizer_state": (
                optimizer.state_dict()
                if optimizer is not None and not fsdp
                else None
            ),
            "rng_state": torch.get_rng_state(),
            "config": config_dict,
            "agent_slots": list(population.keys()),
            "slot_to_object": slot_to_object,
            "fsdp": fsdp,
        }
        if extra_meta:
            meta.update(extra_meta)
        torch.save(meta, ckpt_dir / "meta.pt")

    # Symlink "latest" → this dir (rank 0 only; others skip to avoid races).
    if rank == 0:
        latest = ckpt_root / "latest"
        if latest.exists() or latest.is_symlink():
            try:
                latest.unlink()
            except OSError:
                pass
        try:
            latest.symlink_to(name, target_is_directory=True)
        except (OSError, NotImplementedError):
            # Filesystem may not support symlinks; that's OK.
            pass

    return ckpt_dir


def load_population_checkpoint(
    *,
    population: dict[str, Any],
    optimizer: Any,
    ckpt_dir: Path,
    device: torch.device | str | None = None,
    load_optimizer: bool = True,
) -> int:
    """Load a per-agent checkpoint directory into a population.

    Returns the iteration number from ``meta.pt``.
    """
    ckpt_dir = Path(ckpt_dir)
    meta = torch.load(ckpt_dir / "meta.pt", map_location=device or "cpu")

    # FSDP: gather params on every rank with writeback=True so injection
    # edits the full tensor and FSDP re-shards on exit.
    from marlllm.fsdp_utils import is_fsdp, summon_full_params
    first_backbone = getattr(next(iter(population.values())), "_backbone", None)
    fsdp = bool(is_fsdp(first_backbone))

    slot_to_object: dict[str, str] = meta.get("slot_to_object", {})
    loaded_objects: dict[int, bool] = {}
    with summon_full_params(first_backbone, writeback=fsdp):
        for slot, agent in population.items():
            if id(agent) in loaded_objects:
                continue
            canonical = slot_to_object.get(slot, slot)
            path = ckpt_dir / f"{canonical}.pt"
            if not path.exists():
                _LOG.warning("No per-agent checkpoint found for slot %r at %s", slot, path)
                continue
            payload = torch.load(path, map_location=device or "cpu")
            inject_agent_state(agent, payload, strict=False)
            loaded_objects[id(agent)] = True

    if load_optimizer and optimizer is not None and meta.get("optimizer_state") is not None:
        try:
            optimizer.load_state_dict(meta["optimizer_state"])
        except Exception as e:
            _LOG.warning("Could not load optimizer state: %s", e)

    if meta.get("rng_state") is not None:
        try:
            torch.set_rng_state(meta["rng_state"].cpu())
        except Exception:
            pass

    return int(meta.get("iteration", 0))


def load_single_agent_payload(
    ckpt_path: Path,
    *,
    map_location: Any = "cpu",
) -> dict[str, Any]:
    """Load a single per-agent ``.pt`` file (used by the migration CLI)."""
    return torch.load(Path(ckpt_path), map_location=map_location)
