"""FSDP helpers for the LoRA-shared-base training path.

The cultural-emergence stack normally runs DDP: each rank holds a full
backbone replica and gradients are manually all-reduced before the
optimizer step. That caps the trainable model size at what fits on a
single GH200 (≈14B in bf16, leaving room for vLLM + activations).

This module enables a FULL_SHARD FSDP path so the *trainer* backbone is
sharded across the four ranks on one node, freeing memory for larger
bases (e.g. Qwen2.5-32B). Constraints we preserve:

  * ``LoRASharedBaseAgent.parameters()`` filters by name on
    ``named_parameters()`` — so we wrap with ``use_orig_params=True``
    (each parameter ref stays valid; ``.data`` becomes the local shard).
  * The optimizer iterates the same per-adapter parameter refs and steps
    on local shards. AdamW / AdamW8bit work unchanged.
  * vLLM is per-rank and loads weights independently from disk, so its
    setup is unaffected. Only the *adapter saves* needed for vLLM weight
    sync require gathering full LoRA tensors — see ``summon_full_params``.

The PEFT model is wrapped in-place: ``peft_model`` is returned with FSDP
substituted at the auto-wrap granularity (one FSDP unit per transformer
decoder layer, plus a root unit for embeddings / lm_head). ``set_adapter``
and the other PEFT control methods continue to work because FSDP only
intercepts ``forward``.
"""
from __future__ import annotations

import contextlib
from typing import Any, Iterable

import torch
import torch.nn as nn


def _decoder_layer_classes(model: nn.Module) -> set[type]:
    """Find the transformer decoder layer class(es) to use as the FSDP wrap unit.

    HuggingFace causal LMs set ``_no_split_modules`` on the config /
    model to mark the indivisible decoder block class for ``device_map``
    placement. We reuse that as the FSDP auto-wrap granularity.
    """
    # PEFT model exposes the underlying HF model via base_model.model.
    inner = model
    for attr in ("base_model", "model"):
        inner = getattr(inner, attr, inner)
    no_split = getattr(inner, "_no_split_modules", None) or getattr(
        getattr(inner, "config", None), "_no_split_modules", None
    )
    if not no_split:
        raise RuntimeError(
            "Could not determine transformer decoder layer class for FSDP "
            "auto-wrap: the model has no _no_split_modules attribute."
        )
    found: set[type] = set()
    targets = set(no_split)
    for mod in model.modules():
        if type(mod).__name__ in targets:
            found.add(type(mod))
    if not found:
        raise RuntimeError(
            f"FSDP auto-wrap: none of {sorted(targets)} were found in the "
            f"loaded model. Module tree may have a different name."
        )
    return found


def wrap_peft_model_with_fsdp(
    peft_model: nn.Module,
    *,
    local_rank: int,
    param_dtype: torch.dtype = torch.bfloat16,
    reduce_dtype: torch.dtype = torch.float32,
    buffer_dtype: torch.dtype = torch.bfloat16,
) -> nn.Module:
    """Wrap a PEFT (LoRA-attached) model with FULL_SHARD FSDP.

    Returns the wrapped model. Use ``use_orig_params=True`` so the
    trainer's per-adapter parameter selection (string-match on
    ``named_parameters()``) still resolves to the same ``nn.Parameter``
    refs — their ``.data`` becomes the per-rank shard.
    """
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        MixedPrecision,
        ShardingStrategy,
    )
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    from functools import partial

    decoder_classes = _decoder_layer_classes(peft_model)
    policy = partial(
        transformer_auto_wrap_policy, transformer_layer_cls=decoder_classes
    )
    mp = MixedPrecision(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        buffer_dtype=buffer_dtype,
    )
    wrapped = FSDP(
        peft_model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        auto_wrap_policy=policy,
        mixed_precision=mp,
        device_id=local_rank,
        use_orig_params=True,
        limit_all_gathers=True,
    )
    return wrapped


def is_fsdp(model: nn.Module) -> bool:
    """Return True if ``model`` (or any descendant) is wrapped in FSDP."""
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    except ImportError:
        return False
    if isinstance(model, FSDP):
        return True
    for m in model.modules():
        if isinstance(m, FSDP):
            return True
    return False


@contextlib.contextmanager
def summon_full_params(model: nn.Module, *, writeback: bool = False):
    """Gather sharded params on every rank for the duration of the context.

    No-op when ``model`` isn't FSDP-wrapped. Otherwise calls FSDP's
    ``summon_full_params(recurse=True)`` which all-gathers every shard
    so ``.data`` temporarily holds the full tensor. With ``writeback=False``
    edits inside the context are discarded — that's what we want for
    read-only operations like ``save_pretrained`` and state-dict
    extraction.
    """
    if not is_fsdp(model):
        yield
        return
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    with FSDP.summon_full_params(model, writeback=writeback, recurse=True):
        yield
