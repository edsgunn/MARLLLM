"""
Agent ABC and IndependentAgent implementation.

IndependentAgent wraps a HuggingFace CausalLM and attaches a scalar ValueHead.
The value head receives detached hidden states (Option B gradient isolation):
gradient from the value loss never flows into the LM backbone.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase


def _detach_kv_cache(cache):
    """Detach all K/V tensors in an HF cache so the next chunk's autograd
    graph is independent of prior chunks.

    Three supported formats:

    * Modern ``DynamicCache`` (transformers ≥ 4.50ish) stores per-layer
      K/V on ``cache.layers[i].keys`` / ``.values``. The older top-level
      ``cache.key_cache`` / ``.value_cache`` attributes are gone, so a
      ``hasattr`` check on them silently fails — which used to leave the
      cache un-detached and triggered "backward through graph twice"
      errors in the chunked-loss path.
    * Pre-4.50ish DynamicCache exposed ``key_cache`` / ``value_cache``
      as the backing lists; we still support that.
    * Legacy tuple-of-tuples format used by older models.

    Returns ``cache`` (mutated in place for both DynamicCache variants;
    new tuple for the legacy form).
    """
    if cache is None:
        return None
    layers = getattr(cache, "layers", None)
    if layers is not None:
        for layer in layers:
            k = getattr(layer, "keys", None)
            v = getattr(layer, "values", None)
            if k is not None:
                layer.keys = k.detach()
            if v is not None:
                layer.values = v.detach()
        return cache
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        cache.key_cache = [k.detach() if k is not None else k for k in cache.key_cache]
        cache.value_cache = [v.detach() if v is not None else v for v in cache.value_cache]
        return cache
    if isinstance(cache, tuple):
        return tuple(
            tuple(x.detach() if isinstance(x, torch.Tensor) else x for x in layer)
            for layer in cache
        )
    return cache


def _fix_cpu_buffers(model: nn.Module) -> None:
    """Move CPU-resident buffers to the same device as the module's parameters.

    accelerate's device_map="auto" can leave non-parameter buffers (e.g. the
    RoPE inv_freq tensor) on CPU even when the surrounding parameters are on GPU,
    causing a Triton "cpu tensor?" error at runtime.
    """
    for module in model.modules():
        # Find what device this module's own parameters live on (if any).
        param_devices = {p.device for p in module.parameters(recurse=False)}
        gpu_devices = [d for d in param_devices if d.type != "cpu"]
        target = gpu_devices[0] if gpu_devices else None

        for buf_name, buf in list(module.named_buffers(recurse=False)):
            if buf is not None and buf.device.type == "cpu":
                dest = target or (torch.device("cuda:0") if torch.cuda.is_available() else None)
                if dest is not None:
                    setattr(module, buf_name, buf.to(dest))


def _sample_token(
    logits: torch.Tensor,
    temperature: float,
) -> tuple[int, float]:
    """Sample one token from a (1, V) logits tensor.

    temperature == 0 → greedy argmax (avoids dividing by zero).
    """
    if temperature == 0.0:
        token_id = int(logits.argmax(dim=-1).item())
        log_prob = float(F.log_softmax(logits, dim=-1)[0, token_id].item())
    else:
        if temperature != 1.0:
            logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        token_id = int(torch.multinomial(probs, num_samples=1).item())
        log_prob = float(F.log_softmax(logits, dim=-1)[0, token_id].item())
    return token_id, log_prob


def _sample_batch(
    logits: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample one token per row from a (B, V) logits tensor.

    Returns next_toks (B, 1) and log_probs (B, 1).
    temperature == 0 → greedy argmax per row.
    """
    if temperature == 0.0:
        next_toks = logits.argmax(dim=-1, keepdim=True)  # (B, 1)
        log_probs = F.log_softmax(logits, dim=-1).gather(1, next_toks)
    else:
        if temperature != 1.0:
            logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        next_toks = torch.multinomial(probs, num_samples=1)
        log_probs = F.log_softmax(logits, dim=-1).gather(1, next_toks)
    return next_toks, log_probs


@torch.no_grad()
def _generate_batch(
    backbone: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    device: torch.device,
    contexts: list[list[int]],
    n_tokens: int,
    temperature: float,
    eos_token_ids: list[int] | None,
    cache_implementation: str | None = None,
    stop_strings: list[str] | None = None,
) -> tuple[list[list[int]], list[list[float]]]:
    """Batched autoregressive sampling via HF ``model.generate``.

    Replaces the per-token Python loop that the agents used to drive directly,
    handing scheduling/KV-cache/eos-stopping to HF's compiled generate path.
    Faster on small batches because per-step Python overhead is amortised
    inside the C++/CUDA kernels.

    Returns ``(token_ids_per_seq, logprobs_per_seq)`` — same contract as the
    legacy implementation. Logprobs are ``log_softmax`` of the *processed*
    scores (after temperature/top-p warpers), gathered at the sampled token,
    matching what the old ``_sample_batch`` produced.
    """
    if device.type == "cuda":
        torch.cuda.set_device(device.index or 0)

    B = len(contexts)
    eos_set = set(eos_token_ids or [])
    pad_id = tokenizer.pad_token_id
    max_in = max(len(c) for c in contexts)

    input_ids = torch.full((B, max_in), pad_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros(B, max_in, dtype=torch.long, device=device)
    for i, ctx in enumerate(contexts):
        L = len(ctx)
        input_ids[i, max_in - L:] = torch.tensor(ctx, dtype=torch.long, device=device)
        attention_mask[i, max_in - L:] = 1

    gen_kwargs: dict = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=int(n_tokens),
        return_dict_in_generate=True,
        output_scores=True,
        pad_token_id=pad_id,
        use_cache=True,
        # Disable HF's cross-rank generation sync. When WORLD_SIZE > 1, HF
        # auto-enables synced_gpus assuming a model-parallel inference setup
        # where all ranks step through the same sequence together. For our
        # data-parallel rollout (each rank generates its own different
        # sequences), that sync raises a device-side assertion as soon as
        # one rank's batch finishes ahead of another's.
        synced_gpus=False,
    )
    if eos_token_ids:
        gen_kwargs["eos_token_id"] = list(eos_token_ids)
    if stop_strings:
        # HF halts each sequence when any of these strings is generated; the
        # stop string is retained in the output. Requires the tokenizer so HF
        # can match against decoded text. Mirrors the vLLM path's `stop=`.
        gen_kwargs["stop_strings"] = list(stop_strings)
        gen_kwargs["tokenizer"] = tokenizer
    if temperature == 0.0:
        gen_kwargs["do_sample"] = False
    else:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = float(temperature)
        gen_kwargs["top_p"] = 1.0
        gen_kwargs["top_k"] = 0  # disable HF's default top_k=50
    if cache_implementation is not None:
        # "static" pairs with torch.compile to give CUDA-graph-stable shapes.
        # Without compile it's a small allocator win; with compile it's a major
        # speedup because the forward becomes graph-capturable.
        gen_kwargs["cache_implementation"] = cache_implementation

    out = backbone.generate(**gen_kwargs)
    new_tokens = out.sequences[:, max_in:]  # (B, T_out)
    T_out = new_tokens.shape[1]

    if T_out > 0 and out.scores:
        # out.scores: tuple of T_out tensors of shape (B, V), warped.
        scores = torch.stack(out.scores, dim=1)  # (B, T_out, V)
        log_probs_full = F.log_softmax(scores, dim=-1)
        chosen_logp = log_probs_full.gather(2, new_tokens.unsqueeze(-1)).squeeze(-1)
    else:
        chosen_logp = torch.zeros((B, 0), device=device)

    new_tokens_cpu = new_tokens.cpu().tolist()
    chosen_logp_cpu = chosen_logp.cpu().tolist()

    all_ids: list[list[int]] = []
    all_lps: list[list[float]] = []
    for i in range(B):
        ids = new_tokens_cpu[i]
        lps = chosen_logp_cpu[i]
        # When some rows finish early, HF pads finished rows' positions with
        # pad_token_id. Trim each row at the first EOS occurrence (inclusive).
        # If no EOS appears, the row hit max_new_tokens — keep everything.
        cut = len(ids)
        for j, t in enumerate(ids):
            if t in eos_set:
                cut = j + 1
                break
        all_ids.append(ids[:cut])
        all_lps.append(lps[:cut])
    return all_ids, all_lps


class Agent(ABC):
    """
    Interface that the Trainer and Loss see.

    act()      — used during rollout collection (inference only)
    evaluate() — used during training (full differentiable forward pass)
    parameters() — used to build the optimiser
    """

    @property
    @abstractmethod
    def agent_id(self) -> str: ...

    @property
    @abstractmethod
    def character_prompt(self) -> str: ...

    @abstractmethod
    def act(
        self,
        context_token_ids: list[int],
        n_tokens: int,
        temperature: float = 1.0,
        eos_token_ids: list[int] | None = None,
    ) -> tuple[list[int], list[float]]:
        """
        Sample up to n_tokens action tokens autoregressively given the full
        context. If ``eos_token_ids`` is supplied, generation stops as soon as
        any of those ids is sampled; the EOS token is included in the
        returned sequence so the caller can keep the chat-template structure
        intact (assistant turn closed by ``<|im_end|>`` etc.). Otherwise
        generation runs to ``n_tokens``.

        Returns (token_ids, log_probs). Called under torch.no_grad().
        """
        ...

    @abstractmethod
    def evaluate(
        self,
        input_ids: torch.Tensor,       # (B, T)
        attention_mask: torch.Tensor,  # (B, T)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Full differentiable forward pass.
        Returns (logits (B, T, V), values (B, T)).
        """
        ...

    def evaluate_hidden(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning (last_hidden (B, T, H), values (B, T)).

        Skips the lm_head pass so the (B, T, V) logits tensor is never
        materialised — callers reconstruct logits chunk-wise via lm_head().
        """
        raise NotImplementedError

    def evaluate_hidden_ref(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor | None:
        """Hidden-state reference pass. Returns last_hidden (B, T, H) or None."""
        return None

    def lm_head(self, hidden: torch.Tensor) -> torch.Tensor:
        """Apply the (LoRA-active) lm_head to a hidden-state slice."""
        raise NotImplementedError

    def lm_head_ref(self, hidden: torch.Tensor) -> torch.Tensor:
        """Apply the lm_head used for the KL reference (adapters disabled)."""
        raise NotImplementedError

    @abstractmethod
    def parameters(self) -> Iterable[nn.Parameter]: ...


    def train_mode(self) -> None:
        """Switch to training mode."""
        ...

    def eval_mode(self) -> None:
        """Switch to eval/inference mode."""
        ...


def _find_lora_target_modules(model: nn.Module) -> list[str]:
    """
    Walk the model and collect unique leaf-module name suffixes for nn.Linear
    layers whose names suggest attention projections or MLP gates.
    Falls back to a standard set if nothing suitable is found.

    This is needed for models (e.g. Qwen3.5 hybrid) where PEFT cannot
    auto-detect target_modules from the model config alone.
    """
    # Prefer names that look like attention/mlp projections.
    # We collect *last segment* names (e.g. "q_proj", "gate_proj") then
    # verify at least one module with that name exists in the model.
    CANDIDATE_SUFFIXES = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        "qkv_proj", "out_proj",
        "query_key_value",
    ]
    found: set[str] = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            suffix = name.split(".")[-1]
            if suffix in CANDIDATE_SUFFIXES:
                found.add(suffix)

    if found:
        # Always include both q and v at minimum if either is present.
        return sorted(found)

    # Last resort: target ALL linear layers (expensive but correct).
    all_linear: set[str] = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            all_linear.add(name.split(".")[-1])
    return sorted(all_linear)


class ValueHead(nn.Module):
    """
    Scalar value head: Linear(hidden_size -> 1).
    Receives detached hidden states so the value loss gradient stays
    inside the linear layer and never propagates into the LM backbone.

    Optionally carries PopArt (van Hasselt et al. 2016, "Learning values across
    many orders of magnitude"), which exists for exactly the problem this
    project has: the return spans ~250 at gamma=0.99 to ~2300 at gamma=1, a
    plain MSE head over that range is badly conditioned, and standardising the
    TARGET instead throws away the level the baseline is supposed to absorb.

    PopArt splits the difference. ART adaptively rescales the target using
    running moments, so the head always regresses onto a well-conditioned
    quantity. POP then rescales the output layer whenever those moments move,
    so the DENORMALISED output is preserved exactly and the value function
    stays on the return's own scale. The head learns a normalised value;
    ``denormalise`` puts it back in nats for the advantage.
    """

    def __init__(self, hidden_size: int, *, popart: bool = False,
                 popart_beta: float = 3e-4) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_size, 1, bias=True)
        self.popart = bool(popart)
        self.popart_beta = float(popart_beta)
        # Running first and second moments of the TARGET, and the scale derived
        # from them. Buffers so they checkpoint and move with the module.
        self.register_buffer("popart_mu", torch.zeros(()))
        self.register_buffer("popart_nu", torch.ones(()))
        self.register_buffer("popart_sigma", torch.ones(()))
        self.register_buffer("popart_ready", torch.zeros(()))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: (B, T, H) — should already be detached by caller.
        # With PopArt this is the NORMALISED value; use denormalise() for the
        # value in the return's units.
        return self.linear(hidden_states).squeeze(-1)  # (B, T)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        """Tolerate checkpoints written before PopArt existed.

        Adding the four running-moment buffers made every earlier checkpoint
        unloadable — `load_state_dict` reports them as missing keys and raises,
        which breaks resume as well as any analysis over old checkpoints. A
        checkpoint from before PopArt legitimately has no moments; the right
        behaviour is to keep the freshly-initialised ones, which have
        popart_ready = 0 so the first update seeds them from the batch.

        Only the PopArt buffers are filled in. Any other missing or unexpected
        key still surfaces as it did.
        """
        for name in ("popart_mu", "popart_nu", "popart_sigma", "popart_ready"):
            key = prefix + name
            if key not in state_dict:
                state_dict[key] = getattr(self, name).detach().clone()
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def denormalise(self, v_norm: torch.Tensor) -> torch.Tensor:
        """Normalised value -> the return's own scale."""
        if not self.popart:
            return v_norm
        return v_norm * self.popart_sigma + self.popart_mu

    def normalise_target(self, y: torch.Tensor) -> torch.Tensor:
        """A raw return -> the scale the head is actually regressing onto."""
        if not self.popart:
            return y
        return (y - self.popart_mu) / self.popart_sigma

    @torch.no_grad()
    def popart_update(self, batch_mean: float, batch_sq_mean: float) -> dict:
        """ART then POP. Call once per optimiser step, never per micro-batch.

        Rescaling the output layer changes parameters that already-accumulated
        gradients were computed against, so doing this mid-accumulation would
        apply stale gradients to moved weights.
        """
        if not self.popart:
            return {}
        mu_old = self.popart_mu.clone()
        sigma_old = self.popart_sigma.clone()
        b = self.popart_beta
        if float(self.popart_ready) < 1.0:
            # Seed from the first batch rather than crawling up from 0/1, which
            # would leave sigma wrong by orders of magnitude for thousands of
            # steps at beta=3e-4.
            self.popart_mu.fill_(float(batch_mean))
            self.popart_nu.fill_(float(batch_sq_mean))
            self.popart_ready.fill_(1.0)
        else:
            self.popart_mu.mul_(1.0 - b).add_(b * float(batch_mean))
            self.popart_nu.mul_(1.0 - b).add_(b * float(batch_sq_mean))
        var = (self.popart_nu - self.popart_mu * self.popart_mu).clamp_min(1e-4)
        self.popart_sigma.copy_(var.sqrt())
        # POP: preserve outputs precisely.
        #   sigma_new * (W' h + b') + mu_new  ==  sigma_old * (W h + b) + mu_old
        #   =>  W' = W * sigma_old/sigma_new
        #       b' = (sigma_old * b + mu_old - mu_new) / sigma_new
        ratio = sigma_old / self.popart_sigma
        self.linear.weight.mul_(ratio)
        self.linear.bias.mul_(ratio).add_((mu_old - self.popart_mu) / self.popart_sigma)
        return {"popart/mu": float(self.popart_mu),
                "popart/sigma": float(self.popart_sigma),
                "popart/rescale": float(ratio)}


class LearningProgressReference:
    """A frozen second copy of this agent's own policy, lagged in training time.

    Why a second model at all
    -------------------------
    The surrogate action return sums this agent's own surprise over future
    observation tokens, which equals ``KL(p_env || p_theta) + H(p_env)``. The
    entropy term is the dark room: a REINFORCE reward for making the world
    simpler. It is common to *any* cross-entropy against ``p_env``, so it
    cancels exactly in a difference of two of them — which is why a second
    reference removes it no matter how poor a model of ``p_env`` that
    reference is. See ``TrainingConfig.learning_progress_return``.

    The two-slot buffer
    -------------------
    The reference must never be the current weights: at zero lag the return is
    identically zero (the plug-in estimator of
    surprise_minimisation_derivation.md 5.5.4). So we keep two snapshots —
    ``pending``, taken at the last rotation, and the *active* reference, which
    is the snapshot taken at the rotation before that. On every rotation the
    pending snapshot becomes the active reference and a fresh pending snapshot
    is taken. The effective lag therefore runs from ``L+1`` to ``2L``
    iterations and never reaches 0.

    Storage
    -------
    The active reference lives in the shared backbone as an extra, frozen LoRA
    adapter; ``pending`` is a plain CPU tensor dict. Neither is ever handed to
    the optimiser.

    The reference adapter's name is ``lpref_<sha1(live)[:8]>`` and deliberately
    does NOT contain the live adapter's name. PEFT selects an adapter's weights
    with a plain substring test —

        {k: v for k, v in sd.items() if "lora_" in k and adapter_name in k}

    (``peft/utils/save_and_load.py``, ``get_peft_model_state_dict``) — so a
    name like ``agent_0__lp_ref`` would be swept into every
    ``save_pretrained(selected_adapters=["agent_0"])``. That call is on the
    vLLM sync path, which runs after every optimiser step, so the rollout
    engine would have been handed a file containing two policies' weights. The
    hash keeps the two names disjoint. It never reaches disk: checkpoints
    re-key the reference to the live adapter's names (see
    ``learning_progress_state``), so the saved format does not depend on it.
    """

    _LP_PREFIX = "lpref_"

    # Set by enable_learning_progress_reference(); None means "not enabled".
    _lp_adapter_name: str | None = None
    _lp_pending: dict | None = None
    # Iteration at which each snapshot was taken, for the lag diagnostic.
    _lp_reference_iteration: int = 0
    _lp_pending_iteration: int = 0

    # -------------------------------------------------------------- #
    # Hooks a concrete agent must supply                               #
    # -------------------------------------------------------------- #

    def _lp_live_adapter_name(self) -> str:
        """Name of the adapter carrying this agent's trainable weights."""
        raise NotImplementedError

    def _lp_backbone(self):
        return self._backbone

    @classmethod
    def _lp_reference_adapter_name(cls, live: str) -> str:
        """A name for the reference adapter that cannot contain ``live``.

        See the class docstring: PEFT's adapter selection is a substring test,
        so the two names must be disjoint or the reference leaks into every
        adapter file the vLLM sync path writes.
        """
        import hashlib
        return cls._LP_PREFIX + hashlib.sha1(live.encode("utf-8")).hexdigest()[:8]

    # -------------------------------------------------------------- #
    # Snapshot plumbing (overridden wholesale in the unit tests)       #
    # -------------------------------------------------------------- #

    def _lp_snapshot_live(self) -> dict:
        """CPU copy of the live adapter's weights, keyed by live-adapter name."""
        live = self._lp_live_adapter_name()
        return {
            name: param.detach().to("cpu", copy=True)
            for name, param in self._lp_backbone().named_parameters()
            if f".{live}." in name
        }

    def _lp_write_reference(self, state: dict) -> None:
        """Copy a snapshot into the frozen reference adapter."""
        live = self._lp_live_adapter_name()
        lag = self._lp_adapter_name
        params = dict(self._lp_backbone().named_parameters())
        missing = []
        with torch.no_grad():
            for name, value in state.items():
                target = params.get(name.replace(f".{live}.", f".{lag}."))
                if target is None:
                    missing.append(name)
                    continue
                target.data.copy_(value.to(target.device, dtype=target.dtype))
        if missing:
            raise KeyError(
                f"{len(missing)} snapshot keys have no counterpart in the "
                f"learning-progress reference adapter (first: {missing[:3]}). "
                "The reference would be a mix of two policies and the return "
                "would be meaningless."
            )

    # -------------------------------------------------------------- #
    # Lifecycle                                                        #
    # -------------------------------------------------------------- #

    def enable_learning_progress_reference(self, iteration: int = 0) -> None:
        """Attach the reference adapter and seed both slots with the current
        weights. Idempotent.

        Seeding both slots with the current weights means the reference is the
        *initial* policy until the second rotation, so early training measures
        progress against the pre-trained prior and only becomes genuinely
        lagged-self later. That is the intended warm start, not an artefact.
        """
        if self._lp_adapter_name is not None:
            return
        backbone = self._lp_backbone()
        peft_config = getattr(backbone, "peft_config", None)
        if not peft_config:
            raise RuntimeError(
                "learning_progress_reference='lagged' needs a PEFT-wrapped "
                "backbone to hold the second adapter, and this agent has none "
                "(lora_r = 0 means full fine-tuning). Use lora_r > 0, or set "
                "learning_progress_reference='base' to use the frozen "
                "pre-trained model as the second reference instead."
            )
        live = self._lp_live_adapter_name()
        if live not in peft_config:
            raise RuntimeError(
                f"live adapter {live!r} is not among the backbone's adapters "
                f"{list(peft_config)}"
            )
        lag = self._lp_reference_adapter_name(live)
        if lag not in peft_config:
            backbone.add_adapter(lag, peft_config[live])
        self._lp_adapter_name = lag
        for name, param in backbone.named_parameters():
            if f".{lag}." in name:
                param.requires_grad_(False)
        # add_adapter can leave the new adapter active; put the live one back
        # before anything else touches the backbone.
        backbone.set_adapter(live)
        self._lp_pending = self._lp_snapshot_live()
        self._lp_write_reference(self._lp_pending)
        self._lp_reference_iteration = int(iteration)
        self._lp_pending_iteration = int(iteration)

    def rotate_learning_progress_reference(self, iteration: int) -> None:
        """pending -> active reference; current weights -> pending."""
        if self._lp_adapter_name is None:
            raise RuntimeError(
                "rotate_learning_progress_reference called before "
                "enable_learning_progress_reference"
            )
        fresh = self._lp_snapshot_live()
        self._lp_write_reference(self._lp_pending)
        self._lp_reference_iteration = self._lp_pending_iteration
        self._lp_pending = fresh
        self._lp_pending_iteration = int(iteration)

    def reseed_learning_progress_reference(self, iteration: int = 0) -> None:
        """Reset both slots to the current weights.

        Used after a checkpoint that carries no reference of its own is loaded
        into a run that has one: the reference must be the weights just
        loaded, not the freshly-initialised adapter they replaced.
        """
        if self._lp_adapter_name is None:
            return
        self._lp_pending = self._lp_snapshot_live()
        self._lp_write_reference(self._lp_pending)
        self._lp_reference_iteration = int(iteration)
        self._lp_pending_iteration = int(iteration)

    def learning_progress_reference_age(self, iteration: int) -> int:
        """Iterations between the active reference's weights and now."""
        if self._lp_adapter_name is None:
            return 0
        return max(0, int(iteration) - int(self._lp_reference_iteration))

    # -------------------------------------------------------------- #
    # Checkpointing                                                    #
    # -------------------------------------------------------------- #

    def learning_progress_state(self) -> dict | None:
        """Both slots, keyed by LIVE adapter name so migration remapping and
        checkpoint loading can use the same key rewrite as the live adapter."""
        if self._lp_adapter_name is None:
            return None
        live = self._lp_live_adapter_name()
        lag = self._lp_adapter_name
        reference = {
            name.replace(f".{lag}.", f".{live}."): param.detach().cpu().clone()
            for name, param in self._lp_backbone().named_parameters()
            if f".{lag}." in name
        }
        return {
            "reference": reference,
            "pending": {k: v.cpu().clone() for k, v in self._lp_pending.items()},
            "reference_iteration": int(self._lp_reference_iteration),
            "pending_iteration": int(self._lp_pending_iteration),
        }

    def load_learning_progress_state(self, state: dict) -> None:
        if self._lp_adapter_name is None:
            self.enable_learning_progress_reference()
        self._lp_write_reference(state["reference"])
        self._lp_pending = {k: v.cpu().clone() for k, v in state["pending"].items()}
        self._lp_reference_iteration = int(state.get("reference_iteration", 0))
        self._lp_pending_iteration = int(state.get("pending_iteration", 0))

    # -------------------------------------------------------------- #
    # Forward passes under the reference                               #
    # -------------------------------------------------------------- #

    @contextmanager
    def _lp_reference_mode(self):
        """Activate the reference adapter, in eval mode, and put everything
        back afterwards.

        Eval mode matters. ``ln p_theta^-(x_s)`` is a property of the reference
        *policy*, and a dropout mask is not part of a policy. Left in train
        mode the reference would be scored under a different random mask from
        the one the training forward used, and the difference of the two
        surprises would carry that noise instead of learning progress. Every
        base model in use has dropout disabled — Qwen2.5 sets
        ``attention_dropout`` to 0 and has no residual dropout — so the two
        coincide there, but that is a property of the checkpoint rather than of
        this code, and gpt2 breaks it.

        Never hold this open across a ``yield``: peft toggles model state on
        enter and exit, so a suspended generator would leave the caller's frame
        on the wrong adapter.
        """
        backbone = self._lp_backbone()
        was_training = backbone.training
        backbone.eval()
        backbone.set_adapter(self._lp_adapter_name)
        try:
            yield
        finally:
            backbone.set_adapter(self._lp_live_adapter_name())
            if was_training:
                backbone.train()

    @torch.no_grad()
    def evaluate_hidden_lp(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """Last hidden states under the lagged reference policy.

        Call this BEFORE the training forward: ``set_adapter`` toggles
        requires_grad on every adapter, and running it first means
        ``evaluate_hidden``'s own ``_activate()`` is what the autograd graph
        is built under.
        """
        if self._lp_adapter_name is None:
            return None
        backbone = self._lp_backbone()
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device.index or 0)
        kwargs: dict = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            logits_to_keep=1,
        )
        if position_ids is not None:
            kwargs["position_ids"] = position_ids
        with self._lp_reference_mode():
            out = backbone(**kwargs)
        return out.hidden_states[-1]

    @torch.no_grad()
    def evaluate_hidden_lp_chunked(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        chunk_size: int,
    ):
        """No-grad chunked counterpart of ``evaluate_hidden_lp``.

        Yields ``(s, e, hidden_chunk)``, or ``(s, e, None)`` when no reference
        is attached — mirroring ``evaluate_hidden_ref_chunked``. The adapter is
        switched around each forward and never across a ``yield``, so a
        suspended generator cannot leave the backbone on the wrong adapter.
        """
        B, T = input_ids.shape
        if self._lp_adapter_name is None:
            for s in range(0, T, chunk_size):
                yield s, min(s + chunk_size, T), None
            return
        backbone = self._lp_backbone()
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device.index or 0)
        past_kv = None
        for s in range(0, T, chunk_size):
            e = min(s + chunk_size, T)
            position_ids = torch.arange(s, e, device=input_ids.device).unsqueeze(0).expand(B, -1)
            cache_position = torch.arange(s, e, device=input_ids.device)
            with self._lp_reference_mode():
                out = backbone(
                    input_ids=input_ids[:, s:e],
                    attention_mask=attention_mask[:, :e],
                    position_ids=position_ids,
                    past_key_values=past_kv,
                    cache_position=cache_position,
                    output_hidden_states=True,
                    use_cache=True,
                )
            hidden_chunk = out.hidden_states[-1]
            past_kv = out.past_key_values  # already no_grad; no detach needed
            del out
            yield s, e, hidden_chunk


class IndependentAgent(LearningProgressReference, Agent):
    """
    Standalone HuggingFace CausalLM with an attached ValueHead.

    evaluate() runs a single forward pass with output_hidden_states=True,
    then detaches the last hidden state before passing it to the value head.
    This is the only place where gradient isolation is enforced.
    """

    def __init__(
        self,
        agent_id: str,
        character_prompt: str,
        model_name_or_path: str,
        device: str = "cpu",
        torch_dtype: torch.dtype | str | None = "auto",
        device_map: str | dict | None = None,
        keep_ref_model: bool = False,
        popart: bool = False,
        popart_beta: float = 3e-4,
        gradient_checkpointing: bool = False,
        lora_r: int = 0,
        lora_alpha: int = 16,
        lora_target_modules: list[str] | None = None,
        compile_model: bool = False,
        attn_implementation: str | None = None,
        context_formatter: str = "auto",
    ) -> None:
        """
        Args:
            torch_dtype: Passed to from_pretrained. "auto" lets HF pick the
                best dtype (bf16 on capable hardware). Set to torch.float32
                to force full precision, or torch.bfloat16 / torch.float16
                explicitly. For CPU-only runs "auto" resolves to float32.
            device_map: Passed to from_pretrained. Set to "auto" to shard a
                large model across all available GPUs/CPU automatically.
                If set, the explicit .to(device) call is skipped.
            keep_ref_model: If True, load a frozen copy of the pretrained model
                to use as the KL reference. Doubles memory usage for the backbone.
            gradient_checkpointing: Recompute activations during backward instead
                of storing them. Trades ~30% compute for significant VRAM savings.
            lora_r: LoRA rank. 0 = full fine-tuning. Requires `peft` package.
            lora_alpha: LoRA scaling factor (lora_alpha / lora_r).
            lora_target_modules: Which linear layer names to apply LoRA to.
                None = PEFT auto-detects standard attention projection names.
            compile_model: Run torch.compile() on the backbone. Adds a ~1-2
                iteration warm-up cost but then speeds up both act_batch() and
                evaluate() by ~20-40% on Ampere/Hopper GPUs.
        """
        self._agent_id = agent_id
        self._character_prompt = character_prompt
        self.device = torch.device(device)

        self._tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

        load_kwargs: dict = {}
        if torch_dtype is not None:
            load_kwargs["dtype"] = torch_dtype
        if device_map is not None:
            load_kwargs["device_map"] = device_map
        if attn_implementation is not None:
            load_kwargs["attn_implementation"] = attn_implementation

        self._backbone = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, **load_kwargs
        )
        if device_map is None:
            self._backbone = self._backbone.to(self.device)
        elif device_map == "auto":
            # accelerate's device_map="auto" can leave non-parameter buffers (e.g.
            # RoPE inv_freq) on CPU. Move each module's CPU buffers to whichever
            # GPU the module's parameters live on (fall back to cuda:0).
            _fix_cpu_buffers(self._backbone)

        if gradient_checkpointing:
            self._backbone.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        if compile_model:
            self._backbone = torch.compile(self._backbone)
        # See LoRASharedBaseAgent: pair compile with static cache so the
        # captured graph stays valid across generate calls.
        self._cache_impl: str | None = "static" if compile_model else None

        self._lora_active = lora_r > 0
        if self._lora_active:
            try:
                from peft import LoraConfig, TaskType, get_peft_model
            except ImportError as e:
                raise ImportError(
                    "LoRA requires the `peft` package: uv add peft"
                ) from e
            target_modules = lora_target_modules or _find_lora_target_modules(self._backbone)
            peft_cfg = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=target_modules,
                lora_dropout=0.0,
                bias="none",
            )
            self._backbone = get_peft_model(self._backbone, peft_cfg)
            self._backbone.print_trainable_parameters()

        hidden_size = self._backbone.config.hidden_size
        model_dtype = next(self._backbone.parameters()).dtype
        self._value_head = ValueHead(
            hidden_size, popart=bool(popart), popart_beta=float(popart_beta),
        ).to(self.device, dtype=model_dtype)
        # Optional second value head, fitted to the environment-likelihood
        # part of the return alone. Its input is detached exactly as the main
        # head's is, so it trains only its own parameters and cannot reach the
        # policy or the backbone. Used to centre E_s when estimating the
        # gradient bias (environment_likelihood_derivation.md).
        self._bias_value_head = None

        if keep_ref_model:
            ref_kwargs: dict = {}
            if torch_dtype is not None:
                ref_kwargs["dtype"] = torch_dtype
            if device_map is not None:
                ref_kwargs["device_map"] = device_map
            self._ref_backbone = AutoModelForCausalLM.from_pretrained(
                model_name_or_path, **ref_kwargs
            )
            if device_map is None:
                self._ref_backbone = self._ref_backbone.to(self.device)
            elif device_map == "auto":
                _fix_cpu_buffers(self._ref_backbone)
            for p in self._ref_backbone.parameters():
                p.requires_grad_(False)
            self._ref_backbone.eval()
        else:
            self._ref_backbone = None

        from marlllm.context_formatter import make_formatter
        self.context_formatter = make_formatter(self._tokenizer, context_formatter)

    # ------------------------------------------------------------------ #
    # Agent interface                                                       #
    # ------------------------------------------------------------------ #

    @property
    def agent_id(self) -> str:
        return self._agent_id

    @property
    def character_prompt(self) -> str:
        return self._character_prompt

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        return self._tokenizer

    def parameters(self) -> Iterable[nn.Parameter]:
        # When LoRA is active only yield the trainable adapter weights, not
        # the frozen base. This keeps the optimizer small and correct.
        #
        # The learning-progress reference adapter is excluded BY NAME as well
        # as by requires_grad: set_adapter() toggles requires_grad every time
        # the reference forward runs, so a requires_grad filter alone would
        # depend on which adapter happened to be active when the optimiser was
        # built.
        if self._lora_active:
            lp = self._lp_adapter_name
            for name, param in self._backbone.named_parameters():
                if lp is not None and f".{lp}." in name:
                    continue
                if param.requires_grad:
                    yield param
        else:
            yield from self._backbone.parameters()
        yield from self._value_head.parameters()
        if self._bias_value_head is not None:
            yield from self._bias_value_head.parameters()

    def _lp_live_adapter_name(self) -> str:
        if not self._lora_active:
            raise RuntimeError(
                "IndependentAgent has no LoRA adapter to snapshot "
                "(lora_r = 0). Use lora_r > 0 or "
                "learning_progress_reference='base'."
            )
        # get_peft_model() installs a single adapter under this name.
        return "default"


    def enable_bias_probe(self) -> None:
        """Attach the environment-likelihood probe head (idempotent)."""
        if self._bias_value_head is not None:
            return
        hs = self._value_head.linear.in_features
        dt = next(self._value_head.parameters()).dtype
        self._bias_value_head = ValueHead(hs).to(self.device, dtype=dt)

    def bias_values(self, last_hidden):
        """Probe value estimate. Input detached: no gradient to the trunk."""
        if self._bias_value_head is None:
            return None
        return self._bias_value_head(last_hidden.detach())

    def train_mode(self) -> None:
        self._backbone.train()
        self._value_head.train()
        if self._bias_value_head is not None:
            self._bias_value_head.train()

    def eval_mode(self) -> None:
        self._backbone.eval()
        self._value_head.eval()
        if self._bias_value_head is not None:
            self._bias_value_head.eval()

    # ------------------------------------------------------------------ #
    # Rollout: autoregressive sampling with KV cache                       #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def act(
        self,
        context_token_ids: list[int],
        n_tokens: int,
        temperature: float = 1.0,
        eos_token_ids: list[int] | None = None,
    ) -> tuple[list[int], list[float]]:
        """Sample up to ``n_tokens`` tokens, reusing KV cache across steps.

        If ``eos_token_ids`` is supplied, generation halts as soon as any of
        those ids is sampled. The EOS token is included in the returned
        sequence so the assistant turn closes naturally (``<|im_end|>``).
        """
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device.index or 0)
        eos_set = set(eos_token_ids or [])
        sampled_ids: list[int] = []
        sampled_lps: list[float] = []

        input_ids = torch.tensor(
            [context_token_ids], dtype=torch.long, device=self.device
        )
        past_key_values = None

        for _ in range(n_tokens):
            out = self._backbone(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits = out.logits[:, -1, :]  # (1, V)
            past_key_values = out.past_key_values

            token_id, log_prob = _sample_token(logits, temperature)
            sampled_ids.append(token_id)
            sampled_lps.append(log_prob)

            if token_id in eos_set:
                break

            # Next step: only feed the new token
            input_ids = torch.tensor([[token_id]], dtype=torch.long, device=self.device)

        return sampled_ids, sampled_lps

    # ------------------------------------------------------------------ #
    # Batched rollout: parallel sampling across multiple contexts          #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def act_batch(
        self,
        contexts: list[list[int]],
        n_tokens: int,
        temperature: float = 1.0,
        eos_token_ids: list[int] | None = None,
        stop_strings: list[str] | None = None,
    ) -> tuple[list[list[int]], list[list[float]]]:
        """Batched autoregressive sampling — delegates to ``_generate_batch``."""
        return _generate_batch(
            self._backbone, self._tokenizer, self.device,
            contexts, n_tokens, temperature, eos_token_ids,
            cache_implementation=self._cache_impl,
            stop_strings=stop_strings,
        )

    # ------------------------------------------------------------------ #
    # Training: differentiable forward pass                                #
    # ------------------------------------------------------------------ #

    def evaluate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Single forward pass returning logits and value estimates.
        The LM backbone runs with output_hidden_states=True.
        Hidden states are detached before the value head (Option B).
        """
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device.index or 0)
        out = self._backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,  # not needed during training; required for gradient checkpointing
        )
        logits = out.logits                       # (B, T, V)
        last_hidden = out.hidden_states[-1]       # (B, T, H)
        values = self._value_head(last_hidden.detach())  # (B, T) — detached
        return logits, values

    def evaluate_hidden(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device.index or 0)
        # logits_to_keep=1 makes the HF lm_head pass produce only (B, 1, V)
        # instead of (B, T, V); the full logits tensor is several GiB at long T.
        # We discard that 1-position slice and apply lm_head ourselves chunk-wise
        # in the loss path.
        kwargs: dict = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            logits_to_keep=1,
        )
        if position_ids is not None:
            kwargs["position_ids"] = position_ids
        out = self._backbone(**kwargs)
        last_hidden = out.hidden_states[-1]              # (B, T, H)
        values = self._value_head(last_hidden.detach())  # (B, T)
        return last_hidden, values

    @torch.no_grad()
    def evaluate_ref(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """
        Forward pass through the frozen reference model.
        Returns logits (B, T, V), or None if no reference model was loaded.
        """
        if self._ref_backbone is None:
            return None
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device.index or 0)
        kwargs: dict = dict(input_ids=input_ids, attention_mask=attention_mask)
        if position_ids is not None:
            kwargs["position_ids"] = position_ids
        out = self._ref_backbone(**kwargs)
        return out.logits

    @torch.no_grad()
    def evaluate_hidden_ref(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        if self._ref_backbone is None:
            return None
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device.index or 0)
        kwargs: dict = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            logits_to_keep=1,
        )
        if position_ids is not None:
            kwargs["position_ids"] = position_ids
        out = self._ref_backbone(**kwargs)
        return out.hidden_states[-1]

    def lm_head(self, hidden: torch.Tensor) -> torch.Tensor:
        return self._backbone.get_output_embeddings()(hidden)

    def lm_head_ref(self, hidden: torch.Tensor) -> torch.Tensor:
        if self._ref_backbone is None:
            raise RuntimeError("No reference model loaded; lm_head_ref unavailable.")
        return self._ref_backbone.get_output_embeddings()(hidden)


# ============================================================================ #
# Shared-base LoRA agent                                                        #
# ============================================================================ #

class LoRASharedBaseAgent(LearningProgressReference, Agent):
    """
    Agent that shares a single PeftModel backbone with one other agent, but
    owns a private LoRA adapter within that backbone.

    Architecture
    ------------
    One AutoModelForCausalLM is loaded once (the 'base').  Two independent
    LoRA adapter sets ('agent_0', 'agent_1') are added to it via PEFT's
    multi-adapter API.  Each LoRASharedBaseAgent:
      - activates its own adapter before every forward pass via set_adapter()
      - owns an independent ValueHead
      - exposes only its adapter parameters + value-head parameters via
        parameters(), so the optimiser never sees the frozen base weights or
        the other agent's adapter

    Memory benefit
    --------------
    One 8 B bfloat16 model ≈ 16 GB.  Two independent 8 B models ≈ 32 GB.
    With this class: 16 GB base + two tiny LoRA adapter sets ≈ 16.05 GB.
    Fits on a single GH200 GPU where two full models would require two.

    Adapter switching thread-safety
    --------------------------------
    The trainer processes agents sequentially in every loop (rollout and loss),
    so set_adapter() is never called concurrently from two threads.  This is
    safe as long as that invariant holds.

    KL reference
    ------------
    When keep_ref_model=True, evaluate_ref() runs the backbone with ALL
    LoRA adapters disabled (PEFT's disable_adapter context manager), exposing
    the frozen pre-trained base as the reference.  Both agents therefore share
    the same KL reference — the pre-trained base — which is exactly what we
    want: each agent's LoRA delta is penalised for drifting from the prior.
    """

    def __init__(
        self,
        agent_id: str,
        character_prompt: str,
        shared_backbone: nn.Module,          # a PEFT PeftModel
        adapter_name: str,
        tokenizer: PreTrainedTokenizerBase,
        device: torch.device | str,
        keep_ref_model: bool = False,
        popart: bool = False,
        popart_beta: float = 3e-4,
        context_formatter: str = "auto",
        compile_rollout: bool = False,
    ) -> None:
        self._agent_id = agent_id
        self._character_prompt = character_prompt
        self._backbone = shared_backbone
        self._adapter_name = adapter_name
        self._tokenizer = tokenizer
        self.device = torch.device(device)
        self._keep_ref_model = keep_ref_model
        # When the trainer has torch.compile-d the shared backbone, the rollout
        # path uses cache_implementation="static" so the compiled graph sees
        # fixed cache shapes (and can stay graph-captured rather than recompiling
        # every call). No-op when compile_rollout is False.
        self._cache_impl: str | None = "static" if compile_rollout else None

        hidden_size = shared_backbone.config.hidden_size
        model_dtype = next(p for p in shared_backbone.parameters() if p.dtype.is_floating_point).dtype
        self._value_head = ValueHead(
            hidden_size, popart=bool(popart), popart_beta=float(popart_beta),
        ).to(self.device, dtype=model_dtype)
        # Optional second value head, fitted to the environment-likelihood
        # part of the return alone. Its input is detached exactly as the main
        # head's is, so it trains only its own parameters and cannot reach the
        # policy or the backbone. Used to centre E_s when estimating the
        # gradient bias (environment_likelihood_derivation.md).
        self._bias_value_head = None

        from marlllm.context_formatter import make_formatter
        self.context_formatter = make_formatter(self._tokenizer, context_formatter)

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _activate(self) -> None:
        """Switch the backbone to this agent's LoRA adapter."""
        self._backbone.set_adapter(self._adapter_name)

    def _lp_live_adapter_name(self) -> str:
        return self._adapter_name

    # ------------------------------------------------------------------ #
    # Agent interface                                                      #
    # ------------------------------------------------------------------ #

    @property
    def agent_id(self) -> str:
        return self._agent_id

    @property
    def character_prompt(self) -> str:
        return self._character_prompt

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        return self._tokenizer

    def parameters(self) -> Iterable[nn.Parameter]:
        # Yield only this adapter's weights plus the private value head.
        # Do NOT filter by requires_grad: PEFT v0.19+ sets the inactive adapter's
        # params to requires_grad=False after add_adapter(), which would cause the
        # second agent's adapter to be missing from the optimizer entirely.
        # The name filter is sufficient — base weights have no adapter name in their path.
        # evaluate() calls _activate() before each forward pass, which restores
        # requires_grad=True for the active adapter so gradients flow correctly.
        for name, param in self._backbone.named_parameters():
            if f".{self._adapter_name}." in name:
                yield param
        yield from self._value_head.parameters()
        if self._bias_value_head is not None:
            yield from self._bias_value_head.parameters()


    def enable_bias_probe(self) -> None:
        """Attach the environment-likelihood probe head (idempotent)."""
        if self._bias_value_head is not None:
            return
        hs = self._value_head.linear.in_features
        dt = next(self._value_head.parameters()).dtype
        self._bias_value_head = ValueHead(hs).to(self.device, dtype=dt)

    def bias_values(self, last_hidden):
        """Probe value estimate. Input detached: no gradient to the trunk."""
        if self._bias_value_head is None:
            return None
        return self._bias_value_head(last_hidden.detach())

    def train_mode(self) -> None:
        self._backbone.train()
        self._value_head.train()
        if self._bias_value_head is not None:
            self._bias_value_head.train()

    def eval_mode(self) -> None:
        self._backbone.eval()
        self._value_head.eval()
        if self._bias_value_head is not None:
            self._bias_value_head.eval()

    # ------------------------------------------------------------------ #
    # Rollout: autoregressive sampling                                     #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def act(
        self,
        context_token_ids: list[int],
        n_tokens: int,
        temperature: float = 1.0,
        eos_token_ids: list[int] | None = None,
    ) -> tuple[list[int], list[float]]:
        self._activate()
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device.index or 0)

        eos_set = set(eos_token_ids or [])
        sampled_ids: list[int] = []
        sampled_lps: list[float] = []
        input_ids = torch.tensor([context_token_ids], dtype=torch.long, device=self.device)
        past_key_values = None

        for _ in range(n_tokens):
            out = self._backbone(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits = out.logits[:, -1, :]
            past_key_values = out.past_key_values
            token_id, log_prob = _sample_token(logits, temperature)
            sampled_ids.append(token_id)
            sampled_lps.append(log_prob)
            if token_id in eos_set:
                break
            input_ids = torch.tensor([[token_id]], dtype=torch.long, device=self.device)

        return sampled_ids, sampled_lps

    @torch.no_grad()
    def act_batch(
        self,
        contexts: list[list[int]],
        n_tokens: int,
        temperature: float = 1.0,
        eos_token_ids: list[int] | None = None,
        stop_strings: list[str] | None = None,
    ) -> tuple[list[list[int]], list[list[float]]]:
        """Batched autoregressive sampling — delegates to ``_generate_batch``.

        Activates this agent's adapter on the shared backbone first so the
        generate path forward-passes through the right LoRA delta.
        """
        self._activate()
        return _generate_batch(
            self._backbone, self._tokenizer, self.device,
            contexts, n_tokens, temperature, eos_token_ids,
            cache_implementation=self._cache_impl,
            stop_strings=stop_strings,
        )

    # ------------------------------------------------------------------ #
    # Training: differentiable forward pass                                #
    # ------------------------------------------------------------------ #

    def evaluate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._activate()
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device.index or 0)
        out = self._backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        logits = out.logits
        last_hidden = out.hidden_states[-1]
        values = self._value_head(last_hidden.detach())
        return logits, values

    def evaluate_hidden(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass yielding (last_hidden, values).

        Accepts a 2D or 4D ``attention_mask``. HuggingFace causal-LM forwards
        treat a 4D ``(B, 1, T_q, T_k)`` additive mask as a complete user-
        supplied mask, bypassing internal causal construction. Packed
        callers should pass the block-diagonal causal mask from
        ``marlllm.loss._build_block_diagonal_causal_mask`` together with
        ``position_ids`` that reset to 0 at each trajectory boundary.
        """
        self._activate()
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device.index or 0)
        kwargs: dict = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            logits_to_keep=1,
        )
        if position_ids is not None:
            kwargs["position_ids"] = position_ids
        out = self._backbone(**kwargs)
        last_hidden = out.hidden_states[-1]
        values = self._value_head(last_hidden.detach())
        return last_hidden, values

    @torch.no_grad()
    def evaluate_ref(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        if not self._keep_ref_model:
            return None
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device.index or 0)
        # The frozen pre-trained base (adapters disabled) serves as the reference
        # for both agents — consistent with anchoring each LoRA delta to the prior.
        kwargs: dict = dict(input_ids=input_ids, attention_mask=attention_mask)
        if position_ids is not None:
            kwargs["position_ids"] = position_ids
        with self._backbone.disable_adapter():
            out = self._backbone(**kwargs)
        return out.logits

    @torch.no_grad()
    def evaluate_hidden_ref(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        if not self._keep_ref_model:
            return None
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device.index or 0)
        kwargs: dict = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            logits_to_keep=1,
        )
        if position_ids is not None:
            kwargs["position_ids"] = position_ids
        with self._backbone.disable_adapter():
            out = self._backbone(**kwargs)
        return out.hidden_states[-1]

    def evaluate_hidden_chunked(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        chunk_size: int,
    ):
        """Generator: yield (s, e, hidden_chunk, values_chunk) per seq chunk.

        Forwards the backbone in chunks along the time axis with KV-cache
        propagation. The cache from prior chunks is detached, so each yielded
        ``hidden_chunk`` carries an autograd graph that touches only the
        backbone parameters used by *this* chunk's forward. Caller is
        expected to compute a per-chunk loss and call backward before
        consuming the next chunk.
        """
        self._activate()
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device.index or 0)
        B, T = input_ids.shape
        past_kv = None
        for s in range(0, T, chunk_size):
            e = min(s + chunk_size, T)
            position_ids = torch.arange(s, e, device=input_ids.device).unsqueeze(0).expand(B, -1)
            cache_position = torch.arange(s, e, device=input_ids.device)
            out = self._backbone(
                input_ids=input_ids[:, s:e],
                attention_mask=attention_mask[:, :e],
                position_ids=position_ids,
                past_key_values=past_kv,
                cache_position=cache_position,
                output_hidden_states=True,
                use_cache=True,
            )
            hidden_chunk = out.hidden_states[-1]
            # Capture and detach the cache *before* the caller backwards through
            # ``hidden_chunk`` — backward releases the chunk's autograd graph and
            # the K/V tensors saved in it.
            past_kv_next = _detach_kv_cache(out.past_key_values)
            # Release per-layer hidden_states tuple now that we've grabbed [-1];
            # the other layer outputs are not needed and would otherwise tie up
            # ~num_layers × B × chunk × H of memory until ``out`` is overwritten.
            del out
            values_chunk = self._value_head(hidden_chunk.detach())
            yield s, e, hidden_chunk, values_chunk
            past_kv = past_kv_next

    @torch.no_grad()
    def evaluate_hidden_ref_chunked(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        chunk_size: int,
    ):
        """No-grad chunked counterpart of ``evaluate_hidden_ref`` for the
        frozen reference (LoRA disabled). Yields (s, e, hidden_ref_chunk).

        Returns ``None`` for ``hidden_ref_chunk`` if no reference model is
        kept (mirrors ``evaluate_hidden_ref``'s ``None`` return).
        """
        if not self._keep_ref_model:
            B, T = input_ids.shape
            for s in range(0, T, chunk_size):
                e = min(s + chunk_size, T)
                yield s, e, None
            return
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device.index or 0)
        B, T = input_ids.shape
        past_kv = None
        for s in range(0, T, chunk_size):
            e = min(s + chunk_size, T)
            position_ids = torch.arange(s, e, device=input_ids.device).unsqueeze(0).expand(B, -1)
            cache_position = torch.arange(s, e, device=input_ids.device)
            # NOTE: open disable_adapter only around the forward — never across
            # the yield. Suspending the generator inside ``with disable_adapter()``
            # would leave the adapter disabled in the caller's frame too, since
            # peft toggles model state on enter/exit.
            with self._backbone.disable_adapter():
                out = self._backbone(
                    input_ids=input_ids[:, s:e],
                    attention_mask=attention_mask[:, :e],
                    position_ids=position_ids,
                    past_key_values=past_kv,
                    cache_position=cache_position,
                    output_hidden_states=True,
                    use_cache=True,
                )
            hidden_ref_chunk = out.hidden_states[-1]
            past_kv = out.past_key_values  # already no_grad; no detach needed
            del out
            yield s, e, hidden_ref_chunk

    def lm_head(self, hidden: torch.Tensor) -> torch.Tensor:
        # Adapter activation only affects LoRA-wrapped modules; lm_head is
        # untouched by LoRA in our configs (target_modules = q/v/etc.), so a
        # bare call is correct regardless of which adapter is active.
        return self._backbone.get_output_embeddings()(hidden)

    def lm_head_ref(self, hidden: torch.Tensor) -> torch.Tensor:
        # Reference path uses the same lm_head (adapters never touch it).
        with self._backbone.disable_adapter():
            return self._backbone.get_output_embeddings()(hidden)
