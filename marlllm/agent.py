"""
Agent ABC and IndependentAgent implementation.

IndependentAgent wraps a HuggingFace CausalLM and attaches a scalar ValueHead.
The value head receives detached hidden states (Option B gradient isolation):
gradient from the value loss never flows into the LM backbone.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase


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
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_size, 1, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: (B, T, H) — should already be detached by caller
        return self.linear(hidden_states).squeeze(-1)  # (B, T)


class IndependentAgent(Agent):
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
        self._value_head = ValueHead(hidden_size).to(self.device, dtype=model_dtype)

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
        if self._lora_active:
            yield from (p for p in self._backbone.parameters() if p.requires_grad)
        else:
            yield from self._backbone.parameters()
        yield from self._value_head.parameters()

    def train_mode(self) -> None:
        self._backbone.train()
        self._value_head.train()

    def eval_mode(self) -> None:
        self._backbone.eval()
        self._value_head.eval()

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
    ) -> tuple[list[list[int]], list[list[float]]]:
        """
        Sample up to ``n_tokens`` for each context in a batched forward pass.

        Contexts are left-padded to the same length so they form a (B, T)
        tensor.  After the first (full-context) step, subsequent steps feed
        only the newly generated token (B, 1) reusing the KV cache, so the
        per-step cost is proportional to B rather than B × T.

        EOS handling
        ------------
        If ``eos_token_ids`` is supplied, each row stops contributing once it
        samples any EOS id (the EOS token is included in that row's output).
        The forward pass continues for the remaining unfinished rows; tokens
        sampled by already-finished rows are dropped from their output. The
        loop exits early when every row has finished.

        Returns:
            (list[list[int]], list[list[float]]) — token IDs and log-probs
            for each context, in the same order as the input.
        """
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device.index or 0)
        B = len(contexts)
        if B == 1:
            ids, lps = self.act(contexts[0], n_tokens, temperature, eos_token_ids)
            return [ids], [lps]

        eos_set = set(eos_token_ids or [])
        max_len = max(len(c) for c in contexts)
        pad_id = self._tokenizer.pad_token_id

        input_ids = torch.full((B, max_len), pad_id, dtype=torch.long, device=self.device)
        attention_mask = torch.zeros(B, max_len, dtype=torch.long, device=self.device)
        for i, ctx in enumerate(contexts):
            L = len(ctx)
            input_ids[i, max_len - L:] = torch.tensor(ctx, dtype=torch.long, device=self.device)
            attention_mask[i, max_len - L:] = 1

        all_ids: list[list[int]] = [[] for _ in range(B)]
        all_lps: list[list[float]] = [[] for _ in range(B)]
        finished: list[bool] = [False] * B
        past_key_values = None

        for step in range(n_tokens):
            out = self._backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits = out.logits[:, -1, :]  # (B, V)
            past_key_values = out.past_key_values

            next_toks, log_probs_t = _sample_batch(logits, temperature)  # (B,1) each

            for i in range(B):
                if finished[i]:
                    continue
                tok = int(next_toks[i, 0])
                all_ids[i].append(tok)
                all_lps[i].append(float(log_probs_t[i, 0]))
                if tok in eos_set:
                    finished[i] = True

            if all(finished):
                break

            input_ids = next_toks  # (B, 1)
            attention_mask = torch.cat(
                [attention_mask, torch.ones(B, 1, dtype=torch.long, device=self.device)],
                dim=1,
            )

        return all_ids, all_lps

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

    @torch.no_grad()
    def evaluate_ref(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor | None:
        """
        Forward pass through the frozen reference model.
        Returns logits (B, T, V), or None if no reference model was loaded.
        """
        if self._ref_backbone is None:
            return None
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device.index or 0)
        out = self._ref_backbone(input_ids=input_ids, attention_mask=attention_mask)
        return out.logits


# ============================================================================ #
# Shared-base LoRA agent                                                        #
# ============================================================================ #

class LoRASharedBaseAgent(Agent):
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
        context_formatter: str = "auto",
    ) -> None:
        self._agent_id = agent_id
        self._character_prompt = character_prompt
        self._backbone = shared_backbone
        self._adapter_name = adapter_name
        self._tokenizer = tokenizer
        self.device = torch.device(device)
        self._keep_ref_model = keep_ref_model

        hidden_size = shared_backbone.config.hidden_size
        model_dtype = next(p for p in shared_backbone.parameters() if p.dtype.is_floating_point).dtype
        self._value_head = ValueHead(hidden_size).to(self.device, dtype=model_dtype)

        from marlllm.context_formatter import make_formatter
        self.context_formatter = make_formatter(self._tokenizer, context_formatter)

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _activate(self) -> None:
        """Switch the backbone to this agent's LoRA adapter."""
        self._backbone.set_adapter(self._adapter_name)

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

    def train_mode(self) -> None:
        self._backbone.train()
        self._value_head.train()

    def eval_mode(self) -> None:
        self._backbone.eval()
        self._value_head.eval()

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
    ) -> tuple[list[list[int]], list[list[float]]]:
        self._activate()
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device.index or 0)

        B = len(contexts)
        if B == 1:
            ids, lps = self.act(contexts[0], n_tokens, temperature, eos_token_ids)
            return [ids], [lps]

        eos_set = set(eos_token_ids or [])
        max_len = max(len(c) for c in contexts)
        pad_id = self._tokenizer.pad_token_id
        input_ids = torch.full((B, max_len), pad_id, dtype=torch.long, device=self.device)
        attention_mask = torch.zeros(B, max_len, dtype=torch.long, device=self.device)
        for i, ctx in enumerate(contexts):
            L = len(ctx)
            input_ids[i, max_len - L:] = torch.tensor(ctx, dtype=torch.long, device=self.device)
            attention_mask[i, max_len - L:] = 1

        all_ids: list[list[int]] = [[] for _ in range(B)]
        all_lps: list[list[float]] = [[] for _ in range(B)]
        finished: list[bool] = [False] * B
        past_key_values = None

        for _ in range(n_tokens):
            out = self._backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits = out.logits[:, -1, :]
            past_key_values = out.past_key_values
            next_toks, log_probs_t = _sample_batch(logits, temperature)
            for i in range(B):
                if finished[i]:
                    continue
                tok = int(next_toks[i, 0])
                all_ids[i].append(tok)
                all_lps[i].append(float(log_probs_t[i, 0]))
                if tok in eos_set:
                    finished[i] = True
            if all(finished):
                break
            input_ids = next_toks
            attention_mask = torch.cat(
                [attention_mask, torch.ones(B, 1, dtype=torch.long, device=self.device)],
                dim=1,
            )

        return all_ids, all_lps

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

    @torch.no_grad()
    def evaluate_ref(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor | None:
        if not self._keep_ref_model:
            return None
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device.index or 0)
        # The frozen pre-trained base (adapters disabled) serves as the reference
        # for both agents — consistent with anchoring each LoRA delta to the prior.
        with self._backbone.disable_adapter():
            out = self._backbone(input_ids=input_ids, attention_mask=attention_mask)
        return out.logits
