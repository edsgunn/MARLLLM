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
    ) -> tuple[list[int], list[float]]:
        """
        Sample n_tokens action tokens autoregressively given the full context.
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

        load_kwargs: dict = {"output_hidden_states": True}
        if torch_dtype is not None:
            load_kwargs["torch_dtype"] = torch_dtype
        if device_map is not None:
            load_kwargs["device_map"] = device_map

        self._backbone = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, **load_kwargs
        )
        if device_map is None:
            self._backbone = self._backbone.to(self.device)

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
                ref_kwargs["torch_dtype"] = torch_dtype
            if device_map is not None:
                ref_kwargs["device_map"] = device_map
            self._ref_backbone = AutoModelForCausalLM.from_pretrained(
                model_name_or_path, **ref_kwargs
            )
            if device_map is None:
                self._ref_backbone = self._ref_backbone.to(self.device)
            for p in self._ref_backbone.parameters():
                p.requires_grad_(False)
            self._ref_backbone.eval()
        else:
            self._ref_backbone = None

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
    ) -> tuple[list[int], list[float]]:
        """Sample n_tokens tokens, reusing KV cache across steps."""
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

            if temperature != 1.0:
                logits = logits / temperature

            probs = F.softmax(logits, dim=-1)
            token_id = torch.multinomial(probs, num_samples=1).item()
            log_prob = F.log_softmax(logits, dim=-1)[0, token_id].item()

            sampled_ids.append(int(token_id))
            sampled_lps.append(float(log_prob))

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
    ) -> tuple[list[list[int]], list[list[float]]]:
        """
        Sample n_tokens for each context in a single batched forward pass.

        Contexts are left-padded to the same length so they form a (B, T)
        tensor.  After the first (full-context) step, subsequent steps feed
        only the newly generated token (B, 1) reusing the KV cache, so the
        per-step cost is proportional to B rather than B × T.

        Returns:
            (list[list[int]], list[list[float]]) — token IDs and log-probs
            for each context, in the same order as the input.
        """
        B = len(contexts)
        if B == 1:
            # Avoid padding overhead for a single context.
            ids, lps = self.act(contexts[0], n_tokens, temperature)
            return [ids], [lps]

        max_len = max(len(c) for c in contexts)
        pad_id = self._tokenizer.pad_token_id

        # Left-pad: real tokens are right-aligned so the last position of
        # each row is the true last token of that context.
        input_ids = torch.full((B, max_len), pad_id, dtype=torch.long, device=self.device)
        attention_mask = torch.zeros(B, max_len, dtype=torch.long, device=self.device)
        for i, ctx in enumerate(contexts):
            L = len(ctx)
            input_ids[i, max_len - L:] = torch.tensor(ctx, dtype=torch.long, device=self.device)
            attention_mask[i, max_len - L:] = 1

        all_ids: list[list[int]] = [[] for _ in range(B)]
        all_lps: list[list[float]] = [[] for _ in range(B)]
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

            if temperature != 1.0:
                logits = logits / temperature

            probs = F.softmax(logits, dim=-1)
            next_toks = torch.multinomial(probs, num_samples=1)          # (B, 1)
            log_probs_t = F.log_softmax(logits, dim=-1).gather(1, next_toks)  # (B, 1)

            for i in range(B):
                all_ids[i].append(int(next_toks[i, 0]))
                all_lps[i].append(float(log_probs_t[i, 0]))

            # Next step: feed only the new token; extend the mask to cover
            # all cached positions + the new one.
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
        out = self._ref_backbone(input_ids=input_ids, attention_mask=attention_mask)
        return out.logits
