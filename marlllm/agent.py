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
