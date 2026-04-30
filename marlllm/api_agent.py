"""
APIAgent: implements the Agent interface using an external chat API.

Why this exists
---------------
The trainer's rollout loop calls `agent.act(context_token_ids, ...)` and expects
back token IDs in the local tokenizer's vocabulary plus per-token log-probs.
APIAgent lets a remote model (Anthropic, OpenAI, ...) play that role.

How the local context maps to API messages
-------------------------------------------
The trainer maintains a per-agent context buffer of token IDs that includes the
character prompt (system) and a history of OBS / ACT spans wrapped by the
context formatter. We do *not* try to parse those formatted token streams back
into structured messages — that round-trip is brittle and tokenizer-dependent.

Instead, the APIAgent keeps its own structured conversation history, populated
*explicitly* by the rollout code via `note_observation()` between act() calls.
For backwards compatibility with rollout code that does not call note_*, we
fall back to a single-user-message blob containing the decoded context.

This means: drop-in compatibility works (decoded blob fallback), and a richer
mode is available if the trainer is taught to call note_observation().

Tokenization
------------
Responses are re-tokenized in the *local* tokenizer's vocabulary so they slot
into the trainer's RolloutBatch unchanged. log-probs are 0.0 for Anthropic
(not provided) or interpolated from provider tokens for OpenAI when available.
We do not try to align provider-token log-probs to local-tokenizer segmentation
— callers should treat APIAgent log-probs as approximate, and APIAgents must
always be in `frozen_agents` so REINFORCE never uses them.

Frozen-only contract
--------------------
APIAgent.parameters() returns nothing. The trainer must list every APIAgent in
`TrainingConfig.frozen_agents`; otherwise the optimiser construction will
filter them out via parameters() but evaluate() will be called and fail.
"""
from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn
from transformers import PreTrainedTokenizerBase

from marlllm.agent import Agent
from marlllm.api_model import APIModel, Message, SamplingParams
from marlllm.context_formatter import make_formatter


class APIAgent(Agent):
    """A frozen Agent backed by an external chat completion API."""

    def __init__(
        self,
        agent_id: str,
        character_prompt: str,
        api_model: APIModel,
        tokenizer: PreTrainedTokenizerBase,
        sampling: SamplingParams | None = None,
        context_formatter: str = "auto",
        max_history_chars: int = 16000,
    ) -> None:
        self._agent_id = agent_id
        self._character_prompt = character_prompt
        self._api = api_model
        self._tokenizer = tokenizer
        self._sampling = sampling or SamplingParams(temperature=1.0, max_tokens=256)
        self._max_history_chars = max_history_chars
        # Dummy device: nothing to place on it, but the trainer reads .device.
        self.device = torch.device("cpu")
        self.context_formatter = make_formatter(tokenizer, context_formatter)

        # Optional structured history (one list per "rollout slot"). Populated
        # by callers via note_observation(). Slot 0 is the default for the
        # un-batched act() path.
        self._history: dict[int, list[Message]] = {}

    # ------------------------------------------------------------------ #
    # Agent interface                                                     #
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
        return iter(())  # frozen by construction

    def train_mode(self) -> None: ...
    def eval_mode(self) -> None: ...

    # ------------------------------------------------------------------ #
    # Optional structured-history hooks                                    #
    # ------------------------------------------------------------------ #

    def reset_history(self, slot: int = 0) -> None:
        self._history[slot] = []

    def note_observation(self, text: str, slot: int = 0) -> None:
        """Record an observation visible to this agent at `slot`."""
        self._history.setdefault(slot, []).append(Message(role="user", content=text))

    def note_own_action(self, text: str, slot: int = 0) -> None:
        """Record this agent's own utterance at `slot`."""
        self._history.setdefault(slot, []).append(Message(role="assistant", content=text))

    # ------------------------------------------------------------------ #
    # Rollout                                                             #
    # ------------------------------------------------------------------ #

    def _build_messages(self, context_token_ids: list[int], slot: int) -> list[Message]:
        msgs: list[Message] = [Message(role="system", content=self._character_prompt)]
        history = self._history.get(slot, [])
        if history:
            msgs.extend(history)
            return msgs
        # Fallback: decode the raw token context as a single user blob. This
        # works without changes to the trainer at the cost of leaking chat
        # template markers from the local tokenizer into the prompt.
        if context_token_ids:
            text = self._tokenizer.decode(context_token_ids, skip_special_tokens=True)
            text = text[-self._max_history_chars:]
            msgs.append(Message(role="user", content=text or "Begin."))
        else:
            msgs.append(Message(role="user", content="Begin."))
        return msgs

    def act(
        self,
        context_token_ids: list[int],
        n_tokens: int,
        temperature: float = 1.0,
    ) -> tuple[list[int], list[float]]:
        sampling = SamplingParams(
            temperature=temperature,
            max_tokens=max(n_tokens, self._sampling.max_tokens),
            top_p=self._sampling.top_p,
            stop=self._sampling.stop,
        )
        messages = self._build_messages(context_token_ids, slot=0)
        result = self._api.complete(messages, sampling)
        # Re-tokenize in local vocabulary; truncate to the budget the trainer
        # asked for so the rollout buffer stays well-shaped.
        ids = self._tokenizer.encode(result.text, add_special_tokens=False)[:n_tokens]
        log_probs = [0.0] * len(ids)
        # Update structured history opportunistically.
        self._history.setdefault(0, []).append(Message(role="assistant", content=result.text))
        return ids, log_probs

    def act_batch(
        self,
        contexts: list[list[int]],
        n_tokens: int,
        temperature: float = 1.0,
    ) -> tuple[list[list[int]], list[list[float]]]:
        # Sequential calls — providers don't reliably batch, and caching makes
        # repeats free. Each context corresponds to one rollout slot.
        all_ids: list[list[int]] = []
        all_lps: list[list[float]] = []
        for slot, ctx in enumerate(contexts):
            sampling = SamplingParams(
                temperature=temperature,
                max_tokens=max(n_tokens, self._sampling.max_tokens),
                top_p=self._sampling.top_p,
                stop=self._sampling.stop,
            )
            messages = self._build_messages(ctx, slot=slot)
            result = self._api.complete(messages, sampling)
            ids = self._tokenizer.encode(result.text, add_special_tokens=False)[:n_tokens]
            all_ids.append(ids)
            all_lps.append([0.0] * len(ids))
            self._history.setdefault(slot, []).append(
                Message(role="assistant", content=result.text)
            )
        return all_ids, all_lps

    # ------------------------------------------------------------------ #
    # Training-time interfaces — not supported, but Trainer skips frozen   #
    # agents so these should never be reached in practice.                #
    # ------------------------------------------------------------------ #

    def evaluate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise RuntimeError(
            f"APIAgent {self._agent_id!r} is non-differentiable; it must be "
            f"listed in TrainingConfig.frozen_agents."
        )

    def evaluate_ref(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor | None:
        return None
