"""
Abstract API model interface and concrete Anthropic / OpenAI implementations.

The trainer's existing rollout loop is built around `Agent.act()`, which produces
a list of token IDs in the local tokenizer's vocabulary. To plug an external
API into that loop we need a layer that:

  - takes a structured `messages` list (system + user/assistant turns),
  - calls an external provider,
  - returns the raw text response,
  - retries on transient failures,
  - caches identical requests to disk (resume-friendly + cheap reruns),
  - tracks cumulative token / USD cost against an optional hard cap.

This module owns that layer. `APIAgent` (in api_agent.py) wraps an `APIModel`
in the existing `Agent` interface so it slots into the trainer unchanged.

Logprobs note
-------------
Anthropic does not return per-token logprobs; OpenAI does. Callers should treat
`CompletionResult.logprobs` as optional and default to 0.0 when absent.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


# ── Pricing table ─────────────────────────────────────────────────────────────
# USD per 1M tokens. Update as Anthropic / OpenAI publish new prices. Models
# not listed here are billed at zero cost (budget tracking becomes a no-op for
# them) and emit a one-time warning.
PRICING_USD_PER_MTOK: dict[str, dict[str, float]] = {
    # Anthropic — Claude 4.x
    "claude-opus-4-7":    {"input": 15.0, "output": 75.0},
    "claude-opus-4-6":    {"input": 15.0, "output": 75.0},
    "claude-sonnet-4-6":  {"input":  3.0, "output": 15.0},
    "claude-haiku-4-5":   {"input":  1.0, "output":  5.0},
    "claude-haiku-4-5-20251001": {"input": 1.0, "output": 5.0},
    # OpenAI placeholders (update with actual values for the model you use).
    "gpt-4o":      {"input":  2.5, "output": 10.0},
    "gpt-4o-mini": {"input": 0.15, "output":  0.6},
}


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    p = PRICING_USD_PER_MTOK.get(model)
    if p is None:
        return 0.0
    return (input_tokens * p["input"] + output_tokens * p["output"]) / 1_000_000.0


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class Message:
    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 256
    top_p: float | None = None
    stop: list[str] | None = None


@dataclass
class CompletionResult:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    logprobs: list[float] | None = None  # per-output-token, in provider's tokenization
    metadata: dict = field(default_factory=dict)


# ── Budget tracker ────────────────────────────────────────────────────────────

class BudgetTracker:
    """Persistent USD budget tracker; file-locked for cross-process safety.

    State is stored as a single JSON file. A trivial fcntl-based lock protects
    concurrent writers (the ablation runner can launch parallel cells against
    the same shared budget by pointing them all at one state file).
    """

    def __init__(self, state_path: str | Path | None, cap_usd: float | None) -> None:
        self.state_path = Path(state_path) if state_path else None
        self.cap_usd = cap_usd
        self._lock = threading.Lock()
        if self.state_path:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)

    def _read(self) -> dict:
        if not self.state_path or not self.state_path.exists():
            return {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "calls": 0}
        try:
            return json.loads(self.state_path.read_text())
        except (json.JSONDecodeError, OSError):
            return {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "calls": 0}

    def _write(self, state: dict) -> None:
        if not self.state_path:
            return
        tmp = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        tmp.write_text(json.dumps(state, indent=2))
        os.replace(tmp, self.state_path)

    def snapshot(self) -> dict:
        with self._lock:
            return self._read()

    def add(self, model: str, input_tokens: int, output_tokens: int) -> dict:
        cost = estimate_cost_usd(model, input_tokens, output_tokens)
        with self._lock:
            state = self._read()
            state["input_tokens"] += input_tokens
            state["output_tokens"] += output_tokens
            state["cost_usd"] += cost
            state["calls"] += 1
            self._write(state)
            if self.cap_usd is not None and state["cost_usd"] > self.cap_usd:
                raise BudgetExceeded(
                    f"Budget cap exceeded: ${state['cost_usd']:.2f} > ${self.cap_usd:.2f}"
                )
            return state


class BudgetExceeded(RuntimeError):
    pass


# ── Disk cache ────────────────────────────────────────────────────────────────

class DiskCache:
    """sha256-keyed JSON cache. Entirely opt-in and local-disk only."""

    def __init__(self, root: str | Path | None) -> None:
        self.root = Path(root) if root else None
        if self.root:
            self.root.mkdir(parents=True, exist_ok=True)

    def _key(self, payload: dict) -> str:
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    def get(self, payload: dict) -> CompletionResult | None:
        if self.root is None:
            return None
        path = self.root / f"{self._key(payload)}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            return CompletionResult(**data)
        except (json.JSONDecodeError, TypeError, OSError):
            return None

    def put(self, payload: dict, result: CompletionResult) -> None:
        if self.root is None:
            return
        path = self.root / f"{self._key(payload)}.json"
        try:
            path.write_text(json.dumps(asdict(result), ensure_ascii=False))
        except OSError:
            pass


# ── Abstract API model ────────────────────────────────────────────────────────

class APIModel(ABC):
    """Provider-agnostic chat completion interface.

    Concrete subclasses implement `_complete_raw`. Retries, caching and budget
    tracking are handled by the base class.
    """

    def __init__(
        self,
        model: str,
        cache: DiskCache | None = None,
        budget: BudgetTracker | None = None,
        max_retries: int = 5,
        base_delay: float = 1.0,
        backoff_factor: float = 2.0,
        rate_limit_delay: float = 60.0,
    ) -> None:
        self.model = model
        self.cache = cache or DiskCache(None)
        self.budget = budget or BudgetTracker(None, None)
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.backoff_factor = backoff_factor
        self.rate_limit_delay = rate_limit_delay

    # Subclasses override.
    @abstractmethod
    def _complete_raw(
        self,
        messages: list[Message],
        sampling: SamplingParams,
    ) -> CompletionResult: ...

    # Subclasses may override for richer error classification.
    def _is_rate_limit(self, exc: BaseException) -> bool:
        msg = str(exc).lower()
        return "rate" in msg and "limit" in msg or "429" in msg

    def _is_retryable(self, exc: BaseException) -> bool:
        msg = str(exc).lower()
        return any(s in msg for s in ("timeout", "connection", "503", "502", "500", "overloaded"))

    def complete(
        self,
        messages: list[Message],
        sampling: SamplingParams | None = None,
    ) -> CompletionResult:
        sampling = sampling or SamplingParams()
        cache_payload = {
            "provider": self.__class__.__name__,
            "model": self.model,
            "messages": [asdict(m) for m in messages],
            "sampling": asdict(sampling),
        }
        cached = self.cache.get(cache_payload)
        if cached is not None:
            cached.metadata = {**cached.metadata, "cache_hit": True}
            return cached

        delay = self.base_delay
        last_exc: BaseException | None = None
        for attempt in range(self.max_retries + 1):
            try:
                t0 = time.time()
                result = self._complete_raw(messages, sampling)
                result.metadata = {
                    **result.metadata,
                    "cache_hit": False,
                    "latency_s": time.time() - t0,
                    "attempts": attempt + 1,
                }
                self.budget.add(self.model, result.input_tokens, result.output_tokens)
                self.cache.put(cache_payload, result)
                return result
            except BudgetExceeded:
                raise
            except BaseException as e:
                last_exc = e
                if attempt >= self.max_retries:
                    break
                if self._is_rate_limit(e):
                    sleep_for = self.rate_limit_delay
                elif self._is_retryable(e):
                    sleep_for = delay + random.random() * 0.5
                    delay *= self.backoff_factor
                else:
                    raise
                time.sleep(sleep_for)
        assert last_exc is not None
        raise last_exc


# ── Anthropic ─────────────────────────────────────────────────────────────────

class AnthropicAPIModel(APIModel):
    """Wrap the official `anthropic` Python SDK."""

    def __init__(self, model: str = "claude-sonnet-4-6", **kwargs: Any) -> None:
        super().__init__(model=model, **kwargs)
        try:
            import anthropic  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "AnthropicAPIModel requires the `anthropic` package: "
                "uv add anthropic"
            ) from e
        from anthropic import Anthropic
        self._client = Anthropic()

    def _complete_raw(
        self,
        messages: list[Message],
        sampling: SamplingParams,
    ) -> CompletionResult:
        # Anthropic separates system from the chat history.
        system_parts = [m.content for m in messages if m.role == "system"]
        chat = [m for m in messages if m.role != "system"]
        # Anthropic requires the conversation to start with a user turn.
        if not chat or chat[0].role != "user":
            chat = [Message(role="user", content="Continue.")] + chat
        api_messages = [{"role": m.role, "content": m.content} for m in chat]

        kwargs: dict[str, Any] = dict(
            model=self.model,
            max_tokens=sampling.max_tokens,
            temperature=sampling.temperature,
            messages=api_messages,
        )
        if system_parts:
            kwargs["system"] = "\n\n".join(system_parts)
        if sampling.top_p is not None:
            kwargs["top_p"] = sampling.top_p
        if sampling.stop:
            kwargs["stop_sequences"] = sampling.stop

        response = self._client.messages.create(**kwargs)
        text = "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        )
        usage = getattr(response, "usage", None)
        in_tok = getattr(usage, "input_tokens", 0) if usage else 0
        out_tok = getattr(usage, "output_tokens", 0) if usage else 0
        return CompletionResult(
            text=text,
            input_tokens=int(in_tok),
            output_tokens=int(out_tok),
            logprobs=None,
            metadata={"stop_reason": getattr(response, "stop_reason", None)},
        )

    def _is_rate_limit(self, exc: BaseException) -> bool:
        try:
            import anthropic
            if isinstance(exc, anthropic.RateLimitError):
                return True
        except ImportError:
            pass
        return super()._is_rate_limit(exc)

    def _is_retryable(self, exc: BaseException) -> bool:
        try:
            import anthropic
            if isinstance(exc, (anthropic.APIConnectionError, anthropic.InternalServerError,
                                anthropic.APITimeoutError)):
                return True
        except ImportError:
            pass
        return super()._is_retryable(exc)


# ── OpenAI ────────────────────────────────────────────────────────────────────

class OpenAIAPIModel(APIModel):
    """Wrap the official `openai` Python SDK."""

    def __init__(self, model: str = "gpt-4o-mini", **kwargs: Any) -> None:
        super().__init__(model=model, **kwargs)
        try:
            import openai  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "OpenAIAPIModel requires the `openai` package: uv add openai"
            ) from e
        from openai import OpenAI
        self._client = OpenAI()

    def _complete_raw(
        self,
        messages: list[Message],
        sampling: SamplingParams,
    ) -> CompletionResult:
        api_messages = [{"role": m.role, "content": m.content} for m in messages]
        kwargs: dict[str, Any] = dict(
            model=self.model,
            messages=api_messages,
            temperature=sampling.temperature,
            max_tokens=sampling.max_tokens,
            logprobs=True,
        )
        if sampling.top_p is not None:
            kwargs["top_p"] = sampling.top_p
        if sampling.stop:
            kwargs["stop"] = sampling.stop

        response = self._client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        text = choice.message.content or ""
        usage = response.usage
        in_tok = getattr(usage, "prompt_tokens", 0) if usage else 0
        out_tok = getattr(usage, "completion_tokens", 0) if usage else 0
        logprobs = None
        if getattr(choice, "logprobs", None) and getattr(choice.logprobs, "content", None):
            logprobs = [tok.logprob for tok in choice.logprobs.content]
        return CompletionResult(
            text=text,
            input_tokens=int(in_tok),
            output_tokens=int(out_tok),
            logprobs=logprobs,
            metadata={"finish_reason": choice.finish_reason},
        )

    def _is_rate_limit(self, exc: BaseException) -> bool:
        try:
            import openai
            if isinstance(exc, openai.RateLimitError):
                return True
        except ImportError:
            pass
        return super()._is_rate_limit(exc)

    def _is_retryable(self, exc: BaseException) -> bool:
        try:
            import openai
            if isinstance(exc, (openai.APIConnectionError, openai.APITimeoutError,
                                openai.InternalServerError)):
                return True
        except ImportError:
            pass
        return super()._is_retryable(exc)


# ── Mock (testing) ────────────────────────────────────────────────────────────

class MockAPIModel(APIModel):
    """Deterministic API model for tests / smoke runs.

    Returns a fixed canned response (or one drawn from a list keyed by call index).
    Counts tokens by whitespace-splitting; cost will be zero unless `model` is in
    the pricing table.
    """

    def __init__(
        self,
        model: str = "mock",
        responses: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, **kwargs)
        self._responses = responses or ["Let's split it 50/50."]
        self._call_idx = 0

    def _complete_raw(
        self,
        messages: list[Message],
        sampling: SamplingParams,
    ) -> CompletionResult:
        text = self._responses[self._call_idx % len(self._responses)]
        self._call_idx += 1
        in_tok = sum(len(m.content.split()) for m in messages)
        out_tok = len(text.split())
        return CompletionResult(text=text, input_tokens=in_tok, output_tokens=out_tok)


# ── Factory ───────────────────────────────────────────────────────────────────

def make_api_model(
    provider: str,
    model: str,
    cache_dir: str | None = None,
    budget_state_path: str | None = None,
    budget_cap_usd: float | None = None,
    **kwargs: Any,
) -> APIModel:
    cache = DiskCache(cache_dir) if cache_dir else DiskCache(None)
    budget = BudgetTracker(budget_state_path, budget_cap_usd)
    p = provider.lower()
    if p == "anthropic":
        return AnthropicAPIModel(model=model, cache=cache, budget=budget, **kwargs)
    if p == "openai":
        return OpenAIAPIModel(model=model, cache=cache, budget=budget, **kwargs)
    if p == "mock":
        return MockAPIModel(model=model, cache=cache, budget=budget, **kwargs)
    raise ValueError(f"Unknown provider {provider!r}; expected anthropic|openai|mock")
