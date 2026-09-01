"""Provider abstraction. Anthropic and OpenAI are both first-class.

API keys come from environment variables only (ANTHROPIC_API_KEY /
OPENAI_API_KEY) and are never logged. Backends return raw model text; parsing
and validation live in `judge.py`.
"""

from __future__ import annotations

import json
import os
from typing import Protocol


# Per-1M-token prices (USD) for the end-of-run cost tally. Update as needed.
PRICING = {
    "claude-opus-4-8": (5.0, 25.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
    # OpenAI examples (fill in current values before relying on the tally):
    "gpt-4.1": (2.0, 8.0),
    "gpt-4.1-mini": (0.4, 1.6),
}


class JudgeBackend(Protocol):
    def label(self, system: str, user: str, *, temperature: float, max_tokens: int) -> "LabelResult":
        ...


class LabelResult:
    __slots__ = ("text", "usage_in", "usage_out")

    def __init__(self, text: str, usage_in: int = 0, usage_out: int = 0):
        self.text = text
        self.usage_in = usage_in
        self.usage_out = usage_out


# ---------------------------------------------------------------------------
# Judge backends
# ---------------------------------------------------------------------------
class AnthropicBackend:
    def __init__(self, model: str, max_retries: int = 4, json_mode: bool = False):
        import anthropic  # noqa: local import so OpenAI-only setups don't need it
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Export it before running extraction."
            )
        self._client = anthropic.Anthropic(max_retries=max_retries)
        self.model = model

    def label(self, system, user, *, temperature, max_tokens) -> LabelResult:
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        u = resp.usage
        return LabelResult(text, getattr(u, "input_tokens", 0), getattr(u, "output_tokens", 0))


class OpenAIBackend:
    def __init__(self, model: str, max_retries: int = 4, json_mode: bool = True):
        import openai
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Export it before running extraction."
            )
        self._client = openai.OpenAI(max_retries=max_retries)
        self.model = model
        self._json_mode = json_mode

    def label(self, system, user, *, temperature, max_tokens) -> LabelResult:
        kwargs = dict(
            model=self.model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        if self._json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = self._client.chat.completions.create(**kwargs)
        text = resp.choices[0].message.content or ""
        u = resp.usage
        return LabelResult(
            text,
            getattr(u, "prompt_tokens", 0),
            getattr(u, "completion_tokens", 0),
        )


def make_judge_backend(cfg) -> JudgeBackend:
    if cfg.provider == "anthropic":
        return AnthropicBackend(cfg.model, cfg.max_retries, cfg.json_mode)
    if cfg.provider == "openai":
        return OpenAIBackend(cfg.model, cfg.max_retries, cfg.json_mode)
    raise ValueError(f"unknown judge provider: {cfg.provider}")


# ---------------------------------------------------------------------------
# Embedding backends (for Rung-1 clustering)
# ---------------------------------------------------------------------------
class OpenAIEmbedding:
    """Per-text embeddings via OpenAI; cached by hash(text+model)."""

    def __init__(self, model: str):
        import openai
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set (embedding.provider=openai).")
        self._client = openai.OpenAI()
        self.model = model

    def embed(self, texts: list) -> list:
        out = []
        # batch in chunks of 256 to stay within request limits
        for i in range(0, len(texts), 256):
            chunk = texts[i : i + 256]
            resp = self._client.embeddings.create(model=self.model, input=chunk)
            out.extend([d.embedding for d in resp.data])
        return out
