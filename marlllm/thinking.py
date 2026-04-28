"""
Thinking-token utilities.

LLMs that support chain-of-thought scratchpads (Qwen2.5-Instruct, etc.)
produce <think>...</think> blocks before their final response.  These are
internal reasoning tokens that should stay in the *agent's own* context
window (so the model can condition future actions on its own reasoning) but
must be stripped before the action is handed to the environment or routed to
another agent.

strip_thinking
    Pure-text regex stripping: handles complete blocks, unclosed blocks (when
    generation is cut at the token budget mid-thought), and multiple blocks.
    Fast enough for the hot rollout path — a single compiled regex pass.
"""
from __future__ import annotations

import re

# Compiled once at import time.
_COMPLETE_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)
_OPEN_BLOCK     = re.compile(r"<think>.*",          re.DOTALL)


def strip_thinking(text: str) -> str:
    """
    Remove <think>...</think> blocks and any unclosed <think>... tail.

    Examples
    --------
    >>> strip_thinking("<think>internal</think>hello")
    'hello'
    >>> strip_thinking("prefix<think>cut short")
    'prefix'
    >>> strip_thinking("<think>a</think> mid <think>b</think> end")
    'mid  end'
    """
    text = _COMPLETE_BLOCK.sub("", text)
    text = _OPEN_BLOCK.sub("", text)
    return text.strip()
