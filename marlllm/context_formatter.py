"""
Context formatters: wrap observation and action tokens with model-appropriate
role markers so the agent's context window matches the chat format the model
was pretrained on.

Without formatting, OBS and ACT tokens are concatenated as a flat stream with
no delimiter — the model has no in-context signal indicating which tokens it
generated versus which came from the environment.  Formatters fix this by
wrapping each OBS block with the model's "user turn" special tokens and
priming the context with the model's "assistant turn" start tokens before each
act() call.

Formatter contract
------------------
Three methods, all pure functions:

  wrap_prompt(ids)       → system-role wrapping for the character prompt
  wrap_observation(ids)  → user-role wrapping + assistant-start primer
  wrap_action(ids)       → typically a no-op; may add an end-of-turn suffix

The formatted sequences are used both in the live context buffer (so the model
generates from a properly-formatted prefix) and in the per-agent training
trajectories (so the forward passes during training see the same context as
during rollout).  The combined "environment overview" trajectory used for
trace writing always stores raw token IDs so that trace_utils routing
detection is unaffected.

Supported formats
-----------------
ChatMLFormatter   : Qwen2, Qwen2.5, Qwen3.x, SmolLM2, and any model whose
                    chat_template contains the string "im_start".
                    Uses  <|im_start|>system / user / assistant  markers.
                    When the chat template references ``enable_thinking``
                    (Qwen3.x native CoT), the assistant primer is extended
                    with an opening ``<think>\\n`` so generation starts
                    in-distribution. The injected primer tokens stay on
                    the OBS side of the OBS/ACT boundary — they were not
                    sampled by the agent. The matching ``</think>`` is
                    sampled and lives in ACT as normal.

Llama3Formatter   : Llama 3.x and any model whose chat_template contains
                    "start_header_id".
                    Uses  <|start_header_id|>system/user/assistant<|end_header_id|>
                    and  <|eot_id|>  markers.

ContextFormatter  : No-op base class.  Used when auto-detection finds no
                    recognised chat template, and as the explicit "none" option.

Usage
-----
    from marlllm.context_formatter import make_formatter
    formatter = make_formatter(tokenizer)           # auto-detect
    formatter = make_formatter(tokenizer, "chatml") # explicit
    formatter = make_formatter(tokenizer, "none")   # disable
"""
from __future__ import annotations


class ContextFormatter:
    """No-op formatter — tokens pass through unchanged (current behaviour)."""

    name: str = "none"

    def wrap_prompt(self, ids: list[int]) -> list[int]:
        """Wrap the character-prompt tokens as a system message."""
        return ids

    def wrap_observation(self, ids: list[int]) -> list[int]:
        """Wrap an observation as a user turn and append the assistant primer."""
        return ids

    def action_close_tokens(self, ids: list[int]) -> list[int]:
        """Tokens the framework *inserts* to close the assistant turn.

        These were not sampled by the model, so under the observation
        principle (generated → ACT/σ=0; inserted → OBS/σ=1) they belong on
        the observation side of the OBS/ACT boundary. ``wrap_action`` is
        ``ids`` followed by exactly these tokens, so the live context stream
        and the training trajectory stay byte-identical while the σ label of
        the close differs based solely on who emitted it. Base formatter
        inserts nothing."""
        return []

    def wrap_action(self, ids: list[int]) -> list[int]:
        """Wrap action tokens (generated tokens + any inserted turn-close)."""
        return list(ids) + self.action_close_tokens(ids)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class ChatMLFormatter(ContextFormatter):
    """
    ChatML format used by Qwen2, Qwen2.5, Qwen3.x, SmolLM2-Instruct, and others.

    Context structure produced::

        <|im_start|>system
        {character prompt}<|im_end|>
        <|im_start|>user
        {observation}<|im_end|>
        <|im_start|>assistant
        {action tokens}
        <|im_start|>user
        {next observation} ...

    Native thinking
    ---------------
    When ``enable_thinking=True`` (Qwen3-style native CoT), an opening
    ``<think>\\n`` is appended to the assistant primer so the model starts
    inside its trained thinking distribution. The injection lives on the
    OBS side of the OBS/ACT boundary (it's part of ``wrap_observation``'s
    output, like ``<|im_start|>assistant\\n`` itself) — these tokens were
    not sampled by the agent and therefore must not receive policy
    gradient. The closing ``</think>`` is sampled by the model and stays
    on the ACT side. ``marlllm.thinking.strip_thinking`` handles both
    closed and truncated-open blocks before routing the action to the env.
    """

    name: str = "chatml"

    def __init__(self, tokenizer, *, enable_thinking: bool = False) -> None:
        enc = lambda s: tokenizer.encode(s, add_special_tokens=False)
        self.enable_thinking = enable_thinking
        self._sys_prefix   = enc("<|im_start|>system\n")
        self._sys_suffix   = enc("<|im_end|>\n")
        self._user_prefix  = enc("<|im_start|>user\n")
        assistant_primer = "<|im_end|>\n<|im_start|>assistant\n"
        if enable_thinking:
            assistant_primer += "<think>\n"
        self._user_suffix  = enc(assistant_primer)
        # End-of-turn marker for the assistant turn. Single token in vocab.
        im_end = enc("<|im_end|>")
        self._im_end_id = im_end[0] if len(im_end) == 1 else None
        self._action_close = enc("<|im_end|>\n")  # canonical close + separator

    def wrap_prompt(self, ids: list[int]) -> list[int]:
        return self._sys_prefix + ids + self._sys_suffix

    def wrap_observation(self, ids: list[int]) -> list[int]:
        return self._user_prefix + ids + self._user_suffix

    def action_close_tokens(self, ids: list[int]) -> list[int]:
        """Framework-inserted tokens that close the assistant turn (σ=1/OBS).

        Whether ``<|im_end|>`` is here depends solely on who generated it: if
        the model emitted it (it's the last sampled token), only the separator
        ``\\n`` is inserted; if generation was cut short (e.g. stopped at an
        information-returning tag like ``</read>``), the full ``<|im_end|>\\n``
        is inserted. Either way these tokens were authored by the framework,
        not sampled by the policy, so they are perception, not action."""
        if ids and self._im_end_id is not None and ids[-1] == self._im_end_id:
            # Model stopped on <|im_end|>; only the separator newline is ours.
            return [tid for tid in self._action_close if tid != self._im_end_id]
        return list(self._action_close)


class Llama3Formatter(ContextFormatter):
    """
    Llama 3 / 3.1 / 3.2 instruct format.

    Context structure produced::

        <|start_header_id|>system<|end_header_id|>\\n\\n
        {character prompt}<|eot_id|>
        <|start_header_id|>user<|end_header_id|>\\n\\n
        {observation}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\\n\\n
        {action tokens}
        <|start_header_id|>user<|end_header_id|>\\n\\n
        ...
    """

    name: str = "llama3"

    def __init__(self, tokenizer) -> None:
        enc = lambda s: tokenizer.encode(s, add_special_tokens=False)
        self._sys_prefix  = enc("<|start_header_id|>system<|end_header_id|>\n\n")
        self._sys_suffix  = enc("<|eot_id|>")
        self._user_prefix = enc("<|start_header_id|>user<|end_header_id|>\n\n")
        self._user_suffix = enc("<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")
        eot = enc("<|eot_id|>")
        self._eot_id = eot[0] if len(eot) == 1 else None

    def wrap_prompt(self, ids: list[int]) -> list[int]:
        return self._sys_prefix + ids + self._sys_suffix

    def wrap_observation(self, ids: list[int]) -> list[int]:
        return self._user_prefix + ids + self._user_suffix

    def action_close_tokens(self, ids: list[int]) -> list[int]:
        """Framework-inserted ``<|eot_id|>`` turn-close (σ=1/OBS), if the model
        didn't already emit it. Inserted ⇒ perception, not action."""
        if self._eot_id is None:
            return []
        if ids and ids[-1] == self._eot_id:
            return []
        return [self._eot_id]


# ── Registry and factory ──────────────────────────────────────────────────────

_REGISTRY: dict[str, type[ContextFormatter]] = {
    "none":   ContextFormatter,
    "chatml": ChatMLFormatter,
    "llama3": Llama3Formatter,
}


def _detect_native_thinking(tokenizer) -> bool:
    """Return True if the tokenizer's chat template carries Qwen3-style
    native thinking (i.e. references an ``enable_thinking`` variable and
    emits a ``<think>`` opener). Used by the ``"auto"`` path so Qwen3.x
    runs get the thinking primer without any config change."""
    template = getattr(tokenizer, "chat_template", "") or ""
    return "enable_thinking" in template and "<think>" in template


def make_formatter(tokenizer, name: str = "auto") -> ContextFormatter:
    """
    Return a ContextFormatter for *tokenizer*.

    Parameters
    ----------
    tokenizer:
        A HuggingFace tokenizer.  Used for auto-detection and for encoding
        the special-token strings in the concrete formatters.
    name:
        One of ``"auto"`` (default), ``"none"``, ``"chatml"``,
        ``"chatml-think"``, ``"chatml-nothink"``, ``"llama3"``.
        ``"auto"`` inspects ``tokenizer.chat_template`` to pick the right
        class and to decide whether to inject a ``<think>`` opener (Qwen3.x).
        ``"chatml-think"`` / ``"chatml-nothink"`` force the injection on/off
        for cases where auto-detection is wrong or undesired.
    """
    if name != "auto":
        if name == "chatml-think":
            return ChatMLFormatter(tokenizer, enable_thinking=True)
        if name == "chatml-nothink":
            return ChatMLFormatter(tokenizer, enable_thinking=False)
        cls = _REGISTRY.get(name)
        if cls is None:
            raise ValueError(
                f"Unknown context_formatter {name!r}. "
                f"Choose from: {list(_REGISTRY) + ['chatml-think', 'chatml-nothink']}"
            )
        return cls(tokenizer) if cls is not ContextFormatter else ContextFormatter()

    template = getattr(tokenizer, "chat_template", "") or ""
    if "im_start" in template:
        return ChatMLFormatter(
            tokenizer, enable_thinking=_detect_native_thinking(tokenizer)
        )
    if "start_header_id" in template:
        return Llama3Formatter(tokenizer)
    return ContextFormatter()
