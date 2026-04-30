"""
DialogueContext — a single agent's view of an episode as a chat-template
message list.

Architectural contract
----------------------
This class is environment-agnostic and **per-agent**. It knows nothing about
other agents, multi-agent routing, or speaker identities. From the agent's
perspective there are exactly two kinds of tokens:

  * observation — anything the agent receives from outside itself, which we
                  represent as a ``"user"`` chat-template message
  * action      — anything the agent generates, represented as ``"assistant"``

Other agents in a PettingZoo AEC environment are part of the environment from
this layer's point of view. If their utterances need a speaker tag (e.g.
``[Vex]: …``) the *environment* attaches it when constructing the observation;
this class does not synthesise such tags and never auto-routes content from
one agent's history into another's.

The rollout loop typically maintains ``dict[agent_id, DialogueContext]`` and
drives each context independently using ``env.last()`` (observations) and the
returned ``act`` text (actions).

The class is the single point of truth for what the model sees on the next
generation step. ``get_input_ids()`` runs the tokeniser's chat template with
``add_generation_prompt=True`` so the prompt always ends with the assistant
primer.
"""
from __future__ import annotations

from typing import Iterable

from transformers import PreTrainedTokenizerBase


# Special tokens that may legally end an assistant turn. ChatML uses
# <|im_end|>; Qwen pretrained models sometimes emit <|endoftext|>; Llama 3
# uses <|eot_id|>. Generation must terminate on any of these — and the
# recorded utterance must be decoded with skip_special_tokens=True so they
# cannot leak into the next turn's observation content.
_CHAT_EOS_TOKEN_STRINGS: tuple[str, ...] = (
    "<|im_end|>",
    "<|endoftext|>",
    "<|eot_id|>",
)


def chat_eos_token_ids(tokenizer: PreTrainedTokenizerBase) -> list[int]:
    """Return ids for chat-template end-of-turn tokens that exist in vocab.

    The tokeniser's own ``eos_token_id`` is appended if not already present.
    Tokens that are absent or aren't single tokens are skipped silently.
    """
    ids: list[int] = []
    seen: set[int] = set()

    for s in _CHAT_EOS_TOKEN_STRINGS:
        encoded = tokenizer.encode(s, add_special_tokens=False)
        if len(encoded) != 1:
            continue
        tid = encoded[0]
        if tid == tokenizer.unk_token_id:
            continue
        if tid in seen:
            continue
        ids.append(tid)
        seen.add(tid)

    if tokenizer.eos_token_id is not None and tokenizer.eos_token_id not in seen:
        ids.append(tokenizer.eos_token_id)

    return ids


def verify_special_tokens(
    tokenizer: PreTrainedTokenizerBase,
    required: Iterable[str] = ("<|im_start|>", "<|im_end|>"),
) -> None:
    """Sanity-check that chat-template special tokens are single-token in vocab.

    Raises ``RuntimeError`` if not — the harness should abort, since no
    training run can produce meaningful results with a misconfigured tokeniser.
    """
    for tok in required:
        ids = tokenizer.encode(tok, add_special_tokens=False)
        if len(ids) != 1:
            raise RuntimeError(
                f"Special token {tok!r} encodes to {len(ids)} subword tokens "
                f"({ids}); expected 1. Tokeniser is misconfigured."
            )
        if ids[0] == tokenizer.unk_token_id:
            raise RuntimeError(
                f"Special token {tok!r} encodes to UNK; not in vocabulary."
            )


class DialogueContext:
    """A single agent's chat-template message list for one episode.

    Parameters
    ----------
    tokenizer:
        HuggingFace tokenizer used to apply the chat template.
    system_prompt:
        Optional system message inserted at the start of the history. May be
        empty, in which case no system message is recorded.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        system_prompt: str = "",
    ) -> None:
        self.tokenizer = tokenizer
        self._messages: list[dict[str, str]] = []
        if system_prompt:
            self._messages.append({"role": "system", "content": system_prompt})

    # ---------------------------------------------------------------- #
    # Recording                                                         #
    # ---------------------------------------------------------------- #

    def add_observation(self, content: str) -> None:
        """Append an observation (a user-role chat message)."""
        self._messages.append({"role": "user", "content": content})

    def add_action(self, content: str) -> None:
        """Append an action (an assistant-role chat message)."""
        self._messages.append({"role": "assistant", "content": content})

    # ---------------------------------------------------------------- #
    # Tokenisation entry point                                          #
    # ---------------------------------------------------------------- #

    def get_input_text(self) -> str:
        """Render the message history to a string with all special tokens
        preserved. Used as both the source for tokenisation and a debug-dump
        artifact (the rendered prompt is what the model actually sees)."""
        return self.tokenizer.apply_chat_template(
            self._messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def get_input_ids(self) -> list[int]:
        """Return tokenised input for the agent's next generation step.

        The full message history is rendered through the tokeniser's chat
        template with ``add_generation_prompt=True`` so the prompt ends
        with the assistant primer (e.g. ``<|im_start|>assistant\\n``), then
        re-encoded into a flat ``list[int]``.
        """
        text = self.get_input_text()
        return self.tokenizer.encode(text, add_special_tokens=False)

    # ---------------------------------------------------------------- #
    # Inspection / serialisation                                        #
    # ---------------------------------------------------------------- #

    @property
    def messages(self) -> list[dict[str, str]]:
        """A copy of the agent's full message history."""
        return [dict(m) for m in self._messages]

    def __len__(self) -> int:
        return len(self._messages)

    def render_pretty(self) -> str:
        """Human-readable view of the history (no chat-template syntax)."""
        out: list[str] = []
        for msg in self._messages:
            out.append(f"[{msg['role']}]")
            out.append(msg["content"])
            out.append("")
        return "\n".join(out)
