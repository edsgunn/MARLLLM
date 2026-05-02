"""
Concordia ``LanguageModel`` backed by an in-process HuggingFace model.

Used by the Concordia env to serve Game-Master LLM calls (formative-memory
initialisation, occasional GM-driven prompts) without loading a second
copy of the base weights.

Why this exists
---------------
Our trained agents share a single ``PeftModel`` backbone with one LoRA
adapter per agent (``LoRASharedBaseAgent``).  The Concordia GameMaster
also needs an LLM, but for a *frozen* base — running the GM through any
agent's adapter would mix the GM into a single agent's training
distribution, which is wrong.  The right thing is to use the same
backbone with **all adapters disabled**, exposing the pre-trained model.

This wrapper does that.  When ``sample_text`` is called we enter
``peft_model.disable_adapter()`` and run a standard ``generate()`` call.
The agents' adapters are unaffected because PEFT's adapter switching
state is per-call, not per-instance.

Thread model
~~~~~~~~~~~~
Concordia's GM runs on the simulation thread (same as agent entities).
``MARLLLMLanguageModel.sample_text`` for trained agents posts a prompt
to the trainer and blocks on the act_queue; this adapter, in contrast,
generates synchronously in the sim thread.  GM calls are infrequent in
the Robotic Athanor scenario (one-shot setup + occasional forum
operations) so synchronous generation is acceptable.

Limitations
~~~~~~~~~~~
- ``sample_choice`` returns the highest-likelihood option by scoring each
  candidate's full token sequence — adequate for the choice-style GM
  prompts in the upstream scenarios but not optimised.
- No batching: each call runs a fresh generation.  If GM-LLM throughput
  becomes a bottleneck, plumb the GM through the same PettingZoo queue
  used by trained agents (see notes in concordia_env.py).
"""
from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from typing import Any

import torch
from concordia.language_model import language_model as lm_lib
from transformers import PreTrainedTokenizerBase


class SharedBackboneLanguageModel(lm_lib.LanguageModel):
    """Concordia LanguageModel running on a shared (PEFT) backbone with adapters off.

    Parameters
    ----------
    backbone:
        The PEFT ``PeftModel`` (or any HF causal LM with a ``.generate()``
        method) used by the trained agents.  Adapters are disabled for
        the duration of every call.
    tokenizer:
        Matching HuggingFace tokenizer.
    device:
        Device on which generation runs.  Inputs are moved here.
    """

    def __init__(
        self,
        backbone: Any,
        tokenizer: PreTrainedTokenizerBase,
        device: torch.device | str = "cuda:0",
    ) -> None:
        self._model = backbone
        self._tok = tokenizer
        self._device = torch.device(device)
        # Cached check: does the backbone expose PEFT's disable_adapter context manager?
        self._has_disable_adapter = hasattr(backbone, "disable_adapter")

    # ------------------------------------------------------------------ #
    # Internal generation helper                                          #
    # ------------------------------------------------------------------ #

    def _run_generate(
        self,
        prompt_ids: torch.Tensor,
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        do_sample: bool,
    ) -> torch.Tensor:
        gen_kwargs: dict = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=self._tok.pad_token_id,
        )
        if do_sample:
            gen_kwargs.update(temperature=temperature, top_p=top_p, top_k=top_k)
        ctx = self._model.disable_adapter() if self._has_disable_adapter else _NullContext()
        with ctx:
            with torch.no_grad():
                out = self._model.generate(prompt_ids, **gen_kwargs)
        return out

    # ------------------------------------------------------------------ #
    # LanguageModel interface                                             #
    # ------------------------------------------------------------------ #

    def sample_text(
        self,
        prompt: str,
        *,
        max_tokens: int = lm_lib.DEFAULT_MAX_TOKENS,
        terminators: Collection[str] = lm_lib.DEFAULT_TERMINATORS,
        temperature: float = lm_lib.DEFAULT_TEMPERATURE,
        top_p: float = lm_lib.DEFAULT_TOP_P,
        top_k: int = lm_lib.DEFAULT_TOP_K,
        timeout: float = lm_lib.DEFAULT_TIMEOUT_SECONDS,
        seed: int | None = None,
    ) -> str:
        ids = self._tok.encode(prompt, return_tensors="pt").to(self._device)
        out = self._run_generate(
            ids,
            max_new_tokens=max_tokens,
            temperature=max(temperature, 1e-3),
            top_p=top_p,
            top_k=top_k,
            do_sample=temperature > 0.0,
        )
        gen = out[0, ids.shape[1]:]
        text = self._tok.decode(gen, skip_special_tokens=True)
        # Apply Concordia's terminator convention: trim at the first occurrence.
        for term in terminators or ():
            cut = text.find(term)
            if cut != -1:
                text = text[:cut]
                break
        return text

    def sample_choice(
        self,
        prompt: str,
        responses: Sequence[str],
        *,
        seed: int | None = None,
    ) -> tuple[int, str, Mapping[str, Any]]:
        """Return the highest-log-likelihood response under the frozen base.

        Scores each ``prompt + response`` by the sum of log-probs of the
        response tokens conditioned on the prompt.  Adequate for the
        choice-style GM prompts; not optimised.
        """
        if not responses:
            raise ValueError("sample_choice requires at least one response.")

        ctx = self._model.disable_adapter() if self._has_disable_adapter else _NullContext()
        best_i = 0
        best_score = -float("inf")
        with ctx:
            with torch.no_grad():
                prompt_ids = self._tok.encode(prompt, return_tensors="pt").to(self._device)
                for i, resp in enumerate(responses):
                    full_ids = self._tok.encode(prompt + resp, return_tensors="pt").to(self._device)
                    if full_ids.shape[1] <= prompt_ids.shape[1]:
                        continue
                    logits = self._model(full_ids).logits  # (1, T, V)
                    # Shift: predict token t from logits at t-1
                    target = full_ids[0, prompt_ids.shape[1]:]
                    pred_logits = logits[0, prompt_ids.shape[1] - 1: -1]
                    log_probs = pred_logits.log_softmax(-1)
                    score = log_probs.gather(1, target.unsqueeze(-1)).sum().item()
                    if score > best_score:
                        best_score = score
                        best_i = i
        return best_i, responses[best_i], {"score": best_score}


class _NullContext:
    """Context manager that does nothing — used when backbone has no adapters."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False
