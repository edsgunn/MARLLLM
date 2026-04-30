"""
Sanity checks for the overhauled per-agent dialogue context manager.

Run before any training or eval campaign to verify the harness is correctly
assembling contexts and terminating generation cleanly. The checks here
treat DialogueContext as strictly per-agent: the only signals it knows about
are observations (user-role) and actions (assistant-role). Anything to do
with multi-agent routing or speaker labels is the environment's concern and
is intentionally not tested here.

Usage
-----
    uv run python scripts/test_dialogue_context.py
    uv run python scripts/test_dialogue_context.py --skip-generation   # no model load
    uv run python scripts/test_dialogue_context.py --model Qwen/Qwen2.5-1.5B-Instruct
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Repo-root-importable when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from marlllm.dialogue import (
    DialogueContext,
    chat_eos_token_ids,
    verify_special_tokens,
)


_OK = "\033[92m✓\033[0m"
_FAIL = "\033[91m✗\033[0m"


def _result(name: str, ok: bool, detail: str = "") -> bool:
    mark = _OK if ok else _FAIL
    print(f"  {mark} {name}{(' — ' + detail) if detail else ''}")
    return ok


def _generate(
    model,
    tokenizer,
    input_ids: list[int],
    max_new_tokens: int,
    eos_ids: list[int],
    temperature: float,
    do_sample: bool = True,
) -> tuple[str, list[int]]:
    """Generate one completion. Returns (decoded text with skip_special_tokens,
    raw new token ids)."""
    inp = torch.tensor([input_ids], dtype=torch.long, device=model.device)
    out = model.generate(
        inp,
        max_new_tokens=max_new_tokens,
        eos_token_id=eos_ids,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        do_sample=do_sample,
        temperature=temperature if do_sample else 1.0,
    )
    new_tokens = out[0, inp.shape[1]:].tolist()
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return text, new_tokens


# ────────────────────────────────────────────────────────────────────────── #
# Checks                                                                     #
# ────────────────────────────────────────────────────────────────────────── #


def check_special_tokens(tokenizer) -> bool:
    print("§3.6  Special-token verification")
    try:
        verify_special_tokens(tokenizer)
        ok = _result("required tokens are single-token, non-UNK", True)
    except RuntimeError as e:
        ok = _result("required tokens are single-token, non-UNK", False, str(e))
    eos_ids = chat_eos_token_ids(tokenizer)
    _result("chat_eos_token_ids returns non-empty list", bool(eos_ids), str(eos_ids))
    return ok


def check_role_types(tokenizer) -> bool:
    print("§3.1/3.2  Two token types only: observation→user, action→assistant")
    ctx = DialogueContext(tokenizer, system_prompt="You are a trader.")
    ctx.add_observation("Task: divide the items.")
    ctx.add_action("I'll start by asking what you want.")
    ctx.add_observation("I want the books.")
    ctx.add_action("OK, then I take the hats.")

    msgs = ctx.messages
    ok1 = _result(
        "system prompt at index 0",
        msgs[0]["role"] == "system" and msgs[0]["content"] == "You are a trader.",
    )
    ok2 = _result(
        "observations are user-role",
        msgs[1]["role"] == "user" and msgs[3]["role"] == "user",
    )
    ok3 = _result(
        "actions are assistant-role",
        msgs[2]["role"] == "assistant" and msgs[4]["role"] == "assistant",
    )
    ok4 = _result(
        "DialogueContext exposes no notion of 'other agent'",
        not any(hasattr(ctx, name)
                for name in ("record_agent_utterance", "speaker_labels", "agent_ids")),
    )
    return all([ok1, ok2, ok3, ok4])


def check_generation_prompt(tokenizer) -> bool:
    print("§3.3  Tokenisation entry point: ends with assistant primer")
    ctx = DialogueContext(tokenizer, system_prompt="You are a trader.")
    ctx.add_observation("Hello.")
    text = ctx.get_input_text()
    ok = _result(
        "rendered prompt ends with '<|im_start|>assistant\\n'",
        text.endswith("<|im_start|>assistant\n"),
        f"...{text[-40:]!r}",
    )
    # Token-level: last few ids should match.
    ids = ctx.get_input_ids()
    primer = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
    ok2 = _result(
        "tokenised input ends with the assistant primer ids",
        ids[-len(primer):] == primer,
    )
    return ok and ok2


def check_round_trip(tokenizer) -> bool:
    print("§7.2  Round-trip integrity (turn N+1 prompt extends turn N)")
    ctx = DialogueContext(tokenizer, system_prompt="You are a trader.")
    ctx.add_observation("task")
    ids_before = ctx.get_input_ids()
    ctx.add_action("hello")
    ctx.add_observation("hi back")
    ids_after = ctx.get_input_ids()
    text_after = ctx.get_input_text()

    # All prior content should still be in the rendered output.
    ok1 = _result(
        "all prior content present in new prompt",
        ("task" in text_after) and ("hello" in text_after) and ("hi back" in text_after),
    )
    # The new prompt must be strictly longer (we added two messages).
    ok2 = _result(
        "new prompt is longer than prior prompt (more tokens)",
        len(ids_after) > len(ids_before),
        f"{len(ids_before)} → {len(ids_after)} tokens",
    )
    return ok1 and ok2


def check_eos_termination(model, tokenizer) -> bool:
    print("§7.4  EOS termination stress test (temp=1.5, 20 turns)")
    eos_ids = chat_eos_token_ids(tokenizer)
    forbidden = ["<|im_start|>", "<|im_end|>", "<|endoftext|>"]
    leaks = 0
    hit_budget = 0
    for i in range(20):
        ctx = DialogueContext(
            tokenizer,
            system_prompt="You are a chatty trader. Keep your answer short.",
        )
        ctx.add_observation(f"Say something brief about trade. (#{i})")
        text, new_tokens = _generate(
            model, tokenizer, ctx.get_input_ids(),
            max_new_tokens=64, eos_ids=eos_ids, temperature=1.5,
        )
        for tok in forbidden:
            if tok in text:
                leaks += 1
                break
        # Did we end on an EOS token (good) or hit the budget (no natural stop)?
        if not new_tokens or new_tokens[-1] not in eos_ids:
            hit_budget += 1

    ok = _result(
        "no special-token strings in any of 20 high-temp completions",
        leaks == 0,
        f"leaks={leaks}, no_natural_stop={hit_budget}/20",
    )
    return ok


def check_submit_temperature(model, tokenizer) -> bool:
    print("§3.5  SUBMIT-style structured prompt at low temperature")
    eos_ids = chat_eos_token_ids(tokenizer)
    ctx = DialogueContext(
        tokenizer,
        system_prompt="You are a trader. Reply only in the requested format.",
    )
    ctx.add_observation(
        "There are 5 books, 3 hats, 1 ball available. Submit your allocation "
        "in the exact format `books=N hats=N balls=N`. No other text."
    )
    text, _ = _generate(
        model, tokenizer, ctx.get_input_ids(),
        max_new_tokens=32, eos_ids=eos_ids, temperature=0.0, do_sample=False,
    )
    ok = _result(
        "greedy SUBMIT response contains 'books=' and 'hats=' and 'balls='",
        all(s in text for s in ("books=", "hats=", "balls=")),
        text.replace("\n", " ")[:120],
    )
    return ok


# ────────────────────────────────────────────────────────────────────────── #
# Main                                                                       #
# ────────────────────────────────────────────────────────────────────────── #


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="bfloat16",
                   choices=["float32", "bfloat16", "float16"])
    p.add_argument("--skip-generation", action="store_true",
                   help="Skip checks that require loading the model.")
    args = p.parse_args()

    print(f"Loading tokenizer for {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    results: list[bool] = []
    results.append(check_special_tokens(tokenizer));     print()
    results.append(check_role_types(tokenizer));         print()
    results.append(check_generation_prompt(tokenizer));  print()
    results.append(check_round_trip(tokenizer));         print()

    if not args.skip_generation:
        dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
        print(f"Loading model {args.model} on {args.device} ({args.dtype}) ...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=dtype_map[args.dtype]
        ).to(args.device).eval()
        results.append(check_eos_termination(model, tokenizer));  print()
        results.append(check_submit_temperature(model, tokenizer)); print()

    n_pass = sum(results)
    n_total = len(results)
    print(f"Result: {n_pass}/{n_total} checks passed.")
    sys.exit(0 if n_pass == n_total else 1)


if __name__ == "__main__":
    main()
