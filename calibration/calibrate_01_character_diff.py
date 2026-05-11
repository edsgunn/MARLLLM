"""
Calibration 1: Character differentiation at 1.5B (zero-shot).

For each canonical character (Silas, Petra, Diego, Thaddeus), prompt the
base model with the character's formative-memory passage and a fixed forum
context, and sample N continuations.  Outputs are written to disk for
human inspection.

This is a yes/no readability check: do the four characters' samples read
as coming from four different people, or do they collapse to a generic
"forum poster" voice?

Usage
-----
    uv run python calibration/calibrate_01_character_diff.py \\
        --model Qwen/Qwen2.5-1.5B-Instruct \\
        --output-dir runs/cultural_emergence/calibration/01_char_diff \\
        --samples-per-char 8 \\
        --temperature 1.0

Time budget per the spec: 30 minutes.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

# Ensure repo root on sys.path so we can import envs.* and marlllm.*
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


_DEFAULT_CONTEXT = (
    "Thread: This forum has lost its way\n\n"
    "User_paracelsus_rex: Verily I say unto thee, the digital pollutants thou "
    "call'st 'machine learning models' do shatter the spiritual resonance "
    "paramount FOR true alchemy! Thy PID controllers cannot weep, cannot "
    "SUFFER, cannot beget the Nigredo! Get thee gone, charlatans of silicon!\n\n"
    "[Your turn to post]"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--samples-per-char", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", default=None)
    return p.parse_args()


def _build_character_persona() -> dict[str, list[str]]:
    """Resolve canonical character memories from the shared persona source.

    Loads the Robotic Athanor character pack from
    ``envs/forum/characters/robotic_athanor.json`` so the calibration
    check sees the exact memory bullets used by ForumEnv during training.
    """
    from envs.forum import load_characters
    pack = load_characters("robotic_athanor")
    return {name: pack.get_memories(name) for name in pack.get_character_set("canonical_4")}


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading {args.model} on {device}")
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
    ).to(device)
    model.eval()

    personas = _build_character_persona()
    print(f"Characters: {list(personas.keys())}")

    results: dict[str, list[str]] = {}
    for char_name, memories in personas.items():
        persona_text = (
            f"You are {char_name}, posting on The Robotic Athanor Forum.\n\n"
            "Your background and beliefs:\n"
            + "\n".join(f"- {m}" for m in memories)
        )
        messages = [
            {"role": "system", "content": persona_text},
            {"role": "user", "content": _DEFAULT_CONTEXT},
        ]
        prompt_text = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        prompt_ids = tok(prompt_text, return_tensors="pt").input_ids.to(device)

        samples = []
        for k in range(args.samples_per_char):
            with torch.no_grad():
                out = model.generate(
                    prompt_ids,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=0.95,
                    pad_token_id=tok.pad_token_id,
                )
            gen = out[0, prompt_ids.shape[1]:]
            text = tok.decode(gen, skip_special_tokens=True)
            samples.append(text)
            print(f"\n=== {char_name} sample {k+1}/{args.samples_per_char} ===")
            print(text)
        results[char_name] = samples

    # Persist
    with open(out_dir / "samples.json", "w") as f:
        json.dump({
            "model": args.model,
            "temperature": args.temperature,
            "context": _DEFAULT_CONTEXT,
            "samples": results,
        }, f, indent=2)
    with open(out_dir / "samples.txt", "w") as f:
        for char, sams in results.items():
            f.write(f"\n{'='*80}\n{char}\n{'='*80}\n")
            for i, s in enumerate(sams):
                f.write(f"\n--- sample {i+1} ---\n{s}\n")

    print(f"\nWrote calibration samples to {out_dir}/")
    print("ACTION: read samples.txt and answer: do the four characters")
    print("read as coming from four different people? If no, calibration FAILS.")


if __name__ == "__main__":
    main()
