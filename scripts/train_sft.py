"""
Vanilla SFT trainer for Cell A of the distillation ablation.

Consumes the JSONL produced by `scripts/dump_transcripts.py` and trains a
HuggingFace causal LM with a standard next-token-prediction objective masked
to the action tokens of one designated learner role.

This is the bare-minimum baseline: no LoRA hooks, no RL, no value head — just
"predict the strong partner's next utterance given the conversation so far."
Cell A's role is to make every other cell's gain interpretable.

Usage
-----
    uv run python scripts/train_sft.py \
        --transcripts runs/cellA_transcripts.jsonl \
        --model Qwen/Qwen2.5-0.5B \
        --learner-role agent_0 \
        --output-dir runs/cellA_sft \
        --epochs 3 --batch-size 4 --lr 5e-6
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

# Ensure repo root is importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer

from marlllm.context_formatter import make_formatter


class TranscriptDataset(Dataset):
    """One example per transcript turn-of-the-learner.

    For each (episode, learner-action turn t), we build:
        input_ids   = wrap_prompt(prompt) + Σ_{s<t} wrap_*(turn_s) + wrap_obs(prefix to act)
        target_ids  = action tokens
        loss_mask   = 1 on target_ids, 0 on the rest
    """

    def __init__(
        self,
        path: str,
        tokenizer,
        learner_role: str,
        learner_prompt: str,
        max_length: int = 4096,
    ) -> None:
        self.tokenizer = tokenizer
        self.formatter = make_formatter(tokenizer)
        self.learner_role = learner_role
        self.learner_prompt = learner_prompt
        self.max_length = max_length
        self.examples: list[tuple[list[int], list[int]]] = []  # (input_ids, loss_mask)

        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                ep = json.loads(line)
                self._add_episode(ep)

    def _add_episode(self, ep: dict) -> None:
        prompt_ids = self.tokenizer.encode(self.learner_prompt, add_special_tokens=False)
        ctx = self.formatter.wrap_prompt(prompt_ids)
        # Track which positions are "this is what the learner should predict".
        loss_mask = [0] * len(ctx)
        ids = list(ctx)

        for turn in ep["turns"]:
            tok_ids = turn.get("tokens") or self.tokenizer.encode(
                turn.get("text", ""), add_special_tokens=False
            )
            if not tok_ids:
                continue
            if turn["role"] == "obs":
                # Observations to the learner are wrapped as user turns — but
                # we only need the learner's POV. Our transcripts attribute
                # obs to the receiving agent_id, so we filter by that.
                if turn["agent_id"] != self.learner_role:
                    continue
                wrapped = self.formatter.wrap_observation(tok_ids)
                ids.extend(wrapped)
                loss_mask.extend([0] * len(wrapped))
            elif turn["role"] == "act":
                if turn["agent_id"] != self.learner_role:
                    # Other agent acted; in the AEC env, the next obs to the
                    # learner will already include this utterance. Skip.
                    continue
                # Snapshot a training example: predict this action given the
                # context built so far.
                wrapped = self.formatter.wrap_action(tok_ids)
                example_input = ids + wrapped
                example_mask = loss_mask + [1] * len(wrapped)
                if len(example_input) > self.max_length:
                    # Left-truncate to keep the action visible.
                    over = len(example_input) - self.max_length
                    example_input = example_input[over:]
                    example_mask = example_mask[over:]
                self.examples.append((example_input, example_mask))
                ids.extend(wrapped)
                loss_mask.extend([0] * len(wrapped))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        return self.examples[idx]


def collate(batch, pad_id: int):
    max_len = max(len(x[0]) for x in batch)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    attn = torch.zeros(len(batch), max_len, dtype=torch.long)
    mask = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, (ids, m) in enumerate(batch):
        input_ids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        attn[i, :len(ids)] = 1
        mask[i, :len(m)] = torch.tensor(m, dtype=torch.long)
    return input_ids, attn, mask


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--transcripts", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--learner-role", default="agent_0", choices=["agent_0", "agent_1"])
    p.add_argument("--learner-prompt",
                   default="You are Agent A, negotiating to maximise your score.")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--max-length", type=int, default=4096)
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", default="auto", choices=["auto", "float32", "bfloat16", "float16"])
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = {"auto": "auto", "float32": torch.float32,
             "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]

    print(f"Loading {args.model} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device)
    model.train()

    print(f"Loading transcripts from {args.transcripts}")
    ds = TranscriptDataset(
        args.transcripts, tokenizer, args.learner_role,
        args.learner_prompt, max_length=args.max_length,
    )
    print(f"  {len(ds)} (context, action) examples")
    if len(ds) == 0:
        raise RuntimeError(
            f"No training examples found for learner_role={args.learner_role}. "
            f"Check that transcripts attribute action turns to that agent_id."
        )

    pad_id = tokenizer.pad_token_id
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=lambda b: collate(b, pad_id),
    )
    opt = AdamW(model.parameters(), lr=args.lr)

    metrics_path = out / "metrics.jsonl"
    mf = metrics_path.open("w")
    step = 0
    for epoch in range(args.epochs):
        for input_ids, attn, mask in loader:
            input_ids = input_ids.to(device)
            attn = attn.to(device)
            mask = mask.to(device)
            outputs = model(input_ids=input_ids, attention_mask=attn)
            logits = outputs.logits  # (B, T, V)
            shifted_logits = logits[:, :-1, :]
            shifted_targets = input_ids[:, 1:]
            shifted_mask = mask[:, 1:].float()
            loss_per_tok = F.cross_entropy(
                shifted_logits.reshape(-1, shifted_logits.size(-1)),
                shifted_targets.reshape(-1),
                reduction="none",
            ).reshape(shifted_targets.shape)
            denom = shifted_mask.sum().clamp(min=1.0)
            loss = (loss_per_tok * shifted_mask).sum() / denom
            (loss / args.grad_accum).backward()
            if (step + 1) % args.grad_accum == 0:
                opt.step()
                opt.zero_grad()
            mf.write(json.dumps({"epoch": epoch, "step": step, "loss": loss.item()}) + "\n")
            mf.flush()
            if step % 25 == 0:
                print(f"epoch {epoch} step {step:5d} loss {loss.item():.4f}")
            step += 1

    mf.close()
    ckpt_dir = out / "checkpoint"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)
    print(f"Saved SFT checkpoint to {ckpt_dir}")


if __name__ == "__main__":
    main()
