#!/usr/bin/env python3
"""
Interactive forum chat: take part as a human alongside checkpoint-loaded agents.

Loads the same population/env that ``train_population.py`` would build from a
YAML config (LoRA-shared base, forum env), optionally restores agent weights
from a checkpoint directory, then runs a single episode where one (or more)
slots are played by you through the terminal. All model weights stay fixed —
this is generation only, no training, no logging.

Usage
-----
    # On a compute node with a GPU:
    uv run python scripts/forum_chat.py \\
        --config configs/cultural_emergence/run6_2agent_7B_study_group.yaml \\
        --checkpoint runs/cultural_emergence/run6_2agent_7B_study_group/checkpoints/iter_000100 \\
        --human "Priya Shah"

Pass ``--human NAME`` once per slot you want to take over. Slots not marked
human are driven by the loaded checkpoint. Without ``--checkpoint`` the agents
run on the base model with freshly-initialised (untrained) LoRA adapters.

Multi-line input: type your post, then a single ``.`` on its own line to send.
A single blank line sends an empty post. Ctrl-C / Ctrl-D aborts.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

# Match train_population.py's preflight.
os.environ.setdefault("CC", "gcc")
os.environ.setdefault("CXX", "g++")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402
import yaml  # noqa: E402


# ── ANSI helpers ──────────────────────────────────────────────────────────────

_PALETTE = ["\033[36m", "\033[33m", "\033[35m", "\033[32m",
            "\033[34m", "\033[31m", "\033[96m", "\033[93m"]
_RESET = "\033[0m"
_DIM = "\033[2m"
_BOLD = "\033[1m"


def _color_for(name: str, names: list[str]) -> str:
    return _PALETTE[names.index(name) % len(_PALETTE)] if name in names else ""


# ── Config ────────────────────────────────────────────────────────────────────

def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _resolve_characters(cfg: dict) -> dict[str, str]:
    """Extract {name: character_prompt} from a YAML config.

    Mirrors train_population._build_character_prompts but works directly
    off the raw YAML dict (we don't go through argparse here).
    """
    prompts: dict[str, str | list[str]] = {}
    for name in (cfg.get("characters") or []):
        prompts[name] = f"You are {name}."
    yaml_prompts = cfg.get("character_prompts") or {}
    if isinstance(yaml_prompts, dict):
        prompts.update(yaml_prompts)
    # Flatten list-of-variants to first variant for an interactive run.
    return {n: (p[0] if isinstance(p, list) else p) for n, p in prompts.items()}


# ── Build population (LoRA-shared base) ───────────────────────────────────────

def _build_population(cfg: dict, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    from marlllm import LoRASharedBaseAgent
    from marlllm.agent import _find_lora_target_modules

    if not cfg.get("lora_shared_base", False):
        raise SystemExit(
            "This script only supports lora_shared_base configs (the standard "
            "cultural-emergence setup). Pass a config with lora_shared_base: true."
        )

    model_id = cfg["model"]
    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    torch_dtype = dtype_map.get(cfg.get("dtype", "auto"), "auto")
    attn_impl = cfg.get("attn_impl")

    print(f"Loading base model: {model_id}  →  {device}")
    load_kwargs: dict[str, Any] = {}
    if torch_dtype != "auto":
        load_kwargs["torch_dtype"] = torch_dtype
    if attn_impl:
        load_kwargs["attn_implementation"] = attn_impl

    base_model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs).to(device)
    base_model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    lora_modules_str = cfg.get("lora_modules")
    if lora_modules_str:
        target_modules = [m.strip() for m in lora_modules_str.split(",")]
    else:
        target_modules = _find_lora_target_modules(base_model)

    lora_config = LoraConfig(
        r=int(cfg["lora_r"]),
        lora_alpha=int(cfg.get("lora_alpha", 16)),
        target_modules=target_modules,
        lora_dropout=0.0,
        bias="none",
    )

    char_prompts = _resolve_characters(cfg)
    if not char_prompts:
        raise SystemExit("No characters found in config (need 'characters' or 'character_prompts').")
    names = list(char_prompts.keys())

    print(f"Attaching LoRA adapter '{names[0]}' (r={cfg['lora_r']})")
    peft_model = get_peft_model(base_model, lora_config, adapter_name=names[0])
    for name in names[1:]:
        print(f"Attaching LoRA adapter '{name}' (r={cfg['lora_r']})")
        peft_model.add_adapter(name, lora_config)

    population: dict[str, Any] = {}
    for name, prompt in char_prompts.items():
        population[name] = LoRASharedBaseAgent(
            agent_id=name,
            character_prompt=prompt,
            shared_backbone=peft_model,
            adapter_name=name,
            tokenizer=tokenizer,
            device=device,
            keep_ref_model=False,
            compile_rollout=False,
        )
    for agent in population.values():
        agent.eval_mode()
    return population, tokenizer


# ── Build forum env ───────────────────────────────────────────────────────────

def _build_forum_env(cfg: dict, tokenizer: Any) -> Any:
    envs = cfg.get("environments") or []
    forum_specs = [e for e in envs if (e.get("type") == "forum")]
    if not forum_specs:
        raise SystemExit("Config has no forum environment in 'environments'.")
    spec = forum_specs[0]

    from envs.forum import ForumEnv, load_characters, load_environment
    if "environment" not in spec:
        raise SystemExit(
            "Forum env spec must set 'environment: <name>' to select a prompt "
            "from envs/forum/environments/."
        )
    environment = load_environment(spec["environment"])

    if "personas" in spec:
        personas = dict(spec["personas"])
        agent_names = list(personas.keys())
    else:
        if "characters" not in spec:
            raise SystemExit(
                "Forum env spec must set 'characters: <pack>' (with "
                "'character_set:') or provide an explicit 'personas:' dict."
            )
        characters = load_characters(spec["characters"])
        cset = spec.get("character_set", "canonical_4")
        agent_names = characters.get_character_set(cset)
        personas = {
            name: environment.render_persona(name, characters.get_memories(name))
            for name in agent_names
        }

    return ForumEnv(
        agent_names=agent_names,
        agent_personas=personas,
        tokenizer=tokenizer,
        forum_description=spec.get("forum_description") or environment.forum_description,
        initial_invitation=spec.get("initial_invitation") or environment.default_invitation,
        action_token_budget=spec.get("token_budget"),
        max_posts=int(spec.get("max_posts", 12)),
        post_order=spec.get("post_order", "round_robin"),
        seed=int(spec.get("seed", 0)),
        post_length_note=spec.get("post_length_note"),
        post_open_tag=spec.get("post_open_tag", "<post>"),
        post_close_tag=spec.get("post_close_tag", "</post>"),
        post_token_budget=spec.get("post_token_budget"),
        total_token_budget=spec.get("total_token_budget"),
    ), spec, personas


# ── Human input ───────────────────────────────────────────────────────────────

def _read_human_post(prompt_label: str) -> str:
    print(f"\n{_BOLD}>>> Your turn as {prompt_label}{_RESET}  "
          f"{_DIM}(end with a single '.' on its own line; blank → empty post){_RESET}")
    lines: list[str] = []
    while True:
        try:
            line = input()
        except EOFError:
            print()
            raise SystemExit(0)
        if line.strip() == ".":
            break
        if line == "" and not lines:
            break
        lines.append(line)
    return "\n".join(lines).rstrip()


# ── Main loop ─────────────────────────────────────────────────────────────────

def run_episode(
    *,
    population: dict[str, Any],
    env: Any,
    tokenizer: Any,
    persona_for_role: dict[str, str],
    human_roles: set[str],
    temperature: float,
    seed: int,
) -> None:
    from marlllm.dialogue import chat_eos_token_ids
    eos_token_ids = chat_eos_token_ids(tokenizer)

    env.reset(seed=seed)
    role_names = list(env.possible_agents)
    name_color = {n: _color_for(n, role_names) for n in role_names}

    # Per-agent context windows (token IDs) for the model-driven slots.
    contexts: dict[str, list[int]] = {}
    for role in role_names:
        if role in human_roles:
            continue
        agent = population.get(role)
        if agent is None:
            raise SystemExit(
                f"Slot '{role}' has no agent in the population — config lists "
                f"{list(population.keys())}, env expects {role_names}. The "
                f"YAML's 'characters' must match the forum env's persona set."
            )
        formatter = agent.context_formatter
        # System prompt = the character's persona for this scenario.
        prompt_text = persona_for_role.get(role) or agent.character_prompt
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        contexts[role] = formatter.wrap_prompt(prompt_ids)

    # Print framing for the human(s).
    print()
    print(f"{_BOLD}── Forum episode opening ──{_RESET}")
    for role in role_names:
        tag = "(YOU)" if role in human_roles else "(model)"
        print(f"  {name_color[role]}{role}{_RESET} {_DIM}{tag}{_RESET}")
    print()

    while env.agents:
        role = env.agent_selection
        # Pull fresh observation tokens for whoever is acting.
        obs_ids = env.observe(role)
        obs_text = tokenizer.decode(obs_ids, skip_special_tokens=True) if obs_ids else ""

        if role in human_roles:
            if obs_text.strip():
                print(f"{_DIM}── New on the forum ──{_RESET}")
                print(obs_text)
            text = _read_human_post(role)
            # Encode the human's text as plain tokens (no chat template) — the
            # env decodes with skip_special_tokens, so this is just bytes-in.
            act_ids = tokenizer.encode(text, add_special_tokens=False)
        else:
            agent = population[role]
            formatter = agent.context_formatter
            if obs_ids:
                contexts[role].extend(formatter.wrap_observation(obs_ids))
            n_tokens = getattr(env, "action_token_budget", 256)
            with torch.no_grad():
                batch_ids, _ = agent.act_batch(
                    contexts=[contexts[role]],
                    n_tokens=int(n_tokens),
                    temperature=temperature,
                    eos_token_ids=eos_token_ids,
                )
            act_ids = batch_ids[0]
            contexts[role].extend(formatter.wrap_action(act_ids))

        env.step(act_ids)

        # Find the post we just placed on the canonical thread — it's the most
        # recent one — and render it (post-thinking-strip, as everyone else
        # will see it).
        if env._thread:
            last = env._thread[-1]
            speaker = last["speaker"]
            text = last["text"] or "(empty post)"
            color = name_color.get(speaker, "")
            print(f"\n{color}{_BOLD}{speaker}{_RESET} "
                  f"{_DIM}#{last['post_index']}{_RESET}")
            print(text)
            print(_DIM + "─" * 60 + _RESET)

    print(f"\n{_BOLD}── Episode complete ({len(env._thread)} posts) ──{_RESET}")


# ── Entry ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", required=True, type=Path,
                   help="YAML config (e.g. configs/cultural_emergence/run6_*.yaml).")
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="Checkpoint directory to load agent weights from "
                        "(e.g. .../checkpoints/iter_000100). Optional.")
    p.add_argument("--human", action="append", default=[], metavar="NAME",
                   help="Slot name to play as a human. Repeat to play multiple. "
                        "If omitted, you are prompted to pick interactively.")
    p.add_argument("--temperature", type=float, default=1.0,
                   help="Sampling temperature for model-driven agents.")
    p.add_argument("--max-posts", type=int, default=None,
                   help="Override max_posts from the env spec.")
    p.add_argument("--seed", type=int, default=0,
                   help="Env reset seed (controls round-robin opener).")
    p.add_argument("--device", default=None,
                   help="Override torch device (default: cuda:0 if available).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = _load_yaml(args.config)

    device = args.device or cfg.get("device") or (
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )

    population, tokenizer = _build_population(cfg, device=device)

    if args.checkpoint is not None:
        if not args.checkpoint.is_dir():
            raise SystemExit(f"--checkpoint must be a directory: {args.checkpoint}")
        from marlllm.checkpoint_utils import load_population_checkpoint
        iteration = load_population_checkpoint(
            population=population,
            optimizer=None,
            ckpt_dir=args.checkpoint,
            device=device,
            load_optimizer=False,
        )
        print(f"Loaded checkpoint {args.checkpoint} (iteration {iteration})")
    else:
        print("No --checkpoint passed: using freshly-initialised LoRA adapters.")

    env, env_spec, personas = _build_forum_env(cfg, tokenizer)
    if args.max_posts is not None:
        env._max_posts = args.max_posts

    role_names = list(env.possible_agents)
    human_roles: set[str] = set(args.human or [])
    for h in human_roles:
        if h not in role_names:
            raise SystemExit(
                f"--human {h!r} not in env roles {role_names}."
            )
    if not human_roles:
        # Interactive picker.
        print("\nAvailable slots:")
        for i, n in enumerate(role_names):
            print(f"  [{i}] {n}")
        sel = input("Which slot(s) do you want to play? "
                    "(comma-separated indices or names, blank = none): ").strip()
        if sel:
            for tok in [s.strip() for s in sel.split(",") if s.strip()]:
                if tok.isdigit() and 0 <= int(tok) < len(role_names):
                    human_roles.add(role_names[int(tok)])
                elif tok in role_names:
                    human_roles.add(tok)
                else:
                    print(f"  ignoring unknown selector: {tok!r}")

    run_episode(
        population=population,
        env=env,
        tokenizer=tokenizer,
        persona_for_role=personas,
        human_roles=human_roles,
        temperature=args.temperature,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
