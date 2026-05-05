"""
Regenerate episode traces from a saved negotiation experiment.

Reconstructs the environment and agents from a run directory's saved config,
optionally loads a checkpoint, then runs episodes and writes traces in the
multi-agent format (environment overview + per-agent context sections).

Usage
-----
# Use latest checkpoint, write 10 traces to <run-dir>/traces_regen/
python generate_traces.py --run-dir runs/negotiation_prompt_experiments/08_...

# Specific checkpoint, custom output dir, 20 traces
python generate_traces.py \\
    --run-dir runs/negotiation_prompt_experiments/08_... \\
    --checkpoint checkpoints/iter_000200.pt \\
    --n-traces 20 \\
    --output-dir runs/negotiation_prompt_experiments/08_.../my_traces

# No checkpoint (random/untrained weights), useful for sanity-checking format
python generate_traces.py --run-dir runs/... --no-checkpoint
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path
import torch
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

def _auto_device(idx: int, fallback: str) -> str:
    if torch.cuda.is_available() and torch.cuda.device_count() > idx:
        return f'cuda:{idx}'
    return fallback

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Regenerate multi-agent episode traces from a saved checkpoint.', formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument('--run-dir', required=True, help='Path to an experiment output directory that contains config.json and (optionally) experiment_config.yaml and checkpoints/.')
    p.add_argument('--checkpoint', default=None, help='Path to a .pt checkpoint file.  May be absolute or relative to --run-dir.  Defaults to checkpoints/latest.pt inside --run-dir.')
    p.add_argument('--no-checkpoint', action='store_true', help='Skip loading any checkpoint (run with randomly initialised or pre-trained base weights).')
    p.add_argument('--n-traces', type=int, default=10, help='Number of episodes to run and save as traces (default 10).')
    p.add_argument('--output-dir', default=None, help='Directory to write trace files into.  Defaults to <run-dir>/traces_regen/.')
    p.add_argument('--device', default=None, help="PyTorch device override (e.g. 'cpu', 'cuda:0').  If omitted the device from config.json is used.")
    p.add_argument('--seed', type=int, default=None, help='RNG seed for episode generation.  Defaults to the seed in config.json.')
    p.add_argument('--temperature', type=float, default=None, help='Sampling temperature.  Defaults to the value in config.json.')
    p.add_argument('--dialogue-turns', type=int, default=None)
    p.add_argument('--token-budget', type=int, default=None)
    p.add_argument('--max-episode-tokens', type=int, default=None)
    return p.parse_args()

def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)

def _load_yaml_if_present(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        import yaml
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        print('Warning: PyYAML not installed; cannot read experiment_config.yaml.')
        return {}

def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    config_json_path = run_dir / 'config.json'
    if not config_json_path.exists():
        sys.exit(f'ERROR: {config_json_path} not found.  Is --run-dir correct?')
    cfg = _load_json(config_json_path)
    yaml_cfg = _load_yaml_if_present(run_dir / 'experiment_config.yaml')

    def _pick(cli_val, yaml_key, json_key=None, default=None):
        if cli_val is not None:
            return cli_val
        if yaml_key and yaml_key in yaml_cfg:
            return yaml_cfg[yaml_key]
        if json_key and json_key in cfg:
            return cfg[json_key]
        return default
    model_name = cfg['model_name_or_path']
    prompts = cfg.get('character_prompts', {})
    prompt_0 = prompts.get('agent_0', 'You are Agent A, negotiating to maximise your score.')
    prompt_1 = prompts.get('agent_1', 'You are Agent B, negotiating to maximise your score.')
    device_str = args.device or cfg.get('device', 'cpu')
    seed = args.seed if args.seed is not None else cfg.get('seed', 42)
    temperature = args.temperature if args.temperature is not None else cfg.get('temperature', 1.0)
    max_ep_tok = _pick(args.max_episode_tokens, 'max_episode_tokens', 'max_episode_tokens', 1024)
    dialogue_turns = _pick(args.dialogue_turns, 'dialogue_turns', None, 10)
    token_budget = _pick(args.token_budget, 'token_budget', None, 64)
    role_shuffle = yaml_cfg.get('role_shuffle', False)
    shared_weights = yaml_cfg.get('shared_weights', False)
    lora_r = _pick(None, 'lora_r', 'lora_r', 0)
    lora_alpha = _pick(None, 'lora_alpha', 'lora_alpha', 16)
    lora_shared = yaml_cfg.get('lora_shared_base', False)
    dtype_str = yaml_cfg.get('dtype', 'auto')
    attn_impl = yaml_cfg.get('attn_impl', None)
    kl_coef = cfg.get('kl_coef', 0.0)
    torch.manual_seed(seed)
    dtype_map = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}
    torch_dtype = dtype_map.get(dtype_str, 'auto')
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / 'traces_regen'
    output_dir.mkdir(parents=True, exist_ok=True)
    from marlllm import IndependentAgent, LoRASharedBaseAgent, OnPolicyStore, TextTokeniser, Trainer, TrainingConfig, CCSMLoss
    from envs.deal_or_no_deal_env import DealOrNoDealEnv
    device_0 = device_str
    device_1 = _auto_device(1, device_0)
    load_kwargs: dict = {}
    if torch_dtype != 'auto':
        load_kwargs['torch_dtype'] = torch_dtype
    if attn_impl is not None:
        load_kwargs['attn_implementation'] = attn_impl
    extra_agent_kwargs: dict = {}
    if attn_impl is not None:
        extra_agent_kwargs['attn_implementation'] = attn_impl
    if lora_shared:
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError as e:
            sys.exit('ERROR: --lora-shared-base requires `peft`. Run: uv add peft')
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from marlllm.agent import _find_lora_target_modules
        print(f'Loading shared base model: {model_name}  →  {device_0}')
        base_model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs).to(device_0)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        target_modules = _find_lora_target_modules(base_model)
        lora_cfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha, target_modules=target_modules, lora_dropout=0.0, bias='none')
        peft_model = get_peft_model(base_model, lora_cfg, adapter_name='agent_0')
        peft_model.add_adapter('agent_1', lora_cfg)
        agent_0 = LoRASharedBaseAgent('agent_0', prompt_0, peft_model, 'agent_0', tokenizer, device_0, keep_ref_model=False)
        agent_1 = LoRASharedBaseAgent('agent_1', prompt_1, peft_model, 'agent_1', tokenizer, device_0, keep_ref_model=False)
    elif shared_weights:
        print(f'Loading shared agent backbone: {model_name}  →  {device_0}')
        shared = IndependentAgent(agent_id='agent_0', character_prompt=prompt_0, model_name_or_path=model_name, device=device_0, torch_dtype=torch_dtype, keep_ref_model=False, lora_r=lora_r, lora_alpha=lora_alpha, **extra_agent_kwargs)
        agent_0 = shared
        agent_1 = shared
    else:
        print(f'Loading agent_0: {model_name}  →  {device_0}')
        agent_0 = IndependentAgent(agent_id='agent_0', character_prompt=prompt_0, model_name_or_path=model_name, device=device_0, torch_dtype=torch_dtype, keep_ref_model=False, lora_r=lora_r, lora_alpha=lora_alpha, **extra_agent_kwargs)
        print(f'Loading agent_1: {model_name}  →  {device_1}')
        agent_1 = IndependentAgent(agent_id='agent_1', character_prompt=prompt_1, model_name_or_path=model_name, device=device_1, torch_dtype=torch_dtype, keep_ref_model=False, lora_r=lora_r, lora_alpha=lora_alpha, **extra_agent_kwargs)
    checkpoint_iter = 'pretrained'
    if not args.no_checkpoint:
        if args.checkpoint:
            ckpt_path = Path(args.checkpoint)
            if not ckpt_path.is_absolute():
                ckpt_path = run_dir / ckpt_path
        else:
            ckpt_path = run_dir / 'checkpoints' / 'latest.pt'
        if ckpt_path.exists():
            print(f'Loading checkpoint: {ckpt_path}')
            payload = torch.load(ckpt_path, map_location=device_0)
            agent_states = payload.get('agent_states', {})
            loaded_ids: set[int] = set()
            for aid, agent in [('agent_0', agent_0), ('agent_1', agent_1)]:
                if id(agent) in loaded_ids:
                    continue
                if aid in agent_states:
                    states = agent_states[aid]
                    agent._backbone.load_state_dict(states['backbone'])
                    agent._value_head.load_state_dict(states['value_head'])
                    loaded_ids.add(id(agent))
            checkpoint_iter = payload.get('iteration', 'unknown')
            print(f'  → loaded iteration {checkpoint_iter}')
        else:
            print(f'Warning: checkpoint not found at {ckpt_path}, using base weights.')
    print(f'Building DealOrNoDealEnv (dialogue_turns={dialogue_turns}, token_budget={token_budget}, role_shuffle={role_shuffle})')
    env = DealOrNoDealEnv(tokenizer=agent_0.tokenizer, max_dialogue_turns=dialogue_turns, action_token_budget=token_budget, seed=seed, role_shuffle=role_shuffle)
    scratch_dir = output_dir / '_trainer_scratch'
    scratch_dir.mkdir(exist_ok=True)
    train_cfg = TrainingConfig(model_name_or_path=model_name, character_prompts={'agent_0': prompt_0, 'agent_1': prompt_1}, episodes_per_iter=args.n_traces, max_episode_tokens=max_ep_tok, num_iterations=1, output_dir=str(scratch_dir), device=device_0, seed=seed, temperature=temperature, kl_coef=kl_coef)
    tokeniser = TextTokeniser(agent_0.tokenizer)
    trainer = Trainer(agents={'agent_0': agent_0, 'agent_1': agent_1}, env=env, loss=CCSMLoss(), tokeniser=tokeniser, store=OnPolicyStore(), config=train_cfg)
    from marlllm.trace_utils import make_episode_record, write_records_json, write_records_txt
    print(f'Collecting {args.n_traces} episodes …')
    episodes = trainer._collect_episodes_batched(args.n_traces)
    success = sum((1 for _traj, info, _ctx, _env in episodes if info.get('result') == 'success'))
    print(f'  {success}/{args.n_traces} successful deals')
    records = [make_episode_record(episode_idx=idx, agent_context_tokens=ctx_snapshot, tokenizer=agent_0.tokenizer, env_trace=env_trace or ep_info or None) for idx, (_traj, ep_info, ctx_snapshot, env_trace) in enumerate(episodes)]
    label = f'ckpt{checkpoint_iter}'
    write_records_json(records, output_dir / f'{label}.json')
    write_records_txt(records, output_dir / f'{label}.txt')
    print(f'Wrote {args.n_traces} traces to {output_dir}/')
if __name__ == '__main__':
    main()