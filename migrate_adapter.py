"""
CLI: copy a single agent's adapter from one population checkpoint into another.

Usage
-----
Copy the trained Silas adapter from run2/checkpoints/iter_000400 into
run3a/checkpoints/iter_000200 (overwriting run3a's Silas slot in-place):

    uv run python migrate_adapter.py \\
        --source-checkpoint runs/cultural_emergence/run2/checkpoints/iter_000400 \\
        --source-agent "Silas Varnham" \\
        --target-checkpoint runs/cultural_emergence/run3a/checkpoints/iter_000200 \\
        --target-agent "Silas Varnham" \\
        --output runs/cultural_emergence/run3b/checkpoints/iter_000200_migrated

The script copies every per-agent file from the target checkpoint to the
output directory, then overwrites the target-agent file with a payload
constructed from the source agent's adapter (with adapter-name remapping
if the source and target adapter_name differ).

The output directory can then be loaded by the trainer as a normal
per-agent checkpoint via ``--resume-from <output>``.
"""
from __future__ import annotations
import argparse
import shutil
import sys
from pathlib import Path
import torch

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Migrate one agent's adapter between population checkpoints.")
    p.add_argument('--source-checkpoint', required=True, help='Path to source population checkpoint dir (contains per-agent .pt files).')
    p.add_argument('--source-agent', required=True, help="Agent slot name in source checkpoint (e.g. 'Silas Varnham').")
    p.add_argument('--target-checkpoint', required=True, help='Path to target population checkpoint dir.')
    p.add_argument('--target-agent', default=None, help='Agent slot in target to overwrite. Defaults to --source-agent.')
    p.add_argument('--output', required=True, help='Output directory for the migrated checkpoint.')
    p.add_argument('--reset-optimizer', action='store_true', help="Drop the target's optimizer state in the migrated copy (recommended: optimizer momentum is incoherent post-migration).")
    return p.parse_args()

def _resolve_agent_file(ckpt_dir: Path, agent_slot: str) -> Path:
    """Find <agent_slot>.pt inside ckpt_dir, considering slot_to_object."""
    direct = ckpt_dir / f'{agent_slot}.pt'
    if direct.exists():
        return direct
    meta_path = ckpt_dir / 'meta.pt'
    if not meta_path.exists():
        raise FileNotFoundError(f'Neither {direct} nor {meta_path} found. Is this a per-agent checkpoint dir?')
    meta = torch.load(meta_path, map_location='cpu')
    slot_to_object = meta.get('slot_to_object', {})
    canonical = slot_to_object.get(agent_slot, agent_slot)
    cand = ckpt_dir / f'{canonical}.pt'
    if cand.exists():
        return cand
    raise FileNotFoundError(f"Could not find a .pt file for agent slot '{agent_slot}' in {ckpt_dir}. Available files: {sorted((p.name for p in ckpt_dir.glob('*.pt')))}")

def main() -> None:
    args = parse_args()
    src_dir = Path(args.source_checkpoint)
    tgt_dir = Path(args.target_checkpoint)
    out_dir = Path(args.output)
    target_agent = args.target_agent or args.source_agent
    if not src_dir.is_dir():
        sys.exit(f'Source checkpoint must be a directory: {src_dir}')
    if not tgt_dir.is_dir():
        sys.exit(f'Target checkpoint must be a directory: {tgt_dir}')
    src_file = _resolve_agent_file(src_dir, args.source_agent)
    tgt_file = _resolve_agent_file(tgt_dir, target_agent)
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in tgt_dir.iterdir():
        if f.is_file():
            shutil.copy2(f, out_dir / f.name)
    src_payload = torch.load(src_file, map_location='cpu')
    tgt_payload = torch.load(tgt_file, map_location='cpu')
    src_type = src_payload.get('agent_type')
    tgt_type = tgt_payload.get('agent_type')
    if src_type != tgt_type:
        sys.exit(f'Source agent_type ({src_type}) != target agent_type ({tgt_type}). Cross-type migration is not supported.')
    if src_type == 'lora_shared':
        src_adapter = src_payload.get('adapter_name')
        tgt_adapter = tgt_payload.get('adapter_name')
        adapter = src_payload['adapter']
        if src_adapter != tgt_adapter:
            adapter = {k.replace(f'.{src_adapter}.', f'.{tgt_adapter}.'): v for k, v in adapter.items()}
            print(f'Adapter-name remap: {src_adapter} -> {tgt_adapter} ({len(adapter)} tensors)')
        new_payload = {'agent_id': target_agent, 'agent_type': 'lora_shared', 'adapter_name': tgt_adapter, 'adapter': adapter, 'adapter_param_count': len(adapter), 'value_head': src_payload['value_head']}
    else:
        new_payload = dict(src_payload)
        new_payload['agent_id'] = target_agent
    out_agent_file = out_dir / tgt_file.name
    torch.save(new_payload, out_agent_file)
    print(f'Wrote migrated agent payload: {out_agent_file}')
    meta_path = out_dir / 'meta.pt'
    if meta_path.exists():
        meta = torch.load(meta_path, map_location='cpu')
        meta['migrated_from'] = {'source_checkpoint': str(src_dir), 'source_agent': args.source_agent, 'target_agent': target_agent}
        if args.reset_optimizer:
            meta['optimizer_state'] = None
            print('Optimizer state cleared in migrated checkpoint.')
        torch.save(meta, meta_path)
    print(f'Migration complete. Output: {out_dir}')
    print('Resume training with --resume-from', out_dir)
if __name__ == '__main__':
    main()