"""
Construct a migration-experiment starting checkpoint by splicing one agent
from a *source* population's iter_100 checkpoint into a *target* population's
iter_100 checkpoint, preserving full per-agent identity (slot name, persona,
LoRA adapter, value head) AND splicing the optimiser state so all 8 agents
keep their Adam moments.

Output layout matches save_population_checkpoint(): meta.pt + one <slot>.pt
per agent. The trainer can then resume from --resume-from <output_dir>.

Why splice the optimiser state per-agent (rather than just resetting it):
the user wants every agent to keep momentum from its home population so the
migrant doesn't get an artificial cold-start advantage relative to the natives.
Adam state in the saved meta.pt is a single param-group with N*K param entries,
where N = #agents and K = #params per agent (LoRA + value-head). Both
populations share the same base model + LoRA config (q_proj,v_proj, r=64,
alpha=128), so K is identical. Per-agent ordering inside K is determined by
named_parameters() traversal of the shared backbone, which is identical
across populations because the layer / module structure is identical and
adapter-name only appears in the param key (not its iteration order).

Example
-------
    uv run python scripts/build_migration_checkpoint.py \\
        --source-ckpt   runs/cultural_emergence/run7_8agent_7B_study_group/checkpoints/iter_000100 \\
        --source-agent  "Priya Shah" \\
        --target-ckpt   runs/cultural_emergence/run7_8agent_7B_study_group_strathearn/checkpoints/iter_000100 \\
        --target-agent  "Mhairi Buchanan" \\
        --output        runs/migration_experiments/priya_to_strathearn/checkpoints/iter_000100
"""
from __future__ import annotations
import argparse
import shutil
import sys
from pathlib import Path
from typing import Any
import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--source-ckpt', required=True, help='Source population iter_100 checkpoint dir.')
    p.add_argument('--source-agent', required=True, help='Migrant slot name in the source population.')
    p.add_argument('--target-ckpt', required=True, help='Target (host) population iter_100 checkpoint dir.')
    p.add_argument('--target-agent', required=True, help='Role-matched slot in the target population to be replaced.')
    p.add_argument('--output', required=True, help='Output directory for the migrated checkpoint.')
    return p.parse_args()


def _load_meta(ckpt_dir: Path) -> dict[str, Any]:
    meta_path = ckpt_dir / 'meta.pt'
    if not meta_path.exists():
        sys.exit(f'Missing meta.pt in {ckpt_dir}')
    return torch.load(meta_path, map_location='cpu', weights_only=False)


def _splice_optimizer_state(
    *,
    target_state_dict: dict[str, Any],
    source_state_dict: dict[str, Any],
    target_slot_index: int,
    source_slot_index: int,
    n_target_slots: int,
    n_source_slots: int,
) -> dict[str, Any]:
    """Replace the migrant slot's K param entries in target_state_dict with the
    corresponding K entries from source_state_dict.

    Both optimisers are single-param-group AdamW with N*K params. K is inferred
    from len(state) / N. Splice per-agent, keeping all other slots untouched.
    """
    tgt_state = target_state_dict.get('state') or {}
    src_state = source_state_dict.get('state') or {}
    if not tgt_state:
        # Optimizer wasn't saved — nothing to splice; trainer will start fresh
        # for moments but adapter weights are still migrated correctly.
        print('[warn] target optimizer_state is empty; spliced state will be empty.')
        return target_state_dict
    if not src_state:
        sys.exit('Source optimizer_state is empty; cannot splice migrant moments.')

    n_tgt = len(tgt_state)
    n_src = len(src_state)
    if n_tgt % n_target_slots != 0 or n_src % n_source_slots != 0:
        sys.exit(
            f'Optimizer state size not divisible by slot count: '
            f'tgt={n_tgt}/{n_target_slots}, src={n_src}/{n_source_slots}'
        )
    k_tgt = n_tgt // n_target_slots
    k_src = n_src // n_source_slots
    if k_tgt != k_src:
        sys.exit(
            f'Per-agent param count mismatch: target K={k_tgt}, source K={k_src}. '
            'LoRA configs / value heads must match across populations for splicing.'
        )

    keys_sorted = sorted(tgt_state.keys())
    if keys_sorted != list(range(n_tgt)):
        sys.exit(f'Unexpected non-contiguous optimizer state keys: {keys_sorted[:5]}...')

    new_state = {k: v for k, v in tgt_state.items()}
    for j in range(k_tgt):
        tgt_idx = target_slot_index * k_tgt + j
        src_idx = source_slot_index * k_src + j
        if src_idx not in src_state:
            sys.exit(f'Missing source optimizer entry at index {src_idx}')
        new_state[tgt_idx] = src_state[src_idx]
    print(f'Spliced {k_tgt} optimizer-state entries: '
          f'src[{source_slot_index*k_src}:{(source_slot_index+1)*k_src}] '
          f'-> tgt[{target_slot_index*k_tgt}:{(target_slot_index+1)*k_tgt}]')
    out = dict(target_state_dict)
    out['state'] = new_state
    return out


def main() -> None:
    args = parse_args()
    src_dir = Path(args.source_ckpt)
    tgt_dir = Path(args.target_ckpt)
    out_dir = Path(args.output)

    if not src_dir.is_dir():
        sys.exit(f'Source checkpoint must be a directory: {src_dir}')
    if not tgt_dir.is_dir():
        sys.exit(f'Target checkpoint must be a directory: {tgt_dir}')

    src_meta = _load_meta(src_dir)
    tgt_meta = _load_meta(tgt_dir)

    src_slots: list[str] = list(src_meta.get('agent_slots') or [])
    tgt_slots: list[str] = list(tgt_meta.get('agent_slots') or [])
    if args.source_agent not in src_slots:
        sys.exit(f'{args.source_agent!r} not in source agent_slots: {src_slots}')
    if args.target_agent not in tgt_slots:
        sys.exit(f'{args.target_agent!r} not in target agent_slots: {tgt_slots}')
    s_idx = src_slots.index(args.source_agent)
    t_idx = tgt_slots.index(args.target_agent)
    print(f'Source slot index: {s_idx} ({args.source_agent})')
    print(f'Target slot index: {t_idx} ({args.target_agent}) -> will become {args.source_agent}')

    out_dir.mkdir(parents=True, exist_ok=True)

    # Copy all of target's per-agent .pt files except the target_agent's,
    # which is replaced by the migrant's source payload.
    new_slots = list(tgt_slots)
    new_slots[t_idx] = args.source_agent
    new_slot_to_object = {slot: slot for slot in new_slots}

    for slot in tgt_slots:
        if slot == args.target_agent:
            continue
        # Resolve via slot_to_object in case of weight-tied slots.
        canonical = (tgt_meta.get('slot_to_object') or {}).get(slot, slot)
        src_path = tgt_dir / f'{canonical}.pt'
        if not src_path.exists():
            sys.exit(f'Missing target per-agent file: {src_path}')
        shutil.copy2(src_path, out_dir / f'{canonical}.pt')

    # Migrant: copy the source's <source_agent>.pt verbatim. Agent IDs and
    # adapter names already match (full-identity migration: slot name = adapter
    # name = persona owner), so no remapping is needed.
    src_canonical = (src_meta.get('slot_to_object') or {}).get(args.source_agent, args.source_agent)
    src_payload_path = src_dir / f'{src_canonical}.pt'
    if not src_payload_path.exists():
        sys.exit(f'Missing source per-agent file: {src_payload_path}')
    payload = torch.load(src_payload_path, map_location='cpu', weights_only=False)
    payload_adapter_name = payload.get('adapter_name')
    if payload_adapter_name != args.source_agent:
        # Defensive — full-identity expects the adapter name = source slot name.
        sys.exit(
            f'Source payload adapter_name={payload_adapter_name!r} != source slot '
            f'{args.source_agent!r}. Full-identity migration requires they match.'
        )
    payload['agent_id'] = args.source_agent
    torch.save(payload, out_dir / f'{args.source_agent}.pt')
    print(f'Wrote migrant per-agent file: {out_dir / (args.source_agent + ".pt")}')

    # Build new meta.pt: target's meta with agent_slots / slot_to_object updated
    # and optimizer_state spliced.
    src_opt = src_meta.get('optimizer_state') or {}
    tgt_opt = tgt_meta.get('optimizer_state') or {}
    new_opt = _splice_optimizer_state(
        target_state_dict=tgt_opt,
        source_state_dict=src_opt,
        target_slot_index=t_idx,
        source_slot_index=s_idx,
        n_target_slots=len(tgt_slots),
        n_source_slots=len(src_slots),
    )

    new_meta = dict(tgt_meta)
    new_meta['agent_slots'] = new_slots
    new_meta['slot_to_object'] = new_slot_to_object
    new_meta['optimizer_state'] = new_opt
    new_meta['migrated_from'] = {
        'source_checkpoint': str(src_dir),
        'source_agent': args.source_agent,
        'replaced_target_agent': args.target_agent,
        'source_slot_index': s_idx,
        'target_slot_index': t_idx,
    }
    torch.save(new_meta, out_dir / 'meta.pt')
    print(f'Wrote meta.pt with spliced optimizer state and updated agent_slots:')
    print(f'  {new_slots}')
    print(f'Migration checkpoint ready at: {out_dir}')


if __name__ == '__main__':
    main()
