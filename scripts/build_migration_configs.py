"""
Generate the 8 migration-experiment YAML configs and matching slurm scripts.

For each (migrant_name, source_population, target_population) triple:

  - Clone the host's iter-100->150 continuation YAML structure.
  - Substitute the role-matched host slot with the migrant's slot name,
    keeping the slot's positional index (so optimiser-state splicing in
    build_migration_checkpoint.py is positional and clean).
  - Inject explicit `personas:` for all 8 agents: 7 from the host scenario
    JSON (so they read identically to the within-pop control), 1 (the
    migrant) from the source-population scenario JSON.
  - Set environment.scenario = host scenario, so the forum description /
    invitation are the host environment.
  - Set output_dir = runs/migration_experiments/<run_name>.
  - The slurm script invokes train_population.py with
        --resume-from <prebuilt migration ckpt dir>
    (the dir produced by build_migration_checkpoint.py, run separately
    once both populations have an iter_100 checkpoint on disk).

Usage:
    uv run python scripts/build_migration_configs.py
        # generates configs/migration_experiments/*.yaml
        # and slurm_scripts/migration_*.sh
"""
from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime
import sys

PROJECT_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_DIR / 'configs' / 'migration_experiments'
SLURM_DIR = PROJECT_DIR / 'slurm_scripts'
RUNS_DIR_REL = 'runs/migration_experiments'

# Reformulated migration experiments on the new <post>-tag env format.
# Source/target are the *_server (Strathearn) and *_gc (Ashbourne) cultural-
# emergence runs — same agent rosters as the original Strathearn / Ashbourne
# runs, but framed by the new "the server" / "the gc" environment prompts
# and trained without thinking_enabled (the env now always extracts <post>).
ASHBOURNE_GC = {
    'name': 'ashbourne_gc',
    'characters': [
        'Priya Shah', 'Tom Whitaker', 'Hana Yilmaz', 'Olu Adeyemi',
        'Beatrice Okafor', 'Sam Pritchard', 'Imogen Carter', 'Marcus Webb',
    ],
    'character_pack_json': 'envs/forum/characters/study_group_ashbourne.json',
    'environment_json': 'envs/forum/environments/study_group_ashbourne_gc.json',
    'character_pack_name': 'study_group_ashbourne',
    'environment_name': 'study_group_ashbourne_gc',
    'iter100_ckpt': 'runs/cultural_emergence/run7_8agent_7B_study_group_ashbourne_gc/checkpoints/iter_000100',
}
STRATHEARN_SERVER = {
    'name': 'strathearn_server',
    'characters': [
        'Mhairi Buchanan', 'Callum Reid', 'Niamh Donnelly', 'Daniyal Khan',
        'Eilidh MacGregor', 'Joseph Akingbola', 'Ada Whitfield', 'Finn Nakamura',
    ],
    'character_pack_json': 'envs/forum/characters/study_group_strathearn.json',
    'environment_json': 'envs/forum/environments/study_group_strathearn_server.json',
    'character_pack_name': 'study_group_strathearn',
    'environment_name': 'study_group_strathearn_server',
    'iter100_ckpt': 'runs/cultural_emergence/run7_8agent_7B_study_group_strathearn_server/checkpoints/iter_000100',
}

# (migrant_name, source_pop, target_pop, role_matched_target_slot, priority_tier)
# Priority tier 1 = convenor + newcomer (run first); tier 2 = checker + pattern-spotter.
MIGRATIONS = [
    # Direction A: Ashbourne (gc) -> Strathearn (server)
    ('Priya Shah',   ASHBOURNE_GC,      STRATHEARN_SERVER, 'Mhairi Buchanan', 1),
    ('Marcus Webb',  ASHBOURNE_GC,      STRATHEARN_SERVER, 'Finn Nakamura',   1),
    ('Tom Whitaker', ASHBOURNE_GC,      STRATHEARN_SERVER, 'Ada Whitfield',   2),
    ('Hana Yilmaz',  ASHBOURNE_GC,      STRATHEARN_SERVER, 'Callum Reid',     2),
    # Direction B: Strathearn (server) -> Ashbourne (gc)
    ('Mhairi Buchanan', STRATHEARN_SERVER, ASHBOURNE_GC, 'Priya Shah',     1),
    ('Finn Nakamura',   STRATHEARN_SERVER, ASHBOURNE_GC, 'Marcus Webb',    1),
    ('Ada Whitfield',   STRATHEARN_SERVER, ASHBOURNE_GC, 'Tom Whitaker',   2),
    ('Callum Reid',     STRATHEARN_SERVER, ASHBOURNE_GC, 'Hana Yilmaz',    2),
]


def _short(name: str) -> str:
    """First-name slug for run identifiers (e.g. 'Priya Shah' -> 'priya')."""
    return name.split()[0].lower()


def _load_template(environment_json_path: Path) -> str:
    return json.loads(environment_json_path.read_text())['system_prompt_template']


def _load_bullets(character_pack_json_path: Path, character_name: str) -> list[str]:
    data = json.loads(character_pack_json_path.read_text())
    chars = data.get('characters', {})
    if character_name not in chars:
        sys.exit(f'{character_name!r} not found in {character_pack_json_path}')
    return list(chars[character_name]['description'])


def _render_persona(template: str, character_name: str, bullets: list[str]) -> str:
    """Mirror envs/forum/personas.py: template.format(name, bulleted-description)."""
    body = '\n'.join(f'- {m}' for m in bullets)
    return template.format(character_name=character_name, character_description_bullets=body)


def _yaml_block_scalar(text: str, indent: int = 4) -> str:
    """Format a multi-line string as a YAML block scalar (|)."""
    pad = ' ' * indent
    lines = text.splitlines()
    return '|-\n' + '\n'.join(pad + ln for ln in lines)


def _yaml_quote(name: str) -> str:
    if any(c in name for c in (':', '#')) or "'" in name:
        return '"' + name.replace('"', '\\"') + '"'
    return repr(name)


def build_config(
    *,
    migrant: str,
    source_pop: dict,
    target_pop: dict,
    target_slot: str,
) -> tuple[str, dict]:
    """Return (yaml_text, metadata) for a migration run config."""
    # New population order: target's order with target_slot replaced by migrant
    # at the same positional index (preserves Adam-state splice positions).
    new_chars = list(target_pop['characters'])
    t_idx = new_chars.index(target_slot)
    new_chars[t_idx] = migrant

    # Personas:
    #   - All 8 use the *host* system_prompt_template, so every agent sees the
    #     host environment description as the framing of their system prompt
    #     ("the maths club at <host school>"). This matches the conversation
    #     they are actually in.
    #   - 7 host natives use their own bio bullets (host character pack).
    #   - The migrant uses their *source* bio bullets — i.e. their personal
    #     history follows them, but the description of the room they're in
    #     is the host room.
    host_env_json = PROJECT_DIR / target_pop['environment_json']
    host_chars_json = PROJECT_DIR / target_pop['character_pack_json']
    src_chars_json = PROJECT_DIR / source_pop['character_pack_json']
    host_template = _load_template(host_env_json)
    personas: dict[str, str] = {}
    for ch in new_chars:
        bullets_json = src_chars_json if ch == migrant else host_chars_json
        bullets = _load_bullets(bullets_json, ch)
        personas[ch] = _render_persona(host_template, ch, bullets)

    direction = f"{source_pop['name']}_to_{target_pop['name']}"
    run_name = f"migration_{_short(migrant)}_{direction}"

    output_dir = f"{RUNS_DIR_REL}/{run_name}"
    migration_ckpt_dir = f"{output_dir}/checkpoints/iter_000100"

    # personas is nested under environments[0], which itself sits at 2-space
    # indent under the top-level "environments:" sequence. The personas dict
    # keys must be indented 4 spaces (under "  personas:"), and block-scalar
    # bodies must be indented 6 spaces (one more level than the keys).
    persona_lines = []
    for name in new_chars:
        persona_lines.append(f"    {_yaml_quote(name)}: {_yaml_block_scalar(personas[name], indent=6)}")
    personas_block = '\n'.join(persona_lines)

    char_lines = '\n'.join(f"- {_yaml_quote(c)}" for c in new_chars)

    yaml = f"""# Migration experiment: {migrant} ({source_pop['name']}) ->
# host population {target_pop['name']}, replacing role-matched slot
# {target_slot} (positional index {t_idx}). Continues training the full
# 8-agent host population (7 natives + 1 migrant) for 50 iters under the
# host's scenario / forum / invitation, with the migrant's persona text
# travelling with them.
#
# Resume target: {migration_ckpt_dir}
# (built by scripts/build_migration_checkpoint.py from the source's and
# target's iter_000100 checkpoints; that builder splices Adam state so all
# 8 agents keep momentum from their home populations.)
name: {run_name}
script: train_population
slurm:
  gpus_per_node: 4
  time: "12:00:00"
model: Qwen/Qwen2.5-7B-Instruct
dtype: bfloat16
device: cuda:0
attn_impl: sdpa
use_vllm: true
vllm_gpu_mem_util: 0.30
vllm_max_num_seqs: 8
vllm_quantization: fp8
lora_modules: q_proj,v_proj
use_8bit_adam: true
characters:
{char_lines}
lora_shared_base: true
lora_r: 64
lora_alpha: 128
pairing_strategy: random_no_self
iters: 150
rollouts: 16
lr: 5e-6
kl_coef: 0.05
beta: 0.01
grad_accum: 16
seed: 1
log_every: 5
checkpoint_every: 25
num_checkpoint_traces: 8
max_episode_tokens: 24576
environments:
- name: {target_pop['environment_name']}
  type: forum
  environment: {target_pop['environment_name']}
  post_token_budget: 128
  total_token_budget: 384
  personas:
{personas_block}
  max_posts: 16
  post_order: round_robin
output_dir: {output_dir}
gradient_checkpointing: true

# Identifying metadata for the analysis pipeline. The trainer ignores
# unknown top-level keys, so this is purely informational.
migration:
  migrant: {_yaml_quote(migrant)}
  source_population: {target_pop['name'] if False else source_pop['name']}
  target_population: {target_pop['name']}
  replaced_target_slot: {_yaml_quote(target_slot)}
  source_iter100_checkpoint: {source_pop['iter100_ckpt']}
  target_iter100_checkpoint: {target_pop['iter100_ckpt']}
  resume_from: {migration_ckpt_dir}
"""
    meta = {
        'run_name': run_name,
        'migrant': migrant,
        'source_pop': source_pop['name'],
        'target_pop': target_pop['name'],
        'target_slot': target_slot,
        'positional_index': t_idx,
        'migration_ckpt_dir': migration_ckpt_dir,
        'source_iter100': source_pop['iter100_ckpt'],
        'target_iter100': target_pop['iter100_ckpt'],
        'output_dir': output_dir,
    }
    return yaml, meta


def main() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    SLURM_DIR.mkdir(parents=True, exist_ok=True)
    summary_lines = ['# Migration experiment configs', '', '| Run | Migrant | Direction | Replaces | Tier |', '|---|---|---|---|---|']
    cmds_to_build_ckpt = []
    cmds_to_submit = []
    for migrant, src, tgt, target_slot, tier in MIGRATIONS:
        yaml_text, meta = build_config(
            migrant=migrant, source_pop=src, target_pop=tgt, target_slot=target_slot,
        )
        cfg_path = CONFIG_DIR / f"{meta['run_name']}.yaml"
        cfg_path.write_text(yaml_text)
        print(f'wrote {cfg_path}')
        summary_lines.append(
            f"| `{meta['run_name']}` | {migrant} | {src['name']} → {tgt['name']} | {target_slot} | {tier} |"
        )
        cmds_to_build_ckpt.append(
            f"uv run python scripts/build_migration_checkpoint.py \\\n"
            f"    --source-ckpt {meta['source_iter100']} \\\n"
            f"    --source-agent '{migrant}' \\\n"
            f"    --target-ckpt {meta['target_iter100']} \\\n"
            f"    --target-agent '{target_slot}' \\\n"
            f"    --output {meta['migration_ckpt_dir']}"
        )
        cmds_to_submit.append(
            f"# tier {tier}: {meta['run_name']}\n"
            f"sbatch slurm_scripts/{meta['run_name']}.sh"
        )
    (CONFIG_DIR / 'README.md').write_text(
        '\n'.join(summary_lines)
        + '\n\n## Step 1 — build resume checkpoints (run after both iter_100 ckpts exist)\n\n```bash\n'
        + '\n\n'.join(cmds_to_build_ckpt)
        + '\n```\n\n## Step 2 — submit slurm jobs (run tier 1 first, tier 2 after compute permits)\n\n```bash\n'
        + '\n'.join(cmds_to_submit)
        + '\n```\n'
    )
    print(f'wrote {CONFIG_DIR / "README.md"}')


if __name__ == '__main__':
    main()
