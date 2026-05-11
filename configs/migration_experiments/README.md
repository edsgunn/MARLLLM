# Migration experiment configs

| Run | Migrant | Direction | Replaces | Tier |
|---|---|---|---|---|
| `migration_priya_ashbourne_gc_to_strathearn_server` | Priya Shah | ashbourne_gc → strathearn_server | Mhairi Buchanan | 1 |
| `migration_marcus_ashbourne_gc_to_strathearn_server` | Marcus Webb | ashbourne_gc → strathearn_server | Finn Nakamura | 1 |
| `migration_tom_ashbourne_gc_to_strathearn_server` | Tom Whitaker | ashbourne_gc → strathearn_server | Ada Whitfield | 2 |
| `migration_hana_ashbourne_gc_to_strathearn_server` | Hana Yilmaz | ashbourne_gc → strathearn_server | Callum Reid | 2 |
| `migration_mhairi_strathearn_server_to_ashbourne_gc` | Mhairi Buchanan | strathearn_server → ashbourne_gc | Priya Shah | 1 |
| `migration_finn_strathearn_server_to_ashbourne_gc` | Finn Nakamura | strathearn_server → ashbourne_gc | Marcus Webb | 1 |
| `migration_ada_strathearn_server_to_ashbourne_gc` | Ada Whitfield | strathearn_server → ashbourne_gc | Tom Whitaker | 2 |
| `migration_callum_strathearn_server_to_ashbourne_gc` | Callum Reid | strathearn_server → ashbourne_gc | Hana Yilmaz | 2 |

## Step 1 — build resume checkpoints (run after both iter_100 ckpts exist)

```bash
uv run python scripts/build_migration_checkpoint.py \
    --source-ckpt runs/cultural_emergence/run7_8agent_7B_study_group_ashbourne_gc/checkpoints/iter_000100 \
    --source-agent 'Priya Shah' \
    --target-ckpt runs/cultural_emergence/run7_8agent_7B_study_group_strathearn_server/checkpoints/iter_000100 \
    --target-agent 'Mhairi Buchanan' \
    --output runs/migration_experiments/migration_priya_ashbourne_gc_to_strathearn_server/checkpoints/iter_000100

uv run python scripts/build_migration_checkpoint.py \
    --source-ckpt runs/cultural_emergence/run7_8agent_7B_study_group_ashbourne_gc/checkpoints/iter_000100 \
    --source-agent 'Marcus Webb' \
    --target-ckpt runs/cultural_emergence/run7_8agent_7B_study_group_strathearn_server/checkpoints/iter_000100 \
    --target-agent 'Finn Nakamura' \
    --output runs/migration_experiments/migration_marcus_ashbourne_gc_to_strathearn_server/checkpoints/iter_000100

uv run python scripts/build_migration_checkpoint.py \
    --source-ckpt runs/cultural_emergence/run7_8agent_7B_study_group_ashbourne_gc/checkpoints/iter_000100 \
    --source-agent 'Tom Whitaker' \
    --target-ckpt runs/cultural_emergence/run7_8agent_7B_study_group_strathearn_server/checkpoints/iter_000100 \
    --target-agent 'Ada Whitfield' \
    --output runs/migration_experiments/migration_tom_ashbourne_gc_to_strathearn_server/checkpoints/iter_000100

uv run python scripts/build_migration_checkpoint.py \
    --source-ckpt runs/cultural_emergence/run7_8agent_7B_study_group_ashbourne_gc/checkpoints/iter_000100 \
    --source-agent 'Hana Yilmaz' \
    --target-ckpt runs/cultural_emergence/run7_8agent_7B_study_group_strathearn_server/checkpoints/iter_000100 \
    --target-agent 'Callum Reid' \
    --output runs/migration_experiments/migration_hana_ashbourne_gc_to_strathearn_server/checkpoints/iter_000100

uv run python scripts/build_migration_checkpoint.py \
    --source-ckpt runs/cultural_emergence/run7_8agent_7B_study_group_strathearn_server/checkpoints/iter_000100 \
    --source-agent 'Mhairi Buchanan' \
    --target-ckpt runs/cultural_emergence/run7_8agent_7B_study_group_ashbourne_gc/checkpoints/iter_000100 \
    --target-agent 'Priya Shah' \
    --output runs/migration_experiments/migration_mhairi_strathearn_server_to_ashbourne_gc/checkpoints/iter_000100

uv run python scripts/build_migration_checkpoint.py \
    --source-ckpt runs/cultural_emergence/run7_8agent_7B_study_group_strathearn_server/checkpoints/iter_000100 \
    --source-agent 'Finn Nakamura' \
    --target-ckpt runs/cultural_emergence/run7_8agent_7B_study_group_ashbourne_gc/checkpoints/iter_000100 \
    --target-agent 'Marcus Webb' \
    --output runs/migration_experiments/migration_finn_strathearn_server_to_ashbourne_gc/checkpoints/iter_000100

uv run python scripts/build_migration_checkpoint.py \
    --source-ckpt runs/cultural_emergence/run7_8agent_7B_study_group_strathearn_server/checkpoints/iter_000100 \
    --source-agent 'Ada Whitfield' \
    --target-ckpt runs/cultural_emergence/run7_8agent_7B_study_group_ashbourne_gc/checkpoints/iter_000100 \
    --target-agent 'Tom Whitaker' \
    --output runs/migration_experiments/migration_ada_strathearn_server_to_ashbourne_gc/checkpoints/iter_000100

uv run python scripts/build_migration_checkpoint.py \
    --source-ckpt runs/cultural_emergence/run7_8agent_7B_study_group_strathearn_server/checkpoints/iter_000100 \
    --source-agent 'Callum Reid' \
    --target-ckpt runs/cultural_emergence/run7_8agent_7B_study_group_ashbourne_gc/checkpoints/iter_000100 \
    --target-agent 'Hana Yilmaz' \
    --output runs/migration_experiments/migration_callum_strathearn_server_to_ashbourne_gc/checkpoints/iter_000100
```

## Step 2 — submit slurm jobs (run tier 1 first, tier 2 after compute permits)

```bash
# tier 1: migration_priya_ashbourne_gc_to_strathearn_server
sbatch slurm_scripts/migration_priya_ashbourne_gc_to_strathearn_server.sh
# tier 1: migration_marcus_ashbourne_gc_to_strathearn_server
sbatch slurm_scripts/migration_marcus_ashbourne_gc_to_strathearn_server.sh
# tier 2: migration_tom_ashbourne_gc_to_strathearn_server
sbatch slurm_scripts/migration_tom_ashbourne_gc_to_strathearn_server.sh
# tier 2: migration_hana_ashbourne_gc_to_strathearn_server
sbatch slurm_scripts/migration_hana_ashbourne_gc_to_strathearn_server.sh
# tier 1: migration_mhairi_strathearn_server_to_ashbourne_gc
sbatch slurm_scripts/migration_mhairi_strathearn_server_to_ashbourne_gc.sh
# tier 1: migration_finn_strathearn_server_to_ashbourne_gc
sbatch slurm_scripts/migration_finn_strathearn_server_to_ashbourne_gc.sh
# tier 2: migration_ada_strathearn_server_to_ashbourne_gc
sbatch slurm_scripts/migration_ada_strathearn_server_to_ashbourne_gc.sh
# tier 2: migration_callum_strathearn_server_to_ashbourne_gc
sbatch slurm_scripts/migration_callum_strathearn_server_to_ashbourne_gc.sh
```
