# Migration experiment configs

| Run | Migrant | Direction | Replaces | Tier |
|---|---|---|---|---|
| `migration_priya_ashbourne_to_strathearn` | Priya Shah | ashbourne → strathearn | Mhairi Buchanan | 1 |
| `migration_marcus_ashbourne_to_strathearn` | Marcus Webb | ashbourne → strathearn | Finn Nakamura | 1 |
| `migration_tom_ashbourne_to_strathearn` | Tom Whitaker | ashbourne → strathearn | Ada Whitfield | 2 |
| `migration_hana_ashbourne_to_strathearn` | Hana Yilmaz | ashbourne → strathearn | Callum Reid | 2 |
| `migration_mhairi_strathearn_to_ashbourne` | Mhairi Buchanan | strathearn → ashbourne | Priya Shah | 1 |
| `migration_finn_strathearn_to_ashbourne` | Finn Nakamura | strathearn → ashbourne | Marcus Webb | 1 |
| `migration_ada_strathearn_to_ashbourne` | Ada Whitfield | strathearn → ashbourne | Tom Whitaker | 2 |
| `migration_callum_strathearn_to_ashbourne` | Callum Reid | strathearn → ashbourne | Hana Yilmaz | 2 |

## Step 1 — build resume checkpoints (run after both iter_100 ckpts exist)

```bash
uv run python scripts/build_migration_checkpoint.py \
    --source-ckpt runs/cultural_emergence/run7_8agent_7B_study_group/checkpoints/iter_000100 \
    --source-agent 'Priya Shah' \
    --target-ckpt runs/cultural_emergence/run7_8agent_7B_study_group_strathearn/checkpoints/iter_000100 \
    --target-agent 'Mhairi Buchanan' \
    --output runs/migration_experiments/migration_priya_ashbourne_to_strathearn/checkpoints/iter_000100

uv run python scripts/build_migration_checkpoint.py \
    --source-ckpt runs/cultural_emergence/run7_8agent_7B_study_group/checkpoints/iter_000100 \
    --source-agent 'Marcus Webb' \
    --target-ckpt runs/cultural_emergence/run7_8agent_7B_study_group_strathearn/checkpoints/iter_000100 \
    --target-agent 'Finn Nakamura' \
    --output runs/migration_experiments/migration_marcus_ashbourne_to_strathearn/checkpoints/iter_000100

uv run python scripts/build_migration_checkpoint.py \
    --source-ckpt runs/cultural_emergence/run7_8agent_7B_study_group/checkpoints/iter_000100 \
    --source-agent 'Tom Whitaker' \
    --target-ckpt runs/cultural_emergence/run7_8agent_7B_study_group_strathearn/checkpoints/iter_000100 \
    --target-agent 'Ada Whitfield' \
    --output runs/migration_experiments/migration_tom_ashbourne_to_strathearn/checkpoints/iter_000100

uv run python scripts/build_migration_checkpoint.py \
    --source-ckpt runs/cultural_emergence/run7_8agent_7B_study_group/checkpoints/iter_000100 \
    --source-agent 'Hana Yilmaz' \
    --target-ckpt runs/cultural_emergence/run7_8agent_7B_study_group_strathearn/checkpoints/iter_000100 \
    --target-agent 'Callum Reid' \
    --output runs/migration_experiments/migration_hana_ashbourne_to_strathearn/checkpoints/iter_000100

uv run python scripts/build_migration_checkpoint.py \
    --source-ckpt runs/cultural_emergence/run7_8agent_7B_study_group_strathearn/checkpoints/iter_000100 \
    --source-agent 'Mhairi Buchanan' \
    --target-ckpt runs/cultural_emergence/run7_8agent_7B_study_group/checkpoints/iter_000100 \
    --target-agent 'Priya Shah' \
    --output runs/migration_experiments/migration_mhairi_strathearn_to_ashbourne/checkpoints/iter_000100

uv run python scripts/build_migration_checkpoint.py \
    --source-ckpt runs/cultural_emergence/run7_8agent_7B_study_group_strathearn/checkpoints/iter_000100 \
    --source-agent 'Finn Nakamura' \
    --target-ckpt runs/cultural_emergence/run7_8agent_7B_study_group/checkpoints/iter_000100 \
    --target-agent 'Marcus Webb' \
    --output runs/migration_experiments/migration_finn_strathearn_to_ashbourne/checkpoints/iter_000100

uv run python scripts/build_migration_checkpoint.py \
    --source-ckpt runs/cultural_emergence/run7_8agent_7B_study_group_strathearn/checkpoints/iter_000100 \
    --source-agent 'Ada Whitfield' \
    --target-ckpt runs/cultural_emergence/run7_8agent_7B_study_group/checkpoints/iter_000100 \
    --target-agent 'Tom Whitaker' \
    --output runs/migration_experiments/migration_ada_strathearn_to_ashbourne/checkpoints/iter_000100

uv run python scripts/build_migration_checkpoint.py \
    --source-ckpt runs/cultural_emergence/run7_8agent_7B_study_group_strathearn/checkpoints/iter_000100 \
    --source-agent 'Callum Reid' \
    --target-ckpt runs/cultural_emergence/run7_8agent_7B_study_group/checkpoints/iter_000100 \
    --target-agent 'Hana Yilmaz' \
    --output runs/migration_experiments/migration_callum_strathearn_to_ashbourne/checkpoints/iter_000100
```

## Step 2 — submit slurm jobs (run tier 1 first, tier 2 after compute permits)

```bash
# tier 1: migration_priya_ashbourne_to_strathearn
sbatch slurm_scripts/migration_priya_ashbourne_to_strathearn.sh
# tier 1: migration_marcus_ashbourne_to_strathearn
sbatch slurm_scripts/migration_marcus_ashbourne_to_strathearn.sh
# tier 2: migration_tom_ashbourne_to_strathearn
sbatch slurm_scripts/migration_tom_ashbourne_to_strathearn.sh
# tier 2: migration_hana_ashbourne_to_strathearn
sbatch slurm_scripts/migration_hana_ashbourne_to_strathearn.sh
# tier 1: migration_mhairi_strathearn_to_ashbourne
sbatch slurm_scripts/migration_mhairi_strathearn_to_ashbourne.sh
# tier 1: migration_finn_strathearn_to_ashbourne
sbatch slurm_scripts/migration_finn_strathearn_to_ashbourne.sh
# tier 2: migration_ada_strathearn_to_ashbourne
sbatch slurm_scripts/migration_ada_strathearn_to_ashbourne.sh
# tier 2: migration_callum_strathearn_to_ashbourne
sbatch slurm_scripts/migration_callum_strathearn_to_ashbourne.sh
```
