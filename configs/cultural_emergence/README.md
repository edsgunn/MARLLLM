# Cultural Emergence experimental campaign

Three-day experimental campaign on Concordia's Robotic Athanor scenario
using Qwen2.5-1.5B-Instruct with LoRA r64 adapters per agent.

## Order of operations

1. **Calibration (Day 1 morning)** — must pass before committing cluster time.
   ```
   uv run python calibration/calibrate_01_character_diff.py \
       --output-dir runs/cultural_emergence/calibration/01_char_diff
   uv run python calibration/calibrate_02_concordia_smoke.py \
       --output-dir runs/cultural_emergence/calibration/02_smoke
   # calibrate_03 can be deferred until any two short runs exist
   ```

2. **Run 1 (Day 1 afternoon → Day 2 morning)** — 2-agent collapse case.
   ```
   uv run python train_population.py \
       --config configs/cultural_emergence/run1_2agent_robotic_athanor.yaml
   ```

3. **Run 2 (Day 2)** — 8-agent stable population (Population A).
   ```
   uv run python train_population.py \
       --config configs/cultural_emergence/run2_8agent_seedA.yaml
   ```

4. **Run 3a (Day 3 morning)** — 8-agent baseline at different seed (Population B).
   ```
   uv run python train_population.py \
       --config configs/cultural_emergence/run3a_8agent_seedB.yaml
   ```

5. **Calibration 3 (between 3a and 3b)** — confirms migration mechanics.
   ```
   uv run python calibration/calibrate_03_adapter_migration.py \
       --pop-a-checkpoint runs/cultural_emergence/run2_8agent_seedA/checkpoints/iter_000400 \
       --pop-b-checkpoint runs/cultural_emergence/run3a_8agent_seedB/checkpoints/iter_000200 \
       --pop-b-config configs/cultural_emergence/run3a_8agent_seedB.yaml \
       --agent-name "Silas Varnham" \
       --output-dir runs/cultural_emergence/calibration/03_migration
   ```

6. **Run 3b (Day 3 afternoon)** — migrate Silas A → B, continue training.
   ```
   # Step 1: build the migrated start checkpoint
   uv run python migrate_adapter.py \
       --source-checkpoint runs/cultural_emergence/run2_8agent_seedA/checkpoints/iter_000400 \
       --source-agent "Silas Varnham" \
       --target-checkpoint runs/cultural_emergence/run3a_8agent_seedB/checkpoints/iter_000200 \
       --target-agent "Silas Varnham" \
       --output runs/cultural_emergence/run3b_migration/start_checkpoint \
       --reset-optimizer

   # Step 2: continue training for 150 iters
   uv run python train_population.py \
       --config configs/cultural_emergence/run3b_migration.yaml \
       --resume-from runs/cultural_emergence/run3b_migration/start_checkpoint
   ```

## Artefacts produced per run

Each run directory contains:
- `train.log`                   — per-iteration metrics + warnings
- `metrics.jsonl`               — machine-readable training metrics
- `experiment_config.yaml`      — copy of the YAML used
- `config.json`                 — resolved TrainingConfig
- `traces/iter_NNNNNN.{json,txt}` — single-trace dump at log intervals
- `traces/ckpt_NNNNNN/traces.{json,txt}` — multi-trace dump per checkpoint
- `checkpoints/iter_NNNNNN/`     — per-agent checkpoint dir
    - `meta.pt`                  — iteration, optimizer, rng, config, slot map
    - `<agent_id>.pt`            — per-agent payload (LoRA adapter only for shared-base)
- `checkpoints/latest -> iter_NNNNNN`  — symlink to the most recent checkpoint
- `snapshots/iter_NNNNNN.json`   — held-out behavioural-distribution samples
                                    (K=8 per context per agent at training temp)

## Per-agent checkpoint layout

For LoRA-shared-base populations, each `<agent_id>.pt` contains *only* that
agent's LoRA adapter weights plus its value head — typically 30 MB per
agent at LoRA r64 on Qwen2.5-1.5B.  This makes adapter migration cheap and
lets `migrate_adapter.py` operate on individual agent files without
ever loading the full population.
