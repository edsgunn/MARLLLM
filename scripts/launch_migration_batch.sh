#!/bin/bash
# ============================================================
# Migration-experiment batch launcher.
#
# Prereq: both populations have an iter_000100 checkpoint on disk:
#   runs/cultural_emergence/run7_8agent_7B_study_group/checkpoints/iter_000100
#   runs/cultural_emergence/run7_8agent_7B_study_group_strathearn/checkpoints/iter_000100
#
# Usage:
#   bash scripts/launch_migration_batch.sh smoke
#       Build only the Priya->Strathearn migration checkpoint.
#       Print the smoke-test verification commands.
#
#   bash scripts/launch_migration_batch.sh build [tier1|tier2|all]
#       Build the migration starting checkpoint(s) (no slurm submission).
#
#   bash scripts/launch_migration_batch.sh submit [tier1|tier2|all]
#       sbatch the slurm scripts (assumes checkpoints are already built).
# ============================================================
set -euo pipefail
cd "$(dirname "$0")/.."

ASHBOURNE_CKPT="runs/cultural_emergence/run7_8agent_7B_study_group/checkpoints/iter_000100"
STRATHEARN_CKPT="runs/cultural_emergence/run7_8agent_7B_study_group_strathearn/checkpoints/iter_000100"

# (run_name, source_ckpt_var, source_agent, target_ckpt_var, target_agent, tier)
MIGRATIONS=(
  "migration_priya_ashbourne_to_strathearn|ASHBOURNE_CKPT|Priya Shah|STRATHEARN_CKPT|Mhairi Buchanan|1"
  "migration_marcus_ashbourne_to_strathearn|ASHBOURNE_CKPT|Marcus Webb|STRATHEARN_CKPT|Finn Nakamura|1"
  "migration_mhairi_strathearn_to_ashbourne|STRATHEARN_CKPT|Mhairi Buchanan|ASHBOURNE_CKPT|Priya Shah|1"
  "migration_finn_strathearn_to_ashbourne|STRATHEARN_CKPT|Finn Nakamura|ASHBOURNE_CKPT|Marcus Webb|1"
  "migration_tom_ashbourne_to_strathearn|ASHBOURNE_CKPT|Tom Whitaker|STRATHEARN_CKPT|Ada Whitfield|2"
  "migration_hana_ashbourne_to_strathearn|ASHBOURNE_CKPT|Hana Yilmaz|STRATHEARN_CKPT|Callum Reid|2"
  "migration_ada_strathearn_to_ashbourne|STRATHEARN_CKPT|Ada Whitfield|ASHBOURNE_CKPT|Tom Whitaker|2"
  "migration_callum_strathearn_to_ashbourne|STRATHEARN_CKPT|Callum Reid|ASHBOURNE_CKPT|Hana Yilmaz|2"
)

filter_tier() {
  local want="$1"
  for entry in "${MIGRATIONS[@]}"; do
    IFS='|' read -r run_name src_var src_agent tgt_var tgt_agent tier <<<"$entry"
    if [[ "$want" == "all" || "$tier" == "${want#tier}" ]]; then
      echo "$entry"
    fi
  done
}

cmd="${1:-}"
arg="${2:-all}"

case "$cmd" in
  smoke)
    echo "==> Smoke build: Priya -> Strathearn only (5-iter test)"
    if [[ ! -d "$ASHBOURNE_CKPT" ]]; then
      echo "FAIL: $ASHBOURNE_CKPT does not exist." >&2; exit 1
    fi
    if [[ ! -d "$STRATHEARN_CKPT" ]]; then
      echo "FAIL: $STRATHEARN_CKPT does not exist (Strathearn extension still running?)" >&2; exit 1
    fi
    out_dir="runs/migration_experiments/migration_priya_ashbourne_to_strathearn/checkpoints/iter_000100"
    .venv/bin/python scripts/build_migration_checkpoint.py \
      --source-ckpt "$ASHBOURNE_CKPT" \
      --source-agent "Priya Shah" \
      --target-ckpt "$STRATHEARN_CKPT" \
      --target-agent "Mhairi Buchanan" \
      --output "$out_dir"
    echo
    echo "==> To run the 5-iter smoke test, use a config override (iters: 105 instead of 150):"
    echo "  Run by hand on a single GPU node, OR temporarily edit"
    echo "  configs/migration_experiments/migration_priya_ashbourne_to_strathearn.yaml"
    echo "  to set iters: 105, then submit slurm_scripts/migration_priya_ashbourne_to_strathearn.sh"
    echo
    echo "==> After 5 iters, verify:"
    echo "  - logs show 'Population: [Priya Shah, Callum Reid, ...]' (Mhairi gone)"
    echo "  - traces/iter_000005.txt has Priya's posts attributed to 'Priya Shah'"
    echo "  - traces/ckpt_000000 contains an iter-100-state trace before any training"
    echo "  - checkpoints/iter_000100/Priya Shah.pt exists (and Mhairi Buchanan.pt does not)"
    echo "  - metrics.jsonl includes per-agent loss/KL/entropy for Priya Shah"
    ;;

  build)
    while IFS='|' read -r run_name src_var src_agent tgt_var tgt_agent tier; do
      [[ -z "${run_name:-}" ]] && continue
      src="${!src_var}"
      tgt="${!tgt_var}"
      out_dir="runs/migration_experiments/${run_name}/checkpoints/iter_000100"
      echo "==> [$run_name] (tier $tier)"
      .venv/bin/python scripts/build_migration_checkpoint.py \
        --source-ckpt "$src" \
        --source-agent "$src_agent" \
        --target-ckpt "$tgt" \
        --target-agent "$tgt_agent" \
        --output "$out_dir"
      echo
    done < <(filter_tier "$arg")
    ;;

  submit)
    DEP="${SBATCH_DEPENDENCY:-}"
    dep_arg=()
    if [[ -n "$DEP" ]]; then
      dep_arg=(--dependency="$DEP")
      echo "Using slurm dependency: $DEP"
    fi
    while IFS='|' read -r run_name _ _ _ _ tier; do
      [[ -z "${run_name:-}" ]] && continue
      out_dir="runs/migration_experiments/${run_name}/checkpoints/iter_000100"
      if [[ ! -d "$out_dir" ]]; then
        echo "[skip] $run_name: $out_dir not built yet (run 'build' first)" >&2
        continue
      fi
      echo "==> sbatch slurm_scripts/${run_name}.sh (tier $tier)"
      sbatch "${dep_arg[@]}" "slurm_scripts/${run_name}.sh"
    done < <(filter_tier "$arg")
    ;;

  *)
    echo "Usage: $0 {smoke | build [tier1|tier2|all] | submit [tier1|tier2|all]}" >&2
    exit 2
    ;;
esac
