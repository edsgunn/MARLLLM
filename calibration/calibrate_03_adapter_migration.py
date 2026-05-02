"""
Calibration 3: Adapter migration mechanics.

Procedure
---------
1. Take any two short-trained populations from prior runs (or run two tiny
   training jobs of 5 iterations each as fallback).
2. Extract one agent's adapter from population A.
3. Replace the corresponding agent's adapter in population B with it via
   the migrate_adapter.py CLI.
4. Continue training population B for 5 iterations.

Pass criteria
-------------
- The migrated adapter loads without error.
- The migrated agent produces outputs (does not crash on first rollout).
- 5 post-migration iterations complete with no harness failures.

Usage
-----
    uv run python calibration/calibrate_03_adapter_migration.py \\
        --pop-a-checkpoint runs/.../popA/checkpoints/iter_NNNNNN \\
        --pop-b-checkpoint runs/.../popB/checkpoints/iter_NNNNNN \\
        --pop-b-config configs/cultural_emergence/run3a_8agent_seedB.yaml \\
        --agent-name "Silas Varnham" \\
        --output-dir runs/cultural_emergence/calibration/03_migration

Time budget per the spec: 1-2 hours.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import yaml
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--pop-a-checkpoint", required=True,
                   help="Source checkpoint dir.")
    p.add_argument("--pop-b-checkpoint", required=True,
                   help="Target checkpoint dir.")
    p.add_argument("--pop-b-config", required=True,
                   help="YAML config used to train pop B (used to resume).")
    p.add_argument("--agent-name", default="Silas Varnham")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--continue-iters", type=int, default=5)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    migrated_dir = out / "migrated_checkpoint"

    # Step 1 — run migration CLI
    print("Step 1: extracting adapter and writing migrated checkpoint")
    cmd = [
        "uv", "run", "python", str(_REPO_ROOT / "migrate_adapter.py"),
        "--source-checkpoint", args.pop_a_checkpoint,
        "--source-agent", args.agent_name,
        "--target-checkpoint", args.pop_b_checkpoint,
        "--target-agent", args.agent_name,
        "--output", str(migrated_dir),
        "--reset-optimizer",
    ]
    print("  ", " ".join(cmd))
    rc = subprocess.call(cmd, cwd=str(_REPO_ROOT))
    if rc != 0:
        sys.exit(f"Migration CLI FAILED (exit {rc}).")

    # Step 2 — modify config to resume from migrated dir, run continue-iters more
    with open(args.pop_b_config) as f:
        cfg = yaml.safe_load(f)
    cfg["iters"] = args.continue_iters
    cfg["output_dir"] = str(out / "post_migration_run")
    cfg["log_every"] = 1
    cfg["checkpoint_every"] = max(1, args.continue_iters)
    smoke_cfg_path = out / "post_migration_config.yaml"
    with open(smoke_cfg_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    print(f"Step 2: continuing training for {args.continue_iters} iterations")
    cmd = [
        "uv", "run", "python", str(_REPO_ROOT / "train_population.py"),
        "--config", str(smoke_cfg_path),
        "--resume-from", str(migrated_dir),
    ]
    print("  ", " ".join(cmd))
    rc = subprocess.call(cmd, cwd=str(_REPO_ROOT))
    if rc != 0:
        sys.exit(f"Post-migration training FAILED (exit {rc}).")

    print()
    print("Calibration 3 PASSED.  Migration mechanics work end-to-end.")
    print(f"  migrated checkpoint: {migrated_dir}")
    print(f"  post-migration run:  {out / 'post_migration_run'}")


if __name__ == "__main__":
    main()
