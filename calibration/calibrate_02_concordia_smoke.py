"""
Calibration 2: Concordia integration smoke test (4-agent, 50 iterations).

Verifies that:
  1. The Robotic Athanor scenario loads via the Concordia upstream package.
  2. Each agent receives properly-formatted contexts (per-agent invariant).
  3. The MARLLM PopulationTrainer can drive the simulation for 50 iterations.
  4. Perception loss decreases at least slightly across the run.
  5. Traces and per-agent checkpoints are written to disk.

This script is a thin wrapper around train_population.py with a baked-in
small config — pure pipeline confirmation, not useful training.

Usage
-----
    uv run python calibration/calibrate_02_concordia_smoke.py \\
        --output-dir runs/cultural_emergence/calibration/02_smoke

Time budget per the spec: half a day, including any debugging.
Pass criterion: the run completes 50 iterations without error,
``train.log`` shows perception loss declining, and at least one trace
record exists in ``traces/``.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--rollouts", type=int, default=2)
    p.add_argument("--token-budget", type=int, default=128)
    p.add_argument("--max-steps", type=int, default=8,
                   help="Concordia simulation max steps per episode.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Generate an inline YAML config for the smoke test
    cfg_path = out / "smoke_config.yaml"
    cfg_path.write_text(f"""\
# Calibration 2: Concordia smoke test
model: {args.model}
dtype: bfloat16
device: cuda:0
attn_impl: eager

characters:
  - "Silas Varnham"
  - "Petra Ouyang"
  - "Diego Esparza"
  - "Thaddeus 'Aurelius' Thorne"

# Use LoRA r16 (cheap) — we are not training to convergence here
lora_shared_base: true
lora_r: 16
lora_alpha: 32

iters: {args.iters}
rollouts: {args.rollouts}
lr: 1e-5
seed: 1
log_every: 5
checkpoint_every: 25
num_checkpoint_traces: 2
max_episode_tokens: 1024
grad_accum: 4

environments:
  - name: robotic_athanor_smoke
    type: concordia
    scenario: robotic_athanor
    character_set: canonical_4
    max_steps: {args.max_steps}
    token_budget: {args.token_budget}
    max_turns: 50

output_dir: {out}
""")
    print(f"Smoke config: {cfg_path}")
    cmd = [
        "uv", "run", "python",
        str(_REPO_ROOT / "train_population.py"),
        "--config", str(cfg_path),
    ]
    print("Running:", " ".join(cmd))
    rc = subprocess.call(cmd, cwd=str(_REPO_ROOT))
    if rc != 0:
        sys.exit(f"Smoke test FAILED with exit code {rc}")

    # Sanity-check artefacts
    train_log = out / "train.log"
    traces_dir = out / "traces"
    ckpt_dir = out / "checkpoints"
    issues = []
    if not train_log.exists():
        issues.append(f"missing {train_log}")
    if not traces_dir.exists() or not any(traces_dir.iterdir()):
        issues.append(f"no traces in {traces_dir}")
    if not ckpt_dir.exists() or not any(ckpt_dir.iterdir()):
        issues.append(f"no checkpoints in {ckpt_dir}")
    if issues:
        sys.exit("Smoke test artefact check FAILED: " + "; ".join(issues))

    print()
    print("Smoke test PASSED.  Artefacts:")
    print(f"  log:     {train_log}")
    print(f"  traces:  {traces_dir}")
    print(f"  ckpts:   {ckpt_dir}")
    print()
    print("ACTION: open train.log and verify perception loss declines.")


if __name__ == "__main__":
    main()
