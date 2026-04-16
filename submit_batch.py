#!/usr/bin/env python3
"""
Submit all experiment configs in a folder as separate Slurm jobs.

Each config file becomes one independent sbatch job, so an OOM crash or
walltime expiry in one run does not affect any other.  The Slurm resource
directives (time, GPUs, memory, etc.) are read from each config's ``slurm``
section, so you can mix small and large jobs in a single batch submission.

Usage
-----
    # Submit everything in configs/experiments/
    python submit_batch.py configs/experiments/

    # Preview what would be submitted without actually calling sbatch
    python submit_batch.py configs/experiments/ --dry-run

    # Override the partition for all jobs
    python submit_batch.py configs/experiments/ --partition gpu_a100

    # Submit only specific configs
    python submit_batch.py configs/experiments/03_qwen_1.5b_baseline.yaml \\
                           configs/experiments/04_qwen_3b.yaml

    # Override account and email for all jobs
    python submit_batch.py configs/experiments/ \\
        --account myproject --mail-user me@university.edu --mail-type END,FAIL

Config YAML format
------------------
Each file must have a ``slurm`` block and a ``script`` key, plus training args:

    name: experiment_name
    description: "What this tests"
    script: train_negotiation          # or "train"

    slurm:
      time: "12:00:00"                 # HH:MM:SS wall-clock limit
      nodes: 1                         # must be 1 if gpus_per_node > 1
      gpus_per_node: 2                 # total GPUs requested
      cpus_per_gpu: 8
      mem: "80G"                       # total memory for the job
      partition: gpu                   # Slurm partition name

    model: Qwen/Qwen2.5-1.5B
    iters: 500
    ...
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_yaml(path: Path) -> dict:
    try:
        import yaml  # type: ignore[import]
    except ImportError as exc:
        sys.exit("PyYAML is required: pip install pyyaml")
    with open(path) as f:
        return yaml.safe_load(f) or {}


def find_configs(paths: list[str]) -> list[Path]:
    """Expand a list of files/directories into a sorted list of YAML paths."""
    result: list[Path] = []
    for p_str in paths:
        p = Path(p_str)
        if p.is_dir():
            result.extend(sorted(p.glob("*.yaml")))
            result.extend(sorted(p.glob("*.yml")))
        elif p.is_file():
            result.append(p)
        else:
            print(f"[WARN] Path not found, skipping: {p}", file=sys.stderr)
    return result


def make_job_script(
    config_path: Path,
    cfg: dict,
    *,
    project_dir: Path,
    log_dir: Path,
    partition_override: str | None,
    account: str | None,
    mail_user: str | None,
    mail_type: str | None,
    extra_modules: list[str],
    venv_activate: str | None,
) -> str:
    """
    Return a complete bash job script as a string.

    The script:
    1. Sets all #SBATCH directives from the config's ``slurm`` section.
    2. Prints rich diagnostic info (host, GPUs, SLURM env) to the job log.
    3. Activates the Python environment.
    4. Runs the appropriate training script with ``--config <path>``.
    5. Captures the exit code and echoes it clearly so post-processing
       can detect failures without parsing stderr.
    """
    slurm_cfg  = cfg.get("slurm", {})
    name       = cfg.get("name", config_path.stem)
    script     = cfg.get("script", "train_negotiation")

    # ── Slurm directives ──────────────────────────────────────────────────
    time_limit    = slurm_cfg.get("time", "24:00:00")
    nodes         = int(slurm_cfg.get("nodes", 1))
    gpus_per_node = int(slurm_cfg.get("gpus_per_node", 1))
    cpus_per_gpu  = int(slurm_cfg.get("cpus_per_gpu", 8))
    mem           = slurm_cfg.get("mem", "80G")
    partition     = partition_override or slurm_cfg.get("partition", "gpu")

    # Enforce single-node when requesting multiple GPUs (multi-node MPI
    # topology is not set up for this framework).
    if gpus_per_node > 1 and nodes != 1:
        print(
            f"[WARN] {name}: gpus_per_node={gpus_per_node} > 1 "
            f"but nodes={nodes}; forcing nodes=1.",
            file=sys.stderr,
        )
        nodes = 1

    out_log = log_dir / f"{name}_%j.out"
    err_log = log_dir / f"{name}_%j.err"

    sbatch_lines = [
        f"#SBATCH --job-name={name}",
        f"#SBATCH --nodes={nodes}",
        f"#SBATCH --gres=gpu:{gpus_per_node}",
        f"#SBATCH --cpus-per-gpu={cpus_per_gpu}",
        f"#SBATCH --mem={mem}",
        f"#SBATCH --time={time_limit}",
        # f"#SBATCH --partition={partition}",
        f"#SBATCH --output={out_log}",
        f"#SBATCH --error={err_log}",
    ]
    if account:
        sbatch_lines.append(f"#SBATCH --account={account}")
    if mail_user:
        sbatch_lines.append(f"#SBATCH --mail-user={mail_user}")
        sbatch_lines.append(f"#SBATCH --mail-type={mail_type or 'END,FAIL'}")

    sbatch_block = "\n".join(sbatch_lines)

    # ── Module loads ──────────────────────────────────────────────────────
    module_block = ""
    if extra_modules:
        module_block = "module purge\n" + "\n".join(
            f"module load {m}" for m in extra_modules
        )

    # ── Venv activation ───────────────────────────────────────────────────
    activate_block = ""
    if venv_activate:
        activate_block = f'source "{venv_activate}"'
    else:
        # Try to detect a uv / venv inside the project directory.
        activate_block = f"""\
if [ -f "{project_dir}/.venv/bin/activate" ]; then
    source "{project_dir}/.venv/bin/activate"
elif command -v uv &>/dev/null; then
    # uv run handles the venv automatically; no explicit activation needed.
    : # no-op
else
    echo "[WARN] No virtual environment found; using system Python."
fi"""

    # Determine the Python invocation: prefer uv run if uv is available.
    python_cmd = (
        f'uv run python "{project_dir}/{script}.py" --config "{config_path.resolve()}"'
        if not venv_activate
        else f'python "{project_dir}/{script}.py" --config "{config_path.resolve()}"'
    )

    script_body = f"""\
#!/bin/bash
# ============================================================
# Auto-generated by submit_batch.py
# Config : {config_path.resolve()}
# Created: {datetime.now().isoformat(timespec='seconds')}
# ============================================================

{sbatch_block}

set -euo pipefail

echo "================================================================"
echo "Job started  : $(date)"
echo "Job ID       : $SLURM_JOB_ID"
echo "Job name     : $SLURM_JOB_NAME"
echo "Node         : $(hostname -s)"
echo "Working dir  : {project_dir}"
echo "Config file  : {config_path.resolve()}"
echo "================================================================"

# ── Print full SLURM environment for reproducibility ─────────────────────────
echo ""
echo "--- SLURM environment ---"
env | grep -E '^SLURM' | sort
echo ""

# ── GPU inventory ─────────────────────────────────────────────────────────────
if command -v nvidia-smi &>/dev/null; then
    echo "--- nvidia-smi ---"
    nvidia-smi --query-gpu=index,name,memory.total,driver_version \\
               --format=csv,noheader
    echo ""
fi

# ── Software environment ──────────────────────────────────────────────────────
{module_block}

cd "{project_dir}"
{activate_block}

echo "--- Python / package versions ---"
python --version 2>&1 || true
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.version.cuda)" 2>&1 || true
python -c "import transformers; print('transformers', transformers.__version__)" 2>&1 || true
echo ""

# ── Run training ──────────────────────────────────────────────────────────────
echo "--- Training command ---"
echo "{python_cmd}"
echo ""

{python_cmd}
EXIT_CODE=$?

echo ""
echo "================================================================"
echo "Job finished : $(date)"
echo "Exit code    : $EXIT_CODE"
echo "================================================================"
exit $EXIT_CODE
"""
    return script_body


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Submit a folder of YAML experiment configs as separate Slurm jobs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "configs", nargs="+",
        help="One or more YAML config files or directories containing them.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print the generated job scripts and sbatch commands without submitting.",
    )
    p.add_argument(
        "--partition", default=None,
        help="Override the Slurm partition for all jobs.",
    )
    p.add_argument(
        "--account", default=None,
        help="Slurm account / project allocation to charge.",
    )
    p.add_argument(
        "--mail-user", default=None,
        help="Email address for job notifications.",
    )
    p.add_argument(
        "--mail-type", default="END,FAIL",
        help="Slurm mail events (default: END,FAIL).",
    )
    p.add_argument(
        "--module", action="append", dest="modules", default=[],
        metavar="MODULE",
        help="Extra `module load` lines to add (repeat for multiple modules).",
    )
    p.add_argument(
        "--venv", default=None,
        metavar="PATH",
        help="Path to a venv activate script to source. "
             "If omitted, the script auto-detects .venv/ or uses `uv run`.",
    )
    p.add_argument(
        "--project-dir", default=None,
        metavar="PATH",
        help="Root directory of the MARLLLM project. "
             "Defaults to the directory containing this script.",
    )
    p.add_argument(
        "--log-dir", default=None,
        metavar="PATH",
        help="Directory for Slurm stdout/stderr logs. "
             "Defaults to <project-dir>/slurm_logs/.",
    )
    p.add_argument(
        "--scripts-dir", default=None,
        metavar="PATH",
        help="Directory where generated job scripts are saved (useful for debugging). "
             "If omitted, scripts are written to a temp directory and deleted after submission.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    project_dir = Path(args.project_dir or Path(__file__).resolve().parent)
    log_dir     = Path(args.log_dir or project_dir / "slurm_logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    scripts_dir: Path | None = None
    if args.scripts_dir:
        scripts_dir = Path(args.scripts_dir)
        scripts_dir.mkdir(parents=True, exist_ok=True)

    configs = find_configs(args.configs)
    if not configs:
        sys.exit("No YAML config files found.")

    print(f"Found {len(configs)} config(s). {'[DRY RUN] ' if args.dry_run else ''}Submitting...")
    print()

    submitted: list[tuple[str, str]] = []  # (name, job_id)
    failed:    list[tuple[str, str]] = []  # (name, reason)

    for config_path in configs:
        cfg  = load_yaml(config_path)
        name = cfg.get("name", config_path.stem)

        script_body = make_job_script(
            config_path,
            cfg,
            project_dir=project_dir,
            log_dir=log_dir,
            partition_override=args.partition,
            account=args.account,
            mail_user=args.mail_user,
            mail_type=args.mail_type,
            extra_modules=args.modules,
            venv_activate=args.venv,
        )

        # Write the job script ─────────────────────────────────────────────
        if scripts_dir:
            script_path = scripts_dir / f"{name}.sh"
            script_path.write_text(script_body)
            script_path.chmod(0o755)
            tmp_path = script_path
            cleanup = False
        else:
            # Temp file — keep until after sbatch so the scheduler can read it.
            tmp = tempfile.NamedTemporaryFile(
                mode="w", suffix=".sh", prefix=f"marlllm_{name}_",
                delete=False,
            )
            tmp.write(script_body)
            tmp.close()
            tmp_path = Path(tmp.name)
            cleanup = True

        if args.dry_run:
            print(f"{'─'*60}")
            print(f"CONFIG  : {config_path}")
            print(f"SCRIPT  : {tmp_path}")
            print(f"COMMAND : sbatch {tmp_path}")
            print()
            print(script_body)
            if cleanup:
                tmp_path.unlink(missing_ok=True)
            continue

        # Submit ───────────────────────────────────────────────────────────
        try:
            result = subprocess.run(
                ["sbatch", str(tmp_path)],
                capture_output=True, text=True, check=True,
            )
            # sbatch prints "Submitted batch job <id>"
            job_id = result.stdout.strip().split()[-1]
            submitted.append((name, job_id))
            print(f"  ✓  {name:<45}  job {job_id}")
        except subprocess.CalledProcessError as exc:
            reason = exc.stderr.strip() or exc.stdout.strip()
            failed.append((name, reason))
            print(f"  ✗  {name:<45}  FAILED: {reason}", file=sys.stderr)
        finally:
            if cleanup:
                tmp_path.unlink(missing_ok=True)

    # ── Summary ───────────────────────────────────────────────────────────
    if not args.dry_run:
        print()
        print(f"Submitted : {len(submitted)}")
        print(f"Failed    : {len(failed)}")
        if submitted:
            print()
            print("Job IDs:")
            for name, jid in submitted:
                print(f"  {jid:>12}  {name}")
        if failed:
            print()
            print("Failures:")
            for name, reason in failed:
                print(f"  {name}: {reason}")

        # Write a submission manifest next to the configs folder
        manifest_path = log_dir / f"submission_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(manifest_path, "w") as f:
            f.write(f"Submitted at: {datetime.now().isoformat()}\n\n")
            f.write("Submitted jobs:\n")
            for name, jid in submitted:
                f.write(f"  {jid}  {name}\n")
            if failed:
                f.write("\nFailed:\n")
                for name, reason in failed:
                    f.write(f"  {name}: {reason}\n")
        print(f"\nManifest written: {manifest_path}")

        if failed:
            sys.exit(1)


if __name__ == "__main__":
    main()
