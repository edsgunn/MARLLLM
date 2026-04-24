#!/usr/bin/env python3
"""
Submit SLURM jobs to regenerate episode traces for every run in a folder.

For each subdirectory that looks like a completed run (contains config.json
and at least one checkpoint), a job is submitted that calls generate_traces.py.
Trace generation only needs inference so jobs are short and light — one GPU,
~30 min walltime by default.

Isambard AI Phase 2 specifics
------------------------------
- Partition : workq
- GPU syntax : --gpus=N  (not --gres=gpu:N)
- Modules   : cuda/12.6, brics/nccl  loaded automatically

Usage examples
--------------
    # Regenerate traces for all runs under runs/negotiation_prompt_experiments/
    python submit_traces.py runs/negotiation_prompt_experiments/ --account <proj>

    # Dry-run to inspect scripts without submitting
    python submit_traces.py runs/ --dry-run

    # Specific checkpoint within each run (relative to run dir)
    python submit_traces.py runs/ --checkpoint checkpoints/iter_000300.pt

    # More traces, different output location
    python submit_traces.py runs/ --n-traces 20 --output-subdir traces_iter300

    # Only runs whose directory name matches a pattern
    python submit_traces.py runs/ --filter 08_

    # Save generated scripts for inspection
    python submit_traces.py runs/ --scripts-dir slurm_scripts/traces/
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

# ── Isambard AI Phase 2 defaults ─────────────────────────────────────────────
ISAMBARD_PARTITION    = "workq"
ISAMBARD_MAX_HOURS    = 24
ISAMBARD_CPUS_PER_GPU = 72
ISAMBARD_DEFAULT_MODULES = ["cuda/12.6", "brics/nccl"]

_DEFAULT_HF_HOME = "/lus/lfs1aip2/projects/a5l/egunn/hf_cache"


# ── Helpers ───────────────────────────────────────────────────────────────────

def find_run_dirs(roots: list[str], name_filter: str | None) -> list[Path]:
    """
    Recursively find directories that contain a config.json and at least one
    .pt checkpoint file, sorted by path for deterministic ordering.
    """
    runs: list[Path] = []
    for root_str in roots:
        root = Path(root_str)
        if not root.exists():
            print(f"[WARN] Path not found, skipping: {root}", file=sys.stderr)
            continue
        # If root itself is a run dir, include it directly.
        if (root / "config.json").exists():
            candidates = [root]
        else:
            # Walk one level down (each immediate child that looks like a run).
            candidates = [d for d in sorted(root.iterdir()) if d.is_dir()]

        for d in candidates:
            if not (d / "config.json").exists():
                continue
            ckpt_dir = d / "checkpoints"
            has_checkpoints = ckpt_dir.exists() and (
                list(ckpt_dir.glob("*.pt")) or (ckpt_dir / "latest.pt").exists()
            )
            if not has_checkpoints:
                print(f"[SKIP] No checkpoints found: {d}", file=sys.stderr)
                continue
            if name_filter and name_filter not in d.name:
                continue
            runs.append(d)

    return sorted(set(runs))


def make_trace_job_script(
    run_dir: Path,
    *,
    project_dir: Path,
    log_dir: Path,
    checkpoint: str | None,
    n_traces: int,
    output_subdir: str | None,
    time_limit: str,
    gpus: int,
    cpus_per_gpu: int,
    partition: str,
    account: str | None,
    qos: str | None,
    mail_user: str | None,
    mail_type: str,
    hf_home: str,
    default_modules: list[str],
    extra_modules: list[str],
    venv_activate: str | None,
    temperature: float | None,
) -> str:
    name = f"traces_{run_dir.name}"

    out_log = log_dir / f"{name}_%j.out"
    err_log = log_dir / f"{name}_%j.err"

    sbatch_lines = [
        f"#SBATCH --job-name={name}",
        f"#SBATCH --nodes=1",
        f"#SBATCH --gpus={gpus}",
        f"#SBATCH --cpus-per-gpu={cpus_per_gpu}",
        f"#SBATCH --time={time_limit}",
        f"#SBATCH --partition={partition}",
        f"#SBATCH --output={out_log}",
        f"#SBATCH --error={err_log}",
    ]
    if account:
        sbatch_lines.append(f"#SBATCH --account={account}")
    if qos:
        sbatch_lines.append(f"#SBATCH --qos={qos}")
    if mail_user:
        sbatch_lines.append(f"#SBATCH --mail-user={mail_user}")
        sbatch_lines.append(f"#SBATCH --mail-type={mail_type}")

    sbatch_block = "\n".join(sbatch_lines)

    all_modules = default_modules + extra_modules
    if all_modules:
        module_block = "module purge\n" + "\n".join(f"module load {m}" for m in all_modules)
    else:
        module_block = "# (no modules loaded)"

    # Build the generate_traces.py command
    output_dir = run_dir / (output_subdir or "traces_regen")
    cmd_parts = [
        f'"{project_dir}/generate_traces.py"',
        f'--run-dir "{run_dir}"',
        f'--n-traces {n_traces}',
        f'--output-dir "{output_dir}"',
    ]
    if checkpoint:
        cmd_parts.append(f'--checkpoint "{checkpoint}"')
    if temperature is not None:
        cmd_parts.append(f'--temperature {temperature}')

    if venv_activate:
        activate_block = f'source "{venv_activate}"'
        python_cmd = "python " + " \\\n    ".join(cmd_parts)
    else:
        activate_block = f"""\
if [ -f "{project_dir}/.venv/bin/activate" ]; then
    source "{project_dir}/.venv/bin/activate"
elif command -v uv &>/dev/null; then
    : # uv run handles the venv automatically
else
    echo "[ERROR] No venv found and uv is not on PATH. Aborting." >&2
    exit 1
fi"""
        python_cmd = "uv run --no-sync python " + " \\\n    ".join(cmd_parts)

    script_body = f"""\
#!/bin/bash -l
# ============================================================
# Auto-generated by submit_traces.py for Isambard AI Phase 2
# Run dir : {run_dir}
# Created : {datetime.now().isoformat(timespec='seconds')}
# ============================================================

{sbatch_block}

set -euo pipefail

echo "================================================================"
echo "Job started  : $(date)"
echo "Job ID       : $SLURM_JOB_ID"
echo "Job name     : $SLURM_JOB_NAME"
echo "Node         : $(hostname -s)"
echo "GPUs         : $SLURM_GPUS"
echo "Run dir      : {run_dir}"
echo "Output dir   : {output_dir}"
echo "================================================================"

if command -v nvidia-smi &>/dev/null; then
    echo "--- nvidia-smi ---"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
    echo ""
fi

# ── Modules ───────────────────────────────────────────────────────────────────
{module_block}

# ── Environment ───────────────────────────────────────────────────────────────
cd "{project_dir}"

export HF_HOME="{hf_home}"
mkdir -p "$HF_HOME"

if [ -z "${{HF_TOKEN:-}}" ]; then
    echo "[WARN] HF_TOKEN is not set — gated models will fail to download."
else
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
    echo "[INFO] HF_TOKEN is set (length ${{#HF_TOKEN}} chars)."
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Python / venv ─────────────────────────────────────────────────────────────
{activate_block}

echo "--- Python ---"
python --version 2>&1 || true
python -c "import torch; print(f'torch {{torch.__version__}} | cuda {{torch.version.cuda}} | gpus {{torch.cuda.device_count()}}')" 2>&1 || true
echo ""

# ── Generate traces ───────────────────────────────────────────────────────────
echo "--- Command ---"
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


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Submit SLURM jobs to regenerate traces for all runs in a folder.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "runs", nargs="+",
        help="Run directory or parent directory of multiple run directories.",
    )
    p.add_argument(
        "--filter", default=None, metavar="SUBSTR",
        help="Only process run directories whose name contains this substring.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print generated job scripts without submitting.",
    )
    p.add_argument(
        "--account", default=None,
        help="Slurm project account (required on Isambard AI).",
    )
    p.add_argument(
        "--partition", default=ISAMBARD_PARTITION,
        help=f"Slurm partition (default: {ISAMBARD_PARTITION}).",
    )
    p.add_argument(
        "--qos", default=None,
        help="Slurm QOS.",
    )
    p.add_argument(
        "--time", default="01:00:00",
        help="Walltime per job (default: 01:00:00). Trace generation is inference-only.",
    )
    p.add_argument(
        "--gpus", type=int, default=1,
        help="Number of GPUs per job (default: 1).",
    )
    p.add_argument(
        "--cpus-per-gpu", type=int, default=ISAMBARD_CPUS_PER_GPU,
        help=f"CPUs per GPU (default: {ISAMBARD_CPUS_PER_GPU}).",
    )
    p.add_argument(
        "--n-traces", type=int, default=10,
        help="Episodes to collect per run (default: 10).",
    )
    p.add_argument(
        "--checkpoint", default=None, metavar="RELPATH",
        help="Checkpoint path relative to each run dir, e.g. "
             "checkpoints/iter_000300.pt.  Defaults to checkpoints/latest.pt.",
    )
    p.add_argument(
        "--output-subdir", default=None, metavar="NAME",
        help="Subdirectory name inside each run dir for trace output "
             "(default: traces_regen).  Useful when generating traces from "
             "multiple checkpoints: set to e.g. traces_iter300.",
    )
    p.add_argument(
        "--temperature", type=float, default=None,
        help="Sampling temperature override (defaults to value in config.json).",
    )
    p.add_argument(
        "--mail-user", default=None,
        help="Email address for job-status notifications.",
    )
    p.add_argument(
        "--mail-type", default="END,FAIL",
        help="Slurm mail events (default: END,FAIL).",
    )
    p.add_argument(
        "--hf-home", default=_DEFAULT_HF_HOME, metavar="PATH",
        help=f"HuggingFace model cache directory (default: {_DEFAULT_HF_HOME}).",
    )
    p.add_argument(
        "--module", action="append", dest="modules", default=[],
        metavar="MODULE",
        help="Extra `module load` line (repeat for multiple).",
    )
    p.add_argument(
        "--no-default-modules", action="store_true",
        help=f"Skip default Isambard modules ({', '.join(ISAMBARD_DEFAULT_MODULES)}).",
    )
    p.add_argument(
        "--venv", default=None, metavar="PATH",
        help="Path to a venv activate script.  "
             "If omitted, auto-detects .venv/ or falls back to `uv run`.",
    )
    p.add_argument(
        "--project-dir", default=None, metavar="PATH",
        help="Root of the MARLLLM project (default: directory containing this script).",
    )
    p.add_argument(
        "--log-dir", default=None, metavar="PATH",
        help="Directory for Slurm stdout/stderr logs (default: <project-dir>/slurm_logs/).",
    )
    p.add_argument(
        "--scripts-dir", default=None, metavar="PATH",
        help="Save generated job scripts here for inspection.",
    )
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    project_dir = Path(args.project_dir or Path(__file__).resolve().parent)
    log_dir     = Path(args.log_dir or project_dir / "slurm_logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    scripts_dir: Path | None = None
    if args.scripts_dir:
        scripts_dir = Path(args.scripts_dir)
        scripts_dir.mkdir(parents=True, exist_ok=True)

    if not args.account and not args.dry_run:
        print(
            "[WARN] --account not set.  Jobs on Isambard AI require a project account.",
            file=sys.stderr,
        )

    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        print(
            "[WARN] $HF_TOKEN is not set.  Gated models will fail to download inside jobs.",
            file=sys.stderr,
        )

    default_modules = [] if args.no_default_modules else list(ISAMBARD_DEFAULT_MODULES)

    run_dirs = find_run_dirs(args.runs, args.filter)
    if not run_dirs:
        sys.exit("No valid run directories found (need config.json + at least one checkpoint).")

    print(
        f"Found {len(run_dirs)} run(s). "
        f"{'[DRY RUN] ' if args.dry_run else ''}Submitting trace-regen jobs...",
    )
    print()

    submitted: list[tuple[str, str]] = []
    failed:    list[tuple[str, str]] = []

    for run_dir in run_dirs:
        name = f"traces_{run_dir.name}"

        script_body = make_trace_job_script(
            run_dir,
            project_dir=project_dir,
            log_dir=log_dir,
            checkpoint=args.checkpoint,
            n_traces=args.n_traces,
            output_subdir=args.output_subdir,
            time_limit=args.time,
            gpus=args.gpus,
            cpus_per_gpu=args.cpus_per_gpu,
            partition=args.partition,
            account=args.account,
            qos=args.qos,
            mail_user=args.mail_user,
            mail_type=args.mail_type,
            hf_home=args.hf_home,
            default_modules=default_modules,
            extra_modules=args.modules,
            venv_activate=args.venv,
            temperature=args.temperature,
        )

        if scripts_dir:
            script_path = scripts_dir / f"{name}.sh"
            script_path.write_text(script_body)
            script_path.chmod(0o700)
            tmp_path = script_path
            cleanup = False
        else:
            tmp = tempfile.NamedTemporaryFile(
                mode="w", suffix=".sh", prefix=f"traces_{run_dir.name}_",
                delete=False,
            )
            tmp.write(script_body)
            tmp.close()
            tmp_path = Path(tmp.name)
            tmp_path.chmod(0o700)
            cleanup = True

        if args.dry_run:
            print(f"{'─' * 60}")
            print(f"RUN DIR : {run_dir}")
            print(f"SCRIPT  : {tmp_path}")
            export_arg = f"--export=ALL,HF_TOKEN=$HF_TOKEN" if hf_token else "--export=ALL"
            print(f"COMMAND : sbatch {export_arg} {tmp_path}")
            print()
            print(script_body)
            if cleanup:
                tmp_path.unlink(missing_ok=True)
            continue

        export_arg = f"ALL,HF_TOKEN={hf_token}" if hf_token else "ALL"
        try:
            result = subprocess.run(
                ["sbatch", f"--export={export_arg}", str(tmp_path)],
                capture_output=True, text=True, check=True,
            )
            job_id = result.stdout.strip().split()[-1]
            submitted.append((name, job_id))
            print(f"  ✓  {name:<60}  job {job_id}")
        except subprocess.CalledProcessError as exc:
            reason = exc.stderr.strip() or exc.stdout.strip()
            failed.append((name, reason))
            print(f"  ✗  {name:<60}  FAILED: {reason}", file=sys.stderr)
        finally:
            if cleanup:
                tmp_path.unlink(missing_ok=True)

    if args.dry_run:
        return

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

    manifest_path = log_dir / f"trace_submission_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(manifest_path, "w") as f:
        f.write(f"Submitted at : {datetime.now().isoformat()}\n")
        f.write(f"Account      : {args.account or '(not set)'}\n")
        f.write(f"n_traces     : {args.n_traces}\n")
        f.write(f"checkpoint   : {args.checkpoint or 'latest.pt'}\n")
        f.write(f"HF_TOKEN set : {'yes' if hf_token else 'NO'}\n\n")
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
