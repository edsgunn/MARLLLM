#!/usr/bin/env python3
"""
Submit MARLLLM experiment configs as separate Slurm jobs on Isambard AI Phase 2.

Isambard AI Phase 2 specifics
------------------------------
- Hardware    : 1,320 nodes × 4 × NVIDIA GH200 Grace Hopper (96 GB HBM3 each)
- CPU         : 4 × ARM Grace, 72 cores each (288 cores / node, aarch64)
- Partition   : workq  (max walltime 24 h)
- GPU syntax  : --gpus=N  (not --gres=gpu:N)
- Modules     : cuda/12.6, brics/nccl  (loaded automatically)
- Interconnect: Slingshot 11 — NCCL requires the brics/nccl module

Gated HuggingFace models (Llama etc.)
--------------------------------------
HF_TOKEN is read from your shell environment at submit time and forwarded to
every job automatically (Slurm exports all env vars by default on Isambard).
Set it once before submitting:

    export HF_TOKEN="hf_..."
    python submit_batch.py configs/experiments/

The token is NEVER written into the generated job scripts to avoid leaking it
into world-readable files on shared storage.  It is forwarded at the sbatch
call level via --export so it only lives in the Slurm job environment.

Usage examples
--------------
    # Submit everything in configs/experiments/
    python submit_batch.py configs/experiments/ --account <your_project>

    # Dry-run: print scripts without submitting
    python submit_batch.py configs/experiments/ --dry-run

    # Save generated job scripts to a directory for inspection
    python submit_batch.py configs/experiments/ --scripts-dir slurm_scripts/

    # Redirect model cache to project storage (avoids filling $HOME)
    python submit_batch.py configs/experiments/ \\
        --account <proj> --hf-home /projects/<proj>/hf_cache

    # Submit a subset
    python submit_batch.py configs/experiments/03_qwen_1.5b_baseline.yaml \\
                           configs/experiments/05_qwen_7b_lora_r16.yaml

Config YAML slurm section
--------------------------
    slurm:
      time: "12:00:00"      # max 24:00:00 on Isambard AI
      nodes: 1              # always 1 for this framework
      gpus_per_node: 2      # GH200 GPUs requested (max 4 per node)
      cpus_per_gpu: 72      # 72 ARM Grace cores per GH200
      partition: workq      # default and only general partition
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path


# ── Isambard AI Phase 2 defaults ──────────────────────────────────────────────
ISAMBARD_PARTITION   = "workq"
ISAMBARD_MAX_HOURS   = 24
ISAMBARD_GPUS_PER_NODE = 4          # max GH200s available per node
ISAMBARD_CPUS_PER_GPU  = 72         # ARM Grace cores per GH200 NUMA domain
# Modules loaded automatically unless --no-default-modules is passed.
ISAMBARD_DEFAULT_MODULES = ["cuda/12.6", "brics/nccl", "gcc-native/12.3"]


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_yaml(path: Path) -> dict:
    try:
        import yaml  # type: ignore[import]
    except ImportError:
        sys.exit("PyYAML is required: pip install pyyaml")
    with open(path) as f:
        return yaml.safe_load(f) or {}


def find_configs(paths: list[str]) -> list[Path]:
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


def _parse_hours(time_str: str) -> float:
    """Return walltime as fractional hours from HH:MM:SS or D-HH:MM:SS."""
    try:
        if "-" in time_str:
            days_part, hms = time_str.split("-", 1)
            days = int(days_part)
        else:
            days, hms = 0, time_str
        parts = hms.split(":")
        h, m, s = (int(parts[0]), int(parts[1]), int(parts[2])) if len(parts) == 3 \
                  else (int(parts[0]), int(parts[1]), 0)
        return days * 24 + h + m / 60 + s / 3600
    except Exception:
        return 0.0


def make_job_script(
    config_path: Path,
    cfg: dict,
    *,
    project_dir: Path,
    log_dir: Path,
    partition_override: str | None,
    account: str | None,
    qos: str | None,
    mail_user: str | None,
    mail_type: str,
    hf_token_var: str,          # name of the env-var to forward (usually HF_TOKEN)
    hf_home: str | None,
    extra_modules: list[str],
    default_modules: list[str],
    venv_activate: str | None,
    output_dir_prefix: str | None,
    extra_args: str = "",
) -> str:
    slurm_cfg  = cfg.get("slurm", {})
    name       = cfg.get("name", config_path.stem)
    script     = cfg.get("script", "train_negotiation")

    # ── Slurm resource directives ─────────────────────────────────────────
    time_limit    = slurm_cfg.get("time", "12:00:00")
    nodes         = int(slurm_cfg.get("nodes", 1))
    gpus_per_node = int(slurm_cfg.get("gpus_per_node", 1))
    cpus_per_gpu  = int(slurm_cfg.get("cpus_per_gpu", ISAMBARD_CPUS_PER_GPU))
    partition     = partition_override or slurm_cfg.get("partition", ISAMBARD_PARTITION)

    # Cap at Isambard's 24-hour limit and warn.
    if _parse_hours(time_limit) > ISAMBARD_MAX_HOURS:
        print(
            f"[WARN] {name}: time={time_limit} exceeds Isambard max "
            f"{ISAMBARD_MAX_HOURS}h; capping at 24:00:00.",
            file=sys.stderr,
        )
        time_limit = "24:00:00"

    # Enforce single-node for multi-GPU jobs (framework uses per-agent device
    # assignment within a single process, not MPI across nodes).
    if nodes != 1:
        print(
            f"[WARN] {name}: nodes={nodes} — framework only supports single-node; "
            f"forcing nodes=1.",
            file=sys.stderr,
        )
        nodes = 1

    if gpus_per_node > ISAMBARD_GPUS_PER_NODE:
        print(
            f"[WARN] {name}: gpus_per_node={gpus_per_node} > {ISAMBARD_GPUS_PER_NODE} "
            f"(max per Isambard node); capping.",
            file=sys.stderr,
        )
        gpus_per_node = ISAMBARD_GPUS_PER_NODE

    out_log = log_dir / f"{name}_%j.out"
    err_log = log_dir / f"{name}_%j.err"

    # Isambard uses --gpus=N (not --gres=gpu:N).
    sbatch_lines = [
        f"#SBATCH --job-name={name}",
        f"#SBATCH --nodes={nodes}",
        f"#SBATCH --gpus={gpus_per_node}",
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

    # ── Module loads ──────────────────────────────────────────────────────
    all_modules = default_modules + extra_modules
    if all_modules:
        module_lines = ["module purge"] + [f"module load {m}" for m in all_modules]
        module_block = "\n".join(module_lines)
    else:
        module_block = "# (no modules loaded)"

    # ── HF_HOME for model cache ───────────────────────────────────────────
    # Default: project storage on Isambard AI (/lus/lfs1aip2 is not quota-limited).
    # Override with --hf-home if running elsewhere.
    _default_hf_home = "/lus/lfs1aip2/projects/a5l/egunn/hf_cache"
    hf_home_line = f'export HF_HOME="{hf_home or _default_hf_home}"'

    # ── Venv / uv activation ──────────────────────────────────────────────
    # When gpus_per_node > 1 we launch via torchrun for data-parallel training.
    # The trainer detects RANK/LOCAL_RANK/WORLD_SIZE from torchrun and shards
    # episodes across ranks with manual gradient all-reduce.
    use_ddp = gpus_per_node > 1
    if use_ddp:
        launch_prefix = f"torchrun --standalone --nnodes=1 --nproc_per_node={gpus_per_node}"
    else:
        launch_prefix = "python"
    if venv_activate:
        activate_block = f'source "{venv_activate}"'
        python_cmd = f'{launch_prefix} "{project_dir}/{script}.py" --config "{config_path.resolve()}"'
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
        # `uv run` runs a sub-command; for torchrun we run it via uv run too so
        # the PATH/venv is consistent.
        if use_ddp:
            python_cmd = (
                f'uv run --no-sync {launch_prefix} "{project_dir}/{script}.py" '
                f'--config "{config_path.resolve()}"'
            )
        else:
            python_cmd = (
                f'uv run --no-sync python "{project_dir}/{script}.py" '
                f'--config "{config_path.resolve()}"'
            )

    # ── Output-dir prefix remapping ───────────────────────────────────────
    # Allows redirecting run outputs to $PROJECTDIR or $SCRATCHDIR at submit
    # time without editing every config file.
    if output_dir_prefix:
        out_dir_cfg = cfg.get("output_dir", f"runs/{name}")
        # Strip any leading runs/ so it remaps cleanly.
        run_subdir = Path(out_dir_cfg).name if "/" in out_dir_cfg else out_dir_cfg
        output_dir_override = f'--output-dir "{output_dir_prefix}/{run_subdir}"'
        python_cmd += f" {output_dir_override}"

    if extra_args:
        python_cmd += f" {extra_args}"

    script_body = f"""\
#!/bin/bash -l
# ============================================================
# Auto-generated by submit_batch.py for Isambard AI Phase 2
# Config : {config_path.resolve()}
# Created: {datetime.now().isoformat(timespec='seconds')}
# ============================================================
#
# Hardware: GH200 Grace Hopper — 4 × 96 GB HBM3 GPU, ARM Grace CPU (aarch64)
# Docs   : https://docs.isambard.ac.uk

{sbatch_block}

set -euo pipefail

echo "================================================================"
echo "Job started  : $(date)"
echo "Job ID       : $SLURM_JOB_ID"
echo "Job name     : $SLURM_JOB_NAME"
echo "Node         : $(hostname -s)"
echo "GPUs         : $SLURM_GPUS"
echo "CPUs/GPU     : ${{SLURM_CPUS_PER_GPU:-N/A}}"
echo "Working dir  : {project_dir}"
echo "Config file  : {config_path.resolve()}"
echo "================================================================"

# ── SLURM environment ─────────────────────────────────────────────────────────
echo ""
echo "--- SLURM environment ---"
env | grep -E '^SLURM' | sort
echo ""

# ── GPU inventory (GH200 shows as H100 in nvidia-smi) ────────────────────────
if command -v nvidia-smi &>/dev/null; then
    echo "--- nvidia-smi ---"
    nvidia-smi --query-gpu=index,name,memory.total,driver_version,compute_mode \\
               --format=csv,noheader
    echo ""
fi

# ── Modules ───────────────────────────────────────────────────────────────────
{module_block}

# ── Environment setup ─────────────────────────────────────────────────────────
cd "{project_dir}"

# Model cache — keep large HF downloads off $HOME quota.
{hf_home_line}
mkdir -p "$HF_HOME"

# HuggingFace token (forwarded from submission environment; required for gated
# models like Llama).  Job will fail here with a clear message if not set.
# To set: export HF_TOKEN="hf_..." before running submit_batch.py.
if [ -z "${{HF_TOKEN:-}}" ]; then
    echo "[WARN] HF_TOKEN is not set — gated models (e.g. Llama) will fail to download."
else
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"   # legacy alias used by some tools
    echo "[INFO] HF_TOKEN is set (length ${{#HF_TOKEN}} chars)."
fi

# NCCL tuning for Slingshot 11 / GH200 NVLink topology.
# brics/nccl module sets NCCL_NET_PLUGIN=ofi; these complement it.
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=hsn             # use Slingshot high-speed NICs
export FI_CXI_ATS=0                       # disable address translation for CXI
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Per-job torch.compile / vLLM compile caches. The default location is on
# shared NFS ($HOME/.cache); when several jobs compile the same model
# concurrently they race on the artifact files and the reader hits
# "Bytes object is corrupted, checksum does not match" during vLLM
# profile_run, killing the job before training starts.
export VLLM_CACHE_ROOT="/tmp/vllm_cache_${{SLURM_JOB_ID:-$$}}"
export TORCHINDUCTOR_CACHE_DIR="/tmp/torchinductor_${{SLURM_JOB_ID:-$$}}"
mkdir -p "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR"
trap 'rm -rf "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR"' EXIT

# vLLM ZMQ IPC socket directory. If VLLM_RPC_BASE_PATH is unset OR exported
# empty, vLLM constructs `ipc:///{uuid}` which resolves relative to cwd and
# litters the project root with UUID-named socket files on every run.
export VLLM_RPC_BASE_PATH="${{TMPDIR:-/tmp}}"

# ── Pre-flight cleanup ────────────────────────────────────────────────────────
# A previous failed job on this node can leave behind orphaned python/torchrun/
# vLLM workers that still hold GPU memory and /dev/shm segments. Without this,
# the next job allocated to the same node OOMs before training even starts.
echo "--- Pre-flight cleanup ---"

# 1. Kill any stray training/inference processes belonging to this user.
#    `pkill -9 -u $USER -f <pat>` only matches our own processes, so this is
#    safe on shared nodes. Patterns cover torchrun, vLLM workers, and our
#    training entrypoints. `|| true` because pkill exits 1 when nothing matches.
for pat in "torchrun" "torch.distributed.run" "train_population.py" "train_negotiation.py" "train_concordia.py" "train.py" "vllm" "multiprocessing.spawn" "multiprocessing.resource_tracker"; do
    pkill -9 -u "$USER" -f "$pat" 2>/dev/null || true
done

# 2. Clear shared-memory segments left by torch DataLoader / NCCL / vLLM.
find /dev/shm -maxdepth 1 -user "$USER" -mmin +0 \( -name 'torch_*' -o -name 'nccl-*' -o -name 'cuda.*' -o -name 'vllm*' -o -name 'sem.*' -o -name 'pymp-*' \) -exec rm -rf {{}} + 2>/dev/null || true

# 3. Wait for GPU memory to actually drain. If the kernel still has dying
#    processes attached to a context, `nvidia-smi --query-gpu=memory.used`
#    will report non-zero for a few seconds. Abort if it never clears so the
#    job fails fast instead of OOMing inside torch.cuda.init.
if command -v nvidia-smi &>/dev/null; then
    for attempt in $(seq 1 30); do
        max_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
        # Treat <500 MiB as "clean" — driver itself uses ~0–100 MiB.
        if [ "${{max_used:-99999}}" -lt 500 ]; then
            echo "[INFO] GPU memory clean after ${{attempt}}s (max ${{max_used}} MiB used)."
            break
        fi
        if [ "$attempt" -eq 30 ]; then
            echo "[ERROR] GPU memory still busy after 30s (max ${{max_used}} MiB). Aborting." >&2
            nvidia-smi
            exit 1
        fi
        sleep 1
    done
fi
echo ""

# ── Python / venv ─────────────────────────────────────────────────────────────
{activate_block}

echo ""
echo "--- Python / package versions ---"
python --version 2>&1 || true
python -c "
import torch, platform
print(f'torch       {{torch.__version__}}')
print(f'cuda        {{torch.version.cuda}}')
print(f'gpu count   {{torch.cuda.device_count()}}')
print(f'arch        {{platform.machine()}}')
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f'  gpu{{i}}     {{p.name}} | {{p.total_memory/1024**3:.1f}} GiB')
" 2>&1 || true
python -c "import transformers; print(f'transformers {{transformers.__version__}}')" 2>&1 || true
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
        description="Submit MARLLLM configs as Slurm jobs on Isambard AI Phase 2.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "configs", nargs="+",
        help="YAML config files or directories containing them.",
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
        "--partition", default=None,
        help=f"Slurm partition (default: {ISAMBARD_PARTITION}).",
    )
    p.add_argument(
        "--qos", default=None,
        help="Slurm QOS (e.g. 32gpu_qos for Isambard AI).",
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
        "--hf-token-var", default="HF_TOKEN", metavar="VAR",
        help="Name of the shell variable holding the HuggingFace token "
             "(default: HF_TOKEN).  The value is forwarded to each job "
             "via --export so it is never written into job scripts.",
    )
    p.add_argument(
        "--hf-home", default=None, metavar="PATH",
        help="Path for the HuggingFace model cache (HF_HOME).  "
             "Defaults to /lus/lfs1aip2/projects/a5l/egunn/hf_cache "
             "(project storage, not quota-limited).  Override only if "
             "running on a different system.",
    )
    p.add_argument(
        "--module", action="append", dest="modules", default=[],
        metavar="MODULE",
        help="Extra `module load` line (repeat for multiple).",
    )
    p.add_argument(
        "--no-default-modules", action="store_true",
        help=f"Skip the default Isambard modules ({', '.join(ISAMBARD_DEFAULT_MODULES)}). "
             "Use if you have loaded them another way.",
    )
    p.add_argument(
        "--venv", default=None, metavar="PATH",
        help="Path to a venv activate script.  "
             "If omitted, auto-detects .venv/ or falls back to `uv run`.",
    )
    p.add_argument(
        "--output-dir-prefix", default=None, metavar="PATH",
        help="Prefix prepended to each config's output_dir, e.g. "
             "/projects/myproject/runs or $SCRATCHDIR.  "
             "Useful to redirect all checkpoints to project storage "
             "without editing every config file.",
    )
    p.add_argument(
        "--project-dir", default=None, metavar="PATH",
        help="Root directory of the MARLLLM project. "
             "Defaults to the directory containing this script.",
    )
    p.add_argument(
        "--log-dir", default=None, metavar="PATH",
        help="Directory for Slurm stdout/stderr logs. "
             "Defaults to <project-dir>/slurm_logs/.",
    )
    p.add_argument(
        "--scripts-dir", default=None, metavar="PATH",
        help="Save generated job scripts here instead of a temp dir. "
             "Useful for auditing exactly what was submitted.",
    )
    p.add_argument(
        "--extra-args", default="", metavar="ARGS",
        help="Extra CLI args to append to the train_population.py command in "
             "every generated job, e.g. --extra-args=\"--compile\". Quoted as "
             "a single string; split into argv tokens via the shell.",
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

    # ── Warnings ──────────────────────────────────────────────────────────
    if not args.account and not args.dry_run:
        print(
            "[WARN] --account not set.  Jobs on Isambard AI require a project "
            "account.  Specify --account <project_code> or jobs may be rejected.",
            file=sys.stderr,
        )
        print()

    hf_token_value = os.environ.get(args.hf_token_var, "")
    if not hf_token_value:
        print(
            f"[WARN] ${args.hf_token_var} is not set in the current environment.  "
            "Gated models (Llama etc.) will fail to download inside the job.  "
            f"Run: export {args.hf_token_var}='hf_...' before submitting.",
            file=sys.stderr,
        )
        print()

    default_modules = [] if args.no_default_modules else list(ISAMBARD_DEFAULT_MODULES)

    configs = find_configs(args.configs)
    if not configs:
        sys.exit("No YAML config files found.")

    print(f"Found {len(configs)} config(s). {'[DRY RUN] ' if args.dry_run else ''}Submitting...")
    print()

    submitted: list[tuple[str, str]] = []
    failed:    list[tuple[str, str]] = []

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
            qos=args.qos,
            mail_user=args.mail_user,
            mail_type=args.mail_type,
            hf_token_var=args.hf_token_var,
            hf_home=args.hf_home,
            extra_modules=args.modules,
            default_modules=default_modules,
            venv_activate=args.venv,
            output_dir_prefix=args.output_dir_prefix,
            extra_args=args.extra_args,
        )

        # ── Write job script ──────────────────────────────────────────────
        if scripts_dir:
            script_path = scripts_dir / f"{name}.sh"
            script_path.write_text(script_body)
            script_path.chmod(0o700)   # owner-only: script contains env-var refs
            tmp_path = script_path
            cleanup = False
        else:
            tmp = tempfile.NamedTemporaryFile(
                mode="w", suffix=".sh", prefix=f"marlllm_{name}_",
                delete=False,
            )
            tmp.write(script_body)
            tmp.close()
            tmp_path = Path(tmp.name)
            tmp_path.chmod(0o700)
            cleanup = True

        if args.dry_run:
            print(f"{'─'*60}")
            print(f"CONFIG  : {config_path}")
            print(f"SCRIPT  : {tmp_path}")
            # Show the sbatch command that would be run (token forwarded via --export).
            export_arg = (
                f"--export=ALL,{args.hf_token_var}=${args.hf_token_var}"
                if hf_token_value else "--export=ALL"
            )
            print(f"COMMAND : sbatch {export_arg} {tmp_path}")
            print()
            print(script_body)
            if cleanup:
                tmp_path.unlink(missing_ok=True)
            continue

        # ── Submit ────────────────────────────────────────────────────────
        # Forward HF_TOKEN via --export so it never touches the script file.
        export_arg = (
            f"ALL,{args.hf_token_var}={hf_token_value}"
            if hf_token_value else "ALL"
        )
        try:
            result = subprocess.run(
                ["sbatch", f"--export={export_arg}", str(tmp_path)],
                capture_output=True, text=True, check=True,
            )
            job_id = result.stdout.strip().split()[-1]
            submitted.append((name, job_id))
            print(f"  ✓  {name:<50}  job {job_id}")
        except subprocess.CalledProcessError as exc:
            reason = exc.stderr.strip() or exc.stdout.strip()
            failed.append((name, reason))
            print(f"  ✗  {name:<50}  FAILED: {reason}", file=sys.stderr)
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

        manifest_path = log_dir / f"submission_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(manifest_path, "w") as f:
            f.write(f"Submitted at : {datetime.now().isoformat()}\n")
            f.write(f"Account      : {args.account or '(not set)'}\n")
            f.write(f"HF_TOKEN set : {'yes' if hf_token_value else 'NO'}\n\n")
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
