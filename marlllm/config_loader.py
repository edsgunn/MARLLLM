"""
Utilities for loading experiment configs from YAML files and logging system information.

Config file format (YAML):
  name: "experiment_name"
  description: "What this tests"
  script: "train_negotiation"   # "train" or "train_negotiation"

  slurm:
    time: "12:00:00"            # wall-clock limit (HH:MM:SS)
    nodes: 1
    gpus_per_node: 2
    cpus_per_gpu: 8
    mem: "80G"
    partition: "gpu"

  # Training args — key names use underscores; dashes also accepted.
  # These map directly to --arg-name CLI flags.
  model: "Qwen/Qwen2.5-1.5B"
  iters: 500
  rollouts: 8
  lr: 3.0e-5
  ...

Usage in an argparse-based training script:
  from marlllm.config_loader import apply_config_defaults, log_system_info

  def parse_args():
      pre = argparse.ArgumentParser(add_help=False)
      pre.add_argument("--config", default=None)
      pre_args, _ = pre.parse_known_args()

      p = argparse.ArgumentParser(...)
      p.add_argument("--config", default=None, ...)
      # ... rest of args ...
      if pre_args.config:
          apply_config_defaults(p, pre_args.config)
      return p.parse_args()
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from pathlib import Path


# Keys that are metadata / slurm directives — not passed to the training script.
_SKIP_KEYS = {"name", "description", "script", "slurm"}


def load_yaml(path: str) -> dict:
    """Load a YAML config file, returning its full dict."""
    try:
        import yaml  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required for config file support. "
            "Install it with: pip install pyyaml"
        ) from exc
    with open(path) as f:
        return yaml.safe_load(f) or {}


def apply_config_defaults(parser, config_path: str) -> None:
    """
    Set argparse defaults from a YAML config file.

    Call this *before* parser.parse_args() so that explicit CLI flags still
    override the config file values.

    Only keys not in _SKIP_KEYS are applied; key names are normalised from
    dashes to underscores to match argparse dest names.
    """
    cfg = load_yaml(config_path)
    training_args = {
        k.replace("-", "_"): v
        for k, v in cfg.items()
        if k not in _SKIP_KEYS
    }
    # Boolean flags (store_true actions) stored as false in YAML should be
    # skipped so the argparse default of False is preserved.
    filtered = {k: v for k, v in training_args.items() if v is not None}
    parser.set_defaults(**filtered)


def log_system_info(output_dir: str, config_path: str | None = None) -> None:
    """
    Write a system_info.json snapshot to *output_dir*.

    Captures:
      - hostname, platform, Python version
      - Torch version + CUDA availability and GPU inventory
      - Key installed package versions (transformers, peft)
      - All SLURM_* environment variables
      - Git HEAD commit hash (if inside a git repo)
      - Path to the config file used (and a copy of it)
    """
    import torch  # noqa: PLC0415 — deferred to avoid slow import at module level

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── GPU inventory ──────────────────────────────────────────────────────
    gpu_info: list[dict] = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpu_info.append({
                "index": i,
                "name": props.name,
                "total_memory_gb": round(props.total_memory / 1024**3, 2),
                "major": props.major,
                "minor": props.minor,
                "multi_processor_count": props.multi_processor_count,
            })

    # ── Package versions ───────────────────────────────────────────────────
    def _pkg_version(name: str) -> str:
        try:
            import importlib.metadata
            return importlib.metadata.version(name)
        except Exception:
            return "unknown"

    # ── Git commit ─────────────────────────────────────────────────────────
    def _git_hash() -> str:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=5,
            )
            return result.stdout.strip() if result.returncode == 0 else "unknown"
        except Exception:
            return "unknown"

    def _git_dirty() -> bool:
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True, text=True, timeout=5,
            )
            return bool(result.stdout.strip()) if result.returncode == 0 else False
        except Exception:
            return False

    # ── SLURM environment ──────────────────────────────────────────────────
    slurm_env = {k: v for k, v in os.environ.items() if k.startswith("SLURM")}

    info = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "gpus": gpu_info,
        "packages": {
            "transformers": _pkg_version("transformers"),
            "peft": _pkg_version("peft"),
            "pyyaml": _pkg_version("pyyaml"),
            "pettingzoo": _pkg_version("pettingzoo"),
        },
        "git_commit": _git_hash(),
        "git_dirty": _git_dirty(),
        "slurm_env": slurm_env,
        "config_file": str(Path(config_path).resolve()) if config_path else None,
    }

    # Write JSON
    with open(out / "system_info.json", "w") as f:
        json.dump(info, f, indent=2)

    # Copy the config file into the output dir for full reproducibility
    if config_path:
        dest = out / "experiment_config.yaml"
        if not dest.exists():
            shutil.copy2(config_path, dest)
