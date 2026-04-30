"""
Distillation-ablation experiment runner.

Reads a single YAML config and executes Cells A, B, C, D, E, C0 across the
configured seeds. Each cell is run as a subprocess invocation of either
`dump_transcripts.py + train_sft.py` (Cell A) or `train_negotiation.py`
(every other cell). Per-cell `done.flag` files allow the runner to be
re-invoked safely; completed cells are skipped.

This intentionally keeps cells as plain subprocesses so a single hung cell
doesn't take down the whole sweep, and so SLURM users can stop and restart.

Cell-to-flags mapping
---------------------
  A   : SFT on transcripts (separate path).
  B   : --api-partner agent_1 --alpha-act 0 --alpha-val 0 --perception-agents agent_1
  C   : --api-partner agent_1
  D   : --api-partner agent_1 --alpha-perc 0
  E   : --api-partner agent_1 --alpha-act 0 --alpha-val 0
  C0  : (no api-partner; existing co-trained baseline)

Example config: see configs/distillation_ablation/example.yaml.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parent.parent


def _train_neg_args(cfg: dict, cell: str, seed: int, output_dir: Path,
                    transcripts_path: Path | None) -> list[str]:
    sm = cfg["small_model"]
    sp = cfg.get("strong_partners", {}) or {}
    env_cfg = cfg.get("environment", {}) or {}
    ccsm = cfg.get("ccsm", {}) or {}

    args = [
        sys.executable, str(REPO / "train_negotiation.py"),
        "--model", sm["base_checkpoint"],
        "--output-dir", str(output_dir),
        "--seed", str(seed),
        "--iters", str(env_cfg.get("iterations", 150)),
        "--rollouts", str(env_cfg.get("episodes_per_iteration", 8)),
        "--dialogue-turns", str(env_cfg.get("dialogue_turns", 10)),
        "--token-budget", str(env_cfg.get("token_budget", 64)),
        "--lr", str(sm.get("train_config", {}).get("lr", 1e-5)),
        "--grad-accum", str(sm.get("train_config", {}).get("grad_accum", 8)),
        "--kl-coef", str(ccsm.get("kl_coefficient", 0.0)),
        "--log-every", str(env_cfg.get("log_every", 10)),
        "--checkpoint-every", str(env_cfg.get("checkpoint_every", 50)),
    ]

    needs_api = cell in ("B", "C", "D", "E")
    if needs_api:
        args += [
            "--api-partner", "agent_1",
            "--api-provider", sp.get("provider", "anthropic"),
            "--api-model", sp.get("model", "claude-sonnet-4-6"),
        ]
        if sp.get("cache_dir"):
            args += ["--api-cache-dir", str(sp["cache_dir"])]
        if sp.get("budget_state"):
            args += ["--api-budget-state", str(sp["budget_state"])]
        if sp.get("budget_cap_usd") is not None:
            args += ["--api-budget-cap-usd", str(sp["budget_cap_usd"])]
        sampling = sp.get("sampling", {}) or {}
        if "temperature" in sampling:
            args += ["--api-temperature", str(sampling["temperature"])]
        if "max_tokens" in sampling:
            args += ["--api-max-tokens", str(sampling["max_tokens"])]

    if cell == "B":
        args += ["--alpha-act", "0", "--alpha-val", "0",
                 "--perception-agents", "agent_1"]
    elif cell == "D":
        args += ["--alpha-perc", "0"]
    elif cell == "E":
        args += ["--alpha-act", "0", "--alpha-val", "0"]
    return args


def run_cell(cfg: dict, cell: str, seed: int, root: Path) -> tuple[str, int]:
    cell_dir = root / f"cell_{cell}" / f"seed_{seed}"
    cell_dir.mkdir(parents=True, exist_ok=True)
    done = cell_dir / "done.flag"
    if done.exists():
        return cell, 0

    log_path = cell_dir / "run.log"
    print(f"[{time.strftime('%H:%M:%S')}] Running Cell {cell} seed={seed} → {cell_dir}")

    if cell == "A":
        # Two subprocesses: dump transcripts (cached across seeds via --cache-dir),
        # then SFT on the dumped transcripts.
        sp = cfg.get("strong_partners", {}) or {}
        env_cfg = cfg.get("environment", {}) or {}
        sm = cfg["small_model"]
        transcripts_path = root / "cell_A" / "transcripts.jsonl"
        transcripts_path.parent.mkdir(parents=True, exist_ok=True)
        if not transcripts_path.exists():
            dump_cmd = [
                sys.executable, str(REPO / "scripts" / "dump_transcripts.py"),
                "--provider", sp.get("provider", "anthropic"),
                "--model", sp.get("model", "claude-sonnet-4-6"),
                "--tokenizer", sm["base_checkpoint"],
                "--episodes", str(cfg.get("cell_a", {}).get("episodes", 200)),
                "--seed", str(seed),
                "--dialogue-turns", str(env_cfg.get("dialogue_turns", 10)),
                "--token-budget", str(env_cfg.get("token_budget", 128)),
                "--temperature", str((sp.get("sampling") or {}).get("temperature", 1.0)),
                "--max-tokens", str((sp.get("sampling") or {}).get("max_tokens", 256)),
                "--output", str(transcripts_path),
            ]
            if sp.get("cache_dir"):
                dump_cmd += ["--cache-dir", str(sp["cache_dir"])]
            if sp.get("budget_state"):
                dump_cmd += ["--budget-state", str(sp["budget_state"])]
            if sp.get("budget_cap_usd") is not None:
                dump_cmd += ["--budget-cap-usd", str(sp["budget_cap_usd"])]
            with log_path.open("w") as f:
                ret = subprocess.call(dump_cmd, stdout=f, stderr=subprocess.STDOUT)
            if ret != 0:
                return cell, ret

        sft_cmd = [
            sys.executable, str(REPO / "scripts" / "train_sft.py"),
            "--transcripts", str(transcripts_path),
            "--model", sm["base_checkpoint"],
            "--output-dir", str(cell_dir),
            "--epochs", str(cfg.get("cell_a", {}).get("epochs", 3)),
            "--batch-size", str(cfg.get("cell_a", {}).get("batch_size", 4)),
            "--lr", str(sm.get("train_config", {}).get("lr", 5e-6)),
            "--seed", str(seed),
        ]
        with log_path.open("a") as f:
            ret = subprocess.call(sft_cmd, stdout=f, stderr=subprocess.STDOUT)
    else:
        cmd = _train_neg_args(cfg, cell, seed, cell_dir, None)
        with log_path.open("w") as f:
            ret = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)

    if ret == 0:
        done.write_text(json.dumps({"finished_at": time.time()}))
    return cell, ret


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--cells", default=None,
                   help="Comma-separated cell IDs to run; default = config.ablation.cells.")
    p.add_argument("--seeds", default=None,
                   help="Comma-separated seeds to run; default = config.ablation.seeds.")
    p.add_argument("--invalidate-cache", action="store_true",
                   help="Delete API cache dir before running. Use with caution.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    root = Path(cfg["output_dir"])
    root.mkdir(parents=True, exist_ok=True)

    if args.invalidate_cache:
        cache_dir = (cfg.get("strong_partners") or {}).get("cache_dir")
        if cache_dir and Path(cache_dir).exists():
            import shutil
            shutil.rmtree(cache_dir)

    cells = (args.cells.split(",") if args.cells
             else cfg.get("ablation", {}).get("cells", ["A", "B", "C", "D", "E", "C0"]))
    seeds = ([int(s) for s in args.seeds.split(",")] if args.seeds
             else cfg.get("ablation", {}).get("seeds", [0]))

    manifest = {
        "config_path": args.config,
        "started_at": time.time(),
        "cells": cells,
        "seeds": seeds,
        "results": [],
    }
    for cell in cells:
        for seed in seeds:
            c, ret = run_cell(cfg, cell, seed, root)
            manifest["results"].append({"cell": c, "seed": seed, "exit_code": ret})
            (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
            if ret != 0:
                print(f"  ! Cell {cell} seed={seed} failed with exit {ret}")

    print("Ablation runner complete.")


if __name__ == "__main__":
    main()
