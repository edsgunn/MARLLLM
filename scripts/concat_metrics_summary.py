#!/usr/bin/env python3
"""
Compact results summary: config + periodic metric snapshots per experiment.

Usage:
    python scripts/concat_metrics_summary.py runs/concordia_experiments
    python scripts/concat_metrics_summary.py runs/experiments -n 8 -o summary.txt
    python scripts/concat_metrics_summary.py runs/concordia_experiments --exp 09 12
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Metrics to show per agent (others are omitted for brevity)
_AGENT_KEYS = ["mean_return", "entropy", "act_loss", "value_loss", "kl"]
_GLOBAL_KEYS = ["total_loss", "iteration", "wall_time"]


def _fmt(v: float) -> str:
    if abs(v) >= 1000:
        return f"{v:.0f}"
    if abs(v) >= 1:
        return f"{v:.3f}"
    return f"{v:.5f}"


def _summarise_row(row: dict) -> str:
    """Format one metrics row as a compact single line."""
    it = int(row.get("iteration", 0))
    total_loss = row.get("total_loss", float("nan"))
    wall = row.get("wall_time", 0)

    # Collect per-agent return + entropy
    agent_parts = []
    i = 0
    while True:
        prefix = f"agent_{i}/"
        if not any(k.startswith(prefix) for k in row):
            break
        vals = {k.removeprefix(prefix): v for k, v in row.items() if k.startswith(prefix)}
        fields = "  ".join(
            f"{k}={_fmt(vals[k])}" for k in _AGENT_KEYS if k in vals
        )
        agent_parts.append(f"  [{prefix.rstrip('/')}]  {fields}")
        i += 1

    lines = [f"iter {it:>5}  total_loss={_fmt(total_loss)}  wall={wall:.0f}s"]
    lines += agent_parts
    return "\n".join(lines)


def _periodic_rows(lines: list[str], n: int) -> list[dict]:
    """Pick n evenly-spaced rows (always including first and last)."""
    rows = [json.loads(l) for l in lines if l.strip()]
    if len(rows) <= n:
        return rows
    indices = {0, len(rows) - 1}
    step = (len(rows) - 1) / (n - 1)
    indices |= {round(i * step) for i in range(n)}
    return [rows[i] for i in sorted(indices)]


def collect_experiment(exp_dir: Path, n_snapshots: int) -> str:
    parts: list[str] = [f"\n{'=' * 68}", f"  {exp_dir.name}", f"{'=' * 68}"]

    # One-line config summary
    config_path = exp_dir / "experiment_config.yaml"
    if not config_path.exists():
        config_path = exp_dir / "config.json"
    if config_path.exists():
        text = config_path.read_text(errors="replace")
        # Pull out a handful of useful fields heuristically
        interesting = []
        for keyword in ("model:", "scenario:", "lr:", "lora_r:", "rollouts:",
                        "kl_coef:", "iters:", "num_turns:", "agents:"):
            for line in text.splitlines():
                if line.strip().startswith(keyword):
                    interesting.append(line.strip())
                    break
        if interesting:
            parts.append("config: " + "  |  ".join(interesting))

    # Periodic metric snapshots
    metrics_path = exp_dir / "metrics.jsonl"
    if not metrics_path.exists():
        parts.append("  (no metrics.jsonl)")
        return "\n".join(parts)

    raw_lines = metrics_path.read_text(errors="replace").splitlines()
    if not raw_lines:
        parts.append("  (metrics.jsonl is empty)")
        return "\n".join(parts)

    rows = _periodic_rows(raw_lines, n_snapshots)
    parts.append(f"\nmetrics  ({len(raw_lines)} total iters, showing {len(rows)} snapshots)\n")
    for row in rows:
        parts.append(_summarise_row(row))
        parts.append("")

    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite_dir", type=Path)
    parser.add_argument(
        "-n", "--snapshots", type=int, default=6,
        help="Number of evenly-spaced metric snapshots per experiment (default: 6)",
    )
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument(
        "--exp", nargs="+", metavar="PREFIX",
        help="Only include experiments matching these directory prefixes",
    )
    args = parser.parse_args()

    suite_dir = args.suite_dir.resolve()
    if not suite_dir.is_dir():
        sys.exit(f"Error: {suite_dir} is not a directory")

    exp_dirs = sorted(d for d in suite_dir.iterdir() if d.is_dir() and d.name != "checkpoints")
    if args.exp:
        exp_dirs = [d for d in exp_dirs if any(d.name.startswith(p) for p in args.exp)]
    if not exp_dirs:
        sys.exit("No experiment directories found")

    output = args.output or suite_dir.parent / f"{suite_dir.name}_metrics_summary.txt"

    header = (
        f"METRICS SUMMARY: {suite_dir.name}  "
        f"({len(exp_dirs)} experiments, {args.snapshots} snapshots each)\n"
        f"Suite: {suite_dir}\n"
    )
    body = "\n".join(collect_experiment(d, args.snapshots) for d in exp_dirs)
    text = header + body + "\n"

    output.write_text(text, encoding="utf-8")
    print(f"Wrote {len(text):,} chars ({output.stat().st_size / 1024:.0f} KB) → {output}")


if __name__ == "__main__":
    main()
