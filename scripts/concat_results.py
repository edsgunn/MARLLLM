#!/usr/bin/env python3
"""
Concatenate all results files from a suite of experiments into a single text file.

Includes: experiment_config.yaml, config.json, system_info.json, train.log,
          traces/*.txt, and periodic metric snapshots from metrics.jsonl.
Excludes: checkpoints/, raw metrics.jsonl (replaced by snapshots)

Usage:
    python scripts/concat_results.py runs/concordia_experiments
    python scripts/concat_results.py runs/concordia_experiments -o report.txt
    python scripts/concat_results.py runs/experiments --no-traces
    python scripts/concat_results.py runs/concordia_experiments --exp 01 12
    python scripts/concat_results.py runs/concordia_experiments -n 10
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_AGENT_KEYS = ["mean_return", "entropy", "act_loss", "value_loss", "kl"]

# Matches the token-id array that follows every OBS/ACT line in traces.
# e.g.  "  ids=[1234 5678 ...]"  or  "  ids=[1234]"
_IDS_RE = re.compile(r"\s+ids=\[\d[\d\s]*\]")


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _header(text: str, width: int = 72) -> str:
    bar = "=" * width
    return f"\n{bar}\n{text}\n{bar}\n"


def _subheader(text: str) -> str:
    return f"\n--- {text} ---\n"


def _fmt(v: float) -> str:
    if abs(v) >= 1000:
        return f"{v:.0f}"
    if abs(v) >= 1:
        return f"{v:.3f}"
    return f"{v:.5f}"


# ---------------------------------------------------------------------------
# JSON pretty-printer (config.json, system_info.json)
# ---------------------------------------------------------------------------

def _format_json_file(path: Path) -> str:
    try:
        data = json.loads(path.read_text(errors="replace"))
    except json.JSONDecodeError:
        return path.read_text(errors="replace")
    return _render_dict(data, indent=0)


def _render_dict(obj, indent: int) -> str:
    pad = "  " * indent
    if isinstance(obj, dict):
        if not obj:
            return "{}"
        lines = []
        for k, v in obj.items():
            rendered = _render_dict(v, indent + 1)
            if isinstance(v, (dict, list)) and rendered.count("\n") > 0:
                lines.append(f"{pad}  {k}:\n{rendered}")
            else:
                lines.append(f"{pad}  {k}: {rendered}")
        return "\n".join(lines)
    elif isinstance(obj, list):
        if not obj:
            return "[]"
        # Keep short flat lists inline
        if len(obj) <= 6 and all(not isinstance(x, (dict, list)) for x in obj):
            return "[" + ", ".join(str(x) for x in obj) + "]"
        lines = [_render_dict(x, indent + 1) for x in obj]
        pad2 = "  " * (indent + 1)
        return "\n".join(f"{pad2}- {l.lstrip()}" for l in lines)
    else:
        return str(obj)


# ---------------------------------------------------------------------------
# Trace cleaner
# ---------------------------------------------------------------------------

def _clean_trace(raw: str) -> str:
    """Strip token id arrays and de-duplicate accumulated OBS history."""
    lines = raw.splitlines()
    out: list[str] = []
    prev_obs_segments: list[str] = []

    for line in lines:
        # Strip ids=[...] noise
        line = _IDS_RE.sub("", line).rstrip()

        # OBS lines contain the full accumulated conversation — show only new content
        obs_match = re.match(r"^(\[OBS\s*\])\s+'(.*)'$", line, re.DOTALL)
        if obs_match:
            tag = obs_match.group(1)
            content = obs_match.group(2).replace("\\n", "\n")
            # Split into segments by agent label or newline
            segments = [s for s in re.split(r"\n", content) if s.strip()]
            # Find segments not seen in previous OBS
            new_segments = [s for s in segments if s not in prev_obs_segments]
            prev_obs_segments = segments
            if new_segments:
                out.append(f"{tag}")
                for s in new_segments:
                    out.append(f"  {s}")
            continue

        # ACT lines: strip surrounding quotes if present
        act_match = re.match(r"^(\[ACT\s*\])\s+'(.*)'$", line, re.DOTALL)
        if act_match:
            tag = act_match.group(1)
            content = act_match.group(2).replace("\\n", "\n").strip()
            out.append(f"{tag}  {content}")
            continue

        out.append(line)

    return "\n".join(out)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _format_metrics_row(row: dict) -> str:
    it = int(row.get("iteration", 0))
    total_loss = row.get("total_loss", float("nan"))
    wall = row.get("wall_time", 0)
    lines = [f"iter {it:>5}  total_loss={_fmt(total_loss)}  wall={wall:.0f}s"]
    i = 0
    while True:
        prefix = f"agent_{i}/"
        if not any(k.startswith(prefix) for k in row):
            break
        vals = {k.removeprefix(prefix): v for k, v in row.items() if k.startswith(prefix)}
        fields = "  ".join(f"{k}={_fmt(vals[k])}" for k in _AGENT_KEYS if k in vals)
        lines.append(f"  [agent_{i}]  {fields}")
        i += 1
    return "\n".join(lines)


def _periodic_rows(path: Path, n: int) -> tuple[list[dict], int]:
    rows = [json.loads(l) for l in path.read_text(errors="replace").splitlines() if l.strip()]
    total = len(rows)
    if total == 0:
        return [], 0
    if total <= n:
        return rows, total
    step = (total - 1) / (n - 1)
    indices = sorted({0, total - 1} | {round(i * step) for i in range(n)})
    return [rows[i] for i in indices], total


def _format_metrics_section(exp_dir: Path, n: int) -> str:
    path = exp_dir / "metrics.jsonl"
    if not path.exists():
        return _subheader("metrics") + "  (not found)\n"
    rows, total = _periodic_rows(path, n)
    if total == 0:
        return _subheader("metrics") + "  (empty)\n"
    parts = [_subheader(f"metrics  ({total} iters, {len(rows)} snapshots shown)")]
    for row in rows:
        parts.append(_format_metrics_row(row))
        parts.append("")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Per-experiment assembly
# ---------------------------------------------------------------------------

def collect_experiment(exp_dir: Path, include_traces: bool, n_snapshots: int) -> str:
    parts: list[str] = [_header(f"EXPERIMENT: {exp_dir.name}")]

    for fname in ("experiment_config.yaml", "train.log"):
        fpath = exp_dir / fname
        if fpath.exists():
            parts.append(_subheader(fname))
            parts.append(fpath.read_text(errors="replace").rstrip())

    for fname in ("config.json", "system_info.json"):
        fpath = exp_dir / fname
        if fpath.exists():
            parts.append(_subheader(fname))
            parts.append(_format_json_file(fpath))

    parts.append(_format_metrics_section(exp_dir, n_snapshots))

    if include_traces:
        traces_dir = exp_dir / "traces"
        if traces_dir.is_dir():
            all_traces = sorted(traces_dir.glob("*.txt"))
            if all_traces:
                total = len(all_traces)
                if total <= n_snapshots:
                    selected = all_traces
                else:
                    step = (total - 1) / (n_snapshots - 1)
                    indices = sorted({0, total - 1} | {round(i * step) for i in range(n_snapshots)})
                    selected = [all_traces[i] for i in indices]
                parts.append(_subheader(f"traces  ({total} files, {len(selected)} shown)"))
                for tf in selected:
                    parts.append(_subheader(tf.name))
                    parts.append(_clean_trace(tf.read_text(errors="replace")))

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite_dir", type=Path)
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument("-n", "--snapshots", type=int, default=6,
                        help="Snapshots for both metrics and traces (default: 6)")
    parser.add_argument("--no-traces", action="store_true")
    parser.add_argument("--exp", nargs="+", metavar="PREFIX")
    args = parser.parse_args()

    suite_dir = args.suite_dir.resolve()
    if not suite_dir.is_dir():
        sys.exit(f"Error: {suite_dir} is not a directory")

    exp_dirs = sorted(d for d in suite_dir.iterdir() if d.is_dir() and d.name != "checkpoints")
    if args.exp:
        exp_dirs = [d for d in exp_dirs if any(d.name.startswith(p) for p in args.exp)]
    if not exp_dirs:
        sys.exit("No experiment directories found")

    output = args.output or suite_dir.parent / f"{suite_dir.name}_results.txt"

    sections = [
        _header(f"RESULTS: {suite_dir.name}  ({len(exp_dirs)} experiments)"),
        f"Suite  : {suite_dir}\n"
        f"Metrics: {args.snapshots} snapshots per experiment\n"
        f"Traces : {'included' if not args.no_traces else 'omitted'}\n",
    ]
    for exp_dir in exp_dirs:
        sections.append(collect_experiment(exp_dir, not args.no_traces, args.snapshots))

    text = "\n".join(sections)
    output.write_text(text, encoding="utf-8")
    print(f"Wrote {len(text):,} chars ({output.stat().st_size / 1024:.0f} KB) → {output}")


if __name__ == "__main__":
    main()
