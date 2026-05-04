#!/usr/bin/env python3
"""
Convert forum-format trace JSONs into environment-perspective text files.

The output shows the forum thread in posting order, with each agent's private
thinking (extracted from their context) shown in an indented block above the
post they produced.

Usage
-----
    python scripts/format_forum_traces.py <path>... [--out DIR] [--suffix SFX]

    <path>    one or more iter_*.json files, or directories containing them.
              Directories are scanned for iter_*.json (non-recursive).
    --out     output directory (default: alongside each input file).
    --suffix  output suffix (default: .forum.txt).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import textwrap
from pathlib import Path

_ASSISTANT_TURN_RE = re.compile(
    r"<\|im_start\|>assistant\n(.*?)<\|im_end\|>", re.DOTALL
)
_THINK_RE = re.compile(r"\s*<think>(.*?)</think>", re.DOTALL)

RULE = "═" * 80
SUBRULE = "─" * 80


def extract_thinkings(context_text: str) -> list[str]:
    out: list[str] = []
    for turn in _ASSISTANT_TURN_RE.findall(context_text or ""):
        m = _THINK_RE.match(turn)
        out.append(m.group(1).strip() if m else "")
    return out


def indent_block(text: str, prefix: str) -> str:
    if not text:
        return ""
    return textwrap.indent(text.rstrip(), prefix)


def format_episode(rec: dict, ep_idx: int) -> str:
    env_t = rec.get("env_trace") or {}
    agents_raw = rec.get("agents") or {}
    thread = list(env_t.get("thread") or [])
    ep_agents = list(env_t.get("agents") or list(agents_raw.keys()))

    thinkings_by_agent: dict[str, list[str]] = {}
    for aid, at in agents_raw.items():
        ctx = at.get("context_text", "") if isinstance(at, dict) else str(at)
        thinkings_by_agent[aid] = extract_thinkings(ctx)

    lines: list[str] = []
    lines.append(RULE)
    lines.append(f"EPISODE {rec.get('episode', ep_idx)}")
    meta_parts = []
    for k, v in env_t.items():
        if k in ("thread", "agents"):
            continue
        meta_parts.append(f"{k}={v}")
    if meta_parts:
        lines.append("  " + "  ".join(meta_parts))
    lines.append("  participants: " + ", ".join(ep_agents))
    lines.append(RULE)
    lines.append("")

    speaker_counts: dict[str, int] = {}
    for post in thread:
        speaker = post.get("speaker", "(unknown)")
        idx = post.get("post_index", "?")
        text = post.get("text", "") or ""
        n = speaker_counts.get(speaker, 0)
        speaker_counts[speaker] = n + 1
        thinks = thinkings_by_agent.get(speaker, [])
        thinking = thinks[n] if n < len(thinks) else ""

        lines.append(f"#{idx:>3}  {speaker}")
        lines.append(SUBRULE)
        if thinking:
            lines.append("  ┊ thinking:")
            lines.append(indent_block(thinking, "  ┊   "))
            lines.append("")
        lines.append(indent_block(text, "    "))
        lines.append("")

    return "\n".join(lines)


def format_trace(records: list, source: Path) -> str:
    header = [
        RULE,
        f"FORUM TRACE  {source.name}",
        f"  episodes: {len(records)}",
        RULE,
        "",
    ]
    body = [format_episode(rec, i) for i, rec in enumerate(records)]
    return "\n".join(header) + "\n".join(body)


def convert_file(path: Path, out_dir: Path | None, suffix: str) -> Path:
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected list-of-episodes JSON (forum format)")
    formatted = format_trace(data, path)
    out_path = (out_dir or path.parent) / (path.stem + suffix)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(formatted)
    return out_path


def collect_inputs(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        if p.is_dir():
            out.extend(sorted(p.glob("iter_*.json")))
        elif p.is_file():
            out.append(p)
        else:
            print(f"warning: skipping {p} (not found)", file=sys.stderr)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("paths", nargs="+", type=Path,
                    help="iter_*.json files or directories containing them")
    ap.add_argument("--out", type=Path, default=None,
                    help="output directory (default: alongside each input)")
    ap.add_argument("--suffix", default=".forum.txt",
                    help="output filename suffix (default: .forum.txt)")
    args = ap.parse_args()

    inputs = collect_inputs(args.paths)
    if not inputs:
        print("no input files found", file=sys.stderr)
        sys.exit(1)

    for p in inputs:
        try:
            out = convert_file(p, args.out, args.suffix)
            print(f"wrote {out}")
        except Exception as e:
            print(f"error: {p}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
