"""
Pretty-print eval.trajectories.json produced by evaluate_negotiation.py.

The trajectory file stores, per agent per episode:
  * ``messages``   — the chat-template message list (role + content per turn)
  * ``rendered``   — the chat-template-rendered string (with all special
                     tokens preserved) that the model actually saw on the
                     final turn, ending with the assistant primer

This script renders the conversation for inspection. By default it writes
``eval.trajectories.txt`` next to the input.

Usage
-----
    uv run python scripts/view_trajectories.py PATH/eval.trajectories.json
    uv run python scripts/view_trajectories.py PATH/eval.trajectories.json --episode 3
    uv run python scripts/view_trajectories.py PATH/eval.trajectories.json --rendered
    uv run python scripts/view_trajectories.py PATH/eval.trajectories.json --stdout
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def render_episode(ep: dict, show_rendered: bool) -> str:
    lines: list[str] = []
    lines.append("=" * 80)
    lines.append(f"EPISODE {ep['episode']}")
    out = ep["outcome"]
    lines.append(
        f"  deal={out['deal']}  score_a={out['score_a']}  score_b={out['score_b']}"
    )
    lines.append(
        f"  items={out['items']}  values_a={out['values_a']}  values_b={out['values_b']}"
    )
    lines.append("")

    for aid, rec in ep["agents"].items():
        lines.append("-" * 80)
        lines.append(f"AGENT {aid}  ({len(rec['messages'])} messages)")
        lines.append("-" * 80)
        if show_rendered:
            # Raw chat-template view: shows every <|im_start|> / <|im_end|>
            # exactly as the model saw it. Useful for verifying tokeniser
            # behaviour and template structure.
            lines.append(rec["rendered"])
        else:
            # Pretty per-message view.
            for msg in rec["messages"]:
                lines.append(f"[{msg['role']}]")
                lines.append(msg["content"])
                lines.append("")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("path", type=Path, help="Path to eval.trajectories.json")
    p.add_argument("--episode", type=int, default=None,
                   help="Only render this episode index (default: all).")
    p.add_argument("--rendered", action="store_true",
                   help="Show the chat-template-rendered string with all "
                        "special tokens preserved, instead of the pretty "
                        "per-message view.")
    p.add_argument("--output", type=Path, default=None,
                   help="Output file path. Default: same name as input with "
                        ".txt suffix.")
    p.add_argument("--stdout", action="store_true",
                   help="Print to stdout instead of writing a file.")
    args = p.parse_args()

    episodes = json.loads(args.path.read_text())
    if args.episode is not None:
        episodes = [ep for ep in episodes if ep["episode"] == args.episode]
        if not episodes:
            print(f"No episode {args.episode} in {args.path}", file=sys.stderr)
            sys.exit(1)

    rendered = "\n".join(render_episode(ep, args.rendered) for ep in episodes)

    if args.stdout:
        print(rendered)
        return

    out_path = args.output or args.path.with_suffix(".txt")
    out_path.write_text(rendered)
    print(f"Wrote {len(episodes)} episode(s) → {out_path}")


if __name__ == "__main__":
    main()
