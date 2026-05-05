#!/usr/bin/env python3
"""
Format checkpoint traces from a run into a printable HTML document.

Walks all `ckpt_*/traces.json` under a run's traces directory, extracts the
environment-perspective forum thread for each episode (with the per-speaker
private <think>...</think> reasoning recovered from the agent context), and
writes one page per episode. Print to PDF / paper from the browser.

Usage
-----
    python scripts/format_ckpt_traces_for_print.py RUN_DIR [-o OUT.html]
    python scripts/format_ckpt_traces_for_print.py RUN_DIR/traces/ckpt_000025 [-o OUT.html]

RUN_DIR may be either an experiment root (containing `traces/ckpt_*/`) or a
single ckpt directory. With no -o, writes alongside the input.
"""
from __future__ import annotations

import argparse
import json
import re
from html import escape
from pathlib import Path

_ASSISTANT_TURN_RE = re.compile(r"<\|im_start\|>assistant\n(.*?)<\|im_end\|>", re.DOTALL)
_THINK_RE = re.compile(r"\s*<think>(.*?)</think>", re.DOTALL)


def _extract_thinkings(context_text: str) -> list[str]:
    out: list[str] = []
    for turn in _ASSISTANT_TURN_RE.findall(context_text or ""):
        m = _THINK_RE.match(turn)
        out.append(m.group(1).strip() if m else "")
    return out


def _iter_num(name: str) -> int | None:
    m = re.search(r"(\d+)", name or "")
    return int(m.group(1)) if m else None


def _find_ckpt_dirs(path: Path) -> list[Path]:
    if (path / "traces.json").exists() and path.name.startswith("ckpt_"):
        return [path]
    traces_dir = path / "traces" if (path / "traces").is_dir() else path
    return sorted(d for d in traces_dir.iterdir()
                  if d.is_dir() and d.name.startswith("ckpt_") and (d / "traces.json").exists())


def _render_episode_page(*, run_name: str, ckpt_name: str, ckpt_iter: int | None,
                         episode_idx: int, record: dict) -> str:
    env_t = record.get("env_trace") or {}
    agents_raw = record.get("agents") or {}

    # Recover thinkings per speaker, in order of their posts.
    thinkings_by_agent: dict[str, list[str]] = {}
    for aid, at in agents_raw.items():
        ctx_text = at.get("context_text", "") if isinstance(at, dict) else str(at)
        thinkings_by_agent[aid] = _extract_thinkings(ctx_text)

    thread = sorted(env_t.get("thread") or [], key=lambda p: p.get("post_index", 0))
    speaker_counts: dict[str, int] = {}
    posts_html: list[str] = []
    for post in thread:
        speaker = post.get("speaker") or "(unknown)"
        n = speaker_counts.get(speaker, 0)
        speaker_counts[speaker] = n + 1
        thinks = thinkings_by_agent.get(speaker, [])
        thinking = thinks[n] if n < len(thinks) else ""
        idx = post.get("post_index", "?")
        text = post.get("text") or ""
        think_block = (
            f'<div class="think"><div class="think-lbl">thinking</div>'
            f'<div class="think-body">{escape(thinking)}</div></div>'
            if thinking else ""
        )
        posts_html.append(
            f'<div class="post">'
            f'<div class="post-hdr">#{escape(str(idx))} &middot; {escape(speaker)}</div>'
            f'{think_block}'
            f'<div class="post-body">{escape(text)}</div>'
            f'</div>'
        )

    meta_items = [
        ("run", run_name),
        ("checkpoint", ckpt_name),
        ("ckpt iter", ckpt_iter if ckpt_iter is not None else "?"),
        ("episode", record.get("episode", episode_idx)),
        ("env", env_t.get("env", "?")),
        ("pairing", env_t.get("pairing", "?")),
        ("post_order", env_t.get("post_order", "?")),
        ("posts", f'{env_t.get("completed_posts", "?")}/{env_t.get("max_posts", "?")}'),
    ]
    meta_html = "".join(
        f'<div class="meta-item"><span class="k">{escape(str(k))}</span>'
        f'<span class="v">{escape(str(v))}</span></div>'
        for k, v in meta_items
    )

    return (
        '<section class="page">'
        f'<header class="page-hdr"><h1>{escape(run_name)} &mdash; {escape(ckpt_name)} &mdash; episode {escape(str(record.get("episode", episode_idx)))}</h1>'
        f'<div class="meta">{meta_html}</div></header>'
        f'<div class="thread">{"".join(posts_html)}</div>'
        '</section>'
    )


_CSS = """
@page { size: A4; margin: 14mm 12mm; }
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; }
body { font-family: 'Helvetica Neue', Arial, sans-serif; color: #111; font-size: 10.5pt; line-height: 1.4; }
.page { page-break-after: always; padding: 6pt 0; }
.page:last-of-type { page-break-after: auto; }
.page-hdr { border-bottom: 1.5pt solid #222; margin-bottom: 8pt; padding-bottom: 4pt; }
.page-hdr h1 { font-size: 13pt; margin: 0 0 4pt; font-weight: 700; }
.meta { display: flex; flex-wrap: wrap; gap: 4pt 12pt; font-size: 8.5pt; color: #333; }
.meta-item .k { color: #777; margin-right: 3pt; text-transform: uppercase; letter-spacing: 0.4pt; font-size: 7.5pt; }
.meta-item .v { font-weight: 600; }
.thread { display: flex; flex-direction: column; gap: 6pt; }
.post { border: 0.5pt solid #bbb; border-radius: 3pt; overflow: hidden; break-inside: avoid; }
.post-hdr { background: #eee; padding: 2pt 6pt; font-weight: 700; font-size: 9.5pt; }
.post-body { padding: 5pt 7pt; white-space: pre-wrap; word-wrap: break-word; font-family: 'Georgia', serif; }
.think { background: #fafafa; border-bottom: 0.5pt dashed #ccc; padding: 3pt 7pt; }
.think-lbl { font-size: 7pt; text-transform: uppercase; letter-spacing: 0.5pt; color: #888; margin-bottom: 2pt; }
.think-body { font-style: italic; color: #555; font-size: 9pt; white-space: pre-wrap; word-wrap: break-word; font-family: 'Georgia', serif; }
"""


def build_html(run_name: str, ckpt_dirs: list[Path]) -> str:
    pages: list[str] = []
    for ckpt_dir in ckpt_dirs:
        ckpt_name = ckpt_dir.name
        ckpt_iter = _iter_num(ckpt_name)
        with open(ckpt_dir / "traces.json") as f:
            data = json.load(f)
        if not isinstance(data, list):
            continue
        for i, rec in enumerate(data):
            pages.append(_render_episode_page(
                run_name=run_name, ckpt_name=ckpt_name, ckpt_iter=ckpt_iter,
                episode_idx=i, record=rec,
            ))
    title = f"Traces — {run_name}"
    return (
        '<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"/>'
        f'<title>{escape(title)}</title><style>{_CSS}</style></head>'
        f'<body>{"".join(pages)}</body></html>'
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path", type=Path, help="Run directory or single ckpt_* directory")
    p.add_argument("-o", "--output", type=Path, default=None,
                   help="Output HTML file (default: <input>/ckpt_traces_print.html)")
    args = p.parse_args()

    path = args.path.resolve()
    if not path.exists():
        raise SystemExit(f"path not found: {path}")

    ckpt_dirs = _find_ckpt_dirs(path)
    if not ckpt_dirs:
        raise SystemExit(f"no ckpt_*/traces.json found under {path}")

    if path.name.startswith("ckpt_"):
        run_name = path.parent.parent.name
    elif (path / "traces").is_dir():
        run_name = path.name
    else:
        run_name = path.parent.name

    html = build_html(run_name, ckpt_dirs)

    out = args.output
    if out is None:
        base = path if path.is_dir() else path.parent
        out = base / "ckpt_traces_print.html"
    out.write_text(html)

    n_pages = html.count('<section class="page">')
    print(f"Wrote {n_pages} pages from {len(ckpt_dirs)} checkpoint(s) to {out}")


if __name__ == "__main__":
    main()
