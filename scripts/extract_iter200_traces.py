"""Extract per-(run, agent) iter-200 trace material into a structured folder
for qualitative regime analysis.

For each of the 8 iter-200 population runs, we:
- Pull every post by every agent across all logged episodes (clean text only),
  written to `posts/<run>/<agent>.md`.
- Copy the whole-run `traces.txt` for direct inspection of raw assistant
  generations including unclosed tags (the diagnostic for policy collapse).
- Build cross-cutting symlink trees so you can navigate by regime as well as
  by run.

Output: `runs/cultural_emergence/_qualitative_traces/`.
"""
from __future__ import annotations

import ast
import re
import shutil
from collections import defaultdict
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
RUNS = REPO / "runs" / "cultural_emergence"
OUT = RUNS / "_qualitative_traces"
ANN_DIR = RUNS / "_population_figures_annotated"
INDEX_CSV_CANDIDATES = sorted(ANN_DIR.glob("agent_index.csv"))
if not INDEX_CSV_CANDIDATES:
    raise SystemExit(f"No agent_index.csv in {ANN_DIR}; run "
                     "make_annotated_population_figures.py first.")
INDEX_CSV = INDEX_CSV_CANDIDATES[-1]

SUBSTRATE_SHORT = {
    "study_group_ashbourne_gc":      "ashbourne",
    "study_group_strathearn_server": "strathearn",
}


def slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", name).strip("_")


def run_tag(substrate: str, pop_size: int) -> str:
    return f"{pop_size:02d}agent_{SUBSTRATE_SHORT[substrate]}"


def extract_episodes(traces_txt: Path) -> list[dict]:
    """Return list of episode dicts. Each has 'pairing', 'thread' (list of
    {'speaker', 'text'}), and 'agents' (list of names)."""
    text = traces_txt.read_text(errors="replace")
    episodes = []
    # Episode header lines look like:
    # EPISODE 0  ...  pairing=X vs Y  env=...  thread=[{...}, {...}]  agents=[..]
    for m in re.finditer(r"^EPISODE\s+(\d+)\s+(.+)$", text, re.MULTILINE):
        body = m.group(2)
        # Locate the thread=[...] payload. We can't trust json.loads since the
        # original record uses single quotes with embedded newlines/escapes, so
        # use ast.literal_eval on the substring.
        idx = body.find("thread=")
        if idx < 0:
            continue
        # Find matching list bracket span
        start = body.find("[", idx)
        if start < 0:
            continue
        depth = 0
        end = -1
        for i, ch in enumerate(body[start:], start):
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end < 0:
            continue
        thread_str = body[start:end]
        try:
            thread = ast.literal_eval(thread_str)
        except (SyntaxError, ValueError):
            continue
        # Pairing
        pairing_match = re.search(r"pairing=([^\s].*?)\s+env=", body)
        pairing = pairing_match.group(1) if pairing_match else "?"
        agents_match = re.search(r"agents=(\[.*?\])", body)
        agents = []
        if agents_match:
            try:
                agents = ast.literal_eval(agents_match.group(1))
            except Exception:
                pass
        episodes.append(dict(idx=int(m.group(1)),
                             pairing=pairing,
                             thread=thread,
                             agents=agents))
    return episodes


def write_agent_posts(out_dir: Path, run_label: str, agent: str,
                      regime_meta: dict, episodes: list[dict]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    fpath = out_dir / f"{slug(agent)}.md"
    lines = []
    lines.append(f"# {agent} — {run_label}")
    lines.append("")
    lines.append("```")
    for k in ("category", "kl", "perc_loss", "entropy", "final_iter",
              "trace_confirmed_coherent"):
        if k in regime_meta and regime_meta[k] is not None:
            v = regime_meta[k]
            if isinstance(v, float):
                lines.append(f"{k}: {v:.4f}")
            else:
                lines.append(f"{k}: {v}")
    lines.append("```")
    lines.append("")
    other_speakers: dict[str, int] = defaultdict(int)
    n_posts = 0
    for ep in episodes:
        ep_posts = [p for p in ep["thread"] if p.get("speaker") == agent]
        for p in ep["thread"]:
            other_speakers[p.get("speaker", "?")] += 1
        if not ep_posts:
            continue
        lines.append(f"## Episode {ep['idx']} — pairing: {ep['pairing']}")
        for p in ep_posts:
            n_posts += 1
            text = p.get("text", "")
            lines.append(f"### post {p.get('post_index', '?')}")
            lines.append("```")
            lines.append(text)
            lines.append("```")
        lines.append("")
    lines.insert(3, f"**posts by this agent: {n_posts} across "
                    f"{len(episodes)} episodes**")
    lines.insert(4, "")
    fpath.write_text("\n".join(lines))
    return fpath


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    df = pd.read_csv(INDEX_CSV)
    print(f"loaded index: {len(df)} agents")

    # Group rows by run
    by_run: dict[str, pd.DataFrame] = {}
    for run_path, g in df.groupby("run"):
        by_run[run_path] = g

    posts_root = OUT / "posts_by_run"
    by_regime_root = OUT / "posts_by_regime"
    raw_root = OUT / "raw_traces"

    posts_root.mkdir()
    by_regime_root.mkdir()
    raw_root.mkdir()

    # Collect everything
    all_extracted = []
    for run_path, g in by_run.items():
        run_dir = Path(run_path) if Path(run_path).is_absolute() else RUNS / run_path
        traces_txt = run_dir / "traces" / "ckpt_000200" / "traces.txt"
        if not traces_txt.exists():
            print(f"[warn] missing {traces_txt}")
            continue
        substrate = g["substrate"].iloc[0]
        pop_size = int(g["pop_size"].iloc[0])
        run_label = run_tag(substrate, pop_size)
        run_out = posts_root / run_label
        run_out.mkdir(exist_ok=True)
        # Copy the raw traces.txt
        raw_dst = raw_root / f"{run_label}_traces.txt"
        shutil.copy(traces_txt, raw_dst)
        episodes = extract_episodes(traces_txt)
        print(f"  {run_label}: {len(episodes)} episodes")

        for _, row in g.iterrows():
            meta = row.to_dict()
            posts_path = write_agent_posts(run_out, run_label, row["agent"],
                                           meta, episodes)
            all_extracted.append({
                "run_label": run_label,
                "agent": row["agent"],
                "category": row["category"],
                "kl": row["kl"], "perc_loss": row["perc_loss"],
                "entropy": row["entropy"],
                "trace_confirmed_coherent": row["trace_confirmed_coherent"],
                "posts_md": str(posts_path.relative_to(OUT)),
                "raw_traces_txt": str(raw_dst.relative_to(OUT)),
            })

    # Mirror by regime as symlinks (or copies if symlinks not available)
    for entry in all_extracted:
        regime_dir = by_regime_root / entry["category"]
        regime_dir.mkdir(parents=True, exist_ok=True)
        src = OUT / entry["posts_md"]
        dst = regime_dir / f"{entry['run_label']}__{slug(entry['agent'])}.md"
        if dst.exists():
            dst.unlink()
        try:
            dst.symlink_to(src.resolve())
        except OSError:
            shutil.copy(src, dst)

    pd.DataFrame(all_extracted).sort_values(
        ["category", "run_label", "agent"]
    ).to_csv(OUT / "extracted_index.csv", index=False)
    print(f"[ok] wrote {len(all_extracted)} agent files to {OUT}")


if __name__ == "__main__":
    main()
