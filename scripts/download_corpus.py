"""
Download and format corpora for CCSMTerminalEnv.

Writes a directory tree of plain .txt files, one file per document, organized
into subdirectories by topic/category.  The output is immediately usable as
--corpus-path for train_terminal.py.

Supported sources
-----------------
  wikipedia   English Wikipedia articles, organized by topic keyword
  arxiv       ArXiv paper abstracts + body, organized by primary category
  gutenberg   Project Gutenberg books (PG-19), organized by inferred genre
  python      Python source files from GitHub, organized by package name

Requirements
------------
    pip install datasets tqdm

    Some sources (arxiv full text, The Stack) require a HuggingFace account:
        huggingface-cli login

Usage
-----
    python scripts/download_corpus.py wikipedia /data/corpus/wikipedia
    python scripts/download_corpus.py arxiv     /data/corpus/arxiv --max-files 200000
    python scripts/download_corpus.py gutenberg /data/corpus/pg19
    python scripts/download_corpus.py python    /data/corpus/python --max-files 100000

    # In a Slurm job (fast local scratch):
    python scripts/download_corpus.py wikipedia $SCRATCH/corpus/wikipedia

Target directory layout (example: wikipedia)
--------------------------------------------
    {target}/
      science/
        Quantum_mechanics.txt
        Special_relativity.txt
        ...
      history/
        World_War_II.txt
        ...
      general/
        ...

Each .txt file begins with a one-line metadata header (title / authors / etc.)
followed by a blank line, then the document body.  This means a plain `cat`
gives the agent both the title and content — useful signal for Stage 1+.
"""
import argparse
import os
import re
import sys
import unicodedata
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_datasets() -> None:
    try:
        import datasets  # noqa: F401
    except ImportError:
        sys.exit(
            "The 'datasets' package is required:\n"
            "    pip install datasets tqdm\n"
            "or, inside the project venv:\n"
            "    uv add datasets"
        )


def _sanitize(name: str, max_len: int = 80) -> str:
    """Make a string safe to use as a filename."""
    # Normalize unicode, strip control characters
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if unicodedata.category(c) not in ("Cc", "Cf"))
    # Replace filesystem-unsafe characters
    name = re.sub(r'[/\\:*?"<>|]', "_", name)
    name = re.sub(r"\s+", "_", name.strip())
    name = re.sub(r"_+", "_", name)
    return name[:max_len] or "document"


def _write(path: Path, header: str, body: str, min_chars: int, max_chars: int) -> bool:
    """Write a document file.  Returns True if written, False if skipped."""
    body = body.strip()
    if len(body) < min_chars:
        return False
    body = body[:max_chars]
    path.parent.mkdir(parents=True, exist_ok=True)
    # Skip if file already exists (allows resuming interrupted downloads)
    if path.exists():
        return False
    path.write_text(header.strip() + "\n\n" + body, encoding="utf-8", errors="replace")
    return True


def _progress(n: int, total: int | None, source: str) -> None:
    if n % 1000 == 0:
        if total:
            pct = f"  ({100 * n / total:.1f}%)"
        else:
            pct = ""
        print(f"  [{source}] {n:,} files written{pct}", flush=True)


# ---------------------------------------------------------------------------
# Wikipedia
# ---------------------------------------------------------------------------

# Maps broad topic categories to sets of title keywords (lowercased).
# First matching category wins; unmatched articles go to "general".
_WIKI_CATEGORIES = [  # list[tuple[str, frozenset[str]]]
    ("mathematics",     frozenset({
        "mathematics", "algebra", "calculus", "geometry", "topology",
        "statistics", "probability", "number_theory", "theorem", "proof",
        "combinatorics", "differential", "integral", "manifold",
    })),
    ("physics",         frozenset({
        "physics", "quantum", "relativity", "thermodynamics", "mechanics",
        "electromagnetism", "optics", "particle", "nuclear", "astrophysics",
        "cosmology", "gravitation", "string_theory",
    })),
    ("biology",         frozenset({
        "biology", "genetics", "evolution", "ecology", "neuroscience",
        "microbiology", "cell", "dna", "protein", "taxonomy",
        "organism", "species", "genome",
    })),
    ("chemistry",       frozenset({
        "chemistry", "chemical", "molecule", "atom", "reaction",
        "compound", "polymer", "organic", "inorganic", "biochemistry",
    })),
    ("computing",       frozenset({
        "computer", "software", "algorithm", "programming", "internet",
        "network", "artificial_intelligence", "machine_learning", "database",
        "operating_system", "cryptography", "compiler",
    })),
    ("history",         frozenset({
        "history", "war", "battle", "empire", "dynasty", "revolution",
        "ancient", "medieval", "century", "civilization", "conquest",
        "colonialism", "republic", "kingdom",
    })),
    ("geography",       frozenset({
        "geography", "country", "city", "river", "mountain", "island",
        "continent", "capital", "population", "region", "ocean", "lake",
        "peninsula", "desert",
    })),
    ("philosophy",      frozenset({
        "philosophy", "ethics", "logic", "epistemology", "metaphysics",
        "consciousness", "morality", "existentialism", "phenomenology",
    })),
    ("arts",            frozenset({
        "music", "film", "novel", "literature", "painting", "sculpture",
        "theatre", "dance", "poetry", "cinema", "composer", "author",
        "architecture", "photography",
    })),
    ("economics",       frozenset({
        "economics", "market", "trade", "inflation", "gdp", "capitalism",
        "socialism", "currency", "finance", "banking", "fiscal",
    })),
    ("medicine",        frozenset({
        "medicine", "disease", "virus", "bacteria", "treatment", "therapy",
        "surgery", "drug", "vaccine", "syndrome", "disorder", "anatomy",
    })),
]


def _wiki_category(title: str) -> str:
    low = title.lower().replace(" ", "_")
    for cat, keywords in _WIKI_CATEGORIES:
        if any(kw in low for kw in keywords):
            return cat
    return "general"


def download_wikipedia(target: Path, max_files: int, min_chars: int, max_chars: int) -> None:
    _check_datasets()
    from datasets import load_dataset  # type: ignore[import]

    print("Loading Wikipedia (streaming — no full download required)...")
    # wikimedia/wikipedia uses Parquet (no loading script); config selects language+date.
    ds = load_dataset(
        "wikimedia/wikipedia", "20231101.en",
        split="train",
        streaming=True,
    )

    n = 0
    for row in ds:
        if n >= max_files:
            break
        title: str = row["title"]
        text:  str = row["text"]
        cat   = _wiki_category(title)
        fname = _sanitize(title) + ".txt"
        header = f"Title: {title}"
        if _write(target / cat / fname, header, text, min_chars, max_chars):
            n += 1
            _progress(n, max_files, "wikipedia")

    del ds  # close background streaming thread before interpreter shutdown
    print(f"Wikipedia: wrote {n:,} files to {target}")


# ---------------------------------------------------------------------------
# ArXiv
# ---------------------------------------------------------------------------

# Map arXiv primary category prefix to a human-readable subdirectory name.
_ARXIV_CAT_MAP = {
    "cs":      "computer_science",
    "math":    "mathematics",
    "physics": "physics",
    "cond-mat":"condensed_matter",
    "quant-ph":"quantum_physics",
    "gr-qc":   "relativity_cosmology",
    "hep":     "high_energy_physics",
    "astro-ph":"astrophysics",
    "q-bio":   "quantitative_biology",
    "q-fin":   "quantitative_finance",
    "stat":    "statistics",
    "eess":    "electrical_engineering",
    "econ":    "economics",
}


def _arxiv_subdir(categories: str) -> str:
    """Return a subdirectory name from the primary arXiv category string."""
    primary = categories.strip().split()[0].split(".")[0].lower()
    return _ARXIV_CAT_MAP.get(primary, primary or "uncategorized")


def download_arxiv(target: Path, max_files: int, min_chars: int, max_chars: int) -> None:
    """
    Uses 'ccdv/arxiv-summarization' which contains article body + abstract
    for a large subset of arXiv.  No HuggingFace login required.

    Alternatively, 'scientific_papers' (config 'arxiv') provides similar
    content with section-level splits.  Categories are inferred from title
    keywords since this dataset does not include category metadata.
    """
    _check_datasets()
    from datasets import load_dataset  # type: ignore[import]

    print("Loading arXiv (ccdv/arxiv-summarization, streaming)...")
    ds = load_dataset(
        "ccdv/arxiv-summarization",
        split="train",
        streaming=True,
    )

    # This dataset has: article (body), abstract, section_names (when available)
    # It does not carry category metadata, so we bin by keywords in the abstract.
    n = 0
    for row in ds:
        if n >= max_files:
            break

        abstract: str = row.get("abstract", "")
        article:  str = row.get("article",  "")
        body = article if len(article) > len(abstract) else abstract

        # Infer category from abstract keywords (rough but workable)
        low = abstract.lower()
        subdir = "other"
        for kw, cat in [
            ("neural network",     "computer_science"),
            ("machine learning",   "computer_science"),
            ("deep learning",      "computer_science"),
            ("natural language",   "computer_science"),
            ("reinforcement",      "computer_science"),
            ("quantum",            "quantum_physics"),
            ("cosmolog",           "astrophysics"),
            ("galaxy",             "astrophysics"),
            ("black hole",         "astrophysics"),
            ("protein",            "biology"),
            ("gene",               "biology"),
            ("genome",             "biology"),
            ("climate",            "earth_science"),
            ("econom",             "economics"),
            ("market",             "economics"),
            ("graph",              "mathematics"),
            ("topolog",            "mathematics"),
            ("differential equat", "mathematics"),
            ("fluid",              "physics"),
            ("thermodynamic",      "physics"),
            ("polymer",            "chemistry"),
            ("catalyst",           "chemistry"),
        ]:
            if kw in low:
                subdir = cat
                break

        fname = f"arxiv_{n:07d}.txt"
        header = f"Abstract: {abstract[:300].strip()}"
        if _write(target / subdir / fname, header, body, min_chars, max_chars):
            n += 1
            _progress(n, max_files, "arxiv")

    del ds
    print(f"ArXiv: wrote {n:,} files to {target}")


# ---------------------------------------------------------------------------
# Project Gutenberg (PG-19)
# ---------------------------------------------------------------------------

_GUTENBERG_GENRES = [  # list[tuple[str, frozenset[str]]]
    ("science_fiction",  frozenset({
        "science fiction", "sci-fi", "space", "mars", "alien", "robot",
        "time machine", "moon", "war of the worlds",
    })),
    ("adventure",        frozenset({
        "adventure", "treasure", "island", "journey", "voyage", "quest",
        "explorer", "sea", "pirate",
    })),
    ("mystery",          frozenset({
        "mystery", "detective", "murder", "crime", "sherlock", "case of",
        "secret", "ghost",
    })),
    ("romance",          frozenset({
        "romance", "love", "heart", "bride", "marriage", "lady", "gentleman",
        "courtship",
    })),
    ("history",          frozenset({
        "history", "historical", "memoirs", "war", "battle", "napoleon",
        "century", "empire", "ancient",
    })),
    ("philosophy",       frozenset({
        "philosophy", "ethics", "logic", "discourse", "treatise", "republic",
        "nicomachean", "critique",
    })),
    ("poetry",           frozenset({
        "poem", "poems", "poetry", "sonnets", "odes", "ballads", "lyric",
        "verse",
    })),
    ("drama",            frozenset({
        "play", "plays", "comedy", "tragedy", "hamlet", "othello",
        "midsummer", "merchant",
    })),
    ("natural_science",  frozenset({
        "natural history", "origin of species", "geology", "biology",
        "botany", "zoology", "chemistry",
    })),
    ("social_science",   frozenset({
        "economics", "political", "society", "civilisation", "democracy",
        "rights", "liberty",
    })),
]


def _gutenberg_genre(title: str) -> str:
    low = title.lower()
    for genre, keywords in _GUTENBERG_GENRES:
        if any(kw in low for kw in keywords):
            return genre
    # Fallback: first letter of title for rough alphabetical bucketing
    first = re.sub(r"^the\s+|^a\s+|^an\s+", "", low).strip()
    letter = first[0] if first and first[0].isalpha() else "other"
    return f"fiction_{letter}"


def download_gutenberg(target: Path, max_files: int, min_chars: int, max_chars: int) -> None:
    """
    Uses the PG-19 dataset (Project Gutenberg books published before 1919).
    Books are full-length; each is written as a single .txt file.
    Note: individual books can be 300KB–2MB.  The env will truncate per-step
    output to max_output_chars, but the full text is available via head/tail/grep.
    """
    _check_datasets()
    from datasets import load_dataset  # type: ignore[import]

    print("Loading PG-19 (streaming)...")
    ds = load_dataset(
        "pg19",
        split="train",
        streaming=True,
    )

    n = 0
    for row in ds:
        if n >= max_files:
            break

        title: str = row.get("short_book_title") or row.get("book_title") or f"book_{n}"
        text:  str = row.get("text", "")
        genre  = _gutenberg_genre(title)
        fname  = _sanitize(title) + ".txt"
        header = f"Title: {title}"
        if _write(target / genre / fname, header, text, min_chars, max_chars):
            n += 1
            _progress(n, max_files, "gutenberg")

    del ds
    print(f"Gutenberg: wrote {n:,} files to {target}")


# ---------------------------------------------------------------------------
# Python (GitHub code)
# ---------------------------------------------------------------------------

def download_python(target: Path, max_files: int, min_chars: int, max_chars: int) -> None:
    """
    Uses codeparrot/github-code filtered to Python.

    Organizes files into subdirectories by inferred package name (the top-level
    directory component of the file path, if present).  This gives natural
    groupings like numpy/, sklearn/, requests/, etc.

    No HuggingFace login required for this dataset.

    Note: this dataset is very large (~54M files).  Set --max-files to a
    manageable subset (50000–200000) unless you have ample storage.
    """
    _check_datasets()
    from datasets import load_dataset  # type: ignore[import]

    # codeparrot/github-code covers all languages; we filter to Python in the loop.
    # The dataset is ~54M rows; streaming means only matched rows are processed.
    print("Loading codeparrot/github-code (Python, streaming)...")
    ds = load_dataset(
        "codeparrot/github-code",
        split="train",
        streaming=True,
    )

    n = 0
    for row in ds:
        if n >= max_files:
            break
        if row.get("programming_language") != "Python":
            continue

        code:    str = row.get("code", "")
        filepath: str = row.get("path", f"file_{n}.py")
        repo:    str = row.get("repo_name", "unknown")

        # Subdirectory: top-level package name from the file path, or repo name
        parts = Path(filepath).parts
        if len(parts) >= 2 and re.match(r"^[a-zA-Z][a-zA-Z0-9_]*$", parts[0]):
            subdir = parts[0].lower()
        else:
            # Fall back to the repo's project name (last segment of owner/project)
            subdir = _sanitize(repo.split("/")[-1])[:20] or "misc"

        # Use a counter-qualified name to avoid collisions across repos
        fname = _sanitize(filepath.replace("/", "__"))
        if not fname.endswith(".py"):
            fname += ".py"
        # Give each file a unique prefix to avoid cross-repo collisions
        fname = f"{n:07d}_{fname}"

        header = f"# Repository: {repo}\n# Path: {filepath}"
        if _write(target / subdir / fname, header, code, min_chars, max_chars):
            n += 1
            _progress(n, max_files, "python")

    del ds
    print(f"Python: wrote {n:,} files to {target}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "source",
        choices=["wikipedia", "arxiv", "gutenberg", "python"],
        help="Corpus source to download.",
    )
    p.add_argument(
        "target",
        type=Path,
        help="Output directory.  Created if it does not exist.  "
             "Existing files are skipped (safe to resume interrupted downloads).",
    )
    p.add_argument(
        "--max-files", type=int, default=100_000,
        help="Stop after writing this many files.  Default: 100,000.  "
             "Recommended minimums per stage: Stage 1 → 5,000; "
             "Stage 2–3 → 50,000; Stage 3.5+ → 100,000+.",
    )
    p.add_argument(
        "--min-chars", type=int, default=500,
        help="Skip documents shorter than this many characters.  "
             "Very short documents give near-zero observation surprise and "
             "waste episode steps.  Default: 500.",
    )
    p.add_argument(
        "--max-chars", type=int, default=100_000,
        help="Truncate documents to this many characters before writing.  "
             "The env's max_output_chars (default 8192) will truncate per-step "
             "output anyway; this cap prevents enormous files from slowing "
             "CorpusManager.rotate() and Stage-0 streaming.  Default: 100,000.",
    )
    return p.parse_args()


_DOWNLOADERS = {  # dict[str, Callable]
    "wikipedia": download_wikipedia,
    "arxiv":     download_arxiv,
    "gutenberg": download_gutenberg,
    "python":    download_python,
}


def main() -> None:
    args = parse_args()
    target: Path = args.target.resolve()
    target.mkdir(parents=True, exist_ok=True)

    print(f"Source:    {args.source}")
    print(f"Target:    {target}")
    print(f"Max files: {args.max_files:,}")
    print(f"Min chars: {args.min_chars:,}")
    print(f"Max chars: {args.max_chars:,}")

    # Count existing files so resuming shows correct progress
    existing = sum(1 for _ in target.rglob("*.txt"))
    if existing:
        print(f"Resuming: {existing:,} files already present (will be skipped)")
    print()

    _DOWNLOADERS[args.source](
        target=target,
        max_files=args.max_files,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
    )

    # Final summary
    total = sum(1 for _ in target.rglob("*.txt"))
    subdirs = [d for d in target.iterdir() if d.is_dir()]
    print(f"\nDone.  {total:,} total files in {len(subdirs)} subdirectories:")
    for d in sorted(subdirs):
        count = sum(1 for _ in d.rglob("*.txt"))
        print(f"  {d.name}/  ({count:,} files)")

    # datasets streaming keeps a background network thread that cannot be cleanly
    # joined after the iterator is exhausted.  os._exit() bypasses the interpreter
    # teardown that would otherwise cause a GIL crash; data is already on disk.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
