"""
Corpus management and episode-level file rotation.

At each episode reset, a fresh subset of corpus files is selected and
exposed to the agent via symlinks in a per-agent staging directory. The
corpus root is never modified — only the staging dir changes.

Why symlinks instead of copies:
    Symlinks are instantaneous regardless of file size. The corpus can be
    hundreds of gigabytes; copying would make episode resets infeasible.

Why rotation instead of fixed files:
    Rotation is the primary mechanism for preventing exploration collapse.
    An agent that has learned "always cat foo.txt" will see high observation
    surprise the moment foo.txt is absent or replaced. Surprise → gradient
    signal to update the world model → policy pushed away from the collapsed
    action. Collapse is self-punishing without any reward signal.

Overlap fraction:
    Controls continuity vs. novelty across episodes.
      0.0 → every episode is completely foreign; no persistent world model.
      1.0 → same files every episode; collapse is not self-punishing.
    Default 0.3 is a starting point. Tune empirically:
      - If loss variance between episodes is very high, increase overlap.
      - If per-episode exploration is collapsing, decrease overlap.
    The right value is probably stage-dependent (higher at Stage 0/1,
    lower at Stage 2+ where the agent should be actively searching).

Stage 0 text streaming:
    When no commands are enabled, the corpus is streamed as raw text into
    the observation channel. `read_chunk(max_chars)` returns the next slice
    of the concatenated corpus. The stream resets at each rotate() call.
"""

import random
import shutil
from itertools import chain
from pathlib import Path
from typing import Iterator


class CorpusManager:
    """
    Manages per-episode corpus file selection and staging.

    Parameters
    ----------
    corpus_root:
        Read-only directory containing training text files. Recursively
        searched; all regular files are candidates.
    staging_dir:
        Writable directory that will be rebuilt with symlinks at each
        rotate() call. This is the directory the agent's shell sees as $CORPUS.
    files_per_episode:
        Number of corpus files to expose per episode. Should be large
        enough that the agent cannot exhaust them but small enough that
        rotation causes visible change (e.g. 50–200 files).
    overlap_fraction:
        Fraction of the previous episode's file set to carry over.
    rng:
        Random instance for reproducibility. Isolated per agent so that
        agents on a shared corpus get independently rotated views.
    """

    def __init__(
        self,
        corpus_root: str | Path,
        staging_dir: str | Path,
        files_per_episode: int = 50,
        overlap_fraction: float = 0.3,
        rng: random.Random | None = None,
    ) -> None:
        self.corpus_root = Path(corpus_root).resolve()
        self.staging_dir = Path(staging_dir)
        self.files_per_episode = files_per_episode
        self.overlap_fraction = overlap_fraction
        self._rng = rng or random.Random()

        self._all_files: list[Path] = sorted(
            p for p in self.corpus_root.rglob("*") if p.is_file()
        )
        if not self._all_files:
            raise ValueError(f"No files found under corpus root: {self.corpus_root}")

        self._current_files: list[Path] = []
        self._stream_iter: Iterator[str] | None = None

    def rotate(self) -> Path:
        """
        Select a new episode file set, rebuild the staging directory, and
        reset the Stage-0 text stream. Returns the staging directory path.

        Call this at the start of every episode for every agent
        (or once per episode for shared corpus staging).
        """
        # Carry over a fraction of the previous selection.
        n_keep = min(
            int(len(self._current_files) * self.overlap_fraction),
            len(self._current_files),
        )
        kept = self._rng.sample(self._current_files, n_keep)

        # Sample fresh files to fill the remainder.
        fresh_pool = [f for f in self._all_files if f not in set(kept)]
        n_fresh = max(0, self.files_per_episode - len(kept))
        fresh = self._rng.sample(fresh_pool, min(n_fresh, len(fresh_pool)))

        self._current_files = kept + fresh
        self._rng.shuffle(self._current_files)

        # Rebuild staging dir with symlinks into the read-only corpus root.
        if self.staging_dir.exists():
            shutil.rmtree(self.staging_dir)
        self.staging_dir.mkdir(parents=True)

        for src in self._current_files:
            rel = src.relative_to(self.corpus_root)
            dst = self.staging_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.symlink_to(src)

        # Reset the sequential text stream (used by Stage 0).
        self._stream_iter = self._make_stream()

        return self.staging_dir

    def read_chunk(self, max_chars: int) -> str:
        """
        Read up to `max_chars` from the sequential corpus text stream.

        Used only in Stage 0 where the corpus is streamed directly into
        the observation channel without any shell commands. The stream is
        a concatenation of all current episode files in shuffled order.
        Calling this after the stream is exhausted returns an empty string.
        """
        if self._stream_iter is None:
            self._stream_iter = self._make_stream()

        buf: list[str] = []
        total = 0
        for chunk in self._stream_iter:
            buf.append(chunk)
            total += len(chunk)
            if total >= max_chars:
                break
        return "".join(buf)

    def _make_stream(self) -> Iterator[str]:
        """Yield 4096-char chunks from each file in the current selection."""
        for path in self._current_files:
            try:
                text = path.read_text(errors="replace")
            except (OSError, PermissionError):
                yield f"[unreadable: {path.name}]\n"
                continue
            for i in range(0, len(text), 4096):
                yield text[i: i + 4096]
