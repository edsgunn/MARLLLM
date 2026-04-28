"""
Per-agent sandboxed bash subprocess.

Design decisions
----------------
No Docker / Singularity:
    We are already inside a Slurm allocation. Container runtimes add latency
    (startup, exec round-trips) and complexity for no security benefit in a
    research setting. The command gate is the primary enforcement mechanism;
    the shell's restricted PATH and the scratch guard function are backup.

Persistent subprocess:
    One bash process lives for the entire episode. This preserves shell state
    (working directory, shell variables) across steps — necessary for
    multi-step tasks like `mkdir notes && cd notes && echo ... > file.txt`.

Sentinel-delimited output:
    After every command we `printf` a unique sentinel. The reader thread
    collects stdout lines until it sees the sentinel. No polling, no fixed
    sleep. The sentinel is derived from the current monotonic clock, so
    concurrent shells (one per agent) never share sentinels.

stderr merged into stdout:
    `stderr=SUBPROCESS.STDOUT` means error messages appear in the agent's
    observation stream. Errors are signal — "command not found", "permission
    denied", "no such file" all teach the model about the environment. We
    never silence them.

No pty:
    Interactive commands (less, vi, etc.) are blocked by the gate. Without
    a pty, bash does not attempt readline, cursor movement, or job control
    noise — stdout is clean line-buffered text.

Minimal inherited environment:
    We do NOT inherit the Slurm job's environment. SLURM_*, module system
    paths, and user dotfiles would confuse the agent and potentially expose
    host information. PATH is set to the bare system minimum.

Scratch guard:
    A bash function wraps `rm`, `mv`, and `cp` to reject any argument whose
    realpath falls outside $SCRATCH. This prevents Stage 3+ agents from
    accidentally deleting corpus files even if the gate is misconfigured.
    The gate is the primary mechanism; this is defence-in-depth.

Restart on timeout:
    If a command exceeds `timeout_seconds`, the shell process is killed and
    restarted. The agent receives `[timeout: ...]` as an observation. The
    working directory resets to $CORPUS on restart, which is itself a signal
    that something abnormal happened.
"""

import queue
import subprocess
import threading
import time
from pathlib import Path

from .gate import CommandGate
from .post_process import PostProcessor


class AgentShell:
    def __init__(
        self,
        agent_id: str,
        corpus_dir: str | Path,
        scratch_dir: str | Path,
        allowed_commands: frozenset,
        stage_name: str,
        timeout_seconds: int = 30,
        max_output_chars: int = 8192,
    ) -> None:
        self.agent_id = agent_id
        self.corpus_dir = Path(corpus_dir).resolve()
        self.scratch_dir = Path(scratch_dir).resolve()
        self.gate = CommandGate(allowed_commands, stage_name)
        self.post = PostProcessor(max_chars=max_output_chars)
        self.timeout = timeout_seconds

        self._proc: subprocess.Popen | None = None
        self._q: queue.Queue[str | None] = queue.Queue()
        self._sentinel_prefix = "__CCSM__"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the bash subprocess. Safe to call after close()."""
        if self._proc is not None:
            self.close()

        # Minimal environment — do not inherit Slurm's env.
        env = {
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "HOME": str(self.scratch_dir),
            "CORPUS": str(self.corpus_dir),
            "SCRATCH": str(self.scratch_dir),
            "TERM": "dumb",
            "LANG": "en_US.UTF-8",
            "LC_ALL": "en_US.UTF-8",
        }

        self._proc = subprocess.Popen(
            ["bash", "--norc", "--noprofile"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=str(self.corpus_dir),
            text=True,
            bufsize=1,
        )

        reader = threading.Thread(target=self._reader_loop, daemon=True)
        reader.start()

        # Initial setup sent as a single heredoc-style block so the shell
        # processes it atomically before we send any real commands.
        setup = (
            "set +H\n"                          # disable history expansion
            "set +o history\n"
            "unset HISTFILE\n"
            f'SCRATCH="{self.scratch_dir}"\n'
            f'CORPUS="{self.corpus_dir}"\n'
            # Scratch guard: intercepts rm/mv/cp and rejects paths outside $SCRATCH.
            "_guard() {\n"
            "  local cmd=$1; shift\n"
            "  for arg; do\n"
            "    case $arg in -*) continue;; esac\n"
            "    local real\n"
            "    real=$(realpath -m \"$arg\" 2>/dev/null || printf '%s' \"$arg\")\n"
            "    if [[ $real != \"$SCRATCH\"* ]]; then\n"
            "      printf '[gate] %s outside $SCRATCH is not permitted\\n' \"$cmd\" >&2\n"
            "      return 1\n"
            "    fi\n"
            "  done\n"
            "  command \"$cmd\" \"$@\"\n"
            "}\n"
            "rm() { _guard rm \"$@\"; }\n"
            "mv() { _guard mv \"$@\"; }\n"
            "cp() { _guard cp \"$@\"; }\n"
            f'cd "{self.corpus_dir}"\n'
            "printf '__READY__\\n'\n"
        )
        self._raw_send(setup)
        self._wait_for("__READY__", timeout=15.0)

    def close(self) -> None:
        """Terminate the shell process."""
        if self._proc is None:
            return
        try:
            self._proc.stdin.close()
        except Exception:
            pass
        try:
            self._proc.terminate()
            self._proc.wait(timeout=5)
        except Exception:
            try:
                self._proc.kill()
            except Exception:
                pass
        self._proc = None

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, command: str) -> str:
        """
        Send one shell command and return the observation string.

        The returned string is everything the environment produced in response —
        it does NOT contain the command itself. The command is the agent's action
        (σ_t=0); the training loop already has it. Everything returned here is
        σ_t=1 from the agent's perspective.

        The trailing '$ ' is the shell prompt. It signals readiness for the
        next command and is part of the observation stream — a token the model
        will learn to predict as the boundary between steps.
        """
        allowed, gate_msg = self.gate.check(command)
        if not allowed:
            return gate_msg + "$ "

        sentinel = f"{self._sentinel_prefix}{time.monotonic_ns()}"
        self._raw_send(f"{command.strip()}\nprintf '{sentinel}\\n'\n")

        try:
            raw = self._collect_until(sentinel, timeout=float(self.timeout))
        except TimeoutError:
            self._restart()
            return f"[timeout: command exceeded {self.timeout}s limit]\n$ "

        return self.post.process(raw) + "$ "

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _restart(self) -> None:
        self.close()
        self.start()

    def _raw_send(self, text: str) -> None:
        assert self._proc is not None
        self._proc.stdin.write(text)
        self._proc.stdin.flush()

    def _reader_loop(self) -> None:
        """Background thread that drains stdout into the queue."""
        assert self._proc is not None
        try:
            for line in self._proc.stdout:
                self._q.put(line)
        finally:
            self._q.put(None)  # EOF marker

    def _wait_for(self, marker: str, timeout: float) -> None:
        """Block until a line containing `marker` appears."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            try:
                line = self._q.get(timeout=min(0.1, remaining))
            except queue.Empty:
                continue
            if line is None:
                raise RuntimeError(f"Shell exited before marker '{marker}'")
            if marker in line:
                return
        raise TimeoutError(f"Marker '{marker}' not seen within {timeout:.1f}s")

    def _collect_until(self, sentinel: str, timeout: float) -> str:
        """Collect stdout lines until the sentinel appears; return all preceding lines."""
        lines: list[str] = []
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            try:
                line = self._q.get(timeout=min(0.1, remaining))
            except queue.Empty:
                continue
            if line is None:
                raise RuntimeError("Shell exited while collecting command output")
            if sentinel in line:
                return "".join(lines)
            lines.append(line)
        raise TimeoutError(f"Command did not complete within {timeout:.1f}s")
