"""
Command gate: intercepts shell commands before they reach the subprocess.

The gate checks only whether the command name (the first word on the line,
path-stripped) is enabled at this stage. It does not parse flags or
arguments — that is the shell's responsibility.

Every gate response is an observation string, never a silent drop.
The agent must be able to predict gate messages, so they must be
deterministic and attributable. This keeps the σ_t mask honest.

_ALWAYS_BLOCKED covers commands that would block the reader (interactive
pagers/editors), escape the sandbox (sudo, mount), or corrupt the Slurm job
(reboot, package managers). This is a research safeguard, not a security
boundary — the system assumes trusted agents in a trusted Slurm environment.
"""

import shlex

_ALWAYS_BLOCKED: frozenset[str] = frozenset({
    # Interactive — would block the sentinel-based output reader
    "less", "more", "vi", "vim", "nano", "emacs", "pico",
    # Privilege escalation
    "sudo", "su",
    # Filesystem escape / host modification
    "mount", "umount", "chroot",
    # Slurm job safety
    "kill", "killall", "pkill",
    "reboot", "shutdown", "halt", "poweroff",
    # Package managers (network + host modification)
    "apt", "apt-get", "yum", "dnf", "pip", "pip3",
    # Container runtimes
    "docker", "singularity", "podman", "apptainer",
    # Remote execution
    "ssh", "scp", "sftp",
    # Raw device access
    "dd", "mkfs", "fdisk", "parted",
})


class CommandGate:
    """
    Checks whether a raw command string is permitted at the current stage.

    Usage:
        gate = CommandGate(stage_cfg.allowed_commands, stage_cfg.name)
        allowed, msg = gate.check("grep -r pattern .")
        if allowed:
            output = shell.run("grep -r pattern .")
        else:
            output = msg  # return to agent as observation
    """

    def __init__(self, allowed_commands: frozenset, stage_name: str) -> None:
        self._allowed = allowed_commands
        self._stage = stage_name

    def check(self, raw: str) -> tuple[bool, str]:
        """
        Returns (is_allowed, message).

        If is_allowed is True:  message is "" and the caller forwards raw to shell.
        If is_allowed is False: message is the observation string to return instead.
        """
        stripped = raw.strip()

        # Empty line or comment — shell handles it, produces its own output (or none).
        if not stripped or stripped.startswith("#"):
            return True, ""

        # Let the shell produce its own parse error for malformed quoting.
        try:
            parts = shlex.split(stripped)
        except ValueError:
            return True, ""

        if not parts:
            return True, ""

        # Strip any path prefix so "/usr/bin/grep" → "grep"
        cmd = parts[0].rsplit("/", 1)[-1]

        if cmd in _ALWAYS_BLOCKED:
            return False, f"[gate] '{cmd}' is not available in this environment\n"

        if cmd not in self._allowed:
            return False, (
                f"[gate] '{cmd}' not enabled at stage {self._stage} "
                f"(available: {', '.join(sorted(self._allowed)) or 'none'})\n"
            )

        return True, ""
