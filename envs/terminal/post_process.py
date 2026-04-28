"""
Post-processor for shell output.

Every intervention emits a token into the observation stream rather than
silently dropping content. The agent must be able to predict that binary
files get a truncation marker and that very long outputs get cut — these
are part of the environment's response distribution, not exceptions.
"""


class PostProcessor:
    """
    Cleans and truncates raw shell stdout.

    Parameters
    ----------
    max_chars:
        Soft limit on observation length per step. Exceeded output is
        replaced with a trailing [truncated: N chars] marker.
    """

    def __init__(self, max_chars: int = 8192) -> None:
        self.max_chars = max_chars

    def process(self, raw: str) -> str:
        # Normalise line endings from shells that emit \r\n
        text = raw.replace("\r\n", "\n").replace("\r", "\n")

        # Null bytes indicate binary content. Hard-truncate and label.
        if "\x00" in text:
            text = text.replace("\x00", "").strip()
            cut = text[: self.max_chars // 8]
            return cut + "\n[truncated: binary content]\n"

        if len(text) > self.max_chars:
            return text[: self.max_chars] + f"\n[truncated: {len(text)} chars]\n"

        return text
