"""Incremental reader of the subprocess progress JSONL file (Issue #87).

Moved out of :mod:`lizystudio.services.subprocess_runner` (P-0119, file
size); re-exported there under its historical name.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any


class _ProgressReader:
    """Tail-read the progress JSONL file one new chunk at a time.

    Issue #87: the legacy poll loop read the whole progress file via
    ``Path.read_text().splitlines()`` on every 200 ms tick and tracked a
    consumed-line counter. That is O(N) per poll in the number of lines
    already seen, so a long tune with hundreds of trials quadratically
    degrades the parent thread as the file grows. It also silently
    dropped partial writes (a ``write()`` that landed but whose
    terminating newline had not flushed yet was parsed as invalid JSON
    and thrown away, never to be retried).

    This reader keeps an open file handle and reads only the new bytes
    since the last call (``file.read()`` on a regular file with a live
    offset). A small ``_buffer`` holds any unterminated trailing text
    across calls so it is never lost — the next call stitches the
    continuation on and returns the completed line.

    The reader also handles the common "file does not exist yet"
    startup case: ``_ensure_open`` retries on every call until the
    child creates the file, at which point the handle opens and
    subsequent calls stream from offset 0.
    """

    def __init__(self, path: str) -> None:
        self._path = path
        self._file: Any = None  # typing.IO[str] once opened
        self._buffer = ""

    def _ensure_open(self) -> bool:
        """Open the file lazily if it has appeared on disk.

        Returns True if a live handle is available after the call, and
        False if the file still does not exist.
        """
        if self._file is not None:
            return True
        if not Path(self._path).exists():
            return False
        # Text mode with UTF-8 to match _write_progress; errors="replace"
        # so a corrupt byte sequence in the middle of a long tune does
        # not raise UnicodeDecodeError and kill the poll loop. A broken
        # line will fail JSON parsing in _forward_progress and be
        # dropped there with logging, which is the right failure mode.
        self._file = open(  # noqa: SIM115 — handle lives with the reader
            self._path, encoding="utf-8", errors="replace"
        )
        return True

    def read_new_lines(self) -> list[str]:
        """Return complete lines written since the last call.

        Any partial trailing content (no terminating newline) is stashed
        on ``self._buffer`` and only released once its continuation has
        been written.
        """
        if not self._ensure_open():
            return []
        chunk = self._file.read()
        if not chunk:
            return []
        data = self._buffer + chunk
        # If the chunk ends with a newline, every line is complete and
        # the buffer is empty; otherwise the last fragment is partial
        # and we hold it until a later call completes it.
        if data.endswith("\n"):
            self._buffer = ""
            return [ln for ln in data.splitlines() if ln]
        parts = data.split("\n")
        self._buffer = parts[-1]
        return [ln for ln in parts[:-1] if ln]

    def final_flush(self) -> list[str]:
        """Return any buffered partial line and clear the buffer.

        Called by ``_poll_progress`` after the subprocess has exited and
        the poll retries have finished. If the child was killed
        mid-write, its last bytes live here as a best-effort tail; the
        caller decides whether to forward them (they probably fail JSON
        parsing, but at least they land in logs).
        """
        if not self._buffer:
            return []
        tail = self._buffer
        self._buffer = ""
        return [tail]

    def close(self) -> None:
        """Release the file handle. Safe to call multiple times."""
        if self._file is not None:
            with contextlib.suppress(Exception):
                self._file.close()
            self._file = None
