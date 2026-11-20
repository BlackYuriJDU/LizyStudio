"""Cooperative-cancel + pause request flags (#451).

Extracted from ``JobStore``. Each of the two concerns (cancel — H-0011 /
Issue #152; pause — P-0099 v3-20c / R-1.4) is represented by an in-memory
``set[str]`` guarded by its own lock and mirrored to an on-disk flag file
(``<job_dir>/CANCEL`` / ``<job_dir>/PAUSE``).

The in-memory set is the **source of truth** for same-process callers.
The file is an IPC channel only: a subprocess child constructs its own
fresh ``JobStore`` (whose in-memory sets are disjoint from the parent's),
so the on-disk flag is how it observes the parent's request at the
cancel-aware callback boundary in ``services/training.py``.

The two pairs are kept independent (separate sets + locks): a job can be
both cancel- and pause-requested in flight, and the callback raises
whichever check fires first while the un-observed flag must persist for
the next call.
"""

from __future__ import annotations

import contextlib
import logging
import os
import threading
from pathlib import Path

from lizystudio.services._job_metadata import JobMetadataStore

_logger = logging.getLogger(__name__)


class FlagNotPersistedError(OSError):
    """A cancel / pause flag file could not be written (P-0119, Codex r2-2).

    The request is not acknowledged: the in-memory flag is not set either.
    """


class JobControlFlags:
    """Cancel + pause request flags for jobs. Thread-safe.

    Holds a :class:`JobMetadataStore` ref solely for path resolution
    (``jobs_dir`` for the staging tempfile, ``path_for`` for the flag
    files — both apply the traversal guard).
    """

    def __init__(self, metadata: JobMetadataStore) -> None:
        self._metadata = metadata
        self._cancel_requested: set[str] = set()
        self._cancel_lock = threading.Lock()
        # P-0099 v3-20c: pause primitives mirror cancel exactly — same
        # in-memory set + lock + on-disk flag IPC pattern. Kept as a
        # separate set so cancel and pause observations stay independent
        # in the cancel-aware callback (a job can be both cancel- and
        # pause-requested in flight; the callback raises whichever check
        # fires first, but the un-observed flag must persist for the
        # next call).
        self._pause_requested: set[str] = set()
        self._pause_lock = threading.Lock()

    # --- cancel (H-0011 / Issue #152) ---

    def request_cancel(self, job_id: str) -> None:
        """Mark a job for cancellation (H-0011).

        Issue #152: writes ``<job_dir>/CANCEL`` atomically so a fresh
        ``JobStore`` in a subprocess child (whose in-memory set is disjoint)
        observes the cancel, and so an accepted cancel survives a restart
        (P-0119 INV-LC-4). P-0119 (Codex r2-2): the file is written first and
        a failed write raises :class:`FlagNotPersistedError` with the
        in-memory flag left unset, so a cancel is never acknowledged without
        its durable record. A job whose directory does not exist (deleted, or
        never created) has nothing durable to protect: only the in-memory
        flag is set.
        """
        self._persist_flag(job_id, self._cancel_flag_path(job_id), "cancel")
        with self._cancel_lock:
            self._cancel_requested.add(job_id)

    def mark_cancel_in_memory(self, job_id: str) -> None:
        """Set only the in-memory cancel flag (the delete path, whose job
        directory is about to be removed; its worker still sees the flag)."""
        with self._cancel_lock:
            self._cancel_requested.add(job_id)

    def _persist_flag(self, job_id: str, flag_path: Path, kind: str) -> None:
        """Atomically create *flag_path* (staged in ``jobs_dir``, never in the
        job dir, so a concurrent ``rmtree`` never sees a partial entry).

        No-op when the job dir does not exist (no ``mkdir``: that would race a
        concurrent delete). Any other failure raises FlagNotPersistedError.
        """
        if not flag_path.parent.exists():
            return
        tmp_path = self._metadata.jobs_dir / f".{kind}-{job_id}-{os.getpid()}.tmp"
        try:
            tmp_path.write_bytes(b"")
            os.replace(tmp_path, flag_path)
        except OSError as exc:
            with contextlib.suppress(OSError):
                tmp_path.unlink(missing_ok=True)
            _logger.error(
                "event='lifecycle.flag_not_persisted' job=%s flag=%s", job_id, kind
            )
            msg = f"could not persist the {kind} flag of job {job_id}: {exc}"
            raise FlagNotPersistedError(msg) from exc

    def is_cancel_requested(self, job_id: str) -> bool:
        """Check whether cancellation was requested for a job.

        Checks the in-memory set first (cheap, hot-path friendly for
        cooperative cancel callbacks). Falls back to the on-disk flag
        file so a child subprocess with a fresh ``JobStore`` still
        observes the parent's cancel (Issue #152).
        """
        with self._cancel_lock:
            if job_id in self._cancel_requested:
                return True
        # Fallback: check the file flag. OSError (e.g. permissions)
        # or ValueError (malformed job_id rejected by
        # validate_path_within) conservatively return False — the
        # parent's in-process wait loop will eventually escalate to
        # SIGTERM anyway. Critically, this method is called from the
        # hot cooperative-cancel loop in training.py, so it MUST NOT
        # propagate exceptions.
        try:
            return self._cancel_flag_path(job_id).exists()
        except (OSError, ValueError):
            return False

    def clear_cancel(self, job_id: str) -> None:
        """Clear cancellation flag after processing."""
        with self._cancel_lock:
            self._cancel_requested.discard(job_id)
        try:
            self._cancel_flag_path(job_id).unlink(missing_ok=True)
        except (OSError, ValueError):
            _logger.warning(
                "Failed to clear cancel flag for job %s", job_id, exc_info=True
            )

    def _cancel_flag_path(self, job_id: str) -> Path:
        """Resolve the cancel flag path with the traversal guard.

        Thin wrapper over ``JobMetadataStore.path_for`` so a malformed
        job_id cannot escape the jobs_dir root.
        """
        return self._metadata.path_for(job_id, "cancel_flag")

    # --- pause / unpause (P-0099 v3-20c, R-1.4) ---

    def request_pause(self, job_id: str) -> None:
        """Mark a job for pause (R-1.4 / Issue #360).

        Mirrors :meth:`request_cancel`: ``<job_dir>/PAUSE`` is the IPC channel
        a subprocess child reads, written first; a failed write raises
        :class:`FlagNotPersistedError` with the in-memory flag left unset, so
        a pause is never acknowledged without it (P-0119, Codex r2-2). The
        cancel-aware callback raises ``PausedError`` when it observes it.
        """
        self._persist_flag(job_id, self._pause_flag_path(job_id), "pause")
        with self._pause_lock:
            self._pause_requested.add(job_id)

    def is_pause_requested(self, job_id: str) -> bool:
        """Check whether a pause request has been observed.

        In-memory check first (hot cancel-aware-callback path), then
        on-disk flag fallback for subprocess children.  Any OSError /
        ValueError fallback returns ``False`` so a malformed job_id or
        permissions glitch cannot stall the worker.
        """
        with self._pause_lock:
            if job_id in self._pause_requested:
                return True
        try:
            return self._pause_flag_path(job_id).exists()
        except (OSError, ValueError):
            return False

    def clear_pause(self, job_id: str) -> None:
        """Clear the pause observation after the worker has unwound.

        The /unpause endpoint calls this immediately before re-launching
        the resume worker so the next callback iteration does not raise
        ``PausedError`` again on the same flag.
        """
        with self._pause_lock:
            self._pause_requested.discard(job_id)
        try:
            self._pause_flag_path(job_id).unlink(missing_ok=True)
        except (OSError, ValueError):
            _logger.warning(
                "Failed to clear pause flag for job %s", job_id, exc_info=True
            )

    def _pause_flag_path(self, job_id: str) -> Path:
        """Resolve the pause flag path with the traversal guard."""
        return self._metadata.path_for(job_id, "pause_flag")
