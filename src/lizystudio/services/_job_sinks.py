"""Where a worker's outcome goes (P-0119): the commit sinks and the effects.

``_run_job_core`` runs the same code in the parent (thread mode) and in the
subprocess child, so it takes its outcome sink as an explicit argument
instead of testing the mode:

- :class:`ParentCommitSink` — thread mode: results are written status-free
  under the run token, then the status is committed through the primitive,
  and the effects it returns are emitted once.
- :class:`LegacyChildStatusSink` — the subprocess child, until PR-B: the
  child's status writes as before P-0119, except that it never writes
  ``running`` (admission did), only over a non-terminal row, and never
  touches the flags or the slot. The parent adopts the result with one
  commit. PR-B deletes this class.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol

from lizystudio.services._job_lifecycle_types import (
    CANCELLED_CODE,
    CANCELLED_MESSAGE,
    LifecycleEffects,
    Outcome,
    RunToken,
)

if TYPE_CHECKING:
    from lizystudio.services._job_metadata import Job
    from lizystudio.services.jobs import JobStore
    from lizystudio.ws.progress import ProgressBroadcaster

_logger = logging.getLogger(__name__)


def emit_effects(
    job_store: JobStore,
    broadcaster: ProgressBroadcaster | None,
    effects: LifecycleEffects | None,
    duration: float = 0.0,
) -> None:
    """Send the one frame and count the one terminal metric of a transition.

    Called after the lifecycle lock is released, by the caller of the
    operation that published the transition (INV-LC-2).
    """
    if effects is None:
        return
    if broadcaster is not None:
        _send_frame(broadcaster, effects)
    if effects.metric_status is not None:
        job_store.record_job_terminal(
            effects.job_type,  # type: ignore[arg-type]
            effects.metric_status,
            duration=duration,
        )


def _send_frame(broadcaster: ProgressBroadcaster, effects: LifecycleEffects) -> None:
    if effects.frame == "completed":
        broadcaster.send_completed(effects.job_id)
    elif effects.frame == "paused":
        broadcaster.send_paused(effects.job_id)
    elif effects.frame == "error":
        if effects.code is None:
            broadcaster.send_error(effects.job_id, effects.message)
        else:
            broadcaster.send_error(effects.job_id, effects.message, code=effects.code)


def sync_job(job: Job, job_store: JobStore) -> None:
    """Refresh *job*'s lifecycle fields from disk (the worker's copy)."""
    stored = job_store.get(job.job_id)
    if stored is not None:
        job.status = stored.status
        job.error = stored.error
        job.completed_at = stored.completed_at


class OutcomeSink(Protocol):
    """What ``_run_job_core`` hands its results and its outcome to."""

    @property
    def token(self) -> RunToken | None: ...

    def write_results(self, job: Job) -> bool: ...

    def commit(self, job: Job, outcome: Outcome, duration: float) -> None: ...


class ParentCommitSink:
    """Thread mode: commit through the parent's primitive (P-0119)."""

    def __init__(
        self,
        job_store: JobStore,
        broadcaster: ProgressBroadcaster | None,
        token: RunToken,
    ) -> None:
        self._job_store = job_store
        self._broadcaster = broadcaster
        self._token = token

    @property
    def token(self) -> RunToken | None:
        return self._token

    def write_results(self, job: Job) -> bool:
        return self._job_store.write_results(job, self._token)

    def commit(self, job: Job, outcome: Outcome, duration: float) -> None:
        effects = self._job_store.commit(self._token, outcome)
        sync_job(job, self._job_store)
        emit_effects(self._job_store, self._broadcaster, effects, duration)


class LegacyChildStatusSink:
    """The subprocess child's status writer until PR-B (P-0119 residual)."""

    def __init__(
        self, job_store: JobStore, broadcaster: ProgressBroadcaster | None
    ) -> None:
        self._job_store = job_store
        self._broadcaster = broadcaster

    @property
    def token(self) -> RunToken | None:
        return None

    def write_results(self, job: Job) -> bool:
        self._job_store.update(job)
        return True

    def commit(self, job: Job, outcome: Outcome, duration: float) -> None:
        from datetime import datetime, timezone

        completed_at = (
            None
            if outcome.status == "paused"
            else datetime.now(timezone.utc).isoformat()
        )
        written = self._job_store.legacy_child_write_status(
            job.job_id, outcome.status, error=outcome.error, completed_at=completed_at
        )
        sync_job(job, self._job_store)
        if written and self._broadcaster is not None:
            _send_legacy(self._broadcaster, job.job_id, outcome)


def _send_legacy(
    broadcaster: ProgressBroadcaster, job_id: str, outcome: Outcome
) -> None:
    """The child's frame for its own write, as before P-0119 (PR-B removes)."""
    if outcome.status == "completed":
        broadcaster.send_completed(job_id)
    elif outcome.status == "paused":
        broadcaster.send_paused(job_id)
    elif outcome.status == "cancelled":
        broadcaster.send_error(job_id, CANCELLED_MESSAGE, code=CANCELLED_CODE)
    elif outcome.code is None:
        broadcaster.send_error(job_id, outcome.message or "")
    else:
        broadcaster.send_error(job_id, outcome.message or "", code=outcome.code)
