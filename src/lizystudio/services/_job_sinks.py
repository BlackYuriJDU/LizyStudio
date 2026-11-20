"""Where a worker's outcome goes (P-0119): the commit sinks and the effects.

``_run_job_core`` runs the same code in the parent (thread mode) and in the
subprocess child, so it takes its outcome sink as an explicit argument
instead of testing the mode:

- :class:`ParentCommitSink` — thread mode: results are written status-free
  under the run token, then the status is committed through the primitive,
  and the effects it returns are emitted once.
- :class:`ChildOutcomeSink` — the subprocess child: it writes no
  ``meta.json``; its outcome goes to the outcome file the parent's reconcile
  commits (P-0119 PR-B), so a status becomes visible only with that commit.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol

from lizystudio.services._job_lifecycle_types import (
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

    @property
    def persists_warnings(self) -> bool: ...

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

    @property
    def persists_warnings(self) -> bool:
        return True

    def write_results(self, job: Job) -> bool:
        return self._job_store.write_results(job, self._token)

    def commit(self, job: Job, outcome: Outcome, duration: float) -> None:
        effects = self._job_store.commit(self._token, outcome)
        sync_job(job, self._job_store)
        emit_effects(self._job_store, self._broadcaster, effects, duration)


class ChildOutcomeSink:
    """The subprocess child (P-0119 PR-B): it writes no ``meta.json`` at all.

    Results travel in the outcome file, warnings in the JSONL relay (the
    parent persists them under the run token), and the outcome itself goes
    to ``<job_dir>/outcome.json`` tagged with the run generation the parent
    passed in the child's arguments. No frame is sent: the parent's commit
    emits the one terminal / ``paused`` frame and the one metric.
    """

    def __init__(self, job_store: JobStore, run_gen: int) -> None:
        self._job_store = job_store
        self._run_gen = run_gen

    @property
    def token(self) -> RunToken | None:
        return None

    @property
    def persists_warnings(self) -> bool:
        return False

    def write_results(self, job: Job) -> bool:
        return True

    def commit(self, job: Job, outcome: Outcome, duration: float) -> None:
        from lizystudio.services._job_outcome_file import write_outcome

        completed = outcome.status == "completed"
        write_outcome(
            self._job_store.path_for(job.job_id, "outcome"),
            run_gen=self._run_gen,
            outcome=outcome,
            fit_result=job.fit_result if completed else None,
            tune_result=job.tune_result if completed else None,
            model_path=job.model_path if completed else None,
        )
