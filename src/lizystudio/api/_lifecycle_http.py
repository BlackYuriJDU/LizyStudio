"""HTTP glue of the job lifecycle primitive (P-0119).

Maps admission refusals to their HTTP answers, settles a reservation whose
launch failed, and runs the reset choreography (cancel the holder, wait for
its slot, force-release). Shared by ``api/workspace.py``, ``api/jobs.py``
and ``api/retune.py`` so each handler stays a thin sequence of calls.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

from lizystudio.api.errors import JobConflictError, StudioError
from lizystudio.services._job_lifecycle_types import AdmissionRefused
from lizystudio.services._job_sinks import emit_effects
from lizystudio.services._training_core import PreviousJobStillRunningError

if TYPE_CHECKING:
    from lizystudio.services.jobs import Job, JobStore
    from lizystudio.ws.progress import ProgressBroadcaster

_log = logging.getLogger(__name__)


def unpause_refusal(job_id: str, exc: AdmissionRefused) -> StudioError:
    """400 ``JOB_NOT_PAUSED`` when the job is no longer ``paused``; 409
    ``JOB_CONFLICT`` for every other refusal (generation, start gate)."""
    if exc.reason == "status":
        return StudioError(
            "JOB_NOT_PAUSED",
            f"Unpause requires a paused job (status: {exc.status})",
            400,
        )
    return JobConflictError(job_id)


def launch_reserved(
    job_store: JobStore,
    broadcaster: ProgressBroadcaster | None,
    job: Job,
    launch: Callable[[], str],
) -> str:
    """Run *launch* for a reserved ``pending`` job (``/fit``, ``/tune``,
    ``/retune``, ``/resume``).

    A refused admission and a failed join answer 409 ``JOB_CONFLICT``; any
    other error is re-raised. Either way a job still ``pending`` without a
    run is settled ``failed`` (one frame, one metric); a job the launcher
    already admitted and aborted is left as the abort wrote it.
    """
    try:
        return launch()
    except (AdmissionRefused, PreviousJobStillRunningError) as exc:
        _settle(job_store, broadcaster, job, f"{type(exc).__name__}: {exc}")
        raise JobConflictError(job.job_id) from None
    except Exception as exc:
        _settle(job_store, broadcaster, job, f"{type(exc).__name__}: {exc}")
        raise


def _settle(
    job_store: JobStore,
    broadcaster: ProgressBroadcaster | None,
    job: Job,
    error: str,
) -> None:
    emit_effects(
        job_store, broadcaster, job_store.settle_reservation(job.job_id, error)
    )


def teardown_active_job(
    job_store: JobStore,
    broadcaster: ProgressBroadcaster | None,
    *,
    timeout: float,
    interval: float,
) -> None:
    """Reset's lifecycle half (P-0119, INV-LC-6).

    One lock section advances the lifecycle reset generation and cancels the
    slot holder (CANCEL for a live run, a ``cancelled`` settle for a
    token-less ``pending`` / ``paused`` holder). Then, outside the lock, wait
    up to *timeout* for the slot to drain, and force-release whatever still
    holds it (Q1: a thread-mode worker may run on to its next cooperative
    point, committing nothing).
    """
    for effects in job_store.reset_cancel_active():
        emit_effects(job_store, broadcaster, effects)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        holder = job_store.active_job_id
        if holder is None:
            return
        if job_store.holder_is_settled(holder):
            _force_release(job_store, broadcaster, holder, waited=False)
            return
        time.sleep(interval)
    holder = job_store.active_job_id
    if holder is not None:
        _force_release(job_store, broadcaster, holder, waited=True)


def _force_release(
    job_store: JobStore,
    broadcaster: ProgressBroadcaster | None,
    holder: str,
    *,
    waited: bool,
) -> None:
    if waited:
        _log.warning(
            "workspace_reset: active slot %s did not release in time; "
            "force-released so the next fit / tune does not JOB_CONFLICT",
            holder,
        )
    emit_effects(job_store, broadcaster, job_store.force_release(holder))
