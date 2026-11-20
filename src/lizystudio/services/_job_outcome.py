"""Map a worker's exit to the :class:`Outcome` it commits (P-0119).

Extracted from ``_training_core._apply_job_outcome``, which wrote the status
itself. The mapping is unchanged — ``completed`` on success, ``paused`` on
:class:`PausedError`, ``cancelled`` on :class:`CancelledError` /
``KeyboardInterrupt``, ``failed`` otherwise (with the coded refusal of
P-0117 D2 and the P-0116 codes) — but the function now only *describes* the
outcome; the sink commits it.

A ``completed`` outcome writes its results first, status-free, under the run
token; a failure of that write fails the outcome before any status is
published (P-0119, #711's thread-mode window).
"""

from __future__ import annotations

import logging
import traceback
from collections.abc import Callable
from typing import TYPE_CHECKING

from lizystudio.backends.base import ProgressCallback
from lizystudio.backends.exceptions import (
    BackendRefusalError,
    CancelledError,
    PausedError,
    SearchSpaceRefusedError,
    TuningFailedError,
)
from lizystudio.backends.types import FitSummary, TuningSummary
from lizystudio.services._job_lifecycle_types import (
    CANCELLED_CODE,
    CANCELLED_MESSAGE,
    Outcome,
)

if TYPE_CHECKING:
    from lizystudio.services._job_metadata import Job
    from lizystudio.services._job_sinks import OutcomeSink

_logger = logging.getLogger(__name__)

ExecuteFn = Callable[[ProgressCallback], tuple[FitSummary, TuningSummary | None, str]]


def execute_outcome(
    job: Job, execute_fn: ExecuteFn, cb: ProgressCallback, sink: OutcomeSink
) -> Outcome:
    """Run *execute_fn* and return the outcome to commit."""
    try:
        fit_result, tune_result, model_dir = execute_fn(cb)
    except PausedError:
        return Outcome(status="paused")
    except (CancelledError, KeyboardInterrupt):
        return Outcome(
            status="cancelled", code=CANCELLED_CODE, message=CANCELLED_MESSAGE
        )
    except BackendRefusalError as exc:
        return refusal_outcome(job, exc)
    except Exception as exc:  # noqa: BLE001 — mapped to a failed outcome
        return failure_outcome(exc)
    job.fit_result = fit_result
    job.tune_result = tune_result
    job.model_path = model_dir
    try:
        sink.write_results(job)
    except Exception as exc:  # noqa: BLE001 — the results write fails the job
        return failure_outcome(exc)
    return Outcome(status="completed")


def failure_outcome(exc: BaseException) -> Outcome:
    """``failed`` with the traceback in the error and a coded frame."""
    message = f"{type(exc).__name__}: {exc}"
    code = None
    # P-0116 D5 (#691): a tune whose trials all failed keeps its own code, so
    # the UI does not show it as a generic error; so does a refused search
    # space (P-0116 D1).
    if isinstance(exc, TuningFailedError):
        code = "TUNING_FAILED"
    elif isinstance(exc, SearchSpaceRefusedError):
        code = "SEARCH_SPACE_REFUSED"
    return Outcome(
        status="failed",
        error=f"{message}\n{traceback.format_exc()}",
        code=code,
        message=message,
    )


def refusal_error_text(exc: BackendRefusalError) -> str:
    """The job error of a classified backend refusal (P-0117 D2).

    Its Studio code, the backend's code and reason, the message and the
    remedy, on one line and without a traceback.
    """
    return (
        f"{exc.code} ({exc.backend_code}: {exc.reason}): {exc.message} "
        f"Remedy: {exc.remedy}."
    )


def refusal_outcome(job: Job, exc: BackendRefusalError) -> Outcome:
    """``failed`` with a classified backend refusal (P-0117 D2).

    The error and the frame carry the coded reason and the remedy; the
    traceback goes to the job's logger, which ``_capture_job_logs`` persists
    into ``execution.log`` in both worker modes.
    """
    text = refusal_error_text(exc)
    logging.getLogger(f"lizystudio.training.{job.job_id}").error(
        "Job failed: %s", text, exc_info=exc
    )
    return Outcome(status="failed", error=text, code=exc.code, message=text)
