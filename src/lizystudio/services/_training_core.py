"""Shared training-core helpers (A-3 extraction).

Both :mod:`lizystudio.services.training` (fit/tune) and
:mod:`lizystudio.services.training_retune` (retune/resume) need the
same primitives: the cancel-aware progress callback, the job lifecycle
wrapper (``_run_job_core``), the subprocess orchestrator, plus a
handful of utility helpers. Previously the two modules shared them by
having ``training_retune`` import private symbols from ``training`` —
which introduced a logical cycle that was patched with a lazy
``from lizystudio.services.training import _run_subprocess_job``.

This module owns those primitives so neither side imports from the
other. ``training.py`` re-exports the historical names for backwards
compatibility with callers that still reach for
``from lizystudio.services.training import CancelledError`` or
``_run_job_core``.
"""

from __future__ import annotations

import copy
import io
import logging
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lizystudio.backends.base import BackendAdapter, ProgressCallback
from lizystudio.services import _child_barrier
from lizystudio.services._job_lifecycle_types import (
    AdmissionRefused,
    Outcome,
    RunToken,
)
from lizystudio.services._job_outcome import (
    ExecuteFn,
    execute_outcome,
    refusal_error_text,
)
from lizystudio.services._job_sinks import OutcomeSink, emit_effects, sync_job
from lizystudio.services._job_warnings import JobWarningCollector, capture_job_warnings
from lizystudio.services.jobs import Job, JobStore

if TYPE_CHECKING:
    from lizystudio.services.workspace import WorkspaceState
    from lizystudio.ws.progress import ProgressBroadcaster

_logger = logging.getLogger(__name__)

_JOIN_TIMEOUT = 30  # seconds — generous to allow subprocess cleanup


# Re-exported from :mod:`lizystudio.backends.exceptions` (H-0068) so
# the service and backend layers catch the exact same class.  Keeping
# the name importable here preserves ``except CancelledError`` catches
# scattered through training code and tests without an identity break.
# P-0099 v3-20c: ``PausedError`` joins the same re-export so
# ``_run_job_core`` and the cancel-aware callback share one identity
# across in-process and subprocess workers.
from lizystudio.backends.exceptions import (  # noqa: E402, F401
    BackendRefusalError,
    CancelledError,
    PausedError,
    SearchSpaceRefusedError,
    TuningFailedError,
)


class PreviousJobStillRunningError(Exception):
    """Raised when a previous job thread is still alive after join timeout."""


def _make_cancel_aware_cb(
    job_id: str,
    job_store: JobStore,
    broadcaster: ProgressBroadcaster | None,
    token: RunToken | None = None,
) -> ProgressCallback:
    """Create a progress callback that checks for cancellation (H-0011),
    a revoked run token (P-0119 INV-LC-5) and pause (P-0099 v3-20c, R-1.4).

    Every observation short-circuits the broadcaster emit so a pending
    Pause click does not leak a "trial 7/100" frame after the user
    has already requested the unwind. Cancel and a revoked token are checked
    before pause because the user's intent on a co-pending cancel+pause
    click is "stop now", not "stop here so I can resume". A revoked token
    (the run was force-released) stops the worker at its next cooperative
    point; its later commit is dropped by the primitive.
    """

    def callback(
        *,
        current: int,
        total: int,
        message: str,
        **extra: Any,
    ) -> None:
        if job_store.is_cancel_requested(job_id):
            raise CancelledError
        if token is not None and not job_store.is_live(token):
            raise CancelledError
        if job_store.is_pause_requested(job_id):
            raise PausedError
        if broadcaster is None:
            return

        def send() -> None:
            broadcaster.send_progress(
                job_id,
                current=current,
                total=total,
                message=message,
                fold_results=extra.get("fold_results"),
                trial_results=extra.get("trial_results"),
            )

        if token is None:
            # The subprocess child: its frames go to the JSONL relay, which
            # the parent fences by the run token before forwarding them.
            send()
        elif not job_store.emit_if_live(token, send):
            # Revoked after the checks above (Codex r1-4): no frame, and the
            # worker stops at this cooperative point.
            raise CancelledError

    return callback


def _subprocess_duration_seconds(job: Job) -> float:
    """Compute wall-clock duration from a subprocess-owned job.

    H-0066: the subprocess child stamps both ``created_at`` and
    ``completed_at`` in ISO-8601. The parent cannot share the thread-
    mode ``time.monotonic()`` baseline, so this helper reconstructs
    the elapsed seconds from the persisted timestamps.

    Returns 0.0 when either timestamp is missing / unparseable — safer
    than propagating an exception through the finally-block.
    """
    if job.completed_at is None:
        return 0.0
    try:
        created = datetime.fromisoformat(job.created_at)
        completed = datetime.fromisoformat(job.completed_at)
    except ValueError:
        return 0.0
    return max(0.0, (completed - created).total_seconds())


def _fail_escaped_job(
    *,
    ws: WorkspaceState,
    job: Job,
    job_store: JobStore,
    broadcaster: ProgressBroadcaster | None,
    exc: BaseException,
    epoch: int,
    token: RunToken,
) -> None:
    """Settle *job* after an exception escaped its fit/tune worker thread.

    Issue #612 (INV-1): an exception outside the core must not leave the job
    non-terminal and the slot held. P-0119: the settle is a ``failed``
    commit of the run's *token* — conditioned on it, so an old run's late
    crash cannot touch a newer run of the same job id (#765 の片割れ). A
    commit dropped because the token is no longer live does nothing; a
    commit whose own write keeps failing abandons the run (slot and flags
    released, token-conditioned) so the slot is not held forever.

    The effects (one frame, one metric) come from the commit that published
    the status, so a terminal write that failed once and succeeded here is
    still counted and broadcast once.
    """
    message = f"{type(exc).__name__}: {exc}"
    _logger.error(
        "event='training.worker_crash' job=%s type=%s",
        job.job_id,
        job.job_type,
        exc_info=exc,
    )
    error = f"worker crashed: {message}"
    try:
        effects = job_store.commit(
            token, Outcome(status="failed", error=error, message=error)
        )
    except Exception:  # noqa: BLE001 — log it; the token-conditioned cleanup runs
        _logger.exception(
            "event='training.worker_crash_settle_failed' job=%s", job.job_id
        )
        job_store.abandon(token)
        return
    if effects is not None:
        ws.note_current_job(job.job_id, epoch=epoch)
        emit_effects(job_store, broadcaster, effects)


@contextmanager
def _capture_job_logs(job_id: str) -> Iterator[io.StringIO]:
    """Attach a DEBUG ``StreamHandler`` to the per-job logger; detach on exit.

    Yields the in-memory buffer so the caller can persist captured records
    (see :func:`_persist_job_log`) while the handler is still attached.
    """
    log_buffer = io.StringIO()
    handler = logging.StreamHandler(log_buffer)
    handler.setLevel(logging.DEBUG)
    job_logger = logging.getLogger(f"lizystudio.training.{job_id}")
    job_logger.addHandler(handler)
    job_logger.setLevel(logging.DEBUG)
    try:
        yield log_buffer
    finally:
        job_logger.removeHandler(handler)
        handler.close()


def _persist_job_log(job_store: JobStore, job_id: str, captured: str) -> None:
    """Append captured Python-logger records to the job's log artifact.

    OSError here must not propagate — doing so would short-circuit the
    worker thread and leave a zombie thread handle on the workspace.

    Issue #328: append (not overwrite) so any stdout/stderr the parent's
    ``Popen`` captured into the same path stays intact — the child's stdout
    fd is closed by Python's stdio shutdown before this runs, so the parent
    has finished its writes. When *captured* is empty AND the file does not
    yet exist (in-process callers that bypass the subprocess plumbing), touch
    an empty file so callers that assert the artifact path exists keep
    working.
    """
    try:
        log_path = job_store.path_for(job_id, "log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if captured:
            with log_path.open("a", encoding="utf-8") as f:
                f.write("\n--- captured Python logger records ---\n")
                f.write(captured)
        elif not log_path.exists():
            log_path.touch()
    except OSError:
        _logger.warning(
            "Failed to persist execution log for job %s", job_id, exc_info=True
        )


def _run_job_core(
    *,
    job: Job,
    job_store: JobStore,
    broadcaster: ProgressBroadcaster | None,
    execute_fn: ExecuteFn,
    sink: OutcomeSink,
) -> Job:
    """Shared execution wrapper for fit / tune / retune jobs (P-0119).

    The core neither claims the slot nor writes ``running``: admission did
    both before the worker started. It runs *execute_fn* under captured
    logging and warnings, then hands the outcome to *sink* — the parent's
    primitive in thread mode (:class:`ParentCommitSink`), the outcome file
    in the subprocess child (:class:`ChildOutcomeSink`, P-0119 PR-B). The
    slot, the flags and the terminal metric are settled by that commit, not
    here. *job*'s lifecycle fields mirror the committed row on return.
    """
    start_time = time.monotonic()  # H-0066: wall-clock for the duration histogram
    cb = _make_cancel_aware_cb(job.job_id, job_store, broadcaster, token=sink.token)

    with _capture_job_logs(job.job_id) as log_buffer:
        # P-0117 D4: the backend's warnings raised on this thread go to the
        # job log (attached above) and to ``job.warnings``, in the thread
        # worker and in the subprocess child alike.
        collector = JobWarningCollector(
            job=job,
            job_store=job_store,
            broadcaster=broadcaster,
            job_logger=logging.getLogger(f"lizystudio.training.{job.job_id}"),
            token=sink.token,
            persist=sink.persists_warnings,
        )
        try:
            # P-0119: test-only child barrier, outside the outcome mapping, so a
            # KeyboardInterrupt here escapes the core with no outcome written.
            _child_barrier.barrier("core_entry")
            with capture_job_warnings(collector):
                outcome = execute_outcome(job, execute_fn, cb, sink)
            sink.commit(job, outcome, duration=time.monotonic() - start_time)
        finally:
            _persist_job_log(job_store, job.job_id, log_buffer.getvalue())

    return job


def _save_tuning_plot(
    backend: BackendAdapter,
    model: Any,
    job_dir: Path,
) -> None:
    """Persist the tuning plot JSON so it survives model export (H-0002 B)."""
    try:
        plot_data = backend.plot(model, "tuning")
        path = job_dir / "tuning_plot.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(plot_data.plotly_json, encoding="utf-8")
    except Exception:  # noqa: BLE001
        _logger.warning(
            "Failed to save tuning plot for %s", job_dir.name, exc_info=True
        )


def _prepare_autofit_config(
    config: dict[str, Any], best_params: dict[str, Any]
) -> dict[str, Any]:
    """Build a fit-ready config from the original config + best_params.

    A tuned inner split (``validation_ratio`` or ``inner_valid`` in
    *best_params*) is the tune's choice, not the user's, so the
    ``inner_valid_explicit`` marker is forced ``false`` over a picker's
    ``true`` (D-20261020-008 option A, P-0116 D2).  It is written only when
    ``training`` and ``training.early_stopping`` are dicts or absent; a
    malformed section is left for the backend's schema to reject.
    """
    result = copy.deepcopy(config)
    model = dict(result.get("model", {}))
    existing_params = dict(model.get("params", {}))
    model["params"] = {**existing_params, **best_params}
    result["model"] = model
    result.pop("tuning", None)
    if "validation_ratio" in best_params or "inner_valid" in best_params:
        _mark_inner_split_automatic(result)
    return result


def _mark_inner_split_automatic(config: dict[str, Any]) -> None:
    training = config.get("training", {})
    if not isinstance(training, dict):
        return
    early_stopping = training.get("early_stopping", {})
    if not isinstance(early_stopping, dict):
        return
    config["training"] = {
        **training,
        "early_stopping": {**early_stopping, "inner_valid_explicit": False},
    }


def _run_pickle_preflight(backend: BackendAdapter, job_dir: Path) -> None:
    """Run the H-0062 pre-flight check before tune launches (H-0068).

    Delegates to the adapter's :meth:`preflight_checkpoint_dir` so this
    service module never imports backend-specific symbols; translates
    the common :class:`CheckpointPreflightError` into the HTTP-facing
    :class:`PicklePreflightFailedError` envelope.
    """
    from lizystudio.api.errors import PicklePreflightFailedError
    from lizystudio.backends.exceptions import CheckpointPreflightError

    try:
        backend.preflight_checkpoint_dir(job_dir)
    except CheckpointPreflightError as exc:
        raise PicklePreflightFailedError(str(exc)) from exc


def _join_previous_thread(ws: WorkspaceState) -> None:
    """Join the previous job thread if it exists (H-0040).

    Prevents thread/OpenMP thread-pool accumulation by ensuring the prior
    worker is cleaned up before spawning a new one.

    Raises :class:`PreviousJobStillRunningError` if the thread does not
    finish within ``_JOIN_TIMEOUT`` seconds.

    The lock is intentionally released before ``join`` so a new thread
    can be registered concurrently — holding it across a 30-second join
    would serialize every /fit and /tune request.  The trade-off is a
    non-atomic read-then-act window: if another request registers a new
    thread during our join, we still join the one we observed (which is
    correct — the caller's only goal is to avoid spawning on top of a
    live predecessor).
    """
    prev = ws.previous_job_thread()
    if prev is not None and prev.is_alive():
        prev.join(timeout=_JOIN_TIMEOUT)
        if prev.is_alive():
            raise PreviousJobStillRunningError(
                f"Previous job thread did not finish within {_JOIN_TIMEOUT}s"
            )


def record_outcome(ws: WorkspaceState, finished: Job, *, epoch: int) -> None:
    """Write the post-job workspace state for *finished* (P-0114 fenced).

    Only a ``completed`` commit surfaces its results (P-0119 Q2: when a
    cancel won over a completed run, its sidecars stay on disk but the
    workspace does not record them); any other final status records the job
    with no results, as before.
    """
    completed = finished.status == "completed"
    ws.record_completion(
        fit_result=finished.fit_result if completed else None,
        tune_result=finished.tune_result if completed else None,
        job_id=finished.job_id,
        epoch=epoch,
    )


def start_worker(
    *,
    ws: WorkspaceState,
    job_store: JobStore,
    broadcaster: ProgressBroadcaster | None,
    token: RunToken,
    run: Callable[[], None],
    epoch: int,
    join: Callable[[WorkspaceState], None],
    precheck: Callable[[], str | None] | None = None,
) -> bool:
    """Join the previous worker, pass the start gate, start *run* (P-0119).

    No lock is held across the join or the thread start. *precheck* runs
    after the join (the retune launcher's workspace checks); an error it
    returns aborts the admission and starts nothing (returns False). A
    revoked token at the gate starts nothing (the operation that revoked it
    wrote the terminal) and raises :class:`AdmissionRefused`. A failed join
    or thread start aborts the admission — back to ``paused`` for an
    unpause, ``cancelled`` with CANCEL, ``failed`` otherwise — and re-raises.
    """
    try:
        join(ws)
        error = precheck() if precheck is not None else None
        if error is not None:
            emit_effects(job_store, broadcaster, job_store.abort(token, error))
            return False
        if not job_store.start(token):
            raise AdmissionRefused(token.job_id, "revoked", None)
        thread = threading.Thread(target=run, daemon=True)
        thread.start()
    except Exception as exc:
        effects = job_store.abort(token, f"{type(exc).__name__}: {exc}")
        emit_effects(job_store, broadcaster, effects)
        raise
    ws.register_job_thread(thread, epoch=epoch, job_id=token.job_id)
    return True


def _run_subprocess_job(
    ws: WorkspaceState,
    job: Job,
    job_store: JobStore,
    broadcaster: ProgressBroadcaster | None,
    *,
    token: RunToken,
    mode: str | None = None,
    parent_job_id: str | None = None,
    retune_n_trials: int | None = None,
    retune_expand_boundary: bool | None = None,
    retune_boundary_threshold: float | None = None,
    missing_data_error: str = "No data loaded — cannot run subprocess job",
    resume_from_existing_study: bool = False,
    epoch: int,
) -> Job:
    """Run a job via subprocess and update workspace state.

    Shared by fit, tune, and retune subprocess paths. *epoch* is the
    workspace reset generation the launcher captured (P-0114); workspace
    writes from an older generation are dropped. P-0119: a missing dataset
    aborts the admission; otherwise the parent's reconcile commits the
    child's result through *token* (no ``finally`` metric any more — the
    commit's effects carry the one metric).
    """
    from lizystudio.services.subprocess_runner import run_job_in_subprocess
    from lizystudio.services.workspace import get_backend_name

    if ws.data_ref is None or not ws.data_ref.path:
        emit_effects(job_store, broadcaster, job_store.abort(token, missing_data_error))
        ws.note_current_job(job.job_id, epoch=epoch)
        sync_job(job, job_store)
        return job
    extra_kwargs: dict[str, Any] = {
        "mode": mode,
        "parent_job_id": parent_job_id,
        "retune_n_trials": retune_n_trials,
        "retune_expand_boundary": retune_expand_boundary,
        "retune_boundary_threshold": retune_boundary_threshold,
    }
    extra_kwargs = {k: v for k, v in extra_kwargs.items() if v is not None}
    if resume_from_existing_study:
        extra_kwargs["resume_from_existing_study"] = True
    finished = run_job_in_subprocess(
        job=job,
        job_store=job_store,
        broadcaster=broadcaster,
        backend_name=get_backend_name(ws),
        data_path=ws.data_ref.path,
        token=token,
        **extra_kwargs,
    )
    record_outcome(ws, finished, epoch=epoch)
    return finished


__all__ = [
    "CancelledError",
    "PausedError",
    "PreviousJobStillRunningError",
    "_JOIN_TIMEOUT",
    "_join_previous_thread",
    "_make_cancel_aware_cb",
    "_prepare_autofit_config",
    "_run_job_core",
    "_run_pickle_preflight",
    "_run_subprocess_job",
    "_save_tuning_plot",
    "_subprocess_duration_seconds",
    "record_outcome",
    "refusal_error_text",
    "start_worker",
]
