"""Re-tune / Resume launcher service (H-0062 Phase B).

Extracted from ``training.py`` (H-0062 cleanup) so the retune-specific
helpers can evolve without inflating the shared training module.

Public entry points:
- ``run_retune`` -- synchronous execution body shared by both the thread
  and subprocess paths.
- ``start_retune_async`` -- background launcher used by the API layer.

Internal helpers (``_copy_checkpoint_to_child``,
``_retune_worker``, ``_run_retune_subprocess``) are kept as
module-private functions and are not part of the public surface.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from lizystudio.backends.base import BackendAdapter, ProgressCallback
from lizystudio.backends.types import FitSummary, TuningSummary
from lizystudio.services._job_lifecycle_types import AdmissionRefused, RunToken
from lizystudio.services._job_sinks import OutcomeSink, ParentCommitSink, emit_effects
from lizystudio.services._training_core import (
    _fail_escaped_job,
    _join_previous_thread,
    _prepare_autofit_config,
    _run_job_core,
    _run_pickle_preflight,
    _run_subprocess_job,
    _save_tuning_plot,
    record_outcome,
    start_worker,
)
from lizystudio.services.jobs import Job, JobStore

if TYPE_CHECKING:
    from lizystudio.services.workspace import WorkspaceState
    from lizystudio.ws.progress import ProgressBroadcaster

_logger = logging.getLogger(__name__)


def _copy_checkpoint_to_child(parent_dir: Path, child_dir: Path) -> None:
    """Copy the parent's model.pkl and model_meta.json into the child dir."""
    import shutil

    child_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("model.pkl", "model_meta.json"):
        src = parent_dir / filename
        if src.exists():
            shutil.copy2(src, child_dir / filename)


def run_retune(
    *,
    parent_job: Job,
    child_job: Job,
    job_store: JobStore,
    backend: BackendAdapter,
    dataframe: Any,
    n_trials: int,
    expand_boundary: bool | None,
    boundary_threshold: float | None,
    broadcaster: ProgressBroadcaster | None = None,
    sink: OutcomeSink,
) -> Job:
    """Execute a Re-tune / Resume child job (H-0062); outcome to *sink*.

    Copies the parent's ``model.pkl`` into the child's job directory,
    loads it via the adapter's ``load_checkpoint``, then calls
    ``backend.tune`` with a ``re_tune`` override so the Optuna study is
    continued for the requested n_trials. Subsequent trials are saved
    back to the child's own checkpoint via the standard bridge callback.
    """
    parent_dir = job_store.job_dir(parent_job.job_id)
    child_dir = job_store.job_dir(child_job.job_id)
    _copy_checkpoint_to_child(parent_dir, child_dir)
    _run_pickle_preflight(backend, child_dir)

    def execute(
        cb: ProgressCallback,
    ) -> tuple[FitSummary, TuningSummary | None, str]:
        # CRITICAL-1: constrain cloudpickle deserialization to the
        # studio's own jobs directory so a compromised checkpoint file
        # elsewhere on disk cannot trigger arbitrary code execution.
        model = backend.load_checkpoint(child_dir, allowed_root=job_store.jobs_dir)
        # H-0062: single-round resume. We pass resume=True so the loaded
        # Optuna study is continued for the requested n_trials instead
        # of being thrown away. n_trials and the boundary-expand kwargs
        # are applied to the FIRST round (which is the only round here).
        re_tune_block: dict[str, Any] = {
            "n_rounds": 1,
            "n_trials": n_trials,
        }
        if expand_boundary is not None:
            re_tune_block["expand_boundary"] = expand_boundary
        if boundary_threshold is not None:
            re_tune_block["boundary_threshold"] = boundary_threshold

        tune_result: TuningSummary = backend.tune(
            model,
            on_progress=cb,
            re_tune=re_tune_block,
            checkpoint_dir=child_dir,
            resume=True,
        )

        _save_tuning_plot(backend, model, child_dir)

        fit_config = _prepare_autofit_config(child_job.config, tune_result.best_params)
        model2 = backend.create_model(fit_config, dataframe)
        fit_result: FitSummary = backend.fit(model2, on_progress=cb)
        model_dir = str(child_dir / "model")
        backend.export_model(model2, model_dir)
        return fit_result, tune_result, model_dir

    return _run_job_core(
        job=child_job,
        job_store=job_store,
        broadcaster=broadcaster,
        execute_fn=execute,
        sink=sink,
    )


def _run_retune_subprocess(
    *,
    ws: WorkspaceState,
    job_store: JobStore,
    broadcaster: ProgressBroadcaster | None,
    parent_job: Job,
    child_job: Job,
    n_trials: int,
    expand_boundary: bool | None,
    boundary_threshold: float | None,
    epoch: int,
    token: RunToken,
) -> Job:
    """Execute a Re-tune child job in a subprocess (H-0062).

    Delegates to the shared ``_run_subprocess_job`` orchestrator
    (Issue #118); a missing file-backed dataset aborts the admission there.
    """
    return _run_subprocess_job(
        ws,
        child_job,
        job_store,
        broadcaster,
        token=token,
        mode="retune",
        parent_job_id=parent_job.job_id,
        retune_n_trials=n_trials,
        retune_expand_boundary=expand_boundary,
        retune_boundary_threshold=boundary_threshold,
        missing_data_error=(
            "subprocess retune requires a file-backed dataset "
            "(ws.data_ref.path is empty)"
        ),
        epoch=epoch,
    )


def _retune_inline_error(ws: WorkspaceState, use_subprocess: bool) -> str | None:
    """Why the retune cannot start with the workspace as it is, or ``None``."""
    if ws.dataframe is None:
        return "No data loaded in workspace; cannot re-tune"
    if use_subprocess and (ws.data_ref is None or not ws.data_ref.path):
        # Subprocess path reads the dataframe from disk in the child, so an
        # in-memory-only dataset cannot be re-tuned under OpenMP without
        # first being saved; the thread path would be ~8x slower.
        return (
            "Re-tune requires a file-backed dataset when OpenMP is active; "
            "please re-upload or save the current data before retrying"
        )
    return None


def start_retune_async(
    *,
    ws: WorkspaceState,
    job_store: JobStore,
    broadcaster: ProgressBroadcaster,
    parent_job: Job,
    child_job: Job,
    n_trials: int,
    expand_boundary: bool | None,
    boundary_threshold: float | None,
    mode: Literal["retune", "resume"],
    reset_gen: int,
) -> str:
    """Admit the Re-tune / Resume child and spawn its worker (H-0062, P-0119).

    Admission happens here at the launcher entry, before the join: the child
    claims the slot when it is empty (thread and subprocess mode alike), and
    a slot held by another job, or a *reset_gen* older than the current one
    (a reset ran after ``create_child``), refuses it with
    :class:`AdmissionRefused` and settles the child. The per-parent lock the
    API acquired is released when the worker ends; on a refusal the API's
    own ``finally`` releases it.

    When ``should_use_subprocess()`` is True (OpenMP present), the work runs
    in a child process to avoid the daemon-thread OpenMP slowdown (H-0062
    Bugfix 2026-04-14).
    """
    from lizystudio.services.openmp_detect import should_use_subprocess

    # P-0114: the workspace reset generation, captured before the join.
    epoch = ws.current_epoch()
    try:
        token = job_store.admit(
            child_job.job_id, expect="pending", reset_gen=reset_gen, slot_rule="claim"
        )
    except AdmissionRefused as exc:
        emit_effects(job_store, broadcaster, exc.effects)
        raise
    use_subprocess = should_use_subprocess()
    run = _retune_worker(
        ws=ws,
        job_store=job_store,
        broadcaster=broadcaster,
        parent_job=parent_job,
        child_job=child_job,
        params=(n_trials, expand_boundary, boundary_threshold),
        mode=mode,
        epoch=epoch,
        token=token,
        use_subprocess=use_subprocess,
    )
    started = start_worker(
        ws=ws,
        job_store=job_store,
        broadcaster=broadcaster,
        token=token,
        run=run,
        epoch=epoch,
        join=lambda w: _join_previous_thread(w),
        precheck=lambda: _retune_inline_error(ws, use_subprocess),
    )
    if not started:
        # Early-fail path (after the join, as before P-0119): the abort
        # settled the child; point the workspace at it and free the parent.
        ws.note_current_job(child_job.job_id, epoch=epoch)
        job_store.release_parent_lock(parent_job.job_id)
    return child_job.job_id


def _retune_worker(
    *,
    ws: WorkspaceState,
    job_store: JobStore,
    broadcaster: ProgressBroadcaster,
    parent_job: Job,
    child_job: Job,
    params: tuple[int, bool | None, float | None],
    mode: str,
    epoch: int,
    token: RunToken,
    use_subprocess: bool,
) -> Callable[[], None]:
    """The retune worker thread body."""
    n_trials, expand_boundary, boundary_threshold = params
    retune: dict[str, Any] = {
        "ws": ws,
        "job_store": job_store,
        "broadcaster": broadcaster,
        "parent_job": parent_job,
        "child_job": child_job,
        "n_trials": n_trials,
        "expand_boundary": expand_boundary,
        "boundary_threshold": boundary_threshold,
    }

    def _run() -> None:
        # H-0062: always point the workspace at the child so a failed /
        # cancelled / crashed retune still surfaces in the Results Panel.
        ws.note_current_job(child_job.job_id, epoch=epoch)
        try:
            if use_subprocess:
                _run_retune_subprocess(**retune, epoch=epoch, token=token)
            else:
                _retune_in_thread(retune, epoch=epoch, token=token)
        except Exception as exc:  # noqa: BLE001
            _settle_retune_crash(retune, exc, mode=mode, epoch=epoch, token=token)
        finally:
            # Always release the per-parent lock, even when run_retune raised.
            job_store.release_parent_lock(parent_job.job_id)
            _logger.info(
                "event='retune.finished' mode=%s parent=%s child=%s",
                mode,
                parent_job.job_id,
                child_job.job_id,
            )

    return _run


def _retune_in_thread(retune: dict[str, Any], *, epoch: int, token: RunToken) -> None:
    """Thread mode: run the retune in this worker and record its outcome."""
    ws = retune["ws"]
    job_store = retune["job_store"]
    broadcaster = retune["broadcaster"]
    finished = run_retune(
        parent_job=retune["parent_job"],
        child_job=retune["child_job"],
        job_store=job_store,
        backend=ws.backend,
        dataframe=ws.dataframe,
        n_trials=retune["n_trials"],
        expand_boundary=retune["expand_boundary"],
        boundary_threshold=retune["boundary_threshold"],
        broadcaster=broadcaster,
        sink=ParentCommitSink(job_store, broadcaster, token),
    )
    record_outcome(ws, finished, epoch=epoch)


def _settle_retune_crash(
    retune: dict[str, Any],
    exc: Exception,
    *,
    mode: str,
    epoch: int,
    token: RunToken,
) -> None:
    """H-0062 Bugfix 2026-04-14 (6): an unexpected exception must still settle
    the child; P-0119: a ``failed`` commit of its token, which is a no-op once
    the core has committed."""
    _logger.exception(
        "event='retune.worker_crash' mode=%s parent=%s child=%s",
        mode,
        retune["parent_job"].job_id,
        retune["child_job"].job_id,
    )
    _fail_escaped_job(
        ws=retune["ws"],
        job=retune["child_job"],
        job_store=retune["job_store"],
        broadcaster=retune["broadcaster"],
        exc=exc,
        epoch=epoch,
        token=token,
    )
