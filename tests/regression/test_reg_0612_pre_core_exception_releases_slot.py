"""Regression for Issue #612: an exception before ``_run_job_core`` leaks INV-1.

In thread mode, ``start_fit_async`` / ``start_tune_async`` ran ``run_fit`` /
``run_tune`` with no guard. ``run_tune`` performs the pickle preflight before
``_run_job_core``, so an exception there escaped the daemon thread: the job
stayed ``pending`` (or ``paused`` on the unpause path) and the active slot the
API claimed was never released, so every later ``/fit`` and ``/tune`` got
``JOB_CONFLICT`` until a reset or restart.

INV-1 (BLUEPRINT): the active slot is released on completion OR cancel OR
exception. Each test drives an exception through the worker thread body and
asserts the job is terminal ``failed``, the slot is free, a new job can claim
it, and the failure is broadcast exactly once.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pandas as pd
import pytest

import lizystudio.services.openmp_detect as openmp_detect_mod
import lizystudio.services.training as training_mod
from lizystudio.backends.types import DataRef, FitSummary, TuningSummary
from lizystudio.services.jobs import Job, JobStore
from lizystudio.services.training import start_fit_async, start_tune_async
from lizystudio.services.workspace import WorkspaceState
from lizystudio.ws.progress import ProgressBroadcaster

pytestmark = pytest.mark.unit

_CONFIG: dict[str, Any] = {
    "task": "binary",
    "model": {"name": "lgbm", "params": {"n_estimators": 10}},
}


@pytest.fixture()
def job_store(tmp_path: Path) -> JobStore:
    return JobStore(tmp_path / "jobs")


@pytest.fixture()
def backend() -> MagicMock:
    backend = MagicMock()
    backend.create_model.return_value = MagicMock()
    backend.fit.return_value = FitSummary(
        metrics={"auc": 0.9}, fold_count=2, params=[{"n_estimators": 10}]
    )
    backend.tune.return_value = TuningSummary(
        best_params={"lr": 0.01},
        best_score=0.9,
        trials=[{"number": 0, "score": 0.9, "params": {"lr": 0.01}}],
        metric_name="auc",
        direction="maximize",
    )
    return backend


@pytest.fixture(autouse=True)
def _thread_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(openmp_detect_mod, "should_use_subprocess", lambda: False)


@pytest.fixture(autouse=True)
def _no_thread_escapes(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Fail any test whose worker thread lets an exception escape."""
    escaped: list[BaseException | None] = []
    monkeypatch.setattr(
        threading, "excepthook", lambda args: escaped.append(args.exc_value)
    )
    yield
    assert escaped == []


def _count_terminal(job_store: JobStore, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record every terminal-metric outcome the store is asked to count."""
    counted: list[str] = []
    original = job_store.record_job_terminal

    def spy(job_type: str, outcome: str, **kwargs: Any) -> None:
        counted.append(outcome)
        original(job_type, outcome, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(job_store, "record_job_terminal", spy)
    return counted


def _data_ref() -> DataRef:
    return DataRef(
        source_type="path",
        path="/data/x.csv",
        filename="x.csv",
        fingerprint="f",
        shape=(3, 2),
    )


def _claimed_job(job_store: JobStore, job_type: str) -> Job:
    job = job_store.create_and_claim_active(
        backend_name="lizyml",
        config=_CONFIG,
        data_ref=_data_ref(),
        job_type=job_type,  # type: ignore[arg-type]
    )
    assert job is not None
    assert job_store.active_job_id == job.job_id
    return job


def _join(ws: WorkspaceState) -> None:
    thread = ws._job_thread
    assert thread is not None
    thread.join(timeout=15)
    assert not thread.is_alive()


def _assert_failed_and_released(
    job_store: JobStore, broadcaster: MagicMock, job_id: str, needle: str
) -> None:
    refreshed = job_store.get(job_id)
    assert refreshed is not None
    assert refreshed.status == "failed"
    assert refreshed.error is not None and needle in refreshed.error
    assert refreshed.completed_at is not None
    assert job_store.active_job_id is None
    broadcaster.send_error.assert_called_once()
    assert broadcaster.send_error.call_args[0][0] == job_id
    # The slot is reusable: a new job claims it.
    assert (
        job_store.create_and_claim_active(
            backend_name="lizyml",
            config=_CONFIG,
            data_ref=_data_ref(),
            job_type="fit",
        )
        is not None
    )


def test_tune_preflight_exception_fails_job_and_releases_slot(
    job_store: JobStore, backend: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*_: Any, **__: Any) -> None:
        raise RuntimeError("synthetic preflight failure")

    monkeypatch.setattr(training_mod, "_run_pickle_preflight", boom)
    broadcaster = MagicMock(spec=ProgressBroadcaster)
    job = _claimed_job(job_store, "tune")
    ws = WorkspaceState(backend=backend)

    start_tune_async(
        ws=ws,
        job_store=job_store,
        broadcaster=broadcaster,
        config=_CONFIG,
        dataframe=pd.DataFrame({"x": [1, 2, 3], "y": [0, 1, 0]}),
        job=job,
    )
    _join(ws)

    _assert_failed_and_released(
        job_store, broadcaster, job.job_id, "synthetic preflight failure"
    )
    backend.tune.assert_not_called()


def test_unpause_preflight_exception_fails_paused_job_and_releases_slot(
    job_store: JobStore, backend: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The unpause path starts from ``paused`` with the slot still held."""

    def boom(*_: Any, **__: Any) -> None:
        raise RuntimeError("synthetic preflight failure")

    monkeypatch.setattr(training_mod, "_run_pickle_preflight", boom)
    broadcaster = MagicMock(spec=ProgressBroadcaster)
    job = _claimed_job(job_store, "tune")
    job.status = "paused"
    job_store.update(job)
    ws = WorkspaceState(backend=backend)

    start_tune_async(
        ws=ws,
        job_store=job_store,
        broadcaster=broadcaster,
        config=_CONFIG,
        dataframe=pd.DataFrame({"x": [1, 2, 3], "y": [0, 1, 0]}),
        job=job,
        resume_from_existing_study=True,
    )
    _join(ws)

    _assert_failed_and_released(
        job_store, broadcaster, job.job_id, "synthetic preflight failure"
    )


@pytest.mark.parametrize("job_type", ["fit", "tune"])
def test_exception_escaping_run_job_core_fails_job_and_releases_slot(
    job_store: JobStore,
    backend: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
    job_type: str,
) -> None:
    """Defence in depth: anything raised around the core, not only the
    preflight, must still leave the job terminal and the slot free."""

    def boom(**_: Any) -> Any:
        raise RuntimeError("synthetic escape")

    monkeypatch.setattr(training_mod, "_run_job_core", boom)
    broadcaster = MagicMock(spec=ProgressBroadcaster)
    job = _claimed_job(job_store, job_type)
    ws = WorkspaceState(backend=backend)
    start = start_fit_async if job_type == "fit" else start_tune_async

    start(
        ws=ws,
        job_store=job_store,
        broadcaster=broadcaster,
        config=_CONFIG,
        dataframe=pd.DataFrame({"x": [1, 2, 3], "y": [0, 1, 0]}),
        job=job,
    )
    _join(ws)

    _assert_failed_and_released(job_store, broadcaster, job.job_id, "synthetic escape")


def test_guard_does_not_overwrite_a_terminal_status(
    job_store: JobStore, backend: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A job already terminal when the exception escapes keeps its status
    (e.g. ``record_completion`` raising after the core stamped completed)."""

    def explode(*_: Any, **__: Any) -> None:
        raise RuntimeError("synthetic record_completion failure")

    broadcaster = MagicMock(spec=ProgressBroadcaster)
    job = _claimed_job(job_store, "fit")
    ws = WorkspaceState(backend=backend)
    monkeypatch.setattr(ws, "record_completion", explode)

    start_fit_async(
        ws=ws,
        job_store=job_store,
        broadcaster=broadcaster,
        config=_CONFIG,
        dataframe=pd.DataFrame({"x": [1, 2, 3], "y": [0, 1, 0]}),
        job=job,
    )
    _join(ws)

    refreshed = job_store.get(job.job_id)
    assert refreshed is not None
    assert refreshed.status == "completed"
    assert job_store.active_job_id is None
    broadcaster.send_error.assert_not_called()


def test_cleanup_runs_even_when_settling_the_job_raises(
    job_store: JobStore, backend: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The guard's own write fails too: the slot and the job's cancel and
    pause flags are still released, and nothing escapes the thread."""

    def boom(**_: Any) -> Any:
        raise RuntimeError("synthetic escape")

    def broken_update(_job: Job) -> None:
        raise OSError("synthetic disk failure")

    monkeypatch.setattr(training_mod, "_run_job_core", boom)
    broadcaster = MagicMock(spec=ProgressBroadcaster)
    job = _claimed_job(job_store, "tune")
    job_store.request_cancel(job.job_id)
    job_store.request_pause(job.job_id)
    monkeypatch.setattr(job_store, "update", broken_update)
    ws = WorkspaceState(backend=backend)

    start_tune_async(
        ws=ws,
        job_store=job_store,
        broadcaster=broadcaster,
        config=_CONFIG,
        dataframe=pd.DataFrame({"x": [1, 2, 3], "y": [0, 1, 0]}),
        job=job,
    )
    _join(ws)

    assert job_store.active_job_id is None
    assert job_store.is_cancel_requested(job.job_id) is False
    assert job_store.is_pause_requested(job.job_id) is False
    broadcaster.send_error.assert_not_called()


def test_a_failed_terminal_write_is_counted_and_broadcast_once(
    job_store: JobStore, backend: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The core decides ``failed`` and counts it, but persisting it raises.
    The guard persists and broadcasts the failure without counting it again."""
    backend.fit.side_effect = RuntimeError("synthetic fit failure")
    counted = _count_terminal(job_store, monkeypatch)
    original_update = job_store.update
    failed_writes = {"n": 0}

    def flaky_update(job_obj: Job) -> None:
        if job_obj.status == "failed" and failed_writes["n"] == 0:
            failed_writes["n"] += 1
            raise OSError("synthetic disk failure")
        original_update(job_obj)

    monkeypatch.setattr(job_store, "update", flaky_update)
    broadcaster = MagicMock(spec=ProgressBroadcaster)
    job = _claimed_job(job_store, "fit")
    ws = WorkspaceState(backend=backend)

    start_fit_async(
        ws=ws,
        job_store=job_store,
        broadcaster=broadcaster,
        config=_CONFIG,
        dataframe=pd.DataFrame({"x": [1, 2, 3], "y": [0, 1, 0]}),
        job=job,
    )
    _join(ws)

    assert failed_writes["n"] == 1
    refreshed = job_store.get(job.job_id)
    assert refreshed is not None
    assert refreshed.status == "failed"
    assert counted == ["failed"]
    broadcaster.send_error.assert_called_once()
    assert job_store.active_job_id is None


def test_subprocess_reconcile_persists_before_broadcasting(
    job_store: JobStore,
) -> None:
    """If the reconcile write raises, no error frame has been sent yet, so
    the launcher's guard sends the only one."""
    from lizystudio.services.subprocess_runner import _reconcile_subprocess_result

    job = _claimed_job(job_store, "fit")
    job.status = "running"
    job_store.update(job)
    broadcaster = MagicMock(spec=ProgressBroadcaster)

    def broken_update(_job: Job) -> None:
        raise OSError("synthetic disk failure")

    job_store.update = broken_update  # type: ignore[method-assign]
    with pytest.raises(OSError, match="synthetic disk failure"):
        _reconcile_subprocess_result(
            job=job,
            job_store=job_store,
            broadcaster=broadcaster,
            returncode=1,
        )
    broadcaster.send_error.assert_not_called()
