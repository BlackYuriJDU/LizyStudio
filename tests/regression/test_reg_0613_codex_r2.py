"""P-0119, Codex review round 2 (#768 head 9a1b12e): three fault injections.

1. A force-release whose ``cancelled`` write fails keeps the run's
   authority: the token stays live, so the later commit still publishes a
   terminal and releases the slot.
2. A cancel (and a pause) whose flag file cannot be written is not
   acknowledged: no 200, no in-memory flag (DC1; INV-LC-4 needs a durable
   CANCEL before acceptance).
3. ``recover`` returns the effects of every terminal it publishes, and the
   server emits them once its broadcaster exists: one terminal metric and a
   replayable terminal frame per recovered job (INV-LC-2).
"""

from __future__ import annotations

import stat
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from lizystudio.backends.types import DataRef
from lizystudio.services._job_lifecycle_types import Outcome
from lizystudio.services._job_metadata import JobMetadataStore
from lizystudio.services.jobs import JobStore
from tests.regression.p0119_harness import seed_persisted_job, touch_flag

_REF = DataRef(
    source_type="path",
    path="/tmp/x.csv",
    filename="x.csv",
    fingerprint="f",
    shape=(1, 1),
)


def _admitted(store: JobStore, job_type: str = "tune") -> tuple[Any, Any]:
    reservation = store.create_and_reserve(
        backend_name="lizyml", config={}, data_ref=_REF, job_type=job_type
    )
    assert reservation is not None
    job, gen = reservation
    token = store.admit(
        job.job_id, expect="pending", reset_gen=gen, slot_rule="reserved"
    )
    return job, token


@contextmanager
def _read_only(path: Path) -> Iterator[None]:
    """Make *path* unwritable (a real fault: the flag staging file fails)."""
    mode = path.stat().st_mode
    path.chmod(stat.S_IRUSR | stat.S_IXUSR)
    try:
        yield
    finally:
        path.chmod(mode)


# --- 1. force-release keeps authority until its write is published --------


def test_a_failed_force_release_write_keeps_the_run_committable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = JobStore(tmp_path / "jobs")
    job, token = _admitted(store)
    original = JobMetadataStore.write_status_locked
    failures = {"n": 0}

    def flaky(meta: Any, job_id: str, status: str, **kwargs: Any) -> None:
        if status == "cancelled" and failures["n"] == 0:
            failures["n"] += 1
            raise OSError("synthetic disk failure")
        original(meta, job_id, status, **kwargs)

    monkeypatch.setattr(JobMetadataStore, "write_status_locked", flaky)
    with pytest.raises(OSError, match="synthetic disk failure"):
        store.force_release(job.job_id)
    assert failures["n"] == 1
    assert store.is_live(token) is True
    assert store.active_job_id == job.job_id

    effects = store.commit(token, Outcome(status="completed"))
    assert effects is not None
    assert effects.status == "completed"
    assert effects.metric_status == "completed"
    assert store.active_job_id is None


# --- 2. no acknowledgement without a durable flag --------------------------


def test_a_cancel_whose_flag_cannot_be_persisted_is_not_accepted(
    tmp_path: Path,
) -> None:
    store = JobStore(tmp_path / "jobs")
    job, token = _admitted(store)
    with _read_only(store.jobs_dir), pytest.raises(OSError):
        store.cancel(job.job_id)
    assert store.is_cancel_requested(job.job_id) is False
    assert store.is_live(token) is True


def test_a_pause_whose_flag_cannot_be_persisted_is_not_accepted(
    tmp_path: Path,
) -> None:
    store = JobStore(tmp_path / "jobs")
    job, _token = _admitted(store)
    with _read_only(store.jobs_dir), pytest.raises(OSError):
        store.pause(job.job_id)
    assert store.is_pause_requested(job.job_id) is False


@pytest.mark.parametrize(
    ("action", "code"),
    [("cancel", "CANCEL_NOT_PERSISTED"), ("pause", "PAUSE_NOT_PERSISTED")],
)
def test_the_api_answers_5xx_when_the_flag_cannot_be_persisted(
    client: TestClient, action: str, code: str
) -> None:
    store: JobStore = client.app.state.job_store  # type: ignore[attr-defined]
    job, _token = _admitted(store)
    with _read_only(store.jobs_dir):
        response = client.post(f"/api/jobs/{job.job_id}/{action}")
    assert response.status_code == 500, response.text
    assert response.json()["error"]["code"] == code
    flag = store.is_cancel_requested if action == "cancel" else store.is_pause_requested
    assert flag(job.job_id) is False


# --- 3. recover's terminals have effects ------------------------------------


def test_recover_returns_one_effect_per_published_terminal(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs")
    failed = seed_persisted_job(store, job_type="fit", status="running", data_ref=_REF)
    cancelled = seed_persisted_job(
        store, job_type="tune", status="pending", data_ref=_REF
    )
    touch_flag(store, cancelled.job_id, "cancel_flag")
    seed_persisted_job(store, job_type="fit", status="completed", data_ref=_REF)
    fresh = JobStore(tmp_path / "jobs")
    effects = fresh.recover()
    by_job = {e.job_id: e for e in effects}
    assert set(by_job) == {failed.job_id, cancelled.job_id}
    assert by_job[failed.job_id].metric_status == "failed"
    assert by_job[failed.job_id].frame == "error"
    assert by_job[cancelled.job_id].metric_status == "cancelled"
    assert by_job[cancelled.job_id].code == "JOB_CANCELLED"


def test_the_server_counts_and_replays_recovered_terminals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lizystudio.server import create_app

    jobs_dir = tmp_path / "jobs"
    orphan = seed_persisted_job(
        JobStore(jobs_dir), job_type="fit", status="running", data_ref=_REF
    )
    monkeypatch.setenv("LIZYSTUDIO_JOBS_DIR", str(jobs_dir))
    with TestClient(create_app()) as client:
        state: Any = client.app.state
        registry = state.metrics
        value = registry.jobs_total.labels(job_type="fit", status="failed")._value
        assert value.get() == 1
        cached = state.broadcaster._last_terminal.get(orphan.job_id)
        assert cached is not None
        assert cached[1]["type"] == "error"
