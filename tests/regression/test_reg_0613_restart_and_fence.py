"""P-0119 (Issue #613): restart recovery, the token fence and 窓3.

Acceptance items pinned here:

- 再起動 (3 bullets) and "recover after start is refused" (reset and
  reservations);
- #765 の片割れ, 競合する commit (token and generation fence);
- 窓3, thread half: force-release while the worker is held (a) before and
  (b) just after a callback, then another /tune takes the slot;
- 互換: ``update()`` does not change ``status``.

The restart and ``update()`` tests only use persisted rows and the store's
long-standing read/update methods. The fence tests use the primitive's own
API (``admit`` / ``commit`` / ``force_release``), which does not exist on
develop: their RED is an import or attribute error and is not counted.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import pytest

from lizystudio.backends.types import DataRef
from lizystudio.services.jobs import JobStore
from tests.regression.p0119_harness import (
    Env,
    Gate,
    env,
    join_worker,
    seed_persisted_job,
    short_reset_wait,
    touch_flag,
)

__all__ = ["env"]

_REF = DataRef(
    source_type="path",
    path="/tmp/x.csv",
    filename="x.csv",
    fingerprint="f",
    shape=(1, 1),
)


# --- restart ------------------------------------------------------------


def test_restart_cancel_accepted_before_restart_ends_cancelled(tmp_path: Path) -> None:
    """再起動: a ``running`` row whose cancel was accepted (CANCEL file) is
    recovered to ``cancelled``, not ``failed`` (Q6; INV-restart-1 changes)."""
    store = JobStore(tmp_path / "jobs")
    job = seed_persisted_job(store, job_type="fit", status="running", data_ref=_REF)
    touch_flag(store, job.job_id, "cancel_flag")
    fresh = JobStore(tmp_path / "jobs")
    fresh.reconcile_at_startup()
    recovered = fresh.get(job.job_id)
    assert recovered is not None
    assert recovered.status == "cancelled"
    assert fresh.active_job_id is None


def test_restart_paused_with_cancel_is_cancelled_and_not_reattached(
    tmp_path: Path,
) -> None:
    """再起動: a ``paused`` row with CANCEL becomes ``cancelled`` and does not
    get the slot back."""
    store = JobStore(tmp_path / "jobs")
    job = seed_persisted_job(store, job_type="tune", status="paused", data_ref=_REF)
    touch_flag(store, job.job_id, "cancel_flag")
    fresh = JobStore(tmp_path / "jobs")
    fresh.reconcile_at_startup()
    recovered = fresh.get(job.job_id)
    assert recovered is not None
    assert recovered.status == "cancelled"
    assert fresh.active_job_id is None


def test_restart_paused_without_cancel_gets_the_slot_back(tmp_path: Path) -> None:
    """再起動: a CANCEL-less ``paused`` survivor is re-attached to the slot."""
    store = JobStore(tmp_path / "jobs")
    job = seed_persisted_job(store, job_type="tune", status="paused", data_ref=_REF)
    fresh = JobStore(tmp_path / "jobs")
    fresh.reconcile_at_startup()
    recovered = fresh.get(job.job_id)
    assert recovered is not None
    assert recovered.status == "paused"
    assert fresh.active_job_id == job.job_id


def test_startup_logs_the_worker_mode_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Startup names the worker mode in one INFO line (P-0119 plan §0.1),
    and the app's store is started (``recover`` closed) once serving."""
    from fastapi.testclient import TestClient

    from lizystudio.server import create_app

    monkeypatch.setenv("LIZYSTUDIO_JOBS_DIR", str(tmp_path / "jobs"))
    from lizystudio.services._job_lifecycle import RecoverRefusedError

    with (
        caplog.at_level("INFO", logger="uvicorn.error"),
        TestClient(create_app()) as client,
    ):
        store = client.app.state.job_store  # type: ignore[attr-defined]
        with pytest.raises(RecoverRefusedError):
            store.recover()
    lines = [r for r in caplog.records if "server.worker_mode" in r.getMessage()]
    assert len(lines) == 1
    assert lines[0].getMessage().endswith("mode=thread")


def test_recover_after_start_is_refused(tmp_path: Path) -> None:
    """reset and reservations: ``recover`` called after startup is refused."""
    from lizystudio.services._job_lifecycle import RecoverRefusedError

    store = JobStore(tmp_path / "jobs")
    store.recover()
    store.mark_started()
    with pytest.raises(RecoverRefusedError):
        store.recover()


# --- update() is status-free --------------------------------------------


def test_update_does_not_change_status(tmp_path: Path) -> None:
    """互換: ``update()`` persists results and warnings but never ``status``."""
    store = JobStore(tmp_path / "jobs")
    job = store.create(backend_name="lizyml", config={}, data_ref=_REF, job_type="fit")
    job.status = "completed"
    job.warnings = ["w"]
    store.update(job)
    stored = store.get(job.job_id)
    assert stored is not None
    assert stored.status == "pending"
    assert stored.warnings == ["w"]


# --- token fence ----------------------------------------------------------


def _admitted(store: JobStore, job_type: str = "fit") -> tuple[Any, Any]:
    reservation = store.create_and_reserve(
        backend_name="lizyml", config={}, data_ref=_REF, job_type=job_type
    )
    assert reservation is not None
    job, gen = reservation
    token = store.admit(
        job.job_id, expect="pending", reset_gen=gen, slot_rule="reserved"
    )
    return job, token


@pytest.mark.parametrize("status", ["completed", "failed", "cancelled"])
def test_competing_commits_publish_once(tmp_path: Path, status: str) -> None:
    """競合する commit: after a commit, the same token and a revoked token are
    refused; one status write, one effect (frame + metric). A revoked token
    cannot clear CANCEL / PAUSE either."""
    from lizystudio.services._job_lifecycle import Outcome, RunToken

    store = JobStore(tmp_path / "jobs")
    job, token = _admitted(store)
    first = store.commit(token, Outcome(status=status))
    assert first is not None
    assert first.status == status
    assert first.metric_status == status
    store.request_cancel(job.job_id)
    store.request_pause(job.job_id)
    assert store.commit(token, Outcome(status="failed")) is None
    stale = RunToken(job_id=job.job_id, run_gen=token.run_gen - 1)
    assert store.commit(stale, Outcome(status="completed")) is None
    stored = store.get(job.job_id)
    assert stored is not None
    assert stored.status == status
    assert store.is_cancel_requested(job.job_id) is True
    assert store.is_pause_requested(job.job_id) is True


def test_a_paused_commit_revokes_its_token(tmp_path: Path) -> None:
    """pause: after a ``paused`` commit the same token's commit is refused,
    the slot stays with the job, and PAUSE is cleared."""
    from lizystudio.services._job_lifecycle import Outcome

    store = JobStore(tmp_path / "jobs")
    job, token = _admitted(store, "tune")
    store.request_pause(job.job_id)
    effects = store.commit(token, Outcome(status="paused"))
    assert effects is not None
    assert effects.status == "paused"
    assert effects.metric_status is None
    assert store.commit(token, Outcome(status="completed")) is None
    assert store.active_job_id == job.job_id
    assert store.is_pause_requested(job.job_id) is False
    stored = store.get(job.job_id)
    assert stored is not None
    assert stored.status == "paused"


def test_delete_release_refuses_a_root_with_a_live_run(tmp_path: Path) -> None:
    """Q5 (delete through the primitive): the route's ``running`` check reads
    the row outside the lock, so the release refuses in-lock when the job it
    deletes has a live run, and changes nothing (slot, flags, token)."""
    store = JobStore(tmp_path / "jobs")
    job, token = _admitted(store)
    assert store.release_for_delete([job.job_id]) is False
    assert store.active_job_id == job.job_id
    assert store.is_cancel_requested(job.job_id) is False
    assert store.is_live(token) is True


def test_fail_escaped_job_with_a_revoked_token_changes_nothing(tmp_path: Path) -> None:
    """#765 の片割れ: a new run of the same job id holds the slot with a
    cancel pending; ``_fail_escaped_job`` with the old run's revoked token
    leaves the slot, CANCEL and the status unchanged."""
    from lizystudio.backends.registry import get_adapter
    from lizystudio.services._job_lifecycle import Outcome
    from lizystudio.services._training_core import _fail_escaped_job
    from lizystudio.services.workspace import WorkspaceState

    store = JobStore(tmp_path / "jobs")
    job, old = _admitted(store, "tune")
    assert store.commit(old, Outcome(status="paused")) is not None
    new = store.admit(
        job.job_id,
        expect="paused",
        reset_gen=store.capture_reset_gen(),
        slot_rule="own",
    )
    assert store.cancel(job.job_id).accepted is True
    ws = WorkspaceState(backend=get_adapter("lizyml"))
    _fail_escaped_job(
        ws=ws,
        job=job,
        job_store=store,
        broadcaster=None,
        exc=RuntimeError("late crash of the old run"),
        epoch=ws.current_epoch(),
        token=old,
    )
    assert store.active_job_id == job.job_id
    assert store.is_cancel_requested(job.job_id) is True
    assert store.is_live(new) is True
    stored = store.get(job.job_id)
    assert stored is not None
    assert stored.status == "running"


# --- 窓3 (thread half) ----------------------------------------------------


@pytest.mark.parametrize("where", ["before_callback", "after_callback"])
def test_window3_force_released_worker_commits_nothing(
    env: Env, monkeypatch: pytest.MonkeyPatch, where: str
) -> None:
    """窓3 (thread): a worker held (a) before its first callback or (b) just
    after one is force-released by reset; another /tune takes the slot. The
    old worker runs at most the trial it was in, and its commit, warnings and
    frames are dropped: one ``cancelled`` write, the new job untouched."""
    old_gate = Gate()
    if where == "before_callback":
        env.trainer.tune_gate = old_gate
    else:
        env.trainer.tune_cb_calls = 2
        env.trainer.between_trials_gate = old_gate
    old_id = env.post("/api/workspace/tune").json()["job_id"]
    old_gate.wait_reached()
    with short_reset_wait(monkeypatch, 0.3):
        env.reset()
    assert env.status(old_id) == "cancelled"
    env.reseed()
    new_gate = Gate()
    env.trainer.tune_gate = new_gate
    env.trainer.between_trials_gate = None
    new_id = env.post("/api/workspace/tune").json()["job_id"]
    new_gate.wait_reached()

    monkeypatch.setattr(
        env.trainer, "_trial", _warn_then(env.trainer._trial, "late old-run warning")
    )
    old_gate.release.set()
    old_trials = 0 if where == "before_callback" else 1
    _wait_old_worker_done(env, old_id)
    assert env.trainer.trials == old_trials
    assert env.status(old_id) == "cancelled"
    assert env.rec.writes_of(old_id, "cancelled") == 1
    assert env.rec.terminal_frames(old_id) == ["error:JOB_CANCELLED"]
    old = env.store.get(old_id)
    assert old is not None
    assert old.warnings == []
    assert env.store.active_job_id == new_id
    assert env.status(new_id) == "running"
    new_gate.release.set()
    join_worker(env.client)
    assert env.status(new_id) == "completed"


def _warn_then(trial: Any, text: str) -> Any:
    def wrapped(cb: Any, i: int, total: int) -> None:
        warnings.warn(text, UserWarning, stacklevel=1)
        trial(cb, i, total)

    return wrapped


def _wait_old_worker_done(env: Env, old_id: str) -> None:
    """Join every live worker thread except the new job's (held) one."""
    import threading

    for thread in threading.enumerate():
        target = getattr(thread, "_target", None)
        if target is None or "start_tune_async" not in target.__qualname__:
            continue
        if thread is env.ws.previous_job_thread():
            continue
        thread.join(10)
        assert not thread.is_alive(), f"old worker of {old_id} did not finish"
