"""P-0119 (Issue #613): admission, cancel and the launch-failure paths.

Acceptance items pinned here (P-0119 Acceptance Criteria, lock and priority):
Race A, Race B (both orders), the launch failures, and the API compatibility
rows (a second unpause is 400, a join failure on unpause is 409, cancel of a
not-yet-admitted ``pending`` job is 200). Thread mode; the real-child halves
live in ``test_reg_0697_child_windows.py``.

Every interleaving is held with an event barrier (``Gate``) and released by
the test; the assertions count status transitions, terminal frames, terminal
metrics, slot releases and worker starts.
"""

from __future__ import annotations

import threading
from typing import Any

import pytest

from lizystudio.api import workspace as ws_api
from lizystudio.services import training
from lizystudio.services.training import PreviousJobStillRunningError
from tests.regression.p0119_harness import Call, Env, Gate, env, join_worker, metric

__all__ = ["env"]

pytestmark = pytest.mark.integration


def _code(response: Any) -> str:
    return str(response.json()["error"]["code"])


def test_race_a_second_unpause_is_refused_without_waiting(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Race A: the second unpause is 400 while the first waits in its join.

    The first unpause is held inside ``_join_previous_thread``; the second
    must not wait for that join and must not start a second worker.
    """
    job_id = env.start_paused_tune()
    gate = Gate()
    monkeypatch.setattr(
        training, "_join_previous_thread", gate.wrap(training._join_previous_thread)
    )
    first = Call(lambda: env.post(f"/api/jobs/{job_id}/unpause"))
    gate.wait_reached()

    second = env.post(f"/api/jobs/{job_id}/unpause")
    assert second.status_code == 400, second.text
    assert _code(second) == "JOB_NOT_PAUSED"

    gate.release.set()
    assert first.result().status_code == 200
    join_worker(env.client)
    # 2 tune runs: the original tune that paused, and exactly one resume.
    assert env.trainer.tune_calls == 2
    assert env.status(job_id) == "completed"


def test_race_b_cancel_after_admission_ends_cancelled_once(env: Env) -> None:
    """Race B: a worker admitted and held before its first commit, then
    cancelled, ends ``cancelled`` with one write, frame, metric and release."""
    gate = Gate()
    env.trainer.fit_gate = gate
    before = metric(env.client, "fit", "cancelled")
    r = env.post("/api/workspace/fit")
    assert r.status_code == 200, r.text
    job_id = r.json()["job_id"]
    gate.wait_reached()

    cancel = env.post(f"/api/jobs/{job_id}/cancel")
    assert cancel.status_code == 200, cancel.text
    gate.release.set()
    join_worker(env.client)

    assert env.status(job_id) == "cancelled"
    assert env.rec.writes_of(job_id, "cancelled") == 1
    assert env.rec.writes_of(job_id, "completed") == 0
    assert env.rec.terminal_frames(job_id) == ["error:JOB_CANCELLED"]
    assert metric(env.client, "fit", "cancelled") - before == 1
    assert metric(env.client, "fit", "completed") == 0
    assert env.store.active_job_id is None
    assert env.rec.slot_releases == 1
    # Q2: the fit had produced its result before the cancel won. The sidecar
    # stays on disk, but the workspace does not surface it; the job is still
    # the workspace's current job, so a reload re-attaches to it (P-0102).
    assert env.store.path_for(job_id, "fit_result").exists()
    assert env.ws.workspace_fit_result is None
    assert env.ws.current_job_id == job_id


def test_race_b_cancel_before_admission_refuses_the_launch(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Race B, reverse order: cancel of the reserved ``pending`` job is 200,
    settles it ``cancelled``, and the held launch is refused with 0 workers."""
    gate = Gate()
    monkeypatch.setattr(ws_api, "start_fit_async", gate.wrap(ws_api.start_fit_async))
    before = metric(env.client, "fit", "cancelled")
    fit = Call(lambda: env.post("/api/workspace/fit"))
    gate.wait_reached()
    job_id = env.store.active_job_id
    assert job_id is not None

    cancel = env.post(f"/api/jobs/{job_id}/cancel")
    assert cancel.status_code == 200, cancel.text
    assert env.status(job_id) == "cancelled"
    gate.release.set()

    response = fit.result()
    assert response.status_code == 409, response.text
    assert env.trainer.fit_calls == 0
    assert env.status(job_id) == "cancelled"
    assert env.rec.writes_of(job_id, "cancelled") == 1
    assert env.rec.writes_of(job_id, "running") == 0
    assert metric(env.client, "fit", "cancelled") - before == 1
    assert env.store.active_job_id is None


def test_window1_cancel_waits_for_admission_then_cancels_before_any_trial(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """窓1: admission is held inside the lifecycle lock; a cancel waits for it.
    Once admitted, the worker is held before its first callback until the
    cancel has returned 200; released, it runs 0 trials and ends cancelled.

    The seam ``JobLifecycle._admit_checks_locked`` runs under the lock and is
    new in P-0119, so this test has no RED on develop.
    """
    from lizystudio.services._job_lifecycle import JobLifecycle

    admit_gate = Gate()
    monkeypatch.setattr(
        JobLifecycle,
        "_admit_checks_locked",
        admit_gate.wrap(JobLifecycle._admit_checks_locked),
    )
    tune_gate = Gate()
    env.trainer.tune_gate = tune_gate
    tune = Call(lambda: env.post("/api/workspace/tune"))
    admit_gate.wait_reached()
    job_id = env.store.active_job_id
    assert job_id is not None
    cancel = Call(lambda: env.post(f"/api/jobs/{job_id}/cancel"))
    assert cancel.finished_within(0.3) is False, "cancel did not wait for the lock"
    admit_gate.release.set()
    assert tune.result().status_code == 200
    tune_gate.wait_reached()
    assert cancel.result().status_code == 200
    tune_gate.release.set()
    join_worker(env.client)
    assert env.trainer.trials == 0
    assert env.status(job_id) == "cancelled"
    assert env.rec.writes_of(job_id, "cancelled") == 1


def test_unpause_join_failure_is_409_and_the_job_stays_paused(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Launch failure (unpause): a failed join is 409 ``JOB_CONFLICT``; the job
    is back to ``paused`` with its slot, and a later cancel settles it."""
    job_id = env.start_paused_tune()

    def refuse(_ws: Any) -> None:
        raise PreviousJobStillRunningError("still running")

    monkeypatch.setattr(training, "_join_previous_thread", refuse)
    r = env.post(f"/api/jobs/{job_id}/unpause")
    assert r.status_code == 409, r.text
    assert _code(r) == "JOB_CONFLICT"
    assert env.status(job_id) == "paused"
    stored = env.store.get(job_id)
    assert stored is not None
    assert stored.error is None  # a clean paused row, no launch error on it
    assert env.store.active_job_id == job_id
    assert env.trainer.tune_calls == 1

    cancel = env.post(f"/api/jobs/{job_id}/cancel")
    assert cancel.status_code == 200, cancel.text
    assert env.status(job_id) == "cancelled"
    assert env.rec.writes_of(job_id, "cancelled") == 1
    assert env.store.active_job_id is None


def test_unpause_thread_start_failure_restores_paused(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Launch failure (unpause): the worker thread fails to start after
    admission. The job is ``paused`` again, keeps its slot, its run is over,
    and a later cancel settles it ``cancelled``."""
    job_id = env.start_paused_tune()
    original = threading.Thread.start

    def start(self: threading.Thread) -> None:
        target = getattr(self, "_target", None)
        if target is not None and "start_tune_async" in target.__qualname__:
            raise RuntimeError("thread start failed")
        original(self)

    monkeypatch.setattr(threading.Thread, "start", start)
    with pytest.raises(RuntimeError, match="thread start failed"):
        env.post(f"/api/jobs/{job_id}/unpause")
    monkeypatch.setattr(threading.Thread, "start", original)

    assert env.status(job_id) == "paused"
    assert env.store.active_job_id == job_id
    assert env.trainer.tune_calls == 1
    cancel = env.post(f"/api/jobs/{job_id}/cancel")
    assert cancel.status_code == 200, cancel.text
    assert env.status(job_id) == "cancelled"
    assert env.store.active_job_id is None


def test_fit_join_failure_fails_the_job_once(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Launch failure (/fit): a failed join is 409; the job is ``failed`` with
    one slot release and one failed metric."""

    def refuse(_ws: Any) -> None:
        raise PreviousJobStillRunningError("still running")

    monkeypatch.setattr(training, "_join_previous_thread", refuse)
    before = metric(env.client, "fit", "failed")
    r = env.post("/api/workspace/fit")
    assert r.status_code == 409, r.text
    jobs = env.store.list()
    assert len(jobs) == 1
    assert jobs[0].status == "failed"
    assert env.rec.writes_of(jobs[0].job_id, "failed") == 1
    assert metric(env.client, "fit", "failed") - before == 1
    assert env.rec.slot_releases == 1
    assert env.store.active_job_id is None
    assert env.trainer.fit_calls == 0
