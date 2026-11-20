"""P-0119 (Issue #613): pause rules, reset and reservations, 窓2, and the
subprocess retune slot.

Acceptance items pinned here (P-0119 Acceptance Criteria, lock and
priority): pause (4 bullets), reset and reservations (the /fit reservation,
the retune child at its two stop positions, the unpause A/B generation
case), 窓2 (unpause held in its join while reset force-releases), and the
subprocess retune slot. Thread mode unless a test says otherwise.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from lizystudio.api import jobs as jobs_api
from lizystudio.api import workspace as ws_api
from lizystudio.services import subprocess_runner, training
from lizystudio.services._job_lifecycle_types import Outcome
from lizystudio.services._job_outcome_file import write_outcome
from tests.regression.p0119_harness import (
    Call,
    Env,
    Gate,
    env,
    join_worker,
    seed_persisted_job,
    short_reset_wait,
)

__all__ = ["env"]

pytestmark = pytest.mark.integration


def _code(response: Any) -> str:
    return str(response.json()["error"]["code"])


# --- pause --------------------------------------------------------------


def test_pause_and_cancel_both_requested_end_cancelled(env: Env) -> None:
    """pause: PAUSE and CANCEL both set before the callback → ``cancelled``."""
    gate = Gate()
    env.trainer.tune_gate = gate
    job_id = env.post("/api/workspace/tune").json()["job_id"]
    gate.wait_reached()
    assert env.post(f"/api/jobs/{job_id}/pause").status_code == 200
    assert env.post(f"/api/jobs/{job_id}/cancel").status_code == 200
    gate.release.set()
    join_worker(env.client)
    assert env.status(job_id) == "cancelled"
    assert env.rec.writes_of(job_id, "paused") == 0
    assert env.rec.writes_of(job_id, "cancelled") == 1
    assert env.store.active_job_id is None


def test_pause_then_completed_commit_clears_the_pause_flag(env: Env) -> None:
    """pause: a worker that completes with PAUSE set ends ``completed`` and
    PAUSE is cleared (memory and file)."""
    gate = Gate()
    env.trainer.tune_gate = gate
    env.trainer.tune_cb_calls = 0
    job_id = env.post("/api/workspace/tune").json()["job_id"]
    gate.wait_reached()
    assert env.post(f"/api/jobs/{job_id}/pause").status_code == 200
    gate.release.set()
    join_worker(env.client)
    assert env.status(job_id) == "completed"
    assert env.store.is_pause_requested(job_id) is False
    assert not env.store.path_for(job_id, "pause_flag").exists()


def test_pause_of_pending_paused_and_terminal_jobs_is_400(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """pause: a ``pending`` (reserved, not admitted), ``paused`` or terminal
    job refuses pause with 400."""
    gate = Gate()
    monkeypatch.setattr(ws_api, "start_tune_async", gate.wrap(ws_api.start_tune_async))
    tune = Call(lambda: env.post("/api/workspace/tune"))
    gate.wait_reached()
    pending_id = env.store.active_job_id
    assert pending_id is not None
    assert env.post(f"/api/jobs/{pending_id}/pause").status_code == 400
    env.trainer.tune_gate = None
    gate.release.set()
    assert tune.result().status_code == 200
    join_worker(env.client)
    assert env.status(pending_id) == "completed"
    assert env.post(f"/api/jobs/{pending_id}/pause").status_code == 400

    paused_id = env.start_paused_tune()
    r = env.post(f"/api/jobs/{paused_id}/pause")
    assert r.status_code == 400
    assert _code(r) == "JOB_NOT_RUNNING"


# --- reset and reservations ---------------------------------------------


def test_reset_between_reservation_and_admission_cancels_the_fit(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """reset: a /fit held between its reservation and admission is settled
    ``cancelled`` by reset, and its admission is refused (0 workers)."""
    gate = Gate()
    monkeypatch.setattr(ws_api, "start_fit_async", gate.wrap(ws_api.start_fit_async))
    with short_reset_wait(monkeypatch, 0.3):
        fit = Call(lambda: env.post("/api/workspace/fit"))
        gate.wait_reached()
        job_id = env.store.active_job_id
        assert job_id is not None
        env.reset()
        assert env.status(job_id) == "cancelled"
        gate.release.set()
        assert fit.result().status_code == 409
    assert env.trainer.fit_calls == 0
    assert env.status(job_id) == "cancelled"
    assert env.rec.writes_of(job_id, "cancelled") == 1
    assert env.rec.writes_of(job_id, "running") == 0


def _seed_completed_tune_parent(env: Env) -> str:
    parent = seed_persisted_job(
        env.store,
        job_type="tune",
        status="completed",
        config=env.config,
        data_ref=env.ws.data_ref,
    )
    (env.store.job_dir(parent.job_id) / "model.pkl").write_bytes(b"fake")
    return parent.job_id


@pytest.mark.parametrize("position", ["launcher_entry", "after_create"])
def test_reset_refuses_a_retune_child_created_before_it(
    env: Env, monkeypatch: pytest.MonkeyPatch, position: str
) -> None:
    """reset: a retune child created before a reset is refused admission by
    the reset generation, at both stop positions (launcher entry, and after
    ``job_store.create`` before the launcher). 0 workers start."""
    parent_id = _seed_completed_tune_parent(env)
    gate = Gate()
    if position == "launcher_entry":
        monkeypatch.setattr(
            training, "start_retune_async", gate.wrap(training.start_retune_async)
        )
    else:
        monkeypatch.setattr(
            env.store, "rebind_parent_lock", gate.wrap(env.store.rebind_parent_lock)
        )
    with short_reset_wait(monkeypatch, 0.3):
        retune = Call(
            lambda: env.post(f"/api/jobs/{parent_id}/retune", json={"n_trials": 2})
        )
        gate.wait_reached()
        env.reset()
        env.reseed()
        gate.release.set()
        response = retune.result()
    join_worker(env.client)
    assert response.status_code == 409, response.text
    assert env.trainer.tune_calls == 0
    children = env.store.get_child_job_ids(parent_id)
    assert len(children) == 1
    assert env.status(children[0]) == "cancelled"
    assert env.store.active_job_id is None


def test_unpause_with_a_pre_reset_generation_is_refused(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """reset: unpause A holds generation g in its validation; reset moves to
    g+1 and settles the paused job; unpause B is 400; A is refused by its
    generation. 0 workers start and ``cancelled`` is written once."""
    job_id = env.start_paused_tune()
    gate = Gate()
    monkeypatch.setattr(
        jobs_api, "verify_unpause_study", gate.wrap(jobs_api.verify_unpause_study)
    )
    with short_reset_wait(monkeypatch, 0.3):
        first = Call(lambda: env.post(f"/api/jobs/{job_id}/unpause"))
        gate.wait_reached()
        env.reset()
        env.reseed()
        second = env.post(f"/api/jobs/{job_id}/unpause")
        assert second.status_code == 400, second.text
        assert _code(second) == "JOB_NOT_PAUSED"
        gate.release.set()
        assert first.result().status_code == 409
    join_worker(env.client)
    assert env.trainer.tune_calls == 1
    assert env.status(job_id) == "cancelled"
    assert env.rec.writes_of(job_id, "cancelled") == 1


def test_window2_reset_during_the_unpause_join_stops_the_start(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """窓2: unpause held in ``_join_previous_thread``; reset runs to the end
    (force-release after its wait). Released, the start gate sees the revoked
    token and starts no worker; ``cancelled`` is written once."""
    job_id = env.start_paused_tune()
    gate = Gate()
    monkeypatch.setattr(
        training, "_join_previous_thread", gate.wrap(training._join_previous_thread)
    )
    with short_reset_wait(monkeypatch, 0.3):
        unpause = Call(lambda: env.post(f"/api/jobs/{job_id}/unpause"))
        gate.wait_reached()
        env.reset()
        assert env.status(job_id) == "cancelled"
        gate.release.set()
        assert unpause.result().status_code == 409
    join_worker(env.client)
    assert env.trainer.tune_calls == 1
    assert env.status(job_id) == "cancelled"
    assert env.rec.writes_of(job_id, "cancelled") == 1
    assert env.store.active_job_id is None


# --- subprocess retune slot ---------------------------------------------


def test_subprocess_retune_holds_the_parent_slot(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """subprocess retune slot: while a subprocess-mode retune child runs, the
    parent's slot is held by the child job, a /tune is 409, and the slot is
    released exactly once when it ends."""
    monkeypatch.setattr(
        "lizystudio.services.openmp_detect.should_use_subprocess", lambda: True
    )
    parent_id = _seed_completed_tune_parent(env)
    gate = Gate()

    def fake_supervise(*, job: Any, job_store: Any, args_path: str, **_: Any) -> int:
        # The child's run: held, then it reports ``completed`` in its outcome
        # file (P-0119 PR-B: the child writes no status).
        gate.hold()
        write_outcome(
            job_store.path_for(job.job_id, "outcome"),
            run_gen=json.loads(Path(args_path).read_text())["run_gen"],
            outcome=Outcome(status="completed"),
            fit_result=None,
            tune_result=None,
            model_path=None,
        )
        return 0

    monkeypatch.setattr(subprocess_runner, "_supervise_child", fake_supervise)
    r = env.post(f"/api/jobs/{parent_id}/retune", json={"n_trials": 2})
    assert r.status_code == 200, r.text
    child_id = r.json()["job_id"]
    gate.wait_reached()
    assert env.store.active_job_id == child_id
    tune = env.post("/api/workspace/tune")
    assert tune.status_code == 409, tune.text
    releases_before = env.rec.slot_releases
    gate.release.set()
    join_worker(env.client)
    assert env.status(child_id) == "completed"
    assert env.store.active_job_id is None
    assert env.rec.slot_releases - releases_before == 1
