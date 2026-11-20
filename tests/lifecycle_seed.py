"""Test seeding for the P-0119 lifecycle primitive.

``JobStore.update()`` no longer writes ``status`` (P-0119), and the core no
longer claims the slot or writes ``running``. Tests that model *persisted*
state seed it here, straight to ``meta.json``; tests that model a *live* run
drive the real admission and get its token or sink. Nothing here is a src
writer, so the src status-writer census stays clean.
"""

from __future__ import annotations

from typing import Any

from lizystudio.services._job_lifecycle_types import RunToken
from lizystudio.services._job_metadata import Job
from lizystudio.services._job_sinks import ParentCommitSink
from lizystudio.services.jobs import JobStore


def seed_job(store: JobStore, job: Job) -> None:
    """Write *job* as given — status included — plus its result sidecars.

    Models a row a previous run left on disk (persisted state, no live run).
    """
    with store.lifecycle_lock:
        store._meta.save_meta(job)
        store._meta._write_sidecars(job)


def seed_status(
    store: JobStore, job_id: str, status: str, *, error: str | None = None
) -> None:
    """Set the persisted status of *job_id* (no live run, no slot change)."""
    job = store.get(job_id)
    assert job is not None, f"seed_status: job {job_id!r} does not exist"
    job.status = status  # type: ignore[assignment]
    if error is not None:
        job.error = error
    seed_job(store, job)


def reserve_for_launch(store: JobStore, job: Job) -> int:
    """Give *job* the slot reservation a ``/fit`` / ``/tune`` request would
    hold (when the slot is empty) and return the current reset generation,
    for the ``reset_gen`` argument of the launchers."""
    if store.active_job_id is None:
        store._slot.claim(job.job_id)
    return store.capture_reset_gen()


def admit_running(store: JobStore, job: Job) -> RunToken:
    """Admit one live run of *job* (claiming an empty slot) and return its
    token. A row seeded ``running`` is first put back to ``pending``."""
    stored = store.get(job.job_id)
    assert stored is not None
    if stored.status == "running":
        seed_status(store, job.job_id, "pending")
        stored.status = "pending"
    expect: Any = "paused" if stored.status == "paused" else "pending"
    # A test that raw-set CANCEL before the run models a cancel accepted
    # after admission and before the first callback; keep that order.
    cancelled = store.is_cancel_requested(job.job_id)
    if cancelled:
        store._flags.clear_cancel(job.job_id)
    token = store.admit(
        job.job_id,
        expect=expect,
        reset_gen=store.capture_reset_gen(),
        slot_rule="claim",
    )
    if cancelled:
        store._flags.request_cancel(job.job_id)
    job.status = "running"
    return token


def fake_child_exit(
    monkeypatch: Any, *, status: str | None, returncode: int = 0
) -> None:
    """Replace the child's launch and wait (``_supervise_child``) with an
    exit that persisted *status* (``None``: the child died without writing
    one) and returned *returncode*. The parent's real reconcile — the commit
    of the run token — then runs on that row (P-0119)."""

    def supervise(*, job: Job, job_store: JobStore, **_: Any) -> int:
        if status is not None:
            stored = job_store.get(job.job_id)
            assert stored is not None
            stored.status = status  # type: ignore[assignment]
            seed_job(job_store, stored)
        return returncode

    monkeypatch.setattr(
        "lizystudio.services.subprocess_runner._supervise_child", supervise
    )


def guarded_status(store: JobStore, job_id: str, status: str) -> None:
    """Write *status* through the primitive's one status writer, under its
    INV-3 guard (``IllegalTransitionError`` is an ``AssertionError``).

    For the state-machine tests, which pin the guard itself (P-0119 moved it
    from ``JobStore.set_status`` into the primitive).
    """
    with store.lifecycle_lock:
        store._lc._write_status_locked(job_id, status, error=None, completed_at=None)


def admitted_sink(
    store: JobStore, job: Job, broadcaster: Any = None
) -> ParentCommitSink:
    """Admit *job* and return the thread-mode sink of that run."""
    return ParentCommitSink(store, broadcaster, admit_running(store, job))
