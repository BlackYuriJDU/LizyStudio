"""Job state transition tests — FSM-style exhaustive verification.

Verifies that JobStore correctly handles all valid and invalid state
transitions, concurrent operations, and atomicity guarantees.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from lizystudio.backends.types import DataRef
from lizystudio.services._job_lifecycle_types import Outcome, RunToken
from lizystudio.services.jobs import Job, JobStore
from tests.lifecycle_seed import seed_job

pytestmark = pytest.mark.unit

# --- Fixtures ---


@pytest.fixture()
def store(tmp_path: Path) -> JobStore:
    """Create a fresh JobStore backed by a temp directory."""
    return JobStore(tmp_path / "jobs")


def _make_data_ref() -> DataRef:
    return DataRef(
        source_type="path",
        path="/tmp/data.csv",
        filename="data.csv",
        fingerprint="abc123",
        shape=(100, 5),
    )


def _create_pending_job(store: JobStore) -> Job:
    return store.create(
        backend_name="lizyml",
        config={"task": "binary"},
        data_ref=_make_data_ref(),
        job_type="fit",
    )


def _admit(store: JobStore, job: Job) -> RunToken:
    """Admit one run of the ``pending`` *job* (claiming the empty slot)."""
    return store.admit(
        job.job_id,
        expect="pending",
        reset_gen=store.capture_reset_gen(),
        slot_rule="claim",
    )


# --- Valid state transitions ---


class TestValidTransitions:
    """Verify all valid state transition paths persist correctly.

    Overturned by P-0119 (状態と lock: "update() は status を書かない"): the
    transitions are driven through the primitive — ``admit`` writes
    ``running``, ``commit`` writes the outcome — instead of assigning
    ``job.status`` and calling ``update()``, which now keeps the stored status.
    """

    def test_pending_to_running(self, store: JobStore) -> None:
        """pending -> running is written by admission."""
        job = _create_pending_job(store)
        assert job.status == "pending"

        _admit(store, job)

        reloaded = store.get(job.job_id)
        assert reloaded is not None
        assert reloaded.status == "running"

    def test_running_to_completed(self, store: JobStore) -> None:
        """running -> completed sets completed_at."""
        job = _create_pending_job(store)
        token = _admit(store, job)

        assert store.commit(token, Outcome(status="completed")) is not None

        reloaded = store.get(job.job_id)
        assert reloaded is not None
        assert reloaded.status == "completed"
        assert reloaded.completed_at is not None

    def test_running_to_failed(self, store: JobStore) -> None:
        """running -> failed sets error message."""
        job = _create_pending_job(store)
        token = _admit(store, job)

        store.commit(token, Outcome(status="failed", error="Out of memory"))

        reloaded = store.get(job.job_id)
        assert reloaded is not None
        assert reloaded.status == "failed"
        assert reloaded.error == "Out of memory"

    def test_running_to_cancelled(self, store: JobStore) -> None:
        """running -> cancelled via cancel request."""
        job = _create_pending_job(store)
        token = _admit(store, job)

        assert store.cancel(job.job_id).accepted is True
        assert store.is_cancel_requested(job.job_id)
        # Q2: the accepted cancel wins over the run's own outcome.
        effects = store.commit(token, Outcome(status="completed"))
        assert effects is not None
        assert effects.status == "cancelled"

        reloaded = store.get(job.job_id)
        assert reloaded is not None
        assert reloaded.status == "cancelled"
        assert store.is_cancel_requested(job.job_id) is False

    def test_full_lifecycle_fit(self, store: JobStore) -> None:
        """Complete fit lifecycle: pending -> running -> completed."""
        from lizystudio.backends.types import FitSummary

        reservation = store.create_and_reserve(
            backend_name="lizyml",
            config={"task": "binary"},
            data_ref=_make_data_ref(),
            job_type="fit",
        )
        assert reservation is not None
        job, gen = reservation
        assert store.active_job_id == job.job_id
        token = store.admit(
            job.job_id, expect="pending", reset_gen=gen, slot_rule="reserved"
        )

        # Results first (status-free), then the status commit.
        job.fit_result = FitSummary(
            metrics={"accuracy": 0.95},
            fold_count=5,
            params=[{"lr": 0.01}],
        )
        job.model_path = str(store.jobs_dir / job.job_id / "model")
        assert store.write_results(job, token) is True
        store.commit(token, Outcome(status="completed"))

        # The commit released the active slot.
        assert not store.has_active_job()

        # Verify persistence
        reloaded = store.get(job.job_id)
        assert reloaded is not None
        assert reloaded.status == "completed"
        assert reloaded.fit_result is not None
        assert reloaded.fit_result.metrics["accuracy"] == 0.95

    def test_full_lifecycle_tune(self, store: JobStore) -> None:
        """Complete tune lifecycle: pending -> running -> completed."""
        from lizystudio.backends.types import FitSummary, TuningSummary

        job = store.create(
            backend_name="lizyml",
            config={"task": "binary"},
            data_ref=_make_data_ref(),
            job_type="tune",
        )

        job.status = "running"
        seed_job(store, job)

        job.status = "completed"
        job.completed_at = "2026-04-06T02:00:00+00:00"
        job.fit_result = FitSummary(
            metrics={"accuracy": 0.97},
            fold_count=5,
            params=[{"lr": 0.001}],
        )
        job.tune_result = TuningSummary(
            best_params={"lr": 0.001, "depth": 6},
            best_score=0.97,
            trials=[{"trial": 1, "score": 0.95}, {"trial": 2, "score": 0.97}],
            metric_name="accuracy",
            direction="maximize",
        )
        seed_job(store, job)

        reloaded = store.get(job.job_id)
        assert reloaded is not None
        assert reloaded.tune_result is not None
        assert reloaded.tune_result.best_score == 0.97
        assert reloaded.tune_result.best_params["depth"] == 6


# --- Terminal state tests ---


class TestTerminalStates:
    """Verify terminal states (completed, failed, cancelled) are respected.

    Overturned by P-0119 (INV-LC-2: a terminal is written once and never
    rewritten): each terminal is committed through the primitive, then a
    second commit — same token, or ``update()`` with another in-memory
    status — leaves it unchanged.
    """

    @pytest.mark.parametrize("status", ["completed", "failed", "cancelled"])
    def test_terminal_is_written_once(self, store: JobStore, status: str) -> None:
        job = _create_pending_job(store)
        token = _admit(store, job)
        assert store.commit(token, Outcome(status=status)) is not None  # type: ignore[arg-type]

        assert store.commit(token, Outcome(status="failed")) is None
        job.status = "running"
        store.update(job)

        reloaded = store.get(job.job_id)
        assert reloaded is not None
        assert reloaded.status == status

    def test_completed_is_terminal(self, store: JobStore) -> None:
        """A completed job stays completed after re-read."""
        self.test_terminal_is_written_once(store, "completed")

    def test_failed_is_terminal(self, store: JobStore) -> None:
        """A failed job stays failed after re-read."""
        self.test_terminal_is_written_once(store, "failed")

    def test_cancelled_is_terminal(self, store: JobStore) -> None:
        """A cancelled job stays cancelled after re-read."""
        self.test_terminal_is_written_once(store, "cancelled")


# --- Concurrent operations ---


class TestConcurrentOperations:
    """Verify thread-safety of concurrent job operations."""

    def test_concurrent_cancel_and_read(self, store: JobStore) -> None:
        """Cancel request from one thread while another reads the job."""
        job = _create_pending_job(store)
        job.status = "running"
        seed_job(store, job)

        errors: list[Exception] = []
        barrier = threading.Barrier(2, timeout=5)

        def cancel_job() -> None:
            try:
                barrier.wait()
                store.request_cancel(job.job_id)
            except Exception as e:
                errors.append(e)

        def read_job() -> None:
            try:
                barrier.wait()
                loaded = store.get(job.job_id)
                assert loaded is not None
                assert loaded.status in ("running", "cancelled")
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=cancel_job)
        t2 = threading.Thread(target=read_job)
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)
        assert not errors, f"Concurrent errors: {errors}"

    def test_concurrent_cancel_and_delete(self, store: JobStore) -> None:
        """Concurrent cancel + delete should not raise."""
        job = _create_pending_job(store)
        job.status = "running"
        seed_job(store, job)

        errors: list[Exception] = []
        barrier = threading.Barrier(2, timeout=5)

        def cancel_job() -> None:
            try:
                barrier.wait()
                store.request_cancel(job.job_id)
            except Exception as e:
                errors.append(e)

        def delete_job() -> None:
            try:
                barrier.wait()
                store.delete(job.job_id)
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=cancel_job)
        t2 = threading.Thread(target=delete_job)
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)
        assert not errors, f"Concurrent errors: {errors}"

    def test_active_slot_prevents_second_job(self, store: JobStore) -> None:
        """Only one job can claim the active slot at a time."""
        job1 = _create_pending_job(store)
        job2 = _create_pending_job(store)

        assert store.claim_active(job1.job_id)
        assert not store.claim_active(job2.job_id)

        store.release_active(job1.job_id)
        assert store.claim_active(job2.job_id)

    def test_concurrent_active_claims(self, store: JobStore) -> None:
        """Only one thread succeeds in claiming the active slot."""
        jobs = [_create_pending_job(store) for _ in range(10)]
        results: list[bool] = []
        lock = threading.Lock()

        def try_claim(j: Job) -> None:
            result = store.claim_active(j.job_id)
            with lock:
                results.append(result)

        threads = [threading.Thread(target=try_claim, args=(j,)) for j in jobs]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert results.count(True) == 1, f"Expected exactly 1 claim, got {results}"

    def test_status_update_atomicity(self, store: JobStore) -> None:
        """Concurrent read during write returns a consistent state."""
        job = _create_pending_job(store)
        job.status = "running"
        seed_job(store, job)

        valid_statuses = {"running", "completed"}
        errors: list[str] = []

        def writer() -> None:
            j = store.get(job.job_id)
            assert j is not None
            j.status = "completed"
            j.completed_at = "2026-04-06T00:00:00+00:00"
            seed_job(store, j)

        def reader() -> None:
            for _ in range(50):
                loaded = store.get(job.job_id)
                if loaded is not None and loaded.status not in valid_statuses:
                    errors.append(f"Unexpected status: {loaded.status}")

        t1 = threading.Thread(target=writer)
        t2 = threading.Thread(target=reader)
        t2.start()
        t1.start()
        t1.join(timeout=5)
        t2.join(timeout=5)
        assert not errors, f"Atomicity violation: {errors}"


# --- Cancel lifecycle ---


class TestCancelLifecycle:
    """Verify cancel flag management."""

    def test_cancel_flag_lifecycle(self, store: JobStore) -> None:
        """request_cancel -> is_cancel_requested -> clear_cancel."""
        job = _create_pending_job(store)

        assert not store.is_cancel_requested(job.job_id)

        store.request_cancel(job.job_id)
        assert store.is_cancel_requested(job.job_id)

        store.clear_cancel(job.job_id)
        assert not store.is_cancel_requested(job.job_id)

    def test_cancel_nonexistent_job(self, store: JobStore) -> None:
        """Cancelling a nonexistent job does not raise."""
        store.request_cancel("nonexistent")
        assert not store.is_cancel_requested("nonexistent_other")

    def test_double_cancel_is_idempotent(self, store: JobStore) -> None:
        """Requesting cancel twice is idempotent."""
        job = _create_pending_job(store)
        store.request_cancel(job.job_id)
        store.request_cancel(job.job_id)
        assert store.is_cancel_requested(job.job_id)

        store.clear_cancel(job.job_id)
        assert not store.is_cancel_requested(job.job_id)


# --- Edge cases ---


class TestEdgeCases:
    """Edge cases in job state management."""

    def test_delete_running_job_cleans_up(self, store: JobStore) -> None:
        """Deleting a running job removes files from disk."""
        job = _create_pending_job(store)
        job.status = "running"
        seed_job(store, job)

        assert store.delete(job.job_id)
        assert store.get(job.job_id) is None

    def test_delete_nonexistent_returns_false(self, store: JobStore) -> None:
        """Deleting a nonexistent job returns False."""
        assert not store.delete("nonexistent_id")

    def test_list_filters_by_status(self, store: JobStore) -> None:
        """list(status=...) only returns matching jobs."""
        j1 = _create_pending_job(store)
        j2 = _create_pending_job(store)
        j1.status = "completed"
        j1.completed_at = "2026-04-06T00:00:00+00:00"
        seed_job(store, j1)

        pending_jobs = store.list(status="pending")
        completed_jobs = store.list(status="completed")
        assert len(pending_jobs) == 1
        assert pending_jobs[0].job_id == j2.job_id
        assert len(completed_jobs) == 1
        assert completed_jobs[0].job_id == j1.job_id

    def test_release_wrong_job_id_is_noop(self, store: JobStore) -> None:
        """Releasing a job ID that is not active does nothing."""
        job = _create_pending_job(store)
        store.claim_active(job.job_id)
        store.release_active("wrong_id")
        assert store.has_active_job()
        assert store.active_job_id == job.job_id
