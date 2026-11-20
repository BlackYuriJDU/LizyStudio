"""``JobStore``'s face of the P-0119 lifecycle primitive (a mixin).

Kept out of :mod:`lizystudio.services.jobs` so ``JobStore`` stays a thin
orchestrator: every method here delegates to the
:class:`~lizystudio.services._job_lifecycle.JobLifecycle` the store owns
(``self._lc``), or reads / writes ``meta.json`` under its lock.
"""

from __future__ import annotations

import builtins
import logging
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

from lizystudio.backends.types import DataRef
from lizystudio.services._job_lifecycle_types import (
    CancelResult,
    ExpectedStatus,
    LifecycleEffects,
    Outcome,
    RunToken,
    SlotRule,
)

if TYPE_CHECKING:
    from lizystudio.services._job_lifecycle import JobLifecycle
    from lizystudio.services._job_metadata import Job, JobMetadataStore

_logger = logging.getLogger(__name__)


class JobStoreLifecycle:
    """The lifecycle methods of :class:`~lizystudio.services.jobs.JobStore`."""

    _lc: JobLifecycle
    _meta: JobMetadataStore

    # --- P-0119 lifecycle primitive — delegated to JobLifecycle ---
    #
    # Every status / flag / slot transition goes through these methods. The
    # raw flag and slot methods above and below stay for the subprocess
    # child's reads and for test seeding; no src caller outside the
    # primitive mutates through them (BLUEPRINT §3.4.2, the writer list).

    @property
    def lifecycle_lock(self) -> threading.Lock:
        """The lifecycle lock (held by every ``meta.json`` write)."""
        return self._lc.lock

    def capture_reset_gen(self) -> int:
        """See :meth:`JobLifecycle.capture_reset_gen`."""
        return self._lc.capture_reset_gen()

    def create_and_reserve(
        self,
        *,
        backend_name: str,
        config: dict[str, Any],
        data_ref: DataRef,
        job_type: Literal["fit", "tune"],
        parent_job_id: str | None = None,
    ) -> tuple[Job, int] | None:
        """Create a ``pending`` job holding the slot, plus the request's reset
        generation. See :meth:`JobLifecycle.create_and_reserve`."""
        return self._lc.create_and_reserve(
            backend_name=backend_name,
            config=config,
            data_ref=data_ref,
            job_type=job_type,
            parent_job_id=parent_job_id,
        )

    def create_child(
        self,
        *,
        backend_name: str,
        config: dict[str, Any],
        data_ref: DataRef,
        job_type: Literal["fit", "tune"],
        parent_job_id: str | None = None,
    ) -> tuple[Job, int]:
        """Create a retune / resume child and the request's reset generation."""
        return self._lc.create_child(
            backend_name=backend_name,
            config=config,
            data_ref=data_ref,
            job_type=job_type,
            parent_job_id=parent_job_id,
        )

    def admit(
        self,
        job_id: str,
        *,
        expect: ExpectedStatus,
        reset_gen: int,
        slot_rule: SlotRule,
    ) -> RunToken:
        """See :meth:`JobLifecycle.admit`."""
        return self._lc.admit(
            job_id, expect=expect, reset_gen=reset_gen, slot_rule=slot_rule
        )

    def start(self, token: RunToken) -> bool:
        """The start gate. See :meth:`JobLifecycle.start`."""
        return self._lc.start(token)

    def is_live(self, token: RunToken) -> bool:
        """See :meth:`JobLifecycle.is_live`."""
        return self._lc.is_live(token)

    def commit(self, token: RunToken, outcome: Outcome) -> LifecycleEffects | None:
        """See :meth:`JobLifecycle.commit`."""
        return self._lc.commit(token, outcome)

    def abort(self, token: RunToken, error: str) -> LifecycleEffects | None:
        """See :meth:`JobLifecycle.abort`."""
        return self._lc.abort(token, error)

    def abandon(self, token: RunToken) -> bool:
        """See :meth:`LifecycleOps.abandon`."""
        return self._lc.abandon(token)

    def cancel(self, job_id: str) -> CancelResult:
        """See :meth:`LifecycleOps.cancel`."""
        return self._lc.cancel(job_id)

    def pause(self, job_id: str) -> bool:
        """See :meth:`LifecycleOps.pause`."""
        return self._lc.pause(job_id)

    def reset_cancel_active(self) -> builtins.list[LifecycleEffects]:
        """See :meth:`LifecycleOps.reset_cancel_active`."""
        return self._lc.reset_cancel_active()

    def force_release(self, job_id: str) -> LifecycleEffects | None:
        """See :meth:`LifecycleOps.force_release`."""
        return self._lc.force_release(job_id)

    def holder_is_settled(self, job_id: str) -> bool:
        """See :meth:`LifecycleOps.holder_is_settled`."""
        return self._lc.holder_is_settled(job_id)

    def settle_reservation(self, job_id: str, error: str) -> LifecycleEffects | None:
        """See :meth:`LifecycleOps.settle_reservation`."""
        return self._lc.settle_reservation(job_id, error)

    def release_for_delete(self, job_ids: builtins.list[str]) -> bool:
        """See :meth:`LifecycleOps.release_for_delete`."""
        return self._lc.release_for_delete(job_ids)

    def write_results(self, job: Job, token: RunToken | None) -> bool:
        """Persist results / warnings (status-free). With a *token*, the write
        is dropped (and logged) unless the token is live (INV-LC-5)."""
        with self._lc.lock:
            if token is not None and not self._lc._is_live_locked(token):
                _logger.warning(
                    "event='lifecycle.stale_write_dropped' job=%s run=%d",
                    job.job_id,
                    token.run_gen,
                )
                return False
            self._meta.update(job)
            return True

    def emit_if_live(self, token: RunToken, send: Callable[[], None]) -> bool:
        """See :meth:`JobLifecycle.emit_if_live` (the token-fenced frame)."""
        return self._lc.emit_if_live(token, send)

    # --- Startup reconciliation (P-0099 v3-22a, P-0119 Q6) ---

    def recover(self) -> None:
        """See :meth:`LifecycleOps.recover`."""
        self._lc.recover()

    def mark_started(self) -> None:
        """See :meth:`LifecycleOps.mark_started`."""
        self._lc.mark_started()

    def reconcile_at_startup(self) -> None:
        """Alias of :meth:`recover`, kept for one release (P-0119)."""
        self.recover()
