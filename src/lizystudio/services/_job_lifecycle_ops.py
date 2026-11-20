"""The API-facing operations of the job lifecycle primitive (P-0119).

A mixin of :class:`lizystudio.services._job_lifecycle.JobLifecycle`: cancel,
pause, reset, force-release, startup recovery, the delete release and the
settle of a reservation whose launch failed. Every method takes the
lifecycle lock for its whole read-check-write and returns the effects it
owes; nothing here broadcasts or counts a metric.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from lizystudio.services._job_lifecycle_types import (
    CANCELLED_CODE,
    CANCELLED_MESSAGE,
    TERMINAL_STATUSES,
    CancelResult,
    LifecycleEffects,
    Outcome,
    RecoverRefusedError,
    RunToken,
)

if TYPE_CHECKING:
    from lizystudio.services._job_active_slot import ActiveJobSlot
    from lizystudio.services._job_control_flags import JobControlFlags
    from lizystudio.services._job_metadata import Job, JobMetadataStore

_logger = logging.getLogger(__name__)

_RESTART_FAILED = "Server restarted before this job could complete"
_EXTRA_PAUSED = (
    "Multiple paused jobs found at startup; only the newest "
    "is preserved (INV-1: at most one paused job)"
)


class LifecycleOps:
    """Cancel / pause / reset / force-release / recover / delete (P-0119)."""

    _meta: JobMetadataStore
    _slot: ActiveJobSlot
    _flags: JobControlFlags
    lock: threading.Lock
    _tokens: dict[str, RunToken]
    _reset_gen: int
    _started: bool

    if TYPE_CHECKING:

        def _settle_locked(
            self, job: Job, status: str, error: str | None
        ) -> LifecycleEffects: ...

        def _write_status_locked(
            self,
            job_id: str,
            status: str,
            *,
            error: str | None,
            completed_at: str | None,
        ) -> None: ...

        @staticmethod
        def _effects(job: Job, outcome: Outcome) -> LifecycleEffects: ...

    # --- cancel and pause ----------------------------------------------

    def cancel(self, job_id: str) -> CancelResult:
        """``POST /cancel``: flag a live run, settle a token-less job.

        - live token: CANCEL is set; the run's commit writes ``cancelled``;
        - ``pending`` / ``paused`` without a token: settled ``cancelled`` now;
        - terminal: refused (``accepted`` False, the API answers 400).

        A ``running`` row without a live token is not reachable in this
        process (only admission writes ``running``, and every path that drops
        the token writes another status); for such a row the flag is set as
        before P-0119.
        """
        with self.lock:
            job = self._meta.get(job_id)
            if job is None or job.status in TERMINAL_STATUSES:
                return CancelResult(False, "missing" if job is None else job.status)
            if job_id in self._tokens or job.status == "running":
                self._flags.request_cancel(job_id)
                return CancelResult(True, job.status)
            effects = self._settle_locked(job, "cancelled", None)
            return CancelResult(True, "cancelled", effects)

    def pause(self, job_id: str) -> bool:
        """``POST /pause``: set PAUSE for a live run without CANCEL (P-0119)."""
        with self.lock:
            if job_id not in self._tokens:
                return False
            if self._flags.is_cancel_requested(job_id):
                return False
            self._flags.request_pause(job_id)
            return True

    # --- reset and force-release ----------------------------------------

    def reset_cancel_active(self) -> list[LifecycleEffects]:
        """Advance the reset generation and cancel the slot holder, in one
        lock section (INV-LC-6): a live run gets CANCEL; a token-less
        ``pending`` / ``paused`` holder is settled ``cancelled``."""
        with self.lock:
            self._reset_gen += 1
            holder = self._slot.active_job_id
            if holder is None:
                return []
            if holder in self._tokens:
                self._flags.request_cancel(holder)
                return []
            job = self._meta.get(holder)
            if job is None or job.status not in ("pending", "paused"):
                return []
            return [self._settle_locked(job, "cancelled", None)]

    def force_release(self, job_id: str) -> LifecycleEffects | None:
        """Revoke *job_id*'s run, write ``cancelled`` unless it is terminal,
        and release its slot. CANCEL stays (a still-running child reads it)."""
        with self.lock:
            revoked = self._tokens.pop(job_id, None)
            if revoked is not None:
                _logger.warning(
                    "event='lifecycle.force_release' job=%s run=%d",
                    job_id,
                    revoked.run_gen,
                )
            job = self._meta.get(job_id)
            effects = None
            if job is not None and job.status not in TERMINAL_STATUSES:
                self._write_status_locked(
                    job_id, "cancelled", error=None, completed_at=_now()
                )
                outcome = Outcome(
                    status="cancelled", code=CANCELLED_CODE, message=CANCELLED_MESSAGE
                )
                effects = self._effects(job, outcome)
            self._slot.force_release_if(job_id)
            return effects

    def holder_is_settled(self, job_id: str) -> bool:
        """True when *job_id* holds the slot with no live run and a terminal
        row (reset's wait loop then force-releases it at once)."""
        with self.lock:
            if job_id in self._tokens or self._slot.active_job_id != job_id:
                return False
            job = self._meta.get(job_id)
            return job is None or job.status in TERMINAL_STATUSES

    # --- failure settles --------------------------------------------------

    def settle_reservation(self, job_id: str, error: str) -> LifecycleEffects | None:
        """Fail a reserved ``pending`` job whose launch raised before admission
        (no live token); a job already admitted or settled is left alone."""
        with self.lock:
            job = self._meta.get(job_id)
            if job is None or job.status != "pending" or job_id in self._tokens:
                return None
            return self._settle_locked(job, "failed", error)

    def abandon(self, token: RunToken) -> bool:
        """Drop a live run whose terminal write keeps failing: release its slot
        and flags without a status write (#765 cleanup, token-conditioned)."""
        with self.lock:
            if self._tokens.get(token.job_id) != token:
                return False
            del self._tokens[token.job_id]
            self._slot.release(token.job_id)
            self._flags.clear_cancel(token.job_id)
            self._flags.clear_pause(token.job_id)
            _logger.error("event='lifecycle.run_abandoned' job=%s", token.job_id)
            return True

    # --- delete (Q5) -----------------------------------------------------

    def release_for_delete(self, job_ids: list[str]) -> bool:
        """Before ``rmtree``: flag live runs to stop; for token-less jobs,
        release the slot they hold and clear their flags (no status write —
        the rows are about to be removed).

        Refuses (returns False, changing nothing) when the first id — the job
        the request deletes — has a live run: the route's ``running`` check
        read the row outside the lock, and a run admitted since then must not
        lose its directory.
        """
        with self.lock:
            if job_ids and job_ids[0] in self._tokens:
                return False
            for job_id in job_ids:
                if job_id in self._tokens:
                    self._flags.request_cancel(job_id)
                    continue
                self._slot.release(job_id)
                self._flags.clear_cancel(job_id)
                self._flags.clear_pause(job_id)
            return True

    # --- startup recovery (Q6) -------------------------------------------

    def mark_started(self) -> None:
        """Close ``recover``: the API and the launchers may now run."""
        with self.lock:
            self._started = True

    def recover(self) -> None:
        """Settle what a previous process left (INV-restart-1..3, P-0119 Q6).

        ``running`` / ``pending`` / ``paused`` rows with a CANCEL file become
        ``cancelled``; CANCEL-less ``running`` / ``pending`` become ``failed``;
        the newest CANCEL-less ``paused`` survivor gets the slot back (the
        others ``failed``). Terminal rows are never rewritten. Refused once
        :meth:`mark_started` ran.
        """
        with self.lock:
            if self._started:
                msg = "recover() is only allowed before the process starts"
                raise RecoverRefusedError(msg)
            paused = self._recover_rows_locked()
            self._recover_paused_locked(paused)

    def _recover_rows_locked(self) -> list[Job]:
        paused: list[Job] = []
        for job in self._meta.list():
            if job.status in TERMINAL_STATUSES:
                continue
            if self._flags.is_cancel_requested(job.job_id):
                self._recover_write_locked(job, "cancelled", None)
            elif job.status == "paused":
                paused.append(job)
            else:
                self._recover_write_locked(job, "failed", _RESTART_FAILED)
        return paused

    def _recover_paused_locked(self, paused: list[Job]) -> None:
        if not paused:
            return
        paused.sort(key=lambda j: j.created_at, reverse=True)
        survivor = paused[0]
        for job in paused[1:]:
            self._recover_write_locked(job, "failed", _EXTRA_PAUSED)
        self._slot.reattach(survivor.job_id)
        _logger.info(
            "Re-attached paused job %s to active slot at startup", survivor.job_id
        )

    def _recover_write_locked(self, job: Job, status: str, error: str | None) -> None:
        _logger.warning(
            "Reconciling orphaned %s job %s to %s at startup",
            job.status,
            job.job_id,
            status,
        )
        self._write_status_locked(job.job_id, status, error=error, completed_at=_now())
        self._flags.clear_cancel(job.job_id)
        self._flags.clear_pause(job.job_id)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
