"""Value types of the job lifecycle primitive (P-0119).

Kept apart from :mod:`lizystudio.services._job_lifecycle` so the launchers,
the API layer and the tests can import the tokens, outcomes, effects and
refusals without the primitive's machinery.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

OutcomeStatus = Literal["completed", "failed", "cancelled", "paused"]
TerminalStatus = Literal["completed", "failed", "cancelled"]
AdmissionReason = Literal["reset_gen", "status", "cancel", "live", "slot", "revoked"]
SlotRule = Literal["reserved", "own", "claim"]
ExpectedStatus = Literal["pending", "paused"]
FrameKind = Literal["completed", "error", "paused"]

TERMINAL_STATUSES: frozenset[str] = frozenset({"completed", "failed", "cancelled"})

# P-0099 INV-3 as amended by P-0119: ``pending -> failed`` is legal (startup
# ``recover`` of a CANCEL-less ``pending`` row, and a reserved job whose
# launch fails before admission). Pinned in
# ``tests/regression/test_inv_state_machine.py:LEGAL_TRANSITIONS``.
LEGAL_TRANSITIONS: frozenset[tuple[str, str]] = frozenset(
    {
        ("pending", "running"),
        ("pending", "cancelled"),
        ("pending", "failed"),
        ("running", "completed"),
        ("running", "failed"),
        ("running", "cancelled"),
        ("running", "paused"),
        ("paused", "running"),
        ("paused", "cancelled"),
        ("paused", "failed"),
    }
)

CANCELLED_MESSAGE = "Job cancelled"
CANCELLED_CODE = "JOB_CANCELLED"


@dataclass(frozen=True)
class RunToken:
    """The authority of one admitted run of a job (INV-LC-1, INV-LC-5).

    ``run_gen`` is unique per admission in this process; a token is live
    only while the primitive's table maps ``job_id`` to this exact token.
    ``origin`` is the status the run was admitted from, which decides where
    :meth:`JobLifecycle.abort` sends the job back to.
    """

    job_id: str
    run_gen: int
    origin: Literal["pending", "paused"] = "pending"


@dataclass(frozen=True)
class Outcome:
    """What a worker (or the reconcile of a dead child) asks to commit.

    ``adopted`` marks a status the legacy subprocess child already wrote and
    announced (``LegacyChildStatusSink``, removed by PR-B): the commit then
    settles the token, slot, flags and metric without a second write or frame.
    """

    status: OutcomeStatus
    error: str | None = None
    code: str | None = None
    message: str | None = None
    adopted: bool = False


@dataclass(frozen=True)
class LifecycleEffects:
    """The notifications owed by one published transition (INV-LC-2).

    Returned by the primitive and emitted by the caller after the lifecycle
    lock is released: one frame (or none) and one terminal metric (or none).
    """

    job_id: str
    job_type: str
    status: str
    frame: FrameKind | None
    code: str | None
    message: str
    metric_status: TerminalStatus | None


@dataclass(frozen=True)
class CancelResult:
    """The answer of :meth:`JobLifecycle.cancel`.

    ``accepted`` is False only for a job that is already terminal (400).
    ``effects`` is set when the cancel itself settled the job.
    """

    accepted: bool
    status: str
    effects: LifecycleEffects | None = None


class AdmissionRefused(Exception):  # noqa: N818 — a refusal, not a fault
    """An admission (or a start) the primitive refused.

    ``effects`` carries the settle of a refused ``pending`` job, which the
    caller emits before mapping the refusal to its HTTP answer.
    """

    def __init__(
        self,
        job_id: str,
        reason: AdmissionReason,
        status: str | None,
        effects: LifecycleEffects | None = None,
    ) -> None:
        super().__init__(f"admission of {job_id} refused: {reason} (status={status})")
        self.job_id = job_id
        self.reason: AdmissionReason = reason
        self.status = status
        self.effects = effects


class RecoverRefusedError(RuntimeError):
    """``recover`` was called after the process had started (P-0119)."""


class IllegalTransitionError(AssertionError):
    """A status write outside :data:`LEGAL_TRANSITIONS` (INV-3)."""
