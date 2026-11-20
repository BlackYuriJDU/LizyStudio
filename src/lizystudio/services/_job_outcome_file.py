"""The subprocess child's outcome file (P-0119 PR-B, Q3).

The child never writes ``meta.json`` (no status, no results, no warnings).
When its run ends it writes one file, ``<job_dir>/outcome.json``, tagged
with the run generation the parent's admission issued; the parent's
reconcile reads it and publishes the status — and only then does the status
become visible. A file whose generation is not the live run's is a stale
outcome: it is discarded and logged (INV-LC-5).

The file is removed at admission (a previous run's leftover) and by the
reconcile once read.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from lizystudio.backends.types import FitSummary, TuningSummary
from lizystudio.services._job_lifecycle_types import Outcome
from lizystudio.services._job_metadata import read_job_json, write_job_json

_logger = logging.getLogger(__name__)

OUTCOME_FILENAME = "outcome.json"


@dataclass(frozen=True)
class ChildOutcome:
    """What the child reports: the outcome plus its results."""

    run_gen: int
    outcome: Outcome
    fit_result: FitSummary | None
    tune_result: TuningSummary | None
    model_path: str | None


def write_outcome(
    path: Path,
    *,
    run_gen: int,
    outcome: Outcome,
    fit_result: FitSummary | None,
    tune_result: TuningSummary | None,
    model_path: str | None,
) -> None:
    """Write the child's outcome atomically (versioned JSON, tmp + replace)."""
    payload: dict[str, Any] = {
        "run_gen": run_gen,
        "status": outcome.status,
        "error": outcome.error,
        "code": outcome.code,
        "message": outcome.message,
        "fit_result": None if fit_result is None else asdict(fit_result),
        "tune_result": None if tune_result is None else asdict(tune_result),
        "model_path": model_path,
    }
    write_job_json(path, payload)


def read_outcome(path: Path) -> ChildOutcome | None:
    """The outcome at *path*, or ``None`` when there is none or it is
    unreadable (logged: a dead child's partial write is not an outcome)."""
    if not path.exists():
        return None
    try:
        raw = read_job_json(path)
        return _parse(raw)
    except (OSError, ValueError, KeyError, TypeError):
        _logger.warning(
            "event='lifecycle.outcome_unreadable' path=%s", path, exc_info=True
        )
        return None


def _parse(raw: dict[str, Any]) -> ChildOutcome:
    status = raw["status"]
    if status not in ("completed", "failed", "cancelled", "paused"):
        msg = f"unknown outcome status {status!r}"
        raise ValueError(msg)
    fit = raw.get("fit_result")
    tune = raw.get("tune_result")
    return ChildOutcome(
        run_gen=int(raw["run_gen"]),
        outcome=Outcome(
            status=status,
            error=raw.get("error"),
            code=raw.get("code"),
            message=raw.get("message"),
        ),
        fit_result=None if fit is None else FitSummary(**fit),
        tune_result=None if tune is None else TuningSummary(**tune),
        model_path=raw.get("model_path"),
    )


def remove_outcome(path: Path) -> None:
    """Remove the outcome file if present (best effort, logged on error)."""
    try:
        path.unlink(missing_ok=True)
    except OSError:
        _logger.warning(
            "event='lifecycle.outcome_remove_failed' path=%s", path, exc_info=True
        )
