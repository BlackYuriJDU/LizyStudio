"""Test-only file barrier for the subprocess child (P-0119).

The real-child lifecycle tests hold a child process at a named point and
act on the parent meanwhile. The parent and the child share nothing but the
filesystem, so the barrier is a pair of files in a test-owned directory:
the child creates ``<dir>/<point>.reached`` and waits until the test creates
``<dir>/<point>.release``.

Inert in production: only the child calls :func:`activate_from_env`, and
only when ``LIZYSTUDIO_TEST_CHILD_BARRIER_DIR`` is set does :func:`barrier`
do anything; otherwise it returns after one ``is None`` check. The parent
never activates it, so a thread-mode worker never waits here.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

ENV_VAR = "LIZYSTUDIO_TEST_CHILD_BARRIER_DIR"
_POLL_INTERVAL = 0.02
_MAX_WAIT = 60.0

_barrier_dir: Path | None = None


def activate_from_env() -> None:
    """Read the barrier directory once, in the child (``_child_main``)."""
    global _barrier_dir
    raw = os.environ.get(ENV_VAR)
    _barrier_dir = Path(raw) if raw else None


def barrier(point: str) -> None:
    """Hold at *point* until the test releases it (no-op when inactive).

    Raises ``TimeoutError`` after ``_MAX_WAIT`` seconds so a test that forgets
    to release fails loudly instead of leaving a child behind.
    """
    if _barrier_dir is None:
        return
    (_barrier_dir / f"{point}.reached").touch()
    release = _barrier_dir / f"{point}.release"
    deadline = time.monotonic() + _MAX_WAIT
    while not release.exists():
        if time.monotonic() > deadline:
            msg = f"child barrier {point!r} was never released"
            raise TimeoutError(msg)
        time.sleep(_POLL_INTERVAL)
