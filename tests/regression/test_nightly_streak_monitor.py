"""Regression tests for the ``Nightly streak monitor`` workflow logic.

``nightly.yml`` gates its jobs on an idle-night ``guard`` so a scheduled run
against an unchanged ``develop`` does not re-test the same commit. The side
effect is that every tracked job reports ``skipped`` on such a night.

The streak monitor decides whether a non-blocking Nightly job has been red
for ``STREAK_THRESHOLD`` runs in a row. Its original predicate
(``conclusions.every(c => c === "failure")``) reads a ``skipped`` run as
"not a failure", so a quiet week would break a genuine red streak and the
alert would stop firing. That is a silent loss of detection, which is
precisely the failure mode the monitor was introduced (Issue #576) to
prevent.

The monitor is an inline ``actions/github-script`` body and cannot be
imported, so these tests extract it from the YAML and run it under Node
against stubbed Octokit objects (``streak_monitor_harness.mjs``).
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "nightly-streak-monitor.yml"
HARNESS = Path(__file__).parent / "streak_monitor_harness.mjs"


def _extract_script() -> str:
    """Return the ``github-script`` body from the monitor workflow.

    Raises loudly when the workflow's shape changes: a silent "" would make
    the harness pass against an empty program, which is the same false-clean
    class this module exists to guard.
    """
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    steps = workflow["jobs"]["monitor"]["steps"]
    bodies = [
        step["with"]["script"]
        for step in steps
        if str(step.get("uses", "")).startswith("actions/github-script")
    ]
    if len(bodies) != 1:
        raise AssertionError(
            f"expected exactly one github-script step in {WORKFLOW}, "
            f"found {len(bodies)}"
        )
    return bodies[0]


NIGHTLY = REPO_ROOT / ".github" / "workflows" / "nightly.yml"
_LIST_RE = re.compile(r"const TRACKED_JOBS = \[(.*?)\];", re.DOTALL)


def _tracked_jobs(source: str, where: Path) -> list[str]:
    """The quoted names inside a ``const TRACKED_JOBS = [...]`` literal."""
    matches = _LIST_RE.findall(source)
    if len(matches) != 1:
        raise AssertionError(
            f"expected one TRACKED_JOBS literal in {where}, found {len(matches)}"
        )
    names = re.findall(r'"([^"]+)"', matches[0])
    if not names:
        raise AssertionError(f"TRACKED_JOBS in {where} names no job")
    return names


def test_tracked_jobs_name_real_nightly_jobs() -> None:
    """Every job the monitor tracks is a job ``nightly.yml`` runs, and the
    harness tracks the same list (Issue #617). A name that matches no job
    is never "executed", so the monitor would skip it silently."""
    monitor = _tracked_jobs(_extract_script(), WORKFLOW)
    harness = _tracked_jobs(HARNESS.read_text(encoding="utf-8"), HARNESS)
    nightly = yaml.safe_load(NIGHTLY.read_text(encoding="utf-8"))
    job_names = {job.get("name", key) for key, job in nightly["jobs"].items()}

    assert harness == monitor
    missing = [name for name in monitor if name not in job_names]
    assert missing == [], f"tracked jobs absent from {NIGHTLY}: {missing}"


@pytest.mark.skipif(shutil.which("node") is None, reason="node is required")
def test_streak_monitor_survives_skipped_nightlies(tmp_path: Path) -> None:
    script = _extract_script()
    assert "RUN_WINDOW" in script, (
        "monitor no longer defines RUN_WINDOW — the skipped-run filter that "
        "keeps streak detection alive across idle nights may have been removed"
    )

    script_file = tmp_path / "monitor.js"
    script_file.write_text(script, encoding="utf-8")

    result = subprocess.run(
        ["node", str(HARNESS), str(script_file)],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )
    assert result.returncode == 0, (
        f"streak-monitor harness failed\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
