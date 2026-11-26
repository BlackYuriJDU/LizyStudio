"""P-0110: RFC 9457 Problem Details for the workspace endpoints (Issue #536).

Pins the contract that replaced the silent ``200 OK`` +
``{"saved": false, "errors": [...]}`` envelope:

- INV-API-1: a ``2xx`` from ``/api/workspace/**`` means the side effect is
  committed, and the body carries no ``saved`` field.
- INV-API-2: a blocking validation rejection is ``422`` with
  ``application/problem+json`` and ``validation_errors``.
- INV-API-3: ``severity == "warning"`` entries alone stay ``200`` and ride in
  ``validation_errors`` next to the saved config.
- Scope: errors raised under ``/api/jobs/**`` keep the legacy
  ``{"error": {...}}`` envelope (out of P-0110 scope).
"""

from __future__ import annotations

import csv
from http import HTTPStatus
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import yaml
from fastapi.testclient import TestClient

from lizystudio.api.errors import PROBLEM_JSON_MEDIA_TYPE, wants_problem_details
from lizystudio.api.models import ProblemDetails
from lizystudio.backends.types import DataRef
from tests.lifecycle_seed import seed_job

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _binary_csv(tmp_path: Path) -> str:
    csv_path = tmp_path / "train.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "age", "target"])
        for i in range(50):
            writer.writerow([i, 20 + i, i % 2])
    return str(csv_path)


def _load_data(client: TestClient, tmp_path: Path) -> None:
    r = client.post("/api/workspace/data/path", json={"path": _binary_csv(tmp_path)})
    assert r.status_code == 200, r.text


def _valid_config(client: TestClient) -> dict[str, Any]:
    r = client.get("/api/workspace/config/defaults?task=binary&target=target")
    assert r.status_code == 200, r.text
    return r.json()  # type: ignore[no-any-return]


def _blocking_config(client: TestClient) -> dict[str, Any]:
    """A config whose ``split.n_splits`` exceeds the loaded row count."""
    config = _valid_config(client)
    config["split"]["n_splits"] = 1000
    return config


def _warning_only_setup(client: TestClient, tmp_path: Path) -> dict[str, Any]:
    """Regression data with a zero target + ``mape`` (a warning, P-0101)."""
    csv_path = tmp_path / "reg.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["a", "b", "y"])
        for i in range(10):
            writer.writerow([i, i + 1, float(i)])
    r = client.post("/api/workspace/data/path", json={"path": str(csv_path)})
    assert r.status_code == 200, r.text
    r = client.get("/api/workspace/config/defaults?task=regression&target=y")
    assert r.status_code == 200, r.text
    config: dict[str, Any] = r.json()
    config["evaluation"]["metrics"] = ["mae", "mape"]
    return config


def _assert_matches_problem_schema(body: dict[str, Any]) -> None:
    """The served body and the OpenAPI ``ProblemDetails`` model agree (DC3)."""
    assert set(body) == set(ProblemDetails.model_fields), body
    ProblemDetails.model_validate(body)


def _assert_validation_problem(res: Any) -> list[dict[str, Any]]:
    """Assert the RFC 9457 validation body and return ``validation_errors``."""
    assert res.status_code == 422, res.text
    assert res.headers["content-type"].startswith(PROBLEM_JSON_MEDIA_TYPE)
    body = res.json()
    _assert_matches_problem_schema(body)
    assert body["type"] == "about:blank"
    assert body["title"] == HTTPStatus(422).phrase
    assert body["status"] == 422
    assert body["code"] == "VALIDATION_ERROR"
    assert isinstance(body["detail"], str) and body["detail"]
    assert "saved" not in body
    assert "error" not in body, "legacy envelope must not leak into problem+json"
    errors = body["validation_errors"]
    assert isinstance(errors, list) and errors
    assert any(e.get("severity", "error") == "error" for e in errors)
    return errors  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# PUT /config and POST /config/upload (the two silent-200 sites)
# ---------------------------------------------------------------------------


def test_put_config_blocking_returns_422_problem_and_does_not_persist(
    client: TestClient, tmp_path: Path
) -> None:
    _load_data(client, tmp_path)
    res = client.put("/api/workspace/config", json=_blocking_config(client))
    errors = _assert_validation_problem(res)
    assert any("n_splits" in (e.get("path") or "") for e in errors), errors
    assert client.get("/api/workspace/config").json() == {}


def test_put_config_valid_returns_200_without_saved(
    client: TestClient, tmp_path: Path
) -> None:
    _load_data(client, tmp_path)
    config = _valid_config(client)
    res = client.put("/api/workspace/config", json=config)
    assert res.status_code == 200, res.text
    body = res.json()
    assert set(body) == {"config", "validation_errors"}
    assert body["validation_errors"] == []
    assert client.get("/api/workspace/config").json()["task"] == "binary"


def test_put_config_warning_only_stays_200_and_persists(
    client: TestClient, tmp_path: Path
) -> None:
    config = _warning_only_setup(client, tmp_path)
    res = client.put("/api/workspace/config", json=config)
    assert res.status_code == 200, res.text
    body = res.json()
    assert "saved" not in body
    assert body["validation_errors"], body
    assert all(e.get("severity") == "warning" for e in body["validation_errors"])
    assert client.get("/api/workspace/config").json()["task"] == "regression"


def test_upload_config_blocking_returns_422_problem_and_does_not_persist(
    client: TestClient, tmp_path: Path
) -> None:
    _load_data(client, tmp_path)
    content = yaml.dump(_blocking_config(client)).encode()
    res = client.post(
        "/api/workspace/config/upload",
        files={"file": ("config.yaml", content, "application/x-yaml")},
    )
    _assert_validation_problem(res)
    assert client.get("/api/workspace/config").json() == {}


def test_upload_config_valid_returns_200_without_saved(
    client: TestClient, tmp_path: Path
) -> None:
    _load_data(client, tmp_path)
    content = yaml.dump(_valid_config(client)).encode()
    res = client.post(
        "/api/workspace/config/upload",
        files={"file": ("config.yaml", content, "application/x-yaml")},
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert set(body) == {"config", "validation_errors"}
    assert client.get("/api/workspace/config").json()["task"] == "binary"


# ---------------------------------------------------------------------------
# The ``if blocking: raise ValidationError`` sites of /fit and /tune
# ---------------------------------------------------------------------------


def test_fit_body_config_blocking_is_problem(
    client: TestClient, tmp_path: Path
) -> None:
    _load_data(client, tmp_path)
    with patch("lizystudio.api.workspace.start_fit_async") as start:
        res = client.post(
            "/api/workspace/fit", json={"config": _blocking_config(client)}
        )
    _assert_validation_problem(res)
    assert start.call_count == 0


def test_fit_stored_config_blocking_is_problem(
    client: TestClient, tmp_path: Path
) -> None:
    _load_data(client, tmp_path)
    ws = client.app.state.workspace  # type: ignore[union-attr]
    ws.set_config(_blocking_config(client))
    with patch("lizystudio.api.workspace.start_fit_async") as start:
        res = client.post("/api/workspace/fit")
    _assert_validation_problem(res)
    assert start.call_count == 0


def test_tune_body_config_blocking_is_problem(
    client: TestClient, tmp_path: Path
) -> None:
    _load_data(client, tmp_path)
    with patch("lizystudio.api.workspace.start_tune_async") as start:
        res = client.post(
            "/api/workspace/tune", json={"config": _blocking_config(client)}
        )
    _assert_validation_problem(res)
    assert start.call_count == 0


def test_tune_materialized_config_blocking_is_problem(
    client: TestClient, tmp_path: Path
) -> None:
    _load_data(client, tmp_path)
    ws = client.app.state.workspace  # type: ignore[union-attr]
    ws.set_config(_blocking_config(client))
    with patch("lizystudio.api.workspace.start_tune_async") as start:
        res = client.post("/api/workspace/tune")
    _assert_validation_problem(res)
    assert start.call_count == 0


def test_tune_search_space_refusal_is_problem(
    client: TestClient, tmp_path: Path
) -> None:
    _load_data(client, tmp_path)
    config = _valid_config(client)
    config["tuning"] = {
        "optuna": {
            "params": {"n_trials": 3, "timeout": None},
            "space": {"learning_rate": {"type": "float", "low": 0.5, "high": 0.01}},
        }
    }
    with patch("lizystudio.api.workspace.start_tune_async") as start:
        res = client.post("/api/workspace/tune", json={"config": config})
    errors = _assert_validation_problem(res)
    assert "tuning.optuna.space.learning_rate" in {e["path"] for e in errors}
    assert start.call_count == 0


# ---------------------------------------------------------------------------
# Other StudioErrors under /api/workspace
# ---------------------------------------------------------------------------


def test_workspace_locked_is_409_problem(client: TestClient, tmp_path: Path) -> None:
    _load_data(client, tmp_path)
    store = client.app.state.job_store  # type: ignore[union-attr]
    job = store.create(
        backend_name="lizyml",
        config={"task": "binary"},
        data_ref=DataRef(
            source_type="path",
            path="/data/holder.csv",
            filename="holder.csv",
            fingerprint="h",
            shape=(10, 2),
        ),
        job_type="fit",
    )
    job.status = "running"
    seed_job(store, job)
    assert store.claim_active(job.job_id)
    try:
        res = client.put("/api/workspace/config", json=_valid_config(client))
    finally:
        store.release_active(job.job_id)
    assert res.status_code == 409, res.text
    assert res.headers["content-type"].startswith(PROBLEM_JSON_MEDIA_TYPE)
    body = res.json()
    _assert_matches_problem_schema(body)
    assert body["type"] == "about:blank"
    assert body["status"] == 409
    assert body["title"] == HTTPStatus(409).phrase
    assert body["code"] == "WORKSPACE_LOCKED"
    assert body["details"] == {"job_id": job.job_id}
    assert body["validation_errors"] is None


def test_workspace_no_data_is_problem(client: TestClient) -> None:
    res = client.get("/api/workspace/data/preview")
    assert res.status_code == 400, res.text
    assert res.headers["content-type"].startswith(PROBLEM_JSON_MEDIA_TYPE)
    body = res.json()
    _assert_matches_problem_schema(body)
    assert body["code"] == "WORKSPACE_NO_DATA"
    assert body["status"] == 400
    assert body["detail"] == "No data loaded in workspace"


# ---------------------------------------------------------------------------
# Scope boundary: jobs keep the legacy envelope
# ---------------------------------------------------------------------------


def test_job_endpoint_errors_keep_legacy_envelope(client: TestClient) -> None:
    res = client.get("/api/jobs/job_does_not_exist")
    assert res.status_code == 404, res.text
    assert res.headers["content-type"].startswith("application/json")
    body = res.json()
    assert body["error"]["code"] == "JOB_NOT_FOUND"
    assert "type" not in body


def test_request_body_parse_error_keeps_legacy_envelope(client: TestClient) -> None:
    """Pydantic body-parse 422s are separate (INV-API-4) and stay legacy."""
    res = client.put("/api/workspace/config/tuning-overrides", json={"not_a_field": 1})
    assert res.status_code == 422, res.text
    assert res.headers["content-type"].startswith("application/json")
    assert res.json()["error"]["code"] == "VALIDATION_ERROR"


def test_unknown_workspace_route_is_problem(client: TestClient) -> None:
    """The /api catch-all raises a StudioError, so it is Problem Details too."""
    res = client.get("/api/workspace/does-not-exist")
    assert res.status_code == 404, res.text
    assert res.headers["content-type"].startswith(PROBLEM_JSON_MEDIA_TYPE)
    _assert_matches_problem_schema(res.json())


def test_framework_405_is_not_a_studio_error(client: TestClient) -> None:
    """Only StudioErrors are serialized; a framework 405 keeps its body.

    BLUEPRINT §6.1.1 and docs/api.md scope Problem Details to StudioErrors
    under /api/workspace; this pins that boundary.
    """
    res = client.delete("/api/workspace/config")
    assert res.status_code == 405, res.text
    assert res.headers["content-type"].startswith("application/json")
    assert res.json() == {"detail": "Method Not Allowed"}


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("/api/workspace", True),
        ("/api/workspace/config", True),
        ("/api/workspace/data/preview", True),
        ("/api/workspaces", False),
        ("/api/workspace-x/config", False),
        ("/api/jobs/job_1", False),
        ("/workspace/config", False),
    ],
)
def test_wants_problem_details_anchors_the_prefix(path: str, expected: bool) -> None:
    assert wants_problem_details(path) is expected


# ---------------------------------------------------------------------------
# OpenAPI export
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "method"),
    [
        ("/api/workspace/config", "put"),
        ("/api/workspace/config/upload", "post"),
        ("/api/workspace/fit", "post"),
        ("/api/workspace/tune", "post"),
    ],
)
def test_openapi_exports_problem_json_on_422(
    client: TestClient, path: str, method: str
) -> None:
    schema = client.get("/openapi.json").json()
    responses = schema["paths"][path][method]["responses"]
    content = responses["422"]["content"]
    assert content[PROBLEM_JSON_MEDIA_TYPE]["schema"] == {
        "$ref": "#/components/schemas/ProblemDetails"
    }


def test_openapi_config_update_response_has_no_saved(client: TestClient) -> None:
    schemas = client.get("/openapi.json").json()["components"]["schemas"]
    props = schemas["ConfigUpdateResponse"]["properties"]
    assert "saved" not in props
    assert set(props) == {"config", "validation_errors"}
    problem = schemas["ProblemDetails"]["properties"]
    for key in ("type", "title", "status", "detail", "code", "validation_errors"):
        assert key in problem
