# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Verify API baselines match checked-in fixture responses."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from convey import create_app
from tests.verify_api import ENDPOINTS, baseline_path, fetch_endpoint, normalize


@pytest.fixture(scope="module")
def client():
    journal = str(Path(__file__).resolve().parent / "fixtures" / "journal")
    app = create_app(journal)
    app.config["TESTING"] = True
    return app.test_client()


@pytest.fixture(scope="module")
def journal_path():
    return str(Path(__file__).resolve().parent / "fixtures" / "journal")


@pytest.mark.parametrize(
    "endpoint", ENDPOINTS, ids=[endpoint["name"] for endpoint in ENDPOINTS]
)
def test_api_baseline(client, journal_path, endpoint):
    """Verify endpoint response matches stored baseline."""
    if endpoint.get("sandbox_only"):
        pytest.skip("sandbox-only baseline (differs in Flask test client)")
    path = baseline_path(endpoint)
    if not path.exists():
        pytest.skip(f"No baseline file: {path}")

    status, payload = fetch_endpoint(client, endpoint)
    assert status == endpoint["status"], (
        f"Expected status {endpoint['status']}, got {status}"
    )

    actual = normalize(payload, journal_path)
    expected = json.loads(path.read_text())

    assert actual == expected, (
        f"Baseline mismatch for {endpoint['app']}/{endpoint['name']}. "
        "Run 'make update-api-baselines' to update."
    )
