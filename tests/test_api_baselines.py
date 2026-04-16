# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Verify API baselines match checked-in fixture responses."""

from __future__ import annotations

import json

import pytest
from freezegun import freeze_time

from tests._baseline_harness import (
    FROZEN_DATE,
    FROZEN_TZ_OFFSET,
    isolated_app_env,
    make_logged_in_test_client,
    prepare_isolated_journal,
)
from tests.verify_api import ENDPOINTS, baseline_path, fetch_endpoint, normalize


@pytest.fixture(scope="module", autouse=True)
def _freeze_time():
    with freeze_time(FROZEN_DATE, tz_offset=FROZEN_TZ_OFFSET):
        yield


@pytest.fixture(scope="module")
def _baseline_journal(tmp_path_factory):
    dst = tmp_path_factory.mktemp("baseline_journal") / "journal"
    return prepare_isolated_journal(dst)


@pytest.fixture(scope="module")
def client(_baseline_journal):
    with isolated_app_env(_baseline_journal):
        yield make_logged_in_test_client(_baseline_journal)


@pytest.fixture(scope="module")
def journal_path(_baseline_journal):
    return str(_baseline_journal)


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
