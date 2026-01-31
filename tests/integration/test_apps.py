# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for Convey app endpoints.

Tests that all apps with workspace.html can be accessed via /app/{app_name}.
Requires Convey to be running (port discovered from health file).
"""

from pathlib import Path

import pytest
import requests

from think.utils import read_service_port


def get_convey_port() -> int | None:
    """Get the port Convey is running on, or None if not available."""
    return read_service_port("convey")


def is_convey_running() -> tuple[bool, int | None]:
    """Check if Convey is running and return (is_running, port)."""
    port = get_convey_port()
    if port is None:
        return False, None
    try:
        requests.get(f"http://localhost:{port}/", timeout=2)
        return True, port  # Any response means server is running
    except (requests.ConnectionError, requests.Timeout):
        return False, port


def get_app_names() -> list[str]:
    """Get list of app names from apps/ directory.

    Returns app names for all apps with workspace.html.
    """
    project_root = Path(__file__).parent.parent.parent
    apps_dir = project_root / "apps"

    app_names = []
    for app_path in apps_dir.iterdir():
        if app_path.is_dir() and (app_path / "workspace.html").exists():
            app_names.append(app_path.name)

    return sorted(app_names)


@pytest.fixture(scope="module")
def convey_port():
    """Fixture that returns Convey port and skips if not running."""
    running, port = is_convey_running()
    if not running:
        if port is None:
            pytest.skip("Convey port file not found - is Convey running?")
        else:
            pytest.skip(f"Convey is not responding on port {port}")
    return port


@pytest.mark.parametrize("app_name", get_app_names())
def test_app_endpoint(app_name: str, convey_port: int):
    """Test that each app endpoint returns 200."""
    url = f"http://localhost:{convey_port}/app/{app_name}"
    response = requests.get(url, timeout=5)

    assert response.status_code == 200, (
        f"App '{app_name}' at {url} returned {response.status_code}. "
        "Always use 'make screenshot' to validate this failure"
    )
