"""Tests for Convey app endpoints.

Tests that all apps with workspace.html can be accessed via /app/{app_name}.
Requires Convey to be running on localhost:8000.
"""

from pathlib import Path

import pytest
import requests


def is_convey_running(port: int = 8000) -> bool:
    """Check if Convey is running on localhost."""
    try:
        requests.get(f"http://localhost:{port}/", timeout=2)
        return True  # Any response means server is running
    except (requests.ConnectionError, requests.Timeout):
        return False


def get_app_names() -> list[str]:
    """Get list of app names from apps/ directory.

    Returns app names for all apps with workspace.html.
    """
    project_root = Path(__file__).parent.parent
    apps_dir = project_root / "apps"

    app_names = []
    for app_path in apps_dir.iterdir():
        if app_path.is_dir() and (app_path / "workspace.html").exists():
            app_names.append(app_path.name)

    return sorted(app_names)


@pytest.fixture(scope="module")
def convey_running():
    """Fixture that checks if Convey is running and skips if not."""
    if not is_convey_running():
        pytest.skip("Convey is not running on localhost:8000")
    return True


@pytest.mark.parametrize("app_name", get_app_names())
def test_app_endpoint(app_name: str, convey_running):
    """Test that each app endpoint returns 200."""
    url = f"http://localhost:8000/app/{app_name}"
    response = requests.get(url, timeout=5)

    assert response.status_code == 200, (
        f"App '{app_name}' at {url} returned {response.status_code}"
    )
