# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for Convey app endpoints.

Tests that all apps with workspace.html can be accessed via /app/{app_name}
using Flask's in-process test client.
"""

from pathlib import Path

import pytest

from convey import create_app


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


@pytest.fixture(scope="session")
def convey_client():
    """Create an in-process Convey Flask test client."""
    journal = str(
        Path(__file__).resolve().parent.parent.parent / "tests" / "fixtures" / "journal"
    )
    app = create_app(journal)
    return app.test_client()


@pytest.mark.parametrize("app_name", get_app_names())
def test_app_endpoint(convey_client, app_name: str):
    """Test that each app endpoint returns 200."""
    route = f"/app/{app_name}"
    response = convey_client.get(route, follow_redirects=True)

    assert response.status_code == 200, (
        f"App '{app_name}' at {route} returned {response.status_code}. "
        "Always use 'make screenshot' to validate this failure"
    )
