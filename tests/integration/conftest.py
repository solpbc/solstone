# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Configuration and fixtures for integration tests."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest


def require_cli_tool(name: str, binary: str) -> None:
    """Skip test if CLI tool is not available.

    Args:
        name: Human-readable provider name (e.g., "Anthropic").
        binary: CLI binary name (e.g., "claude").
    """
    if not shutil.which(binary):
        pytest.skip(f"{name} CLI ({binary}) not found on PATH")
    try:
        subprocess.run(
            [binary, "--version"],
            timeout=5,
            capture_output=True,
            text=True,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError):
        pytest.skip(f"{name} CLI ({binary}) not responding")


def pytest_configure(config):
    """Configure pytest for integration testing."""
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line(
        "markers", "requires_api: mark test as requiring external API access"
    )


def pytest_collection_modifyitems(config, items):
    """Extend timeout for tests that require external API access."""
    for item in items:
        if item.get_closest_marker("requires_api"):
            # Override global 5s timeout for API tests (real API calls need more time)
            item.add_marker(pytest.mark.timeout(60))


@pytest.fixture(scope="session")
def integration_journal_path(tmp_path_factory):
    """Create a temporary journal path for integration tests."""
    journal_dir = tmp_path_factory.mktemp("integration_journal")
    old_path = os.environ.get("JOURNAL_PATH")
    os.environ["JOURNAL_PATH"] = str(journal_dir)
    yield journal_dir
    if old_path:
        os.environ["JOURNAL_PATH"] = old_path
    else:
        os.environ.pop("JOURNAL_PATH", None)


@pytest.fixture
def integration_test_data():
    """Provide path to integration test data."""
    return Path(__file__).parent / "data"
