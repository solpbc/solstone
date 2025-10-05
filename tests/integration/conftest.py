"""Configuration and fixtures for integration tests."""

import os
from pathlib import Path

import pytest


def pytest_configure(config):
    """Configure pytest for integration testing."""
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line(
        "markers", "requires_api: mark test as requiring external API access"
    )
    config.addinivalue_line(
        "markers", "requires_claude_sdk: mark test as requiring Claude Code SDK"
    )


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


@pytest.fixture(autouse=True)
def disable_cortex_mcp_server(monkeypatch):
    """Prevent CortexService from starting the MCP server during tests."""

    def _noop_start(self):
        self.mcp_server_url = None
        return None

    monkeypatch.setattr(
        "muse.cortex.CortexService._start_mcp_server",
        _noop_start,
    )
    monkeypatch.setattr(
        "muse.cortex.CortexService._wait_for_mcp_server",
        lambda *args, **kwargs: None,
    )
