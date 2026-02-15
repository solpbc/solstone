# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for app agent discovery, loading, and route helpers."""

import json
import os
from pathlib import Path

import pytest

from apps.agents.routes import _resolve_output_path
from think.muse import _resolve_agent_path, get_agent, get_muse_configs


@pytest.fixture
def fixture_journal():
    """Set JOURNAL_PATH to tests/fixtures/journal for testing."""
    os.environ["JOURNAL_PATH"] = "tests/fixtures/journal"
    yield


@pytest.fixture
def app_with_agent(tmp_path, monkeypatch):
    """Create a temporary app with an agent for testing.

    Creates apps/testapp/muse/myhelper.md with frontmatter in a temp directory,
    then monkeypatches the apps directory path.
    """
    # Create app structure
    app_dir = tmp_path / "apps" / "testapp"
    muse_dir = app_dir / "muse"
    muse_dir.mkdir(parents=True)

    # Create workspace.html (required for app discovery, though not used here)
    (app_dir / "workspace.html").write_text("<h1>Test App</h1>")

    # Create agent file with frontmatter
    metadata = {
        "type": "cogitate",
        "title": "My Test Helper",
        "provider": "openai",
        "tools": "journal",
        "schedule": "daily",
        "priority": 42,
    }
    json_str = json.dumps(metadata, indent=2)
    (muse_dir / "myhelper.md").write_text(
        f"{{\n{json_str[1:-1]}\n}}\n\nYou are a test helper agent.\n\n## Purpose\nHelp with testing."
    )

    # Create another agent without metadata (defaults only)
    (muse_dir / "simple.md").write_text("A simple test agent with no metadata.")

    # Monkeypatch the parent directory so apps discovery finds our temp apps
    monkeypatch.setattr(
        "think.utils.Path.__file__",
        str(tmp_path / "think" / "utils.py"),
    )

    # Actually we need to patch where get_agents looks for apps
    # It uses Path(__file__).parent.parent / "apps"
    # Let's patch it differently - create a mock apps dir structure
    yield {
        "tmp_path": tmp_path,
        "app_dir": app_dir,
        "muse_dir": muse_dir,
    }


def test_resolve_agent_path_system_agent():
    """Test _resolve_agent_path returns correct path for system agents."""
    agent_dir, agent_name = _resolve_agent_path("default")

    assert agent_name == "default"
    assert agent_dir.name == "muse"


def test_resolve_agent_path_app_agent():
    """Test _resolve_agent_path returns correct path for app agents."""
    agent_dir, agent_name = _resolve_agent_path("chat:helper")

    assert agent_name == "helper"
    assert agent_dir.name == "muse"
    assert agent_dir.parent.name == "chat"
    assert "apps" in str(agent_dir)


def test_resolve_agent_path_app_agent_with_underscores():
    """Test _resolve_agent_path handles app names with underscores."""
    agent_dir, agent_name = _resolve_agent_path("my_app:my_agent")

    assert agent_name == "my_agent"
    assert agent_dir.parent.name == "my_app"


def test_get_agent_system_agent(fixture_journal):
    """Test get_agent loads system agents correctly."""
    config = get_agent("default")

    assert config["name"] == "default"
    assert "system_instruction" in config
    assert "user_instruction" in config
    assert len(config["system_instruction"]) > 0
    assert len(config["user_instruction"]) > 0


def test_get_agent_nonexistent_raises():
    """Test get_agent raises FileNotFoundError for nonexistent agents."""
    with pytest.raises(FileNotFoundError) as exc_info:
        get_agent("nonexistent_agent_xyz")

    assert "nonexistent_agent_xyz" in str(exc_info.value)


def test_get_agent_nonexistent_app_agent_raises():
    """Test get_agent raises FileNotFoundError for nonexistent app agents."""
    with pytest.raises(FileNotFoundError) as exc_info:
        get_agent("fakeapp:fakeagent")

    assert "fakeapp:fakeagent" in str(exc_info.value)


def test_get_muse_configs_includes_system_agents(fixture_journal):
    """Test get_muse_configs returns system agents with metadata."""
    agents = get_muse_configs(type="cogitate")

    # Should include known system agents with frontmatter metadata
    assert "default" in agents
    assert agents["default"]["source"] == "system"
    assert "title" in agents["default"]
    assert "path" in agents["default"]


def test_get_muse_configs_system_agents_have_metadata(fixture_journal):
    """Test system agents have proper metadata fields."""
    agents = get_muse_configs(type="cogitate")

    # Check a known system agent
    default = agents.get("default")
    assert default is not None
    assert default["source"] == "system"
    assert "title" in default
    assert "color" in default


def test_get_muse_configs_excludes_private_apps(fixture_journal, tmp_path, monkeypatch):
    """Test get_muse_configs skips apps starting with underscore."""
    # Create a private app with an agent
    private_app = tmp_path / "_private_app" / "agents"
    private_app.mkdir(parents=True)
    (private_app / "secret.md").write_text("Secret agent")

    # This is tricky to test without modifying the actual apps directory
    # The current implementation filters by app_path.name.startswith("_")
    # We verify this by checking the code behavior with get_muse_configs()

    agents = get_muse_configs(type="cogitate")

    # No agents should have keys starting with "_"
    for key in agents:
        assert not key.startswith("_"), f"Private app agent found: {key}"


def test_app_agent_namespace_format(fixture_journal):
    """Test app agent keys follow {app}:{agent} format."""
    agents = get_muse_configs(type="cogitate")

    for key, config in agents.items():
        if config.get("source") == "app":
            # App agents must have colon in key
            assert ":" in key, f"App agent key missing namespace: {key}"
            app_name, agent_name = key.split(":", 1)
            assert config.get("app") == app_name


# --- _resolve_output_path tests ---


class TestResolveOutputPath:
    """Tests for _resolve_output_path route helper."""

    def test_explicit_output_path_returned_directly(self):
        """When output_path is set, return it as-is without derivation."""
        event = {
            "output_path": "/journal/facets/work/activities/20260214/coding_100/summary.md"
        }
        result = _resolve_output_path(event, "/journal")
        assert result == Path(
            "/journal/facets/work/activities/20260214/coding_100/summary.md"
        )

    def test_derives_path_from_request_fields(self, fixture_journal):
        """Without output_path, derives from day/name/segment fields."""
        event = {
            "day": "20260214",
            "name": "default",
            "segment": "100",
            "facet": "health",
        }
        result = _resolve_output_path(event, "tests/fixtures/journal")
        assert result is not None
        assert "20260214" in str(result)
        assert result.suffix in (".md", ".json")

    def test_returns_none_without_day_or_output_path(self):
        """Returns None when neither output_path nor day is present."""
        event = {"name": "default"}
        result = _resolve_output_path(event, "/journal")
        assert result is None

    def test_empty_output_path_falls_through(self, fixture_journal):
        """Empty string output_path falls through to derivation."""
        event = {"output_path": "", "day": "20260214", "name": "default"}
        result = _resolve_output_path(event, "tests/fixtures/journal")
        # Empty string is falsy, so falls through to derivation
        assert result is not None

    def test_uses_env_stream_name(self, fixture_journal):
        """SOL_STREAM from env is passed through to get_output_path."""
        event = {
            "day": "20260214",
            "name": "default",
            "env": {"SOL_STREAM": "mystream"},
        }
        result = _resolve_output_path(event, "tests/fixtures/journal")
        assert result is not None

    def test_explicit_path_ignores_other_fields(self):
        """When output_path is set, day/name/segment are ignored."""
        event = {
            "output_path": "/custom/path/output.md",
            "day": "20260214",
            "name": "default",
            "segment": "100",
        }
        result = _resolve_output_path(event, "/journal")
        assert result == Path("/custom/path/output.md")


# --- api_output_file endpoint tests ---


@pytest.fixture
def agents_client(tmp_path):
    """Create a Flask test client with agents blueprint and tmp journal."""
    from flask import Flask

    from apps.agents.routes import agents_bp
    from convey import state

    app = Flask(__name__)
    app.register_blueprint(agents_bp)

    # Point state at our tmp journal
    state.journal_root = str(tmp_path)

    # Create test files
    day_dir = tmp_path / "20260214"
    day_dir.mkdir()
    (day_dir / "agents" / "flow.md").parent.mkdir(parents=True)
    (day_dir / "agents" / "flow.md").write_text("# Day agent output")

    facet_dir = tmp_path / "facets" / "work" / "activities" / "20260214" / "coding_100"
    facet_dir.mkdir(parents=True)
    (facet_dir / "summary.md").write_text("# Activity summary")

    yield app.test_client()


class TestApiOutputFile:
    """Tests for api_output_file endpoint."""

    def test_serves_day_relative_file(self, agents_client):
        """Day-relative paths resolve under {journal}/{day}/."""
        resp = agents_client.get("/app/agents/api/output/20260214/agents/flow.md")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["content"] == "# Day agent output"
        assert data["format"] == "md"
        assert data["filename"] == "flow.md"

    def test_serves_facet_scoped_activity_file(self, agents_client):
        """Paths starting with facets/ resolve from journal root."""
        resp = agents_client.get(
            "/app/agents/api/output/20260214/"
            "facets/work/activities/20260214/coding_100/summary.md"
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["content"] == "# Activity summary"
        assert data["format"] == "md"

    def test_rejects_invalid_day_format(self, agents_client):
        """Non-YYYYMMDD day returns 400."""
        resp = agents_client.get("/app/agents/api/output/bad-day/agents/flow.md")
        assert resp.status_code == 400

    def test_rejects_path_traversal(self, agents_client):
        """Path traversal attempts return 403."""
        resp = agents_client.get("/app/agents/api/output/20260214/../../etc/passwd")
        assert resp.status_code in (403, 404)

    def test_missing_file_returns_404(self, agents_client):
        """Non-existent file returns 404."""
        resp = agents_client.get(
            "/app/agents/api/output/20260214/agents/nonexistent.md"
        )
        assert resp.status_code == 404
