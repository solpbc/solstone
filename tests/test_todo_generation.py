"""Test TODO generation feature."""

import json
from unittest.mock import MagicMock, patch

import pytest
from flask import Flask

from dream.views.calendar import bp


@pytest.fixture
def app():
    """Create test Flask app with calendar blueprint."""
    app = Flask(__name__)
    app.config["TESTING"] = True
    app.register_blueprint(bp)
    return app


@pytest.fixture
def client(app):
    """Create test client."""
    return app.test_client()


def test_todo_generation_endpoint(client, monkeypatch):
    """Test the TODO generation endpoint."""
    # Set journal root
    journal_root = "/tmp/test_journal"
    import dream.state as dream_state

    monkeypatch.setattr(dream_state, "journal_root", journal_root)

    # Mock cortex client
    mock_client = MagicMock()
    mock_client.spawn_agent.return_value = "test_agent_123"

    with patch("dream.cortex_client.get_global_cortex_client") as mock_get_client:
        mock_get_client.return_value = mock_client

        # Test generation request
        response = client.post("/calendar/20250101/todos/generate")
        assert response.status_code == 200

        data = json.loads(response.data)
        assert data["agent_id"] == "test_agent_123"
        assert data["status"] == "started"

        # Verify spawn_agent was called with correct parameters
        mock_client.spawn_agent.assert_called_once()
        call_args = mock_client.spawn_agent.call_args
        assert call_args.kwargs["persona"] == "todo"
        assert "20250101" in call_args.kwargs["prompt"]


def test_todo_generation_status_endpoint(client, monkeypatch, tmp_path):
    """Test the TODO generation status endpoint."""
    # Set journal root
    journal_root = tmp_path / "test_journal"
    journal_root.mkdir()
    agents_dir = journal_root / "agents"
    agents_dir.mkdir()

    # Import state module to ensure it's available
    import dream.state as dream_state

    monkeypatch.setattr(dream_state, "journal_root", str(journal_root))

    # Clear any state from previous tests
    if hasattr(dream_state, "todo_generation_agents"):
        monkeypatch.delattr(dream_state, "todo_generation_agents")

    # Test status when no agent exists
    response = client.get("/calendar/20250101/todos/generation-status")
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data["status"] == "none"
    assert data["agent_id"] is None

    # Test status with agent_id parameter (running)
    mock_client = MagicMock()
    mock_client.list_agents.return_value = {
        "agents": [{"id": "test_agent_123", "status": "running"}]
    }

    with patch("dream.cortex_client.get_global_cortex_client") as mock_get_client:
        mock_get_client.return_value = mock_client

        response = client.get(
            "/calendar/20250101/todos/generation-status?agent_id=test_agent_123"
        )
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["status"] == "running"
        assert data["agent_id"] == "test_agent_123"

    # Test status when agent is finished (create agent file)
    agent_file = agents_dir / "test_agent_123.jsonl"
    agent_file.write_text('{"event": "start", "ts": 1234567890}\n')

    # Create TODO.md file
    day_dir = journal_root / "20250101"
    day_dir.mkdir()
    todo_file = day_dir / "TODO.md"
    todo_file.write_text("# Today\n- [ ] Test task\n")

    response = client.get(
        "/calendar/20250101/todos/generation-status?agent_id=test_agent_123"
    )
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data["status"] == "finished"
    assert data["agent_id"] == "test_agent_123"
    assert data["todo_created"] is True


def test_todo_generation_no_cortex(client, monkeypatch):
    """Test TODO generation when cortex is not available."""
    # Set journal root
    import dream.state as dream_state

    monkeypatch.setattr(dream_state, "journal_root", "/tmp/test_journal")

    with patch("dream.cortex_client.get_global_cortex_client") as mock_get_client:
        mock_get_client.return_value = None

        response = client.post("/calendar/20250101/todos/generate")
        assert response.status_code == 503
        data = json.loads(response.data)
        assert "error" in data
        assert "Cortex service not available" in data["error"]
