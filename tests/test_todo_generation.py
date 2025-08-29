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

    with patch("think.cortex_client_sync.get_global_cortex_client") as mock_get_client:
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

    with patch("think.cortex_client_sync.get_global_cortex_client") as mock_get_client:
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

    with patch("think.cortex_client_sync.get_global_cortex_client") as mock_get_client:
        mock_get_client.return_value = None

        response = client.post("/calendar/20250101/todos/generate")
        assert response.status_code == 503
        data = json.loads(response.data)
        assert "error" in data
        assert "Cortex service not available" in data["error"]


def test_todo_update_adds_timestamp(client, monkeypatch, tmp_path):
    """Test that updating a todo item adds a timestamp."""
    # Set journal root
    journal_root = tmp_path / "test_journal"
    journal_root.mkdir()
    day_dir = journal_root / "20250101"
    day_dir.mkdir()

    # Create initial TODO.md without timestamps
    todo_file = day_dir / "TODO.md"
    todo_file.write_text(
        """# Today
- [ ] **Task**: Test task without timestamp
- [ ] **Meeting**: Another task

# Future
"""
    )

    import dream.state as dream_state

    monkeypatch.setattr(dream_state, "journal_root", str(journal_root))

    # Mock datetime to control the timestamp
    mock_datetime = MagicMock()
    mock_datetime.now.return_value.strftime.return_value = "15:30"

    with patch("datetime.datetime", mock_datetime):

        # Update the first item (mark as completed)
        response = client.post(
            "/calendar/20250101/todos",
            json={
                "action": "update",
                "line_number": 2,  # Line 2 is the first todo item
                "field": "completed",
                "value": True,
            },
        )
        assert response.status_code == 200

        # Read the updated file
        updated_content = todo_file.read_text()
        lines = updated_content.splitlines()

        # First item should now be completed with timestamp
        assert "- [x]" in lines[1]
        assert "(15:30)" in lines[1]

        # Second item should remain unchanged
        assert "- [ ]" in lines[2]
        assert "(15:30)" not in lines[2]


def test_todo_add_includes_timestamp(client, monkeypatch, tmp_path):
    """Test that adding a new todo item includes a timestamp."""
    # Set journal root
    journal_root = tmp_path / "test_journal"
    journal_root.mkdir()
    day_dir = journal_root / "20250101"
    day_dir.mkdir()

    # Create initial TODO.md
    todo_file = day_dir / "TODO.md"
    todo_file.write_text(
        """# Today

# Future
"""
    )

    import dream.state as dream_state

    monkeypatch.setattr(dream_state, "journal_root", str(journal_root))

    # Mock datetime to control the timestamp
    mock_datetime = MagicMock()
    mock_datetime.now.return_value.strftime.return_value = "10:45"

    with patch("datetime.datetime", mock_datetime):

        # Add a new task to today section
        response = client.post(
            "/calendar/20250101/todos",
            json={
                "action": "add",
                "section": "today",
                "text": "Task: New test task #testing",
            },
        )
        assert response.status_code == 200

        # Read the updated file
        updated_content = todo_file.read_text()

        # New task should have timestamp
        assert "**Task**: New test task #testing (10:45)" in updated_content

        # Add a task to future section (should not have timestamp)
        response = client.post(
            "/calendar/20250101/todos",
            json={"action": "add", "section": "future", "text": "Goal: Future goal"},
        )
        assert response.status_code == 200

        updated_content = todo_file.read_text()
        assert "**Goal**: Future goal" in updated_content
        assert "**Goal**: Future goal (10:45)" not in updated_content


def test_todo_timestamp_update_preserves_content(client, monkeypatch, tmp_path):
    """Test that timestamp updates preserve the task content."""

    # Set journal root
    journal_root = tmp_path / "test_journal"
    journal_root.mkdir()
    day_dir = journal_root / "20250101"
    day_dir.mkdir()

    # Create TODO.md with existing timestamp
    todo_file = day_dir / "TODO.md"
    todo_file.write_text(
        """# Today
- [ ] **Task**: Task with domain #hear (09:00)
- [x] **Meeting**: Completed task #dream (10:00)

# Future
"""
    )

    import dream.state as dream_state

    monkeypatch.setattr(dream_state, "journal_root", str(journal_root))

    # Mock datetime for new timestamp
    mock_datetime = MagicMock()
    mock_datetime.now.return_value.strftime.return_value = "16:45"

    with patch("datetime.datetime", mock_datetime):

        # Update the first item
        response = client.post(
            "/calendar/20250101/todos",
            json={
                "action": "update",
                "line_number": 2,  # Line 2 is the first todo item
                "field": "completed",
                "value": True,
            },
        )
        assert response.status_code == 200

        # Read the updated file
        updated_content = todo_file.read_text()
        lines = updated_content.splitlines()

        # Task should be completed with new timestamp but preserve content
        assert "- [x] **Task**: Task with domain #hear (16:45)" in lines[1]

        # Other items should remain unchanged
        assert "- [x] **Meeting**: Completed task #dream (10:00)" in lines[2]
