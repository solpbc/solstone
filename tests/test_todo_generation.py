"""Tests for the convey todos generation endpoints."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from flask import Flask

from convey.views.todos import bp


@pytest.fixture
def app():
    app = Flask(__name__)
    app.config["TESTING"] = True
    app.register_blueprint(bp)
    return app


@pytest.fixture
def client(app):
    return app.test_client()


def test_todo_generation_endpoint(client, monkeypatch, tmp_path):
    journal_root = tmp_path / "journal"
    journal_root.mkdir()
    agents_dir = journal_root / "agents"
    agents_dir.mkdir()

    import convey.state as convey_state

    monkeypatch.setattr(convey_state, "journal_root", str(journal_root))
    monkeypatch.setenv("JOURNAL_PATH", str(journal_root))

    monkeypatch.setattr(convey_state, "todo_generation_agents", {}, raising=False)

    with patch("muse.cortex_client.cortex_request") as mock_request:
        mock_request.return_value = str(agents_dir / "agent_1_active.jsonl")

        resp = client.post("/todos/20240101/generate")
        assert resp.status_code == 200

        data = json.loads(resp.data)
        assert data == {"agent_id": "agent_1", "status": "started"}

        prompt = mock_request.call_args.kwargs["prompt"]
        assert "2024-01-01" in prompt
        assert "domains/personal/todos/20240101.md" in prompt


def test_todo_generation_status_endpoint(client, monkeypatch, tmp_path):
    journal_root = tmp_path / "journal"
    journal_root.mkdir()
    agents_dir = journal_root / "agents"
    agents_dir.mkdir()

    import convey.state as convey_state

    monkeypatch.setattr(convey_state, "journal_root", str(journal_root))
    monkeypatch.setenv("JOURNAL_PATH", str(journal_root))

    # No agent tracking yet
    resp = client.get("/todos/20240101/generation-status")
    data = json.loads(resp.data)
    assert data == {"status": "none", "agent_id": None}

    # Running agent
    active_file = agents_dir / "agent_1_active.jsonl"
    active_file.write_text('{"event": "request"}\n', encoding="utf-8")

    resp = client.get("/todos/20240101/generation-status?agent_id=agent_1")
    data = json.loads(resp.data)
    assert data["status"] == "running"

    # Finished agent with todo file present
    (agents_dir / "agent_1.jsonl").write_text('{"event": "start"}\n', encoding="utf-8")
    todo_file = Path(journal_root) / "domains" / "personal" / "todos"
    todo_file.mkdir(parents=True)
    (todo_file / "20240101.md").write_text("- [ ] Example\n", encoding="utf-8")

    resp = client.get("/todos/20240101/generation-status?agent_id=agent_1")
    data = json.loads(resp.data)
    assert data["status"] == "finished"
    assert data["todo_created"] is True


def test_todo_generation_no_cortex(client, monkeypatch):
    import convey.state as convey_state

    monkeypatch.setattr(convey_state, "journal_root", "/tmp/journal")
    monkeypatch.delenv("JOURNAL_PATH", raising=False)

    resp = client.post("/todos/20240101/generate")
    assert resp.status_code == 500
    data = json.loads(resp.data)
    assert "error" in data
    assert "Failed to spawn agent" in data["error"]
