import importlib
import json
from pathlib import Path


def test_agents_list_all(monkeypatch, tmp_path):
    """Test listing both live and historical agents."""
    from unittest.mock import Mock

    review = importlib.import_module("dream")
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir(parents=True)
    review.journal_root = str(tmp_path)

    # Create a historical agent file
    historical_file = agents_dir / "200000.jsonl"
    historical_file.write_text(
        json.dumps(
            {
                "event": "start",
                "ts": 200000,
                "prompt": "historical test",
                "model": "gpt-4",
                "persona": "default",
            }
        )
        + "\n"
        + json.dumps({"event": "finish", "ts": 201000, "result": "done"})
    )

    # Mock cortex client to return live agent data
    mock_client = Mock()
    mock_client.list_agents.return_value = {
        "agents": [
            {
                "id": "100000",
                "started_at": 100000,  # milliseconds
                "status": "running",
                "pid": 12345,
                "metadata": {"prompt": "hi", "persona": "p", "model": "m"},
            }
        ],
        "pagination": {"limit": 100, "offset": 0, "total": 1},
    }

    # Mock the cortex client getter
    monkeypatch.setattr(
        "think.cortex_client_sync.get_global_cortex_client", lambda: mock_client
    )
    monkeypatch.setattr("time.time", lambda: 300)  # 300 seconds after start

    with review.app.test_request_context("/agents/api/list?type=all"):
        resp = review.agents_list()

    # Check response structure
    assert "agents" in resp.json
    assert "pagination" in resp.json
    assert "live_count" in resp.json
    assert "historical_count" in resp.json

    agents = resp.json["agents"]
    assert len(agents) == 2  # 1 live + 1 historical

    # Check live agent
    live_agent = next((a for a in agents if a["id"] == "100000"), None)
    assert live_agent is not None
    assert live_agent["is_live"] is True
    assert live_agent["status"] == "running"
    assert live_agent["pid"] == 12345

    # Check historical agent
    hist_agent = next((a for a in agents if a["id"] == "200000"), None)
    assert hist_agent is not None
    assert hist_agent["is_live"] is False
    assert hist_agent["status"] == "finished"
    assert hist_agent["pid"] is None

    # Check counts
    assert resp.json["live_count"] == 1
    assert resp.json["historical_count"] == 1


def test_agents_list_historical_only(monkeypatch, tmp_path):
    """Test listing only historical agents when cortex is unavailable."""
    review = importlib.import_module("dream")
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir(parents=True)
    review.journal_root = str(tmp_path)

    # Create historical agent files
    historical_file1 = agents_dir / "200000.jsonl"
    historical_file1.write_text(
        json.dumps(
            {
                "event": "start",
                "ts": 200000,
                "prompt": "test 1",
                "model": "gpt-4",
                "persona": "default",
            }
        )
        + "\n"
        + json.dumps({"event": "finish", "ts": 201000})
    )

    historical_file2 = agents_dir / "300000.jsonl"
    historical_file2.write_text(
        json.dumps(
            {
                "event": "start",
                "ts": 300000,
                "prompt": "test 2",
                "model": "gpt-3.5",
                "persona": "custom",
            }
        )
        + "\n"
        + json.dumps({"event": "error", "ts": 301000, "error": "failed"})
    )

    # Mock cortex client as unavailable
    monkeypatch.setattr(
        "think.cortex_client_sync.get_global_cortex_client", lambda: None
    )
    monkeypatch.setattr("time.time", lambda: 400)

    with review.app.test_request_context("/agents/api/list?type=historical"):
        resp = review.agents_list()

    agents = resp.json["agents"]
    assert len(agents) == 2

    # Check they're sorted by timestamp (newest first)
    assert agents[0]["id"] == "300000"
    assert agents[0]["status"] == "error"
    assert agents[1]["id"] == "200000"
    assert agents[1]["status"] == "finished"

    # Check counts
    assert resp.json["live_count"] == 0
    assert resp.json["historical_count"] == 2


def test_agents_list_with_real_fixtures(monkeypatch):
    """Test listing agents from actual fixture files."""
    review = importlib.import_module("dream")

    # Use the real fixtures directory
    fixtures_path = Path(__file__).parent.parent / "fixtures" / "journal"
    if not fixtures_path.exists():
        import pytest

        pytest.skip("Fixtures directory not found")

    review.journal_root = str(fixtures_path)

    # Mock cortex client as unavailable
    monkeypatch.setattr(
        "think.cortex_client_sync.get_global_cortex_client", lambda: None
    )
    monkeypatch.setattr("time.time", lambda: 1755392200)  # Time after all fixtures

    with review.app.test_request_context("/agents/api/list?type=historical"):
        resp = review.agents_list()

    # Should have loaded agents from fixtures
    assert resp.json["historical_count"] > 0
    agents = resp.json["agents"]

    # Verify agent structure
    if agents:
        agent = agents[0]
        assert "id" in agent
        assert "prompt" in agent
        assert "model" in agent
        assert "persona" in agent
        assert "status" in agent
        assert agent["is_live"] is False

        # Check status values are correct
        assert agent["status"] in ["finished", "error", "interrupted"]
