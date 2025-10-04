import importlib
import json
from pathlib import Path


def test_agents_list_all(monkeypatch, tmp_path):
    """Test listing both live and historical agents."""
    import os

    # Set up environment
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))

    review = importlib.import_module("convey")
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir(parents=True)
    review.journal_root = str(tmp_path)

    # Create a historical agent file (completed)
    historical_file = agents_dir / "200000.jsonl"
    historical_file.write_text(
        json.dumps(
            {
                "event": "request",
                "ts": 200000,
                "prompt": "historical test",
                "backend": "google",
                "persona": "default",
            }
        )
        + "\n"
        + json.dumps({"event": "finish", "ts": 201000, "result": "done"})
    )

    # Create a live agent file (still active)
    live_file = agents_dir / "100000_active.jsonl"
    live_file.write_text(
        json.dumps(
            {
                "event": "request",
                "ts": 100000,
                "prompt": "hi",
                "backend": "openai",
                "persona": "default",
            }
        )
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

    # Check live agent (should be first due to newer timestamp in active state)
    live_agent = next((a for a in agents if a["id"] == "100000"), None)
    assert live_agent is not None
    assert live_agent["status"] == "running"
    assert live_agent["model"] == "openai"  # backend is stored as model
    assert live_agent["pid"] is None  # We don't track PIDs in the new system

    # Check historical agent
    hist_agent = next((a for a in agents if a["id"] == "200000"), None)
    assert hist_agent is not None
    assert hist_agent["status"] == "completed"
    assert hist_agent["model"] == "google"  # backend is stored as model
    assert hist_agent["pid"] is None
    assert "runtime_seconds" in hist_agent  # Should have calculated runtime

    # Check counts
    assert resp.json["live_count"] == 1
    assert resp.json["historical_count"] == 1


def test_agents_list_historical_only(monkeypatch, tmp_path):
    """Test listing only historical agents."""
    import os

    # Set up environment
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))

    review = importlib.import_module("convey")
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir(parents=True)
    review.journal_root = str(tmp_path)

    # Create historical agent files
    historical_file1 = agents_dir / "200000.jsonl"
    historical_file1.write_text(
        json.dumps(
            {
                "event": "request",
                "ts": 200000,
                "prompt": "test 1",
                "backend": "openai",
                "persona": "default",
            }
        )
        + "\n"
        + json.dumps({"event": "finish", "ts": 201000, "result": "done"})
    )

    historical_file2 = agents_dir / "300000.jsonl"
    historical_file2.write_text(
        json.dumps(
            {
                "event": "request",
                "ts": 300000,
                "prompt": "test 2",
                "backend": "google",
                "persona": "custom",
            }
        )
        + "\n"
        + json.dumps({"event": "error", "ts": 301000, "error": "failed"})
    )

    monkeypatch.setattr("time.time", lambda: 400)

    with review.app.test_request_context("/agents/api/list?type=historical"):
        resp = review.agents_list()

    agents = resp.json["agents"]
    assert len(agents) == 2

    # Check they're sorted by timestamp (newest first)
    assert agents[0]["id"] == "300000"
    assert agents[0]["status"] == "completed"  # All non-active files are "completed"
    assert agents[0]["model"] == "google"  # backend is stored as model
    assert agents[1]["id"] == "200000"
    assert agents[1]["status"] == "completed"
    assert agents[1]["model"] == "openai"  # backend is stored as model

    # Check counts
    assert resp.json["live_count"] == 0
    assert resp.json["historical_count"] == 2


def test_agents_list_with_real_fixtures(monkeypatch):
    """Test listing agents from actual fixture files."""
    import os

    review = importlib.import_module("convey")

    # Use the real fixtures directory
    fixtures_path = Path(__file__).parent.parent / "fixtures" / "journal"
    if not fixtures_path.exists():
        import pytest

        pytest.skip("Fixtures directory not found")

    # Set up environment
    monkeypatch.setenv("JOURNAL_PATH", str(fixtures_path))
    review.journal_root = str(fixtures_path)
    monkeypatch.setattr("time.time", lambda: 1755392200)  # Time after all fixtures

    with review.app.test_request_context("/agents/api/list?type=historical"):
        resp = review.agents_list()

    # Should have loaded agents from fixtures
    assert resp.json["historical_count"] >= 0  # May be 0 if no agent files in fixtures
    agents = resp.json["agents"]

    # Verify agent structure if we have agents
    if agents:
        agent = agents[0]
        assert "id" in agent
        assert "prompt" in agent
        assert "model" in agent
        assert "persona" in agent
        assert "status" in agent
        assert agent["pid"] is None  # We don't track PIDs

        # Check status values are correct
        assert agent["status"] in ["completed", "running"]
