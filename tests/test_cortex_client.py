"""Tests for cortex_client module with Callosum."""

import json
import os
import shutil
import tempfile
import threading
import time
from pathlib import Path

import pytest

from muse.cortex_client import (
    cortex_agents,
    cortex_request,
    get_agent_status,
    get_agent_thread,
)
from think.callosum import CallosumConnection, CallosumServer
from think.models import GPT_5


@pytest.fixture
def callosum_server(monkeypatch):
    """Start a Callosum server for testing.

    Uses a short temp path in /tmp to avoid Unix socket path length limits
    (~104 chars on macOS). pytest's tmp_path creates paths that are too long.
    """
    # Create short temp dir to avoid Unix socket path length limits
    tmp_dir = tempfile.mkdtemp(dir="/tmp", prefix="callosum_")
    tmp_path = Path(tmp_dir)

    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    (tmp_path / "agents").mkdir(parents=True, exist_ok=True)

    server = CallosumServer()
    server_thread = threading.Thread(target=server.start, daemon=True)
    server_thread.start()

    # Wait for server to be ready
    socket_path = tmp_path / "health" / "callosum.sock"
    for _ in range(50):
        if socket_path.exists():
            break
        time.sleep(0.1)
    else:
        pytest.fail("Callosum server did not start in time")

    yield tmp_path

    server.stop()
    server_thread.join(timeout=2)
    shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.fixture
def callosum_listener(callosum_server):
    """Provide a CallosumConnection listener that collects received messages.

    Yields (messages, listener) where messages is a list that accumulates
    all broadcast messages received during the test.
    """
    messages = []

    def callback(msg):
        messages.append(msg)

    listener = CallosumConnection()
    listener.start(callback=callback)
    time.sleep(0.1)  # Allow connection to establish

    yield messages

    listener.stop()


def test_cortex_request_broadcasts_to_callosum(callosum_listener):
    """Test that cortex_request broadcasts request to Callosum."""
    messages = callosum_listener

    # Create a request
    agent_id = cortex_request(
        prompt="Test prompt",
        persona="default",
        backend="openai",
        config={"model": GPT_5},
    )

    time.sleep(0.2)

    # Verify broadcast was received
    assert len(messages) == 1
    msg = messages[0]
    assert msg["tract"] == "cortex"
    assert msg["event"] == "request"
    assert msg["prompt"] == "Test prompt"
    assert msg["persona"] == "default"
    assert msg["backend"] == "openai"
    assert msg["model"] == GPT_5
    assert msg["agent_id"] == agent_id
    assert "ts" in msg


def test_cortex_request_returns_agent_id(callosum_server):
    """Test that cortex_request returns agent_id string."""
    _ = callosum_server  # Needed for side effects only

    agent_id = cortex_request(prompt="Test", persona="default", backend="openai")

    # Verify agent_id is a string timestamp
    assert isinstance(agent_id, str)
    assert agent_id.isdigit()
    assert len(agent_id) == 13  # Millisecond timestamp


def test_cortex_request_with_handoff(callosum_listener):
    """Test cortex_request with handoff_from parameter."""
    messages = callosum_listener

    cortex_request(
        prompt="Continue analysis",
        persona="reviewer",
        backend="anthropic",
        handoff_from="1234567890000",
    )

    time.sleep(0.2)

    msg = messages[0]
    assert msg["handoff_from"] == "1234567890000"
    assert msg["persona"] == "reviewer"


def test_cortex_request_unique_agent_ids(callosum_server):
    """Test that cortex_request generates unique agent IDs."""
    _ = callosum_server  # Needed for side effects only

    agent_ids = []
    for i in range(3):
        agent_id = cortex_request(
            prompt=f"Test {i}", persona="default", backend="openai"
        )
        agent_ids.append(agent_id)
        time.sleep(0.002)

    # All agent IDs should be unique
    assert len(set(agent_ids)) == 3


def test_cortex_request_no_journal_path(callosum_server):
    """Test cortex_request fails without JOURNAL_PATH."""
    _ = callosum_server  # Needed for side effects only
    old_path = os.environ.pop("JOURNAL_PATH", None)
    try:
        with pytest.raises(
            ValueError, match="JOURNAL_PATH environment variable not set"
        ):
            cortex_request("test", "default", "openai")
    finally:
        if old_path:
            os.environ["JOURNAL_PATH"] = old_path


# Tests for cortex_agents remain mostly unchanged as they read from files


def test_cortex_agents_empty(tmp_path, monkeypatch):
    """Test cortex_agents with no agents."""
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))

    result = cortex_agents()

    assert result["agents"] == []
    assert result["pagination"]["total"] == 0
    assert result["pagination"]["has_more"] is False
    assert result["live_count"] == 0
    assert result["historical_count"] == 0


def test_cortex_agents_with_active(tmp_path, monkeypatch):
    """Test cortex_agents with active (running) agents."""
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()

    # Create active agent files
    ts1 = int(time.time() * 1000)
    ts2 = ts1 + 1000

    active_file1 = agents_dir / f"{ts1}_active.jsonl"
    with open(active_file1, "w") as f:
        json.dump(
            {
                "event": "request",
                "ts": ts1,
                "prompt": "Task 1",
                "persona": "default",
                "backend": "openai",
            },
            f,
        )
        f.write("\n")

    active_file2 = agents_dir / f"{ts2}_active.jsonl"
    with open(active_file2, "w") as f:
        json.dump(
            {
                "event": "request",
                "ts": ts2,
                "prompt": "Task 2",
                "persona": "tester",
                "backend": "google",
            },
            f,
        )
        f.write("\n")

    result = cortex_agents()

    assert len(result["agents"]) == 2
    assert result["live_count"] == 2
    assert result["historical_count"] == 0


def test_cortex_agents_with_completed(tmp_path, monkeypatch):
    """Test cortex_agents with completed (historical) agents."""
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()

    # Create completed agent files
    ts1 = int(time.time() * 1000)

    completed_file1 = agents_dir / f"{ts1}.jsonl"
    with open(completed_file1, "w") as f:
        json.dump(
            {
                "event": "request",
                "ts": ts1,
                "prompt": "Old task",
                "persona": "reviewer",
                "backend": "anthropic",
            },
            f,
        )
        f.write("\n")
        json.dump({"event": "finish", "ts": ts1 + 100, "result": "Done"}, f)
        f.write("\n")

    result = cortex_agents()

    assert len(result["agents"]) == 1
    assert result["live_count"] == 0
    assert result["historical_count"] == 1
    assert result["agents"][0]["status"] == "completed"


def test_cortex_agents_pagination(tmp_path, monkeypatch):
    """Test cortex_agents pagination."""
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()

    # Create multiple agents
    base_ts = int(time.time() * 1000)
    for i in range(5):
        ts = base_ts + (i * 1000)
        file = agents_dir / f"{ts}.jsonl"
        with open(file, "w") as f:
            json.dump(
                {
                    "event": "request",
                    "ts": ts,
                    "prompt": f"Task {i}",
                    "persona": "default",
                },
                f,
            )
            f.write("\n")

    # Test limit
    result = cortex_agents(limit=2)
    assert len(result["agents"]) == 2
    assert result["pagination"]["limit"] == 2
    assert result["pagination"]["total"] == 5
    assert result["pagination"]["has_more"] is True


def test_cortex_agents_no_journal_path():
    """Test cortex_agents fails without JOURNAL_PATH."""
    old_path = os.environ.pop("JOURNAL_PATH", None)
    try:
        with pytest.raises(
            ValueError, match="JOURNAL_PATH environment variable not set"
        ):
            cortex_agents()
    finally:
        if old_path:
            os.environ["JOURNAL_PATH"] = old_path


def test_get_agent_status_completed(tmp_path, monkeypatch):
    """Test get_agent_status returns 'completed' for finished agents."""
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()

    agent_id = "1234567890123"
    (agents_dir / f"{agent_id}.jsonl").write_text('{"event": "finish"}\n')

    assert get_agent_status(agent_id) == "completed"


def test_get_agent_status_running(tmp_path, monkeypatch):
    """Test get_agent_status returns 'running' for active agents."""
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()

    agent_id = "1234567890123"
    (agents_dir / f"{agent_id}_active.jsonl").write_text('{"event": "start"}\n')

    assert get_agent_status(agent_id) == "running"


def test_get_agent_status_not_found(tmp_path, monkeypatch):
    """Test get_agent_status returns 'not_found' for missing agents."""
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    (tmp_path / "agents").mkdir()

    assert get_agent_status("nonexistent") == "not_found"


def test_get_agent_status_prefers_completed(tmp_path, monkeypatch):
    """Test get_agent_status returns 'completed' when both files exist."""
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()

    # Edge case: both files exist (shouldn't happen, but check precedence)
    agent_id = "1234567890123"
    (agents_dir / f"{agent_id}.jsonl").write_text('{"event": "finish"}\n')
    (agents_dir / f"{agent_id}_active.jsonl").write_text('{"event": "start"}\n')

    assert get_agent_status(agent_id) == "completed"


def test_get_agent_thread_single_agent(tmp_path, monkeypatch):
    """Test get_agent_thread with a single agent (no thread)."""
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()

    agent_id = "1000"
    (agents_dir / f"{agent_id}.jsonl").write_text(
        '{"event": "request", "prompt": "hello"}\n'
        '{"event": "finish", "result": "done"}\n'
    )

    assert get_agent_thread(agent_id) == [agent_id]


def test_get_agent_thread_from_root(tmp_path, monkeypatch):
    """Test get_agent_thread starting from the root of a 3-agent thread."""
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()

    # Create a thread: 1000 -> 2000 -> 3000
    (agents_dir / "1000.jsonl").write_text(
        '{"event": "request", "prompt": "first"}\n'
        '{"event": "finish", "result": "done"}\n'
        '{"event": "continue", "to": "2000"}\n'
    )
    (agents_dir / "2000.jsonl").write_text(
        '{"event": "request", "prompt": "second", "continue_from": "1000"}\n'
        '{"event": "finish", "result": "done"}\n'
        '{"event": "continue", "to": "3000"}\n'
    )
    (agents_dir / "3000.jsonl").write_text(
        '{"event": "request", "prompt": "third", "continue_from": "2000"}\n'
        '{"event": "finish", "result": "done"}\n'
    )

    assert get_agent_thread("1000") == ["1000", "2000", "3000"]


def test_get_agent_thread_from_middle(tmp_path, monkeypatch):
    """Test get_agent_thread starting from the middle of a thread."""
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()

    # Create a thread: 1000 -> 2000 -> 3000
    (agents_dir / "1000.jsonl").write_text(
        '{"event": "request", "prompt": "first"}\n'
        '{"event": "finish", "result": "done"}\n'
        '{"event": "continue", "to": "2000"}\n'
    )
    (agents_dir / "2000.jsonl").write_text(
        '{"event": "request", "prompt": "second", "continue_from": "1000"}\n'
        '{"event": "finish", "result": "done"}\n'
        '{"event": "continue", "to": "3000"}\n'
    )
    (agents_dir / "3000.jsonl").write_text(
        '{"event": "request", "prompt": "third", "continue_from": "2000"}\n'
        '{"event": "finish", "result": "done"}\n'
    )

    assert get_agent_thread("2000") == ["1000", "2000", "3000"]


def test_get_agent_thread_from_end(tmp_path, monkeypatch):
    """Test get_agent_thread starting from the end of a thread."""
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()

    # Create a thread: 1000 -> 2000 -> 3000
    (agents_dir / "1000.jsonl").write_text(
        '{"event": "request", "prompt": "first"}\n'
        '{"event": "finish", "result": "done"}\n'
        '{"event": "continue", "to": "2000"}\n'
    )
    (agents_dir / "2000.jsonl").write_text(
        '{"event": "request", "prompt": "second", "continue_from": "1000"}\n'
        '{"event": "finish", "result": "done"}\n'
        '{"event": "continue", "to": "3000"}\n'
    )
    (agents_dir / "3000.jsonl").write_text(
        '{"event": "request", "prompt": "third", "continue_from": "2000"}\n'
        '{"event": "finish", "result": "done"}\n'
    )

    assert get_agent_thread("3000") == ["1000", "2000", "3000"]


def test_get_agent_thread_not_found(tmp_path, monkeypatch):
    """Test get_agent_thread raises FileNotFoundError for missing agent."""
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    (tmp_path / "agents").mkdir()

    with pytest.raises(FileNotFoundError):
        get_agent_thread("nonexistent")
