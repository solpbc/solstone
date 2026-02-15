# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for cortex_client module with Callosum."""

import json
import os
import shutil
import tempfile
import threading
import time
from pathlib import Path

import pytest

from think.callosum import CallosumConnection, CallosumServer
from think.cortex_client import (
    cortex_agents,
    cortex_request,
    get_agent_end_state,
    get_agent_log_status,
    wait_for_agents,
)
from think.models import GPT_5
from think.utils import now_ms


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
        name="default",
        provider="openai",
        config={"model": GPT_5},
    )

    time.sleep(0.2)

    # Verify broadcast was received
    assert len(messages) == 1
    msg = messages[0]
    assert msg["tract"] == "cortex"
    assert msg["event"] == "request"
    assert msg["prompt"] == "Test prompt"
    assert msg["name"] == "default"
    assert msg["provider"] == "openai"
    assert msg["model"] == GPT_5
    assert msg["agent_id"] == agent_id
    assert "ts" in msg


def test_cortex_request_returns_agent_id(callosum_server):
    """Test that cortex_request returns agent_id string."""
    _ = callosum_server  # Needed for side effects only

    agent_id = cortex_request(prompt="Test", name="default", provider="openai")

    # Verify agent_id is a string timestamp
    assert isinstance(agent_id, str)
    assert agent_id.isdigit()
    assert len(agent_id) == 13  # Millisecond timestamp


def test_cortex_request_with_handoff(callosum_listener):
    """Test cortex_request with handoff_from parameter."""
    messages = callosum_listener

    cortex_request(
        prompt="Continue analysis",
        name="reviewer",
        provider="anthropic",
        handoff_from="1234567890000",
    )

    time.sleep(0.2)

    msg = messages[0]
    assert msg["handoff_from"] == "1234567890000"
    assert msg["name"] == "reviewer"


def test_cortex_request_unique_agent_ids(callosum_server):
    """Test that cortex_request generates unique agent IDs."""
    _ = callosum_server  # Needed for side effects only

    agent_ids = []
    for i in range(3):
        agent_id = cortex_request(prompt=f"Test {i}", name="default", provider="openai")
        agent_ids.append(agent_id)
        time.sleep(0.002)

    # All agent IDs should be unique
    assert len(set(agent_ids)) == 3


def test_cortex_request_returns_none_on_send_failure(callosum_server, monkeypatch):
    """Test cortex_request returns None when callosum_send fails."""
    monkeypatch.setattr("think.cortex_client.callosum_send", lambda *a, **kw: False)

    agent_id = cortex_request(prompt="Test", name="default", provider="openai")

    assert agent_id is None


def test_cortex_request_uses_default_path_when_journal_path_unset(callosum_server):
    """Test cortex_request uses platform default when JOURNAL_PATH unset."""
    _ = callosum_server  # Needed for side effects only
    old_path = os.environ.pop("JOURNAL_PATH", None)
    try:
        # Uses platform default path, which won't match the test server socket.
        agent_id = cortex_request("test", "default", "openai")
        assert agent_id is None
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
    ts1 = now_ms()
    ts2 = ts1 + 1000

    default_dir = agents_dir / "default"
    tester_dir = agents_dir / "tester"
    default_dir.mkdir()
    tester_dir.mkdir()

    active_file1 = default_dir / f"{ts1}_active.jsonl"
    with open(active_file1, "w") as f:
        json.dump(
            {
                "event": "request",
                "ts": ts1,
                "prompt": "Task 1",
                "name": "default",
                "provider": "openai",
            },
            f,
        )
        f.write("\n")

    active_file2 = tester_dir / f"{ts2}_active.jsonl"
    with open(active_file2, "w") as f:
        json.dump(
            {
                "event": "request",
                "ts": ts2,
                "prompt": "Task 2",
                "name": "tester",
                "provider": "google",
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
    ts1 = now_ms()
    reviewer_dir = agents_dir / "reviewer"
    reviewer_dir.mkdir()

    completed_file1 = reviewer_dir / f"{ts1}.jsonl"
    with open(completed_file1, "w") as f:
        json.dump(
            {
                "event": "request",
                "ts": ts1,
                "prompt": "Old task",
                "name": "reviewer",
                "provider": "anthropic",
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
    base_ts = now_ms()
    default_dir = agents_dir / "default"
    default_dir.mkdir()
    for i in range(5):
        ts = base_ts + (i * 1000)
        file = default_dir / f"{ts}.jsonl"
        with open(file, "w") as f:
            json.dump(
                {
                    "event": "request",
                    "ts": ts,
                    "prompt": f"Task {i}",
                    "name": "default",
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


def test_cortex_agents_uses_default_path_when_journal_path_unset(monkeypatch):
    """Test cortex_agents uses platform default when JOURNAL_PATH unset."""
    import think.utils

    monkeypatch.delenv("JOURNAL_PATH", raising=False)
    monkeypatch.setattr(think.utils, "_journal_path_cache", None)

    # Should work (uses platform default) - doesn't raise due to missing path
    result = cortex_agents()
    # Just verify it returns the expected structure
    assert "agents" in result
    assert "pagination" in result
    assert isinstance(result["agents"], list)


def test_get_agent_log_status_completed(tmp_path, monkeypatch):
    """Test get_agent_log_status returns 'completed' for finished agents."""
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    default_dir = agents_dir / "default"
    default_dir.mkdir()

    agent_id = "1234567890123"
    (default_dir / f"{agent_id}.jsonl").write_text('{"event": "finish"}\n')

    assert get_agent_log_status(agent_id) == "completed"


def test_get_agent_log_status_running(tmp_path, monkeypatch):
    """Test get_agent_log_status returns 'running' for active agents."""
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    default_dir = agents_dir / "default"
    default_dir.mkdir()

    agent_id = "1234567890123"
    (default_dir / f"{agent_id}_active.jsonl").write_text('{"event": "start"}\n')

    assert get_agent_log_status(agent_id) == "running"


def test_get_agent_log_status_not_found(tmp_path, monkeypatch):
    """Test get_agent_log_status returns 'not_found' for missing agents."""
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    (tmp_path / "agents").mkdir()

    assert get_agent_log_status("nonexistent") == "not_found"


def test_get_agent_log_status_prefers_completed(tmp_path, monkeypatch):
    """Test get_agent_log_status returns 'completed' when both files exist."""
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    default_dir = agents_dir / "default"
    default_dir.mkdir()

    # Edge case: both files exist (shouldn't happen, but check precedence)
    agent_id = "1234567890123"
    (default_dir / f"{agent_id}.jsonl").write_text('{"event": "finish"}\n')
    (default_dir / f"{agent_id}_active.jsonl").write_text('{"event": "start"}\n')

    assert get_agent_log_status(agent_id) == "completed"


def test_get_agent_end_state_finish(tmp_path, monkeypatch):
    """Test get_agent_end_state returns 'finish' for successful agents."""
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    default_dir = agents_dir / "default"
    default_dir.mkdir()

    agent_id = "1234567890123"
    (default_dir / f"{agent_id}.jsonl").write_text(
        '{"event": "request", "prompt": "hello"}\n'
        '{"event": "finish", "result": "done"}\n'
    )

    assert get_agent_end_state(agent_id) == "finish"


def test_get_agent_end_state_error(tmp_path, monkeypatch):
    """Test get_agent_end_state returns 'error' for failed agents."""
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    default_dir = agents_dir / "default"
    default_dir.mkdir()

    agent_id = "1234567890123"
    (default_dir / f"{agent_id}.jsonl").write_text(
        '{"event": "request", "prompt": "hello"}\n'
        '{"event": "error", "error": "something went wrong"}\n'
    )

    assert get_agent_end_state(agent_id) == "error"


def test_get_agent_end_state_running(tmp_path, monkeypatch):
    """Test get_agent_end_state returns 'running' for active agents."""
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    default_dir = agents_dir / "default"
    default_dir.mkdir()

    agent_id = "1234567890123"
    (default_dir / f"{agent_id}_active.jsonl").write_text(
        '{"event": "request", "prompt": "hello"}\n'
    )

    assert get_agent_end_state(agent_id) == "running"


def test_get_agent_end_state_unknown(tmp_path, monkeypatch):
    """Test get_agent_end_state returns 'unknown' for missing agents."""
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    (tmp_path / "agents").mkdir()

    assert get_agent_end_state("nonexistent") == "unknown"


# Tests for wait_for_agents


def test_wait_for_agents_already_complete(tmp_path, monkeypatch):
    """Test wait_for_agents returns immediately if agents already completed."""
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    default_dir = agents_dir / "default"
    default_dir.mkdir()
    (tmp_path / "health").mkdir()

    # Create completed agents
    agent_ids = ["1000", "2000"]
    for agent_id in agent_ids:
        (default_dir / f"{agent_id}.jsonl").write_text('{"event": "finish"}\n')

    completed, timed_out = wait_for_agents(agent_ids, timeout=1)

    assert set(completed.keys()) == set(agent_ids)
    assert all(v == "finish" for v in completed.values())
    assert timed_out == []


def test_wait_for_agents_event_completion(callosum_server):
    """Test wait_for_agents completes when finish event is received."""
    tmp_path = callosum_server
    agents_dir = tmp_path / "agents"
    default_dir = agents_dir / "default"
    default_dir.mkdir(exist_ok=True)

    agent_id = "1234567890123"

    # Start wait in background thread
    result = {"completed": None, "timed_out": None}

    def wait_thread():
        result["completed"], result["timed_out"] = wait_for_agents(
            [agent_id], timeout=5
        )

    waiter = threading.Thread(target=wait_thread)
    waiter.start()

    # Give the waiter time to set up listener
    time.sleep(0.2)

    # Create the completed file and emit finish event
    (default_dir / f"{agent_id}.jsonl").write_text('{"event": "finish"}\n')

    # Emit finish event via Callosum
    client = CallosumConnection()
    client.start()
    time.sleep(0.1)
    client.emit("cortex", "finish", agent_id=agent_id, result="done")
    time.sleep(0.2)
    client.stop()

    waiter.join(timeout=3)

    assert result["completed"] == {agent_id: "finish"}
    assert result["timed_out"] == []


def test_wait_for_agents_error_event(callosum_server):
    """Test wait_for_agents completes on error event too."""
    tmp_path = callosum_server
    agents_dir = tmp_path / "agents"
    default_dir = agents_dir / "default"
    default_dir.mkdir(exist_ok=True)

    agent_id = "1234567890124"

    result = {"completed": None, "timed_out": None}

    def wait_thread():
        result["completed"], result["timed_out"] = wait_for_agents(
            [agent_id], timeout=5
        )

    waiter = threading.Thread(target=wait_thread)
    waiter.start()
    time.sleep(0.2)

    # Create completed file and emit error event
    (default_dir / f"{agent_id}.jsonl").write_text('{"event": "error"}\n')

    client = CallosumConnection()
    client.start()
    time.sleep(0.1)
    client.emit("cortex", "error", agent_id=agent_id, error="something failed")
    time.sleep(0.2)
    client.stop()

    waiter.join(timeout=3)

    assert result["completed"] == {agent_id: "error"}
    assert result["timed_out"] == []


def test_wait_for_agents_initial_file_check(tmp_path, monkeypatch):
    """Test wait_for_agents finds already-completed agents via initial file check."""
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    default_dir = agents_dir / "default"
    default_dir.mkdir()
    (tmp_path / "health").mkdir()

    agent_id = "1234567890125"

    # Agent already completed before we start waiting
    (default_dir / f"{agent_id}.jsonl").write_text('{"event": "finish"}\n')

    completed, timed_out = wait_for_agents([agent_id], timeout=1)

    # Should find via initial file check
    assert completed == {agent_id: "finish"}
    assert timed_out == []


def test_wait_for_agents_timeout_actual(tmp_path, monkeypatch):
    """Test wait_for_agents times out for agents that never complete."""
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    default_dir = agents_dir / "default"
    default_dir.mkdir()
    (tmp_path / "health").mkdir()

    agent_id = "1234567890126"
    # Create active file (not completed)
    (default_dir / f"{agent_id}_active.jsonl").write_text('{"event": "start"}\n')

    completed, timed_out = wait_for_agents([agent_id], timeout=1)

    assert completed == {}
    assert timed_out == [agent_id]


def test_wait_for_agents_partial(callosum_server):
    """Test wait_for_agents with some completing and some timing out."""
    tmp_path = callosum_server
    agents_dir = tmp_path / "agents"
    default_dir = agents_dir / "default"
    default_dir.mkdir(exist_ok=True)

    completing_agent = "1111"
    timeout_agent = "2222"

    # Create active file for timeout agent
    (default_dir / f"{timeout_agent}_active.jsonl").write_text('{"event": "start"}\n')

    result = {"completed": None, "timed_out": None}

    def wait_thread():
        result["completed"], result["timed_out"] = wait_for_agents(
            [completing_agent, timeout_agent], timeout=2
        )

    waiter = threading.Thread(target=wait_thread)
    waiter.start()
    time.sleep(0.2)

    # Complete one agent
    (default_dir / f"{completing_agent}.jsonl").write_text('{"event": "finish"}\n')

    client = CallosumConnection()
    client.start()
    time.sleep(0.1)
    client.emit("cortex", "finish", agent_id=completing_agent, result="done")
    time.sleep(0.1)
    client.stop()

    waiter.join(timeout=5)

    assert result["completed"] == {completing_agent: "finish"}
    assert result["timed_out"] == [timeout_agent]


def test_wait_for_agents_missed_event_recovery(tmp_path, monkeypatch, caplog):
    """Test that missed events are recovered via final file check with INFO log."""
    import logging

    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    default_dir = agents_dir / "default"
    default_dir.mkdir()
    (tmp_path / "health").mkdir()

    agent_id = "1234567890127"

    # Start with active file
    (default_dir / f"{agent_id}_active.jsonl").write_text('{"event": "start"}\n')

    result = {"completed": None, "timed_out": None}

    def wait_and_complete():
        # Wait a bit then "complete" the agent by renaming file
        time.sleep(0.3)
        (default_dir / f"{agent_id}_active.jsonl").unlink()
        (default_dir / f"{agent_id}.jsonl").write_text('{"event": "finish"}\n')

    completer = threading.Thread(target=wait_and_complete)
    completer.start()

    with caplog.at_level(logging.INFO):
        result["completed"], result["timed_out"] = wait_for_agents(
            [agent_id], timeout=1
        )

    completer.join()

    # Should recover via final file check
    assert result["completed"] == {agent_id: "finish"}
    assert result["timed_out"] == []

    # Should log about missed event
    assert any(
        "completion event not received but agent completed" in record.message
        for record in caplog.records
    )
