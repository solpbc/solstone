"""Tests for cortex_client module."""

import json
import os
import time
from pathlib import Path
from threading import Thread

import pytest

from think.cortex_client import cortex_agents, cortex_request, cortex_run, cortex_watch


def test_cortex_request(tmp_path, monkeypatch):
    """Test creating a Cortex agent request."""
    # Set up test journal path
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))

    # Create a request
    active_file = cortex_request(
        prompt="Test prompt",
        persona="default",
        backend="openai",
        config={"model": "gpt-4o"},
    )

    # Verify file was created
    assert Path(active_file).exists()
    assert active_file.endswith("_active.jsonl")

    # Verify content
    with open(active_file, "r") as f:
        data = json.loads(f.readline())
        assert data["event"] == "request"
        assert data["prompt"] == "Test prompt"
        assert data["persona"] == "default"
        assert data["backend"] == "openai"
        assert data["model"] == "gpt-4o"
        assert "config" not in data
        assert "ts" in data


def test_cortex_request_minimal(tmp_path, monkeypatch):
    """Test cortex_request with minimal parameters."""
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))

    # Create request with only required params
    active_file = cortex_request(
        prompt="Simple test", persona="tester", backend="google"
    )

    with open(active_file, "r") as f:
        data = json.loads(f.readline())
        assert data["prompt"] == "Simple test"
        assert data["persona"] == "tester"
        assert data["backend"] == "google"
        assert "config" not in data
        assert "handoff_from" not in data


def test_cortex_request_with_handoff(tmp_path, monkeypatch):
    """Test cortex_request with handoff_from parameter."""
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))

    active_file = cortex_request(
        prompt="Continue analysis",
        persona="reviewer",
        backend="anthropic",
        handoff_from="1234567890000",
    )

    with open(active_file, "r") as f:
        data = json.loads(f.readline())
        assert data["prompt"] == "Continue analysis"
        assert data["persona"] == "reviewer"
        assert data["backend"] == "anthropic"
        assert data["handoff_from"] == "1234567890000"


def test_cortex_request_file_naming(tmp_path, monkeypatch):
    """Test that cortex_request creates files with correct naming pattern."""
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))

    # Create multiple requests
    files = []
    for i in range(3):
        active_file = cortex_request(
            prompt=f"Test {i}", persona="default", backend="openai"
        )
        files.append(active_file)
        time.sleep(0.002)  # Small delay to ensure different timestamps

    # Verify all files exist and have unique names
    assert len(set(files)) == 3
    for file_path in files:
        assert Path(file_path).exists()
        assert file_path.endswith("_active.jsonl")
        # Extract timestamp from filename
        filename = Path(file_path).name
        ts_str = filename.replace("_active.jsonl", "")
        assert ts_str.isdigit()
        assert len(ts_str) == 13  # Millisecond timestamp


def test_cortex_request_creates_agents_dir(tmp_path, monkeypatch):
    """Test that cortex_request creates the agents directory if it doesn't exist."""
    # Use a fresh path without pre-created agents dir
    journal_path = tmp_path / "new_journal"
    monkeypatch.setenv("JOURNAL_PATH", str(journal_path))

    assert not journal_path.exists()

    active_file = cortex_request(prompt="Test", persona="default", backend="openai")

    # Verify directory was created
    assert journal_path.exists()
    assert (journal_path / "agents").exists()
    assert Path(active_file).exists()


def test_cortex_request_timestamp_matches_filename(tmp_path, monkeypatch):
    """Test that the timestamp in the request matches the filename."""
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))

    active_file = cortex_request(
        prompt="Test timing", persona="default", backend="openai"
    )

    # Extract timestamp from filename
    filename = Path(active_file).name
    file_ts = int(filename.replace("_active.jsonl", ""))

    # Read timestamp from request
    with open(active_file, "r") as f:
        data = json.loads(f.readline())
        request_ts = data["ts"]

    assert file_ts == request_ts


def test_cortex_request_atomic_rename(tmp_path, monkeypatch):
    """Test that cortex_request uses atomic rename from pending to active."""
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    agents_dir = tmp_path / "agents"

    # Monitor for pending files during execution
    pending_files_seen = []

    def check_pending():
        # Quick check for any pending files
        pending = list(agents_dir.glob("*_pending.jsonl"))
        if pending:
            pending_files_seen.extend(pending)

    # We can't easily catch the pending file in action since rename is atomic,
    # but we can verify the final state
    active_file = cortex_request(
        prompt="Test atomic", persona="default", backend="openai"
    )

    # Verify no pending files remain
    assert len(list(agents_dir.glob("*_pending.jsonl"))) == 0

    # Verify active file exists
    assert Path(active_file).exists()
    assert "_active.jsonl" in active_file


def test_cortex_request_no_journal_path():
    """Test cortex_request fails without JOURNAL_PATH."""
    # Temporarily unset JOURNAL_PATH if it exists
    old_path = os.environ.pop("JOURNAL_PATH", None)
    try:
        with pytest.raises(
            ValueError, match="JOURNAL_PATH environment variable not set"
        ):
            cortex_request("test", "default", "openai")
    finally:
        if old_path:
            os.environ["JOURNAL_PATH"] = old_path


def test_cortex_watch_new_file(tmp_path, monkeypatch):
    """Test cortex_watch detecting and reading new active files."""
    # Set up test journal path
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()

    # Track received events
    received_events = []

    def callback(event):
        received_events.append(event)
        # Stop after receiving finish event
        if event.get("event") == "finish":
            return False
        return True

    def simulate_agent():
        """Simulate agent writing events."""
        time.sleep(0.5)  # Let watcher start

        # Create new active file
        ts = int(time.time() * 1000)
        active_file = agents_dir / f"{ts}_active.jsonl"

        with open(active_file, "w") as f:
            # Write request
            json.dump({"event": "request", "ts": ts, "prompt": "Test"}, f)
            f.write("\n")

        time.sleep(0.2)

        with open(active_file, "a") as f:
            # Write start
            json.dump({"event": "start", "ts": ts + 1}, f)
            f.write("\n")

        time.sleep(0.2)

        with open(active_file, "a") as f:
            # Write finish
            json.dump({"event": "finish", "ts": ts + 2, "result": "Done"}, f)
            f.write("\n")

    # Start simulator in background
    simulator = Thread(target=simulate_agent)
    simulator.daemon = True
    simulator.start()

    # Watch for events (will block until callback returns False)
    cortex_watch(callback)

    # Verify events were received
    assert len(received_events) == 3
    assert received_events[0]["event"] == "request"
    assert received_events[1]["event"] == "start"
    assert received_events[2]["event"] == "finish"
    assert received_events[2]["result"] == "Done"


def test_cortex_watch_existing_file(tmp_path, monkeypatch):
    """Test cortex_watch handling existing active files."""
    # Set up test journal path
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()

    # Create an existing active file with some events
    ts = int(time.time() * 1000)
    active_file = agents_dir / f"{ts}_active.jsonl"

    with open(active_file, "w") as f:
        json.dump({"event": "request", "ts": ts, "prompt": "Test"}, f)
        f.write("\n")
        json.dump({"event": "start", "ts": ts + 1}, f)
        f.write("\n")

    # Track received events
    received_events = []

    def callback(event):
        received_events.append(event)
        # Stop after receiving finish event
        if event.get("event") == "finish":
            return False
        return True

    def append_more_events():
        """Append more events to existing file."""
        time.sleep(0.5)  # Let watcher start

        with open(active_file, "a") as f:
            # Write tool event
            json.dump({"event": "tool_start", "ts": ts + 2, "tool": "search"}, f)
            f.write("\n")

        time.sleep(0.2)

        with open(active_file, "a") as f:
            # Write finish
            json.dump({"event": "finish", "ts": ts + 3, "result": "Done"}, f)
            f.write("\n")

    # Start appender in background
    appender = Thread(target=append_more_events)
    appender.daemon = True
    appender.start()

    # Watch for events (will only see new events, not existing ones)
    cortex_watch(callback)

    # Should only receive the newly appended events
    assert len(received_events) == 2
    assert received_events[0]["event"] == "tool_start"
    assert received_events[1]["event"] == "finish"


def test_cortex_watch_callback_stops(tmp_path, monkeypatch):
    """Test that cortex_watch stops when callback returns False."""
    # Set up test journal path
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()

    # Track received events
    received_events = []

    def callback(event):
        received_events.append(event)
        # Stop immediately after first event
        return False

    def simulate_agent():
        """Simulate agent writing multiple events."""
        time.sleep(0.5)  # Let watcher start

        ts = int(time.time() * 1000)
        active_file = agents_dir / f"{ts}_active.jsonl"

        with open(active_file, "w") as f:
            json.dump({"event": "request", "ts": ts}, f)
            f.write("\n")

        time.sleep(0.2)

        with open(active_file, "a") as f:
            json.dump({"event": "start", "ts": ts + 1}, f)
            f.write("\n")
            json.dump({"event": "finish", "ts": ts + 2}, f)
            f.write("\n")

    # Start simulator in background
    simulator = Thread(target=simulate_agent)
    simulator.daemon = True
    simulator.start()

    # Watch for events
    cortex_watch(callback)

    # Should only receive first event before stopping
    assert len(received_events) == 1
    assert received_events[0]["event"] == "request"


def test_cortex_watch_handles_malformed_json(tmp_path, monkeypatch):
    """Test that cortex_watch handles malformed JSON gracefully."""
    # Set up test journal path
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()

    # Track received events
    received_events = []

    def callback(event):
        received_events.append(event)
        if event.get("event") == "finish":
            return False
        return True

    def simulate_agent():
        """Simulate agent with some malformed output."""
        time.sleep(0.5)  # Let watcher start

        ts = int(time.time() * 1000)
        active_file = agents_dir / f"{ts}_active.jsonl"

        with open(active_file, "w") as f:
            json.dump({"event": "request", "ts": ts}, f)
            f.write("\n")
            f.write("This is not valid JSON\n")  # Malformed line
            json.dump({"event": "finish", "ts": ts + 1}, f)
            f.write("\n")

    # Start simulator in background
    simulator = Thread(target=simulate_agent)
    simulator.daemon = True
    simulator.start()

    # Watch for events
    cortex_watch(callback)

    # Should skip malformed line and continue
    assert len(received_events) == 2
    assert received_events[0]["event"] == "request"
    assert received_events[1]["event"] == "finish"


def test_cortex_watch_multiple_active_files(tmp_path, monkeypatch):
    """Test cortex_watch handling multiple active files."""
    # Set up test journal path
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()

    # Track received events
    received_events = []
    finish_count = 0

    def callback(event):
        nonlocal finish_count
        received_events.append(event)
        if event.get("event") == "finish":
            finish_count += 1
            # Stop after both agents finish
            if finish_count >= 2:
                return False
        return True

    def simulate_multiple_agents():
        """Simulate multiple agents writing events."""
        time.sleep(0.5)  # Let watcher start

        # Create first active file
        ts1 = int(time.time() * 1000)
        active_file1 = agents_dir / f"{ts1}_active.jsonl"

        # Create second active file
        ts2 = ts1 + 100
        active_file2 = agents_dir / f"{ts2}_active.jsonl"

        # Write to both files
        with open(active_file1, "w") as f:
            json.dump({"event": "request", "ts": ts1, "agent": "first"}, f)
            f.write("\n")

        with open(active_file2, "w") as f:
            json.dump({"event": "request", "ts": ts2, "agent": "second"}, f)
            f.write("\n")

        time.sleep(0.2)

        # Both agents make progress
        with open(active_file1, "a") as f:
            json.dump({"event": "finish", "ts": ts1 + 1, "agent": "first"}, f)
            f.write("\n")

        with open(active_file2, "a") as f:
            json.dump({"event": "finish", "ts": ts2 + 1, "agent": "second"}, f)
            f.write("\n")

    # Start simulator in background
    simulator = Thread(target=simulate_multiple_agents)
    simulator.daemon = True
    simulator.start()

    # Watch for events
    cortex_watch(callback)

    # Should receive events from both agents
    assert len(received_events) == 4  # 2 requests + 2 finishes

    # Check we got events from both agents
    agents = {e.get("agent") for e in received_events if "agent" in e}
    assert agents == {"first", "second"}


def test_cortex_watch_no_journal_path():
    """Test cortex_watch fails without JOURNAL_PATH."""
    # Temporarily unset JOURNAL_PATH if it exists
    old_path = os.environ.pop("JOURNAL_PATH", None)
    try:
        with pytest.raises(
            ValueError, match="JOURNAL_PATH environment variable not set"
        ):
            cortex_watch(lambda e: None)
    finally:
        if old_path:
            os.environ["JOURNAL_PATH"] = old_path


def test_cortex_run_basic(tmp_path, monkeypatch):
    """Test run_agent basic functionality."""
    # Set up test journal path
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()

    # Track events received by callback
    received_events = []

    def event_callback(event):
        received_events.append(event)

    def simulate_agent():
        """Simulate agent execution after request is created."""
        time.sleep(0.5)  # Let run_agent create request and start watching

        # Find the active file that was created
        active_files = list(agents_dir.glob("*_active.jsonl"))
        assert len(active_files) == 1
        active_file = active_files[0]

        # Append events to simulate agent execution
        with open(active_file, "a") as f:
            # Write start event
            ts = int(time.time() * 1000)
            json.dump({"event": "start", "ts": ts}, f)
            f.write("\n")

            time.sleep(0.1)

            # Write tool event
            json.dump({"event": "tool_start", "ts": ts + 1, "tool": "search"}, f)
            f.write("\n")

            time.sleep(0.1)

            # Write finish event
            json.dump(
                {
                    "event": "finish",
                    "ts": ts + 2,
                    "result": "Task completed successfully",
                },
                f,
            )
            f.write("\n")

    # Start simulator in background
    simulator = Thread(target=simulate_agent)
    simulator.daemon = True
    simulator.start()

    # Run the agent
    result = cortex_run(
        prompt="Test task", persona="default", backend="openai", on_event=event_callback
    )

    # Verify result
    assert result == "Task completed successfully"

    # Verify events were received
    assert len(received_events) >= 2  # At least start and finish
    event_types = [e["event"] for e in received_events]
    assert "start" in event_types
    assert "finish" in event_types


def test_cortex_run_with_error(tmp_path, monkeypatch):
    """Test run_agent handling error events."""
    # Set up test journal path
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()

    def simulate_agent_error():
        """Simulate agent that encounters an error."""
        time.sleep(0.5)  # Let run_agent create request and start watching

        # Find the active file that was created
        active_files = list(agents_dir.glob("*_active.jsonl"))
        assert len(active_files) == 1
        active_file = active_files[0]

        # Write error event
        with open(active_file, "a") as f:
            ts = int(time.time() * 1000)
            json.dump({"event": "error", "ts": ts, "error": "Something went wrong"}, f)
            f.write("\n")

    # Start simulator in background
    simulator = Thread(target=simulate_agent_error)
    simulator.daemon = True
    simulator.start()

    # Run the agent and expect error
    with pytest.raises(RuntimeError, match="Agent error: Something went wrong"):
        cortex_run(prompt="Test task", persona="default", backend="openai")


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

    # Check agents are sorted by newest first
    assert result["agents"][0]["id"] == str(ts2)
    assert result["agents"][0]["persona"] == "tester"
    assert result["agents"][0]["status"] == "running"

    assert result["agents"][1]["id"] == str(ts1)
    assert result["agents"][1]["persona"] == "default"
    assert result["agents"][1]["status"] == "running"


def test_cortex_agents_with_completed(tmp_path, monkeypatch):
    """Test cortex_agents with completed (historical) agents."""
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()

    # Create completed agent files (no _active suffix)
    ts1 = int(time.time() * 1000)
    ts2 = ts1 + 1000

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

    completed_file2 = agents_dir / f"{ts2}.jsonl"
    with open(completed_file2, "w") as f:
        json.dump(
            {
                "event": "request",
                "ts": ts2,
                "prompt": "Another old task",
                "persona": "default",
                "backend": "openai",
            },
            f,
        )
        f.write("\n")

    result = cortex_agents()

    assert len(result["agents"]) == 2
    assert result["live_count"] == 0
    assert result["historical_count"] == 2

    # All should be completed
    for agent in result["agents"]:
        assert agent["status"] == "completed"


def test_cortex_agents_mixed(tmp_path, monkeypatch):
    """Test cortex_agents with both active and completed agents."""
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()

    # Create mix of active and completed
    ts1 = int(time.time() * 1000)
    ts2 = ts1 + 1000
    ts3 = ts1 + 2000

    # Active agent
    active_file = agents_dir / f"{ts1}_active.jsonl"
    with open(active_file, "w") as f:
        json.dump(
            {
                "event": "request",
                "ts": ts1,
                "prompt": "Running",
                "persona": "default",
                "backend": "openai",
            },
            f,
        )
        f.write("\n")

    # Completed agents
    completed_file1 = agents_dir / f"{ts2}.jsonl"
    with open(completed_file1, "w") as f:
        json.dump(
            {
                "event": "request",
                "ts": ts2,
                "prompt": "Done 1",
                "persona": "reviewer",
                "backend": "anthropic",
            },
            f,
        )
        f.write("\n")

    completed_file2 = agents_dir / f"{ts3}.jsonl"
    with open(completed_file2, "w") as f:
        json.dump(
            {
                "event": "request",
                "ts": ts3,
                "prompt": "Done 2",
                "persona": "tester",
                "backend": "google",
            },
            f,
        )
        f.write("\n")

    result = cortex_agents()

    assert len(result["agents"]) == 3
    assert result["live_count"] == 1
    assert result["historical_count"] == 2


def test_cortex_agents_type_filter(tmp_path, monkeypatch):
    """Test cortex_agents with agent_type filtering."""
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()

    # Create mix of active and completed
    ts1 = int(time.time() * 1000)
    ts2 = ts1 + 1000

    active_file = agents_dir / f"{ts1}_active.jsonl"
    with open(active_file, "w") as f:
        json.dump(
            {"event": "request", "ts": ts1, "prompt": "Running", "persona": "default"},
            f,
        )
        f.write("\n")

    completed_file = agents_dir / f"{ts2}.jsonl"
    with open(completed_file, "w") as f:
        json.dump(
            {"event": "request", "ts": ts2, "prompt": "Done", "persona": "reviewer"}, f
        )
        f.write("\n")

    # Test live filter
    result = cortex_agents(agent_type="live")
    assert len(result["agents"]) == 1
    assert result["agents"][0]["status"] == "running"

    # Test historical filter
    result = cortex_agents(agent_type="historical")
    assert len(result["agents"]) == 1
    assert result["agents"][0]["status"] == "completed"

    # Test all (default)
    result = cortex_agents(agent_type="all")
    assert len(result["agents"]) == 2


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

    # Test offset
    result = cortex_agents(limit=2, offset=2)
    assert len(result["agents"]) == 2
    assert result["pagination"]["offset"] == 2
    assert result["pagination"]["has_more"] is True

    # Test last page
    result = cortex_agents(limit=2, offset=4)
    assert len(result["agents"]) == 1
    assert result["pagination"]["has_more"] is False


def test_cortex_agents_limit_validation(tmp_path, monkeypatch):
    """Test cortex_agents limit parameter validation."""
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))

    # Test max limit
    result = cortex_agents(limit=200)
    assert result["pagination"]["limit"] == 100  # Capped at 100

    # Test min limit
    result = cortex_agents(limit=0)
    assert result["pagination"]["limit"] == 1  # Minimum is 1

    # Test negative offset becomes 0
    result = cortex_agents(offset=-10)
    assert result["pagination"]["offset"] == 0


def test_cortex_agents_skip_pending(tmp_path, monkeypatch):
    """Test that cortex_agents skips pending files."""
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()

    ts = int(time.time() * 1000)

    # Create pending file (should be ignored)
    pending_file = agents_dir / f"{ts}_pending.jsonl"
    with open(pending_file, "w") as f:
        json.dump(
            {"event": "request", "ts": ts, "prompt": "Pending", "persona": "default"}, f
        )
        f.write("\n")

    # Create active file
    active_file = agents_dir / f"{ts + 1000}_active.jsonl"
    with open(active_file, "w") as f:
        json.dump(
            {
                "event": "request",
                "ts": ts + 1000,
                "prompt": "Active",
                "persona": "default",
            },
            f,
        )
        f.write("\n")

    result = cortex_agents()

    # Should only see the active file, not pending
    assert len(result["agents"]) == 1
    assert result["agents"][0]["id"] == str(ts + 1000)


def test_cortex_agents_skip_malformed(tmp_path, monkeypatch):
    """Test that cortex_agents skips files with malformed JSON."""
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()

    ts = int(time.time() * 1000)

    # Create file with malformed JSON
    bad_file = agents_dir / f"{ts}.jsonl"
    with open(bad_file, "w") as f:
        f.write("This is not JSON\n")

    # Create valid file
    good_file = agents_dir / f"{ts + 1000}.jsonl"
    with open(good_file, "w") as f:
        json.dump(
            {
                "event": "request",
                "ts": ts + 1000,
                "prompt": "Valid",
                "persona": "default",
            },
            f,
        )
        f.write("\n")

    result = cortex_agents()

    # Should only see the valid file
    assert len(result["agents"]) == 1
    assert result["agents"][0]["id"] == str(ts + 1000)


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
