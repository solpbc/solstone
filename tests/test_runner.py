"""Tests for think.runner and logs tract integration."""

import os
import threading
import time
from pathlib import Path

import pytest

from think.callosum import CallosumConnection, CallosumServer
from think.runner import ManagedProcess, run_task


@pytest.fixture
def journal_path(tmp_path):
    """Set up a temporary journal path."""
    journal = tmp_path / "journal"
    journal.mkdir()
    os.environ["JOURNAL_PATH"] = str(journal)
    yield journal
    # Cleanup
    if "JOURNAL_PATH" in os.environ:
        del os.environ["JOURNAL_PATH"]


@pytest.fixture
def callosum_server(journal_path):
    """Start a Callosum server in a background thread."""
    server = CallosumServer()

    # Start server in background thread
    server_thread = threading.Thread(target=server.start, daemon=True)
    server_thread.start()

    # Wait for server to be ready
    socket_path = journal_path / "health" / "callosum.sock"
    for _ in range(50):  # 5 seconds max
        if socket_path.exists():
            break
        time.sleep(0.1)
    else:
        raise TimeoutError("Server did not start in time")

    yield server

    # Stop server
    server.stop()
    server_thread.join(timeout=2)


def test_managed_process_has_process_id_and_pid(journal_path):
    """Test that ManagedProcess exposes process_id and pid."""
    managed = ManagedProcess.spawn(
        ["echo", "test"],
        name="test-echo",
    )

    # Verify process_id and pid are accessible
    assert managed.process_id is not None
    assert isinstance(managed.process_id, str)
    assert managed.pid > 0
    assert isinstance(managed.pid, int)

    # Wait and cleanup
    managed.wait()
    managed.cleanup()


def test_managed_process_uses_task_id_as_process_id(journal_path):
    """Test that task_id becomes the process_id when provided."""
    task_id = "1730476800123"
    managed = ManagedProcess.spawn(
        ["echo", "test"],
        name="test-echo",
        task_id=task_id,
    )

    # Verify process_id matches task_id
    assert managed.process_id == task_id

    # Wait and cleanup
    managed.wait()
    managed.cleanup()


def test_logs_tract_exec_event(journal_path, callosum_server):
    """Test that exec event is emitted when process starts."""
    received = []
    listener = CallosumConnection(callback=lambda msg: received.append(msg))
    listener.connect()

    time.sleep(0.1)

    # Spawn process
    managed = ManagedProcess.spawn(
        ["echo", "hello"],
        name="test-exec",
    )

    time.sleep(0.2)

    # Find exec event
    exec_events = [msg for msg in received if msg.get("event") == "exec"]
    assert len(exec_events) >= 1

    exec_event = exec_events[0]
    assert exec_event["tract"] == "logs"
    assert exec_event["event"] == "exec"
    assert exec_event["process"] == managed.process_id
    assert exec_event["name"] == "test-exec"
    assert exec_event["pid"] == managed.pid
    assert exec_event["cmd"] == ["echo", "hello"]
    assert "log_path" in exec_event

    # Wait and cleanup
    managed.wait()
    managed.cleanup()
    listener.close()


def test_logs_tract_line_event(journal_path, callosum_server):
    """Test that line events are emitted for stdout/stderr."""
    received = []
    listener = CallosumConnection(callback=lambda msg: received.append(msg))
    listener.connect()

    time.sleep(0.1)

    # Spawn process that outputs text
    managed = ManagedProcess.spawn(
        ["echo", "hello logs tract"],
        name="test-line",
    )

    # Wait for process and events
    managed.wait()
    time.sleep(0.2)

    # Find line events
    line_events = [msg for msg in received if msg.get("event") == "line"]
    assert len(line_events) >= 1

    # Verify line event structure
    line_event = line_events[0]
    assert line_event["tract"] == "logs"
    assert line_event["event"] == "line"
    assert line_event["process"] == managed.process_id
    assert line_event["name"] == "test-line"
    assert line_event["pid"] == managed.pid
    assert line_event["stream"] in ["stdout", "stderr"]
    assert "line" in line_event
    assert "hello logs tract" in line_event["line"]

    # Cleanup
    managed.cleanup()
    listener.close()


def test_logs_tract_exit_event(journal_path, callosum_server):
    """Test that exit event is emitted when process completes."""
    received = []
    listener = CallosumConnection(callback=lambda msg: received.append(msg))
    listener.connect()

    time.sleep(0.1)

    # Spawn and wait for process
    managed = ManagedProcess.spawn(
        ["echo", "test"],
        name="test-exit",
    )
    managed.wait()
    managed.cleanup()

    time.sleep(0.2)

    # Find exit event
    exit_events = [msg for msg in received if msg.get("event") == "exit"]
    assert len(exit_events) >= 1

    exit_event = exit_events[0]
    assert exit_event["tract"] == "logs"
    assert exit_event["event"] == "exit"
    assert exit_event["process"] == managed.process_id
    assert exit_event["name"] == "test-exit"
    assert exit_event["pid"] == managed.pid
    assert exit_event["exit_code"] == 0
    assert "duration_ms" in exit_event
    assert exit_event["duration_ms"] >= 0
    assert exit_event["cmd"] == ["echo", "test"]
    assert "log_path" in exit_event

    listener.close()


def test_logs_tract_all_events_have_common_fields(journal_path, callosum_server):
    """Test that all logs tract events have process, name, and pid."""
    received = []
    listener = CallosumConnection(callback=lambda msg: received.append(msg))
    listener.connect()

    time.sleep(0.1)

    # Run a process
    managed = ManagedProcess.spawn(["echo", "test"], name="test-common")
    managed.wait()
    managed.cleanup()

    time.sleep(0.2)

    # Filter to only logs tract events
    logs_events = [msg for msg in received if msg.get("tract") == "logs"]
    assert len(logs_events) >= 3  # exec, line, exit

    # Verify common fields in all events
    for event in logs_events:
        assert "process" in event
        assert "name" in event
        assert "pid" in event
        assert "ts" in event  # Auto-added by Callosum
        assert event["process"] == managed.process_id
        assert event["name"] == "test-common"
        assert event["pid"] == managed.pid

    listener.close()


def test_run_task_emits_logs_tract_events(journal_path, callosum_server):
    """Test that run_task function emits logs tract events."""
    received = []
    listener = CallosumConnection(callback=lambda msg: received.append(msg))
    listener.connect()

    time.sleep(0.1)

    # Run task
    success, exit_code = run_task(
        ["echo", "run_task test"],
        name="test-run-task",
    )

    time.sleep(0.2)

    # Verify success
    assert success is True
    assert exit_code == 0

    # Verify events were emitted
    logs_events = [msg for msg in received if msg.get("tract") == "logs"]
    event_types = [msg["event"] for msg in logs_events]

    assert "exec" in event_types
    assert "line" in event_types
    assert "exit" in event_types

    listener.close()


def test_task_id_links_to_task_tract(journal_path, callosum_server):
    """Test that providing task_id links logs to task tract."""
    received = []
    listener = CallosumConnection(callback=lambda msg: received.append(msg))
    listener.connect()

    time.sleep(0.1)

    task_id = "1730476800999"
    managed = ManagedProcess.spawn(
        ["echo", "linked"],
        name="test-linked",
        task_id=task_id,
    )
    managed.wait()
    managed.cleanup()

    time.sleep(0.2)

    # Verify all logs events use task_id as process
    logs_events = [msg for msg in received if msg.get("tract") == "logs"]
    assert len(logs_events) >= 3

    for event in logs_events:
        assert event["process"] == task_id

    listener.close()


def test_error_exit_code_in_exit_event(journal_path, callosum_server):
    """Test that non-zero exit codes are captured in exit event."""
    received = []
    listener = CallosumConnection(callback=lambda msg: received.append(msg))
    listener.connect()

    time.sleep(0.1)

    # Run process that exits with error
    managed = ManagedProcess.spawn(
        ["sh", "-c", "exit 42"],
        name="test-error",
    )
    exit_code = managed.wait()
    managed.cleanup()

    time.sleep(0.2)

    # Verify exit code
    assert exit_code == 42

    # Find exit event
    exit_events = [msg for msg in received if msg.get("event") == "exit"]
    assert len(exit_events) >= 1

    exit_event = exit_events[0]
    assert exit_event["exit_code"] == 42

    listener.close()


def test_process_creates_health_log(journal_path):
    """Test that process output is logged to health directory."""
    managed = ManagedProcess.spawn(
        ["echo", "logged output"],
        name="test-log-file",
    )
    managed.wait()
    managed.cleanup()

    # Verify log file was created
    # Log should be in current day's health directory
    from datetime import datetime

    day = datetime.now().strftime("%Y%m%d")
    log_path = journal_path / day / "health" / "test-log-file.log"

    assert log_path.exists()
    content = log_path.read_text()
    assert "logged output" in content
