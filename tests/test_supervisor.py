# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

import importlib
import io
import os
import subprocess
import sys
from unittest.mock import MagicMock

import psutil
import pytest


@pytest.mark.asyncio
async def test_send_notification(monkeypatch):
    mod = importlib.import_module("think.supervisor")
    called = []

    class FakeNotifier:
        async def send(self, title, message, urgency):
            called.append({"title": title, "message": message, "urgency": urgency})
            return "test-notification-id"

    def fake_get_notifier():
        return FakeNotifier()

    monkeypatch.setattr(mod, "_get_notifier", fake_get_notifier)
    await mod.send_notification("test message", alert_key=("test", "key"))
    assert len(called) == 1
    assert called[0]["message"] == "test message"
    assert called[0]["title"] == "solstone Supervisor"
    assert ("test", "key") in mod._notification_ids
    assert mod._notification_ids[("test", "key")] == "test-notification-id"


@pytest.mark.asyncio
async def test_clear_notification(monkeypatch):
    mod = importlib.import_module("think.supervisor")
    cleared = []

    class FakeNotifier:
        async def send(self, title, message, urgency):
            return "test-notification-id"

        async def clear(self, notification_id):
            cleared.append(notification_id)

    def fake_get_notifier():
        return FakeNotifier()

    monkeypatch.setattr(mod, "_get_notifier", fake_get_notifier)

    # First send a notification to track
    await mod.send_notification("test message", alert_key=("test", "key"))
    assert ("test", "key") in mod._notification_ids

    # Now clear it
    await mod.clear_notification(("test", "key"))
    assert len(cleared) == 1
    assert cleared[0] == "test-notification-id"
    assert ("test", "key") not in mod._notification_ids

    # Clearing a non-existent notification should be a no-op
    await mod.clear_notification(("nonexistent", "key"))
    assert len(cleared) == 1  # Still just one clear call


def test_start_sense(tmp_path, mock_callosum, monkeypatch):
    """Test that sense launches correctly."""
    mod = importlib.import_module("think.supervisor")

    started = []

    class DummyProc:
        def __init__(self):
            self.stdout = io.StringIO()
            self.stderr = io.StringIO()
            self.pid = 12345

        def terminate(self):
            pass

        def wait(self, timeout=None):
            pass

    def fake_popen(
        cmd,
        stdout=None,
        stderr=None,
        text=False,
        bufsize=-1,
        start_new_session=False,
        env=None,
    ):
        proc = DummyProc()
        started.append((cmd, stdout, stderr))
        return proc

    monkeypatch.setattr(mod.subprocess, "Popen", fake_popen)
    monkeypatch.setenv("SOLSTONE_JOURNAL", str(tmp_path))

    # Test start_sense()
    sense_proc = mod.start_sense()
    assert sense_proc is not None
    assert any(cmd == ["sol", "sense", "-v"] for cmd, _, _ in started)

    # Check that stdout and stderr capture pipes
    for cmd, stdout, stderr in started:
        assert stdout == subprocess.PIPE
        assert stderr == subprocess.PIPE


def test_parse_args_remote_flag():
    """Test that parse_args includes --remote flag."""
    mod = importlib.reload(importlib.import_module("think.supervisor"))

    parser = mod.parse_args()
    args = parser.parse_args(["--remote", "https://server/ingest/key"])

    assert args.remote == "https://server/ingest/key"


def test_parse_args_remote_flag_optional():
    """Test that --remote is optional."""
    mod = importlib.reload(importlib.import_module("think.supervisor"))

    parser = mod.parse_args()
    args = parser.parse_args([])

    assert args.remote is None


def test_shutdown_stops_in_reverse_order(monkeypatch):
    """Shutdown stops services in reverse order."""
    operations = []

    class MockProc:
        def __init__(self, name):
            self._name = name

        def terminate(self):
            operations.append(("terminate", self._name))

        def wait(self, timeout=None):
            operations.append(("wait", self._name))

        def kill(self):
            pass

        def poll(self):
            return None

    class MockManaged:
        def __init__(self, name):
            self.name = name
            self.process = MockProc(name)
            self.shutdown_timeout = 15

        def cleanup(self):
            operations.append(("cleanup", self.name))

    procs = [
        MockManaged("convey"),
        MockManaged("sense"),
        MockManaged("cortex"),
    ]

    for managed in reversed(procs):
        proc = managed.process
        try:
            proc.terminate()
        except Exception:
            pass
        try:
            proc.wait(timeout=managed.shutdown_timeout)
        except Exception:
            pass
        managed.cleanup()

    assert operations == [
        ("terminate", "cortex"),
        ("wait", "cortex"),
        ("cleanup", "cortex"),
        ("terminate", "sense"),
        ("wait", "sense"),
        ("cleanup", "sense"),
        ("terminate", "convey"),
        ("wait", "convey"),
        ("cleanup", "convey"),
    ]


def test_get_command_name():
    """Test command name extraction for queue serialization."""
    mod = importlib.import_module("think.supervisor")
    get = mod.TaskQueue.get_command_name

    # sol X -> X
    assert get(["sol", "indexer", "--rescan"]) == "indexer"
    assert get(["sol", "insight", "20240101"]) == "insight"
    assert get(["sol", "think", "--day", "20240101"]) == "think"

    # Other commands -> basename
    assert get(["/usr/bin/python", "script.py"]) == "python"
    assert get(["custom-tool"]) == "custom-tool"

    # Empty -> unknown
    assert get([]) == "unknown"


def test_task_queue_same_command_queued(monkeypatch):
    """Test that same command is queued when already running."""
    mod = importlib.import_module("think.supervisor")

    # Create fresh task queue (no callback to avoid callosum events)
    mod._task_queue = mod.TaskQueue(on_queue_change=None)

    spawned = []

    def fake_thread_start(self):
        spawned.append(self._target.__name__)

    monkeypatch.setattr(mod.threading.Thread, "start", fake_thread_start)

    # First request - should run immediately
    msg1 = {
        "tract": "supervisor",
        "event": "request",
        "cmd": ["sol", "indexer", "--rescan"],
    }
    mod._handle_task_request(msg1)

    assert "indexer" in mod._task_queue._running
    assert len(spawned) == 1

    # Second request (different args) - should be queued
    msg2 = {
        "tract": "supervisor",
        "event": "request",
        "cmd": ["sol", "indexer", "--rescan-full"],
    }
    mod._handle_task_request(msg2)

    assert len(spawned) == 1  # No new spawn
    assert "indexer" in mod._task_queue._queues
    assert len(mod._task_queue._queues["indexer"]) == 1
    # Queue entries are {refs, cmd} dicts (refs is a list for coalescing)
    assert mod._task_queue._queues["indexer"][0]["cmd"] == [
        "sol",
        "indexer",
        "--rescan-full",
    ]
    assert len(mod._task_queue._queues["indexer"][0]["refs"]) == 1


def test_task_queue_dedupe_exact_match(monkeypatch):
    """Test that exact same command is deduped in queue."""
    mod = importlib.import_module("think.supervisor")

    # Create fresh task queue (no callback to avoid callosum events)
    mod._task_queue = mod.TaskQueue(on_queue_change=None)

    spawned = []

    def fake_thread_start(self):
        spawned.append(self._target.__name__)

    monkeypatch.setattr(mod.threading.Thread, "start", fake_thread_start)

    # First request - runs
    msg1 = {
        "tract": "supervisor",
        "event": "request",
        "cmd": ["sol", "indexer", "--rescan"],
    }
    mod._handle_task_request(msg1)

    # Second request (same cmd) - queued
    msg2 = {
        "tract": "supervisor",
        "event": "request",
        "cmd": ["sol", "indexer", "--rescan"],
    }
    mod._handle_task_request(msg2)

    assert len(mod._task_queue._queues["indexer"]) == 1

    # Third request (same cmd again) - deduped, not added
    msg3 = {
        "tract": "supervisor",
        "event": "request",
        "cmd": ["sol", "indexer", "--rescan"],
    }
    mod._handle_task_request(msg3)

    assert len(mod._task_queue._queues["indexer"]) == 1  # Still just 1


def test_task_queue_different_commands_independent(monkeypatch):
    """Test that different commands have independent queues."""
    mod = importlib.import_module("think.supervisor")

    # Create fresh task queue (no callback to avoid callosum events)
    mod._task_queue = mod.TaskQueue(on_queue_change=None)

    spawned = []

    def fake_thread_start(self):
        spawned.append(self._target.__name__)

    monkeypatch.setattr(mod.threading.Thread, "start", fake_thread_start)

    # Indexer request - runs
    msg1 = {
        "tract": "supervisor",
        "event": "request",
        "cmd": ["sol", "indexer", "--rescan"],
    }
    mod._handle_task_request(msg1)

    # Insight request - also runs (different command)
    msg2 = {
        "tract": "supervisor",
        "event": "request",
        "cmd": ["sol", "insight", "20240101"],
    }
    mod._handle_task_request(msg2)

    assert len(spawned) == 2  # Both spawned
    assert "indexer" in mod._task_queue._running
    assert "insight" in mod._task_queue._running


def test_process_queue_spawns_next(monkeypatch):
    """Test that _process_next spawns next queued task."""
    mod = importlib.import_module("think.supervisor")

    # Create task queue with pre-set state
    mod._task_queue = mod.TaskQueue(on_queue_change=None)
    mod._task_queue._running = {"indexer": {"ref": "ref123", "thread": None}}
    mod._task_queue._queues = {
        "indexer": [
            {"refs": ["queued-ref"], "cmd": ["sol", "indexer", "--rescan-full"]}
        ]
    }

    spawned = []

    def fake_thread_start(self):
        spawned.append(self._args)  # Capture args (refs, cmd, cmd_name, callosum)

    monkeypatch.setattr(mod.threading.Thread, "start", fake_thread_start)

    # Process queue
    mod._task_queue._process_next("indexer")

    # Should have spawned the queued task with its refs list
    assert len(spawned) == 1
    assert spawned[0][0] == ["queued-ref"]  # refs list preserved from queue
    assert spawned[0][1] == ["sol", "indexer", "--rescan-full"]  # cmd
    assert spawned[0][2] == "indexer"  # cmd_name

    # Queue should be empty now
    assert mod._task_queue._queues["indexer"] == []


def test_process_queue_clears_running_when_empty(monkeypatch):
    """Test that _process_next clears running state when queue is empty."""
    mod = importlib.import_module("think.supervisor")

    # Create task queue with pre-set state (no queued tasks)
    mod._task_queue = mod.TaskQueue(on_queue_change=None)
    mod._task_queue._running = {"indexer": {"ref": "ref123", "thread": None}}
    mod._task_queue._queues = {"indexer": []}

    spawned = []

    def fake_thread_start(self):
        spawned.append(True)

    monkeypatch.setattr(mod.threading.Thread, "start", fake_thread_start)

    # Process queue
    mod._task_queue._process_next("indexer")

    # No spawn (queue was empty)
    assert len(spawned) == 0

    # Running state should be cleared
    assert "indexer" not in mod._task_queue._running


def test_task_request_uses_caller_provided_ref(monkeypatch):
    """Test that caller-provided ref is used and preserved through queue."""
    mod = importlib.import_module("think.supervisor")

    # Create fresh task queue (no callback to avoid callosum events)
    mod._task_queue = mod.TaskQueue(on_queue_change=None)

    spawned = []

    def fake_thread_start(self):
        spawned.append(self._args)  # Capture args (refs, cmd, cmd_name, callosum)

    monkeypatch.setattr(mod.threading.Thread, "start", fake_thread_start)

    # Request with caller-provided ref
    msg = {
        "tract": "supervisor",
        "event": "request",
        "cmd": ["sol", "indexer", "--rescan"],
        "ref": "my-custom-ref-123",
    }
    mod._handle_task_request(msg)

    # Should use the provided ref
    assert mod._task_queue._running["indexer"]["ref"] == "my-custom-ref-123"
    assert spawned[0][0] == ["my-custom-ref-123"]  # refs is a list


def test_task_queue_preserves_caller_ref(monkeypatch):
    """Test that queued requests preserve their caller-provided ref."""
    mod = importlib.import_module("think.supervisor")

    # Create fresh task queue (no callback to avoid callosum events)
    mod._task_queue = mod.TaskQueue(on_queue_change=None)

    spawned = []

    def fake_thread_start(self):
        spawned.append(self._args)

    monkeypatch.setattr(mod.threading.Thread, "start", fake_thread_start)

    # First request runs immediately
    msg1 = {
        "tract": "supervisor",
        "event": "request",
        "cmd": ["sol", "indexer", "--rescan"],
        "ref": "first-ref",
    }
    mod._handle_task_request(msg1)

    # Second request gets queued with its own ref
    msg2 = {
        "tract": "supervisor",
        "event": "request",
        "cmd": ["sol", "indexer", "--rescan-full"],
        "ref": "second-ref",
    }
    mod._handle_task_request(msg2)

    # Verify queued entry has the caller's ref in refs list
    assert len(mod._task_queue._queues["indexer"]) == 1
    assert mod._task_queue._queues["indexer"][0]["refs"] == ["second-ref"]
    assert mod._task_queue._queues["indexer"][0]["cmd"] == [
        "sol",
        "indexer",
        "--rescan-full",
    ]


def test_task_queue_coalesces_refs_on_dedupe(monkeypatch):
    """Test that duplicate queued requests coalesce their refs."""
    mod = importlib.import_module("think.supervisor")

    # Create fresh task queue (no callback to avoid callosum events)
    mod._task_queue = mod.TaskQueue(on_queue_change=None)

    spawned = []

    def fake_thread_start(self):
        spawned.append(self._args)

    monkeypatch.setattr(mod.threading.Thread, "start", fake_thread_start)

    # First request runs immediately
    msg1 = {
        "tract": "supervisor",
        "event": "request",
        "cmd": ["sol", "indexer", "--rescan"],
        "ref": "first-ref",
    }
    mod._handle_task_request(msg1)

    # Second request (same cmd) gets queued
    msg2 = {
        "tract": "supervisor",
        "event": "request",
        "cmd": ["sol", "indexer", "--rescan"],
        "ref": "second-ref",
    }
    mod._handle_task_request(msg2)

    # Third request (same cmd) should coalesce its ref into existing queue entry
    msg3 = {
        "tract": "supervisor",
        "event": "request",
        "cmd": ["sol", "indexer", "--rescan"],
        "ref": "third-ref",
    }
    mod._handle_task_request(msg3)

    # Should still be just one queue entry
    assert len(mod._task_queue._queues["indexer"]) == 1
    # But it should have both refs
    assert mod._task_queue._queues["indexer"][0]["refs"] == [
        "second-ref",
        "third-ref",
    ]


def test_process_queue_spawns_with_multiple_refs(monkeypatch):
    """Test that dequeued task has all coalesced refs."""
    mod = importlib.import_module("think.supervisor")

    # Create task queue with pre-set state (queued task with multiple refs)
    mod._task_queue = mod.TaskQueue(on_queue_change=None)
    mod._task_queue._running = {"indexer": {"ref": "running-ref", "thread": None}}
    mod._task_queue._queues = {
        "indexer": [
            {
                "refs": ["ref-A", "ref-B", "ref-C"],
                "cmd": ["sol", "indexer", "--rescan"],
            }
        ]
    }

    spawned = []

    def fake_thread_start(self):
        spawned.append(self._args)

    monkeypatch.setattr(mod.threading.Thread, "start", fake_thread_start)

    # Process queue
    mod._task_queue._process_next("indexer")

    # Should spawn with all three refs
    assert len(spawned) == 1
    assert spawned[0][0] == ["ref-A", "ref-B", "ref-C"]  # all refs passed
    assert spawned[0][1] == ["sol", "indexer", "--rescan"]


def test_stale_queue_detected_on_submit(monkeypatch):
    """Test that a dead task thread is detected and cleared on next submit."""
    import threading

    mod = importlib.import_module("think.supervisor")

    mod._task_queue = mod.TaskQueue(on_queue_change=None)

    # Create a dead thread BEFORE monkeypatching Thread.start
    dead_thread = threading.Thread(target=lambda: None)
    dead_thread.start()
    dead_thread.join()
    assert not dead_thread.is_alive()

    spawned = []

    def fake_thread_start(self):
        spawned.append(self._target.__name__)

    monkeypatch.setattr(mod.threading.Thread, "start", fake_thread_start)

    mod._task_queue._running = {"indexer": {"ref": "stale-ref", "thread": dead_thread}}
    mod._task_queue._queues = {
        "indexer": [
            {"refs": ["queued-ref"], "cmd": ["sol", "indexer", "--rescan-full"]}
        ]
    }

    # Submit a new indexer task — should detect stale state and start immediately
    msg = {
        "tract": "supervisor",
        "event": "request",
        "cmd": ["sol", "indexer", "--rescan-new"],
        "ref": "new-ref",
    }
    mod._handle_task_request(msg)

    # Stale entry should have been cleared, new task started
    assert mod._task_queue._running["indexer"]["ref"] == "new-ref"
    assert len(spawned) == 1

    # Old queued entries should still be in queue (stale clear only removes _running)
    assert len(mod._task_queue._queues["indexer"]) == 1


def test_supervisor_singleton_lock_acquired(tmp_path, monkeypatch):
    mod = importlib.reload(importlib.import_module("think.supervisor"))

    monkeypatch.setenv("SOLSTONE_JOURNAL", str(tmp_path))
    (tmp_path / "health").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(sys, "argv", ["supervisor"])

    def stop_after_lock():
        raise SystemExit(0)

    # Skip maint discovery/subprocess runs — unrelated to lock acquisition and
    # slow enough on a fresh tmp_path to blow the 5s pytest-timeout under load.
    monkeypatch.setattr(mod, "run_pending_tasks", lambda *a, **k: (0, 0))
    monkeypatch.setattr(mod, "start_callosum_in_process", stop_after_lock)

    with pytest.raises(SystemExit) as exc:
        mod.main()

    assert exc.value.code == 0
    assert (tmp_path / "health" / "supervisor.lock").exists()
    assert (tmp_path / "health" / "supervisor.pid").read_text().strip() == str(
        os.getpid()
    )
    start_time = float(
        (tmp_path / "health" / "supervisor.start_time").read_text().strip()
    )
    assert start_time > 0


def test_supervisor_singleton_lock_blocked(tmp_path, monkeypatch, capsys):
    import fcntl

    mod = importlib.reload(importlib.import_module("think.supervisor"))

    monkeypatch.setenv("SOLSTONE_JOURNAL", str(tmp_path))
    health_dir = tmp_path / "health"
    health_dir.mkdir(parents=True, exist_ok=True)
    lock_file = open(health_dir / "supervisor.lock", "w")
    fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (health_dir / "supervisor.pid").write_text("12345")
    monkeypatch.setattr(sys, "argv", ["supervisor"])

    start_mock = MagicMock()
    monkeypatch.setattr(mod, "start_callosum_in_process", start_mock)

    try:
        with pytest.raises(SystemExit) as exc:
            mod.main()
    finally:
        lock_file.close()

    assert exc.value.code == 1
    output = capsys.readouterr().out
    assert "Supervisor already running" in output
    assert "PID 12345" in output
    start_mock.assert_not_called()


def test_supervisor_singleton_lock_blocked_with_health(tmp_path, monkeypatch, capsys):
    import fcntl

    mod = importlib.reload(importlib.import_module("think.supervisor"))

    monkeypatch.setenv("SOLSTONE_JOURNAL", str(tmp_path))
    health_dir = tmp_path / "health"
    health_dir.mkdir(parents=True, exist_ok=True)
    lock_file = open(health_dir / "supervisor.lock", "w")
    fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (health_dir / "supervisor.pid").write_text("12345")
    (health_dir / "callosum.sock").touch()
    monkeypatch.setattr(sys, "argv", ["supervisor"])

    start_mock = MagicMock()
    health_mock = MagicMock(return_value=0)
    monkeypatch.setattr(mod, "start_callosum_in_process", start_mock)
    monkeypatch.setattr("think.health_cli.health_check", health_mock)

    try:
        with pytest.raises(SystemExit) as exc:
            mod.main()
    finally:
        lock_file.close()

    assert exc.value.code == 1
    output = capsys.readouterr().out
    assert "Supervisor already running" in output
    assert "PID 12345" in output
    health_mock.assert_called_once_with()
    start_mock.assert_not_called()


def test_is_supervisor_up_without_pid_file(tmp_path, monkeypatch):
    mod = importlib.reload(importlib.import_module("think.supervisor"))

    monkeypatch.setenv("SOLSTONE_JOURNAL", str(tmp_path))
    (tmp_path / "health").mkdir(parents=True, exist_ok=True)

    assert mod.is_supervisor_up() is False


def test_is_supervisor_up_with_dead_pid(tmp_path, monkeypatch):
    mod = importlib.reload(importlib.import_module("think.supervisor"))

    monkeypatch.setenv("SOLSTONE_JOURNAL", str(tmp_path))
    health_dir = tmp_path / "health"
    health_dir.mkdir(parents=True, exist_ok=True)

    proc = subprocess.Popen(["true"])
    proc.wait()
    (health_dir / "supervisor.pid").write_text(str(proc.pid))

    assert mod.is_supervisor_up() is False


def test_is_supervisor_up_with_live_pid_missing_start_time(tmp_path, monkeypatch):
    mod = importlib.reload(importlib.import_module("think.supervisor"))

    monkeypatch.setenv("SOLSTONE_JOURNAL", str(tmp_path))
    health_dir = tmp_path / "health"
    health_dir.mkdir(parents=True, exist_ok=True)
    (health_dir / "supervisor.pid").write_text(str(os.getpid()))

    assert mod.is_supervisor_up() is False


def test_is_supervisor_up_with_live_pid_mismatched_start_time(tmp_path, monkeypatch):
    mod = importlib.reload(importlib.import_module("think.supervisor"))

    monkeypatch.setenv("SOLSTONE_JOURNAL", str(tmp_path))
    health_dir = tmp_path / "health"
    health_dir.mkdir(parents=True, exist_ok=True)
    (health_dir / "supervisor.pid").write_text(str(os.getpid()))
    create_time = psutil.Process(os.getpid()).create_time()
    (health_dir / "supervisor.start_time").write_text(str(create_time + 60))

    assert mod.is_supervisor_up() is False


def test_is_supervisor_up_with_matching_process_identity(tmp_path, monkeypatch):
    mod = importlib.reload(importlib.import_module("think.supervisor"))

    monkeypatch.setenv("SOLSTONE_JOURNAL", str(tmp_path))
    health_dir = tmp_path / "health"
    health_dir.mkdir(parents=True, exist_ok=True)
    (health_dir / "supervisor.pid").write_text(str(os.getpid()))
    (health_dir / "supervisor.start_time").write_text(
        str(psutil.Process(os.getpid()).create_time())
    )

    assert mod.is_supervisor_up() is True
