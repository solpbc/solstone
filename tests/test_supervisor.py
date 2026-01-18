# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

import importlib
import io
import logging
import os
import subprocess
import time

import pytest


def test_check_health():
    """Test health checking based on observe.status event freshness.

    Health model is simple: if observer is running, it sends status events.
    If it has problems, it exits and supervisor restarts it (fail-fast).
    """
    mod = importlib.import_module("think.supervisor")

    # Ensure observer is enabled for this test
    mod._observer_enabled = True

    # Reset state for clean test
    mod._observe_status_state["last_ts"] = 0.0
    mod._observe_status_state["ever_received"] = False

    # Startup grace period: no status ever received - returns healthy (no alerts)
    stale = mod.check_health(threshold=60)
    assert stale == []  # Grace period - don't alert until first status received

    # After first status received, stale timestamp triggers alerts
    mod._observe_status_state["ever_received"] = True
    mod._observe_status_state["last_ts"] = 0.0  # Very old timestamp
    stale = mod.check_health(threshold=60)
    assert sorted(stale) == ["hear", "see"]

    # Fresh status event means healthy (observer is running)
    mod._observe_status_state["last_ts"] = time.time()
    stale = mod.check_health(threshold=60)
    assert stale == []  # Healthy - receiving status events

    # Status became stale (old timestamp) - observer stopped sending
    mod._observe_status_state["last_ts"] = time.time() - 100
    stale = mod.check_health(threshold=60)
    assert sorted(stale) == ["hear", "see"]


def test_check_health_observer_disabled(monkeypatch):
    """Test that health checks are skipped when observer is disabled (--no-observers)."""
    mod = importlib.import_module("think.supervisor")

    # Simulate --no-observers mode (monkeypatch auto-restores after test)
    monkeypatch.setattr(mod, "_observer_enabled", False)

    # Even with stale status, should return empty (no health alerts)
    mod._observe_status_state["ever_received"] = True
    mod._observe_status_state["last_ts"] = 0.0  # Very stale
    stale = mod.check_health(threshold=60)
    assert stale == []  # No alerts when observer disabled


def test_handle_observe_status():
    """Test that observe.status events update health state.

    Handler just tracks event freshness - observer is responsible for
    exiting if it's unhealthy (fail-fast model).
    """
    mod = importlib.import_module("think.supervisor")

    # Reset state
    mod._observe_status_state["last_ts"] = 0.0
    mod._observe_status_state["ever_received"] = False

    # Simulate observe.status message (only tract/event are required)
    message = {"tract": "observe", "event": "status"}
    mod._handle_observe_status(message)

    assert mod._observe_status_state["last_ts"] > 0
    assert mod._observe_status_state["ever_received"] is True  # Grace period ended

    # Non-observe messages should be ignored
    old_ts = mod._observe_status_state["last_ts"]
    mod._handle_observe_status({"tract": "supervisor", "event": "status"})
    assert mod._observe_status_state["last_ts"] == old_ts


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


def test_start_observer_and_sense(tmp_path, mock_callosum, monkeypatch):
    """Test that start_observer() and start_sense() launch their respective processes."""
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
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))

    # Test start_observer()
    observer_proc = mod.start_observer()
    assert observer_proc is not None
    assert any(cmd == ["sol", "observer", "-v"] for cmd, _, _ in started)

    # Test start_sense()
    sense_proc = mod.start_sense()
    assert sense_proc is not None
    assert any(cmd == ["sol", "sense", "-v"] for cmd, _, _ in started)

    # Check that stdout and stderr capture pipes
    for cmd, stdout, stderr in started:
        assert stdout == subprocess.PIPE
        assert stderr == subprocess.PIPE


def test_start_sync(tmp_path, mock_callosum, monkeypatch):
    """Test that start_sync() launches sol sync with remote URL."""
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
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))

    # Test start_sync()
    remote_url = "https://server:5000/app/remote/ingest/abc123"
    sync_proc = mod.start_sync(remote_url)
    assert sync_proc is not None

    # Verify the command includes --remote with the URL
    sync_cmds = [cmd for cmd, _, _ in started if "sync" in cmd]
    assert len(sync_cmds) == 1
    cmd = sync_cmds[0]
    assert cmd == ["sol", "sync", "-v", "--remote", remote_url]


def test_parse_args_remote_flag():
    """Test that parse_args includes --remote flag."""
    mod = importlib.import_module("think.supervisor")

    parser = mod.parse_args()
    args = parser.parse_args(["--remote", "https://server/ingest/key"])

    assert args.remote == "https://server/ingest/key"


def test_parse_args_remote_flag_optional():
    """Test that --remote is optional."""
    mod = importlib.import_module("think.supervisor")

    parser = mod.parse_args()
    args = parser.parse_args([])

    assert args.remote is None


@pytest.mark.asyncio
async def test_supervise_logs_recovery(mock_callosum, monkeypatch, caplog):
    mod = importlib.reload(importlib.import_module("think.supervisor"))
    mod.shutdown_requested = False

    health_states = [["hear"], []]
    time_counter = {"value": 0.0}  # Use dict to allow mutation in closure

    def fake_time():
        """Auto-incrementing time mock that won't run out of values."""
        current = time_counter["value"]
        time_counter["value"] += 1.0
        return current

    def fake_check_health(threshold):
        state = health_states.pop(0)
        if not health_states:
            mod.shutdown_requested = True
        return state

    async def fake_send_notification(*args, **kwargs):
        pass

    async def fake_clear_notification(*args, **kwargs):
        pass

    async def fake_sleep(_):
        pass

    def fake_handle_daily_tasks():
        pass

    monkeypatch.setattr(mod, "check_runner_exits", lambda procs: [])
    monkeypatch.setattr(mod, "check_health", fake_check_health)
    monkeypatch.setattr(mod, "send_notification", fake_send_notification)
    monkeypatch.setattr(mod, "clear_notification", fake_clear_notification)
    monkeypatch.setattr(mod, "handle_daily_tasks", fake_handle_daily_tasks)
    monkeypatch.setattr(mod.time, "time", fake_time)
    monkeypatch.setattr(mod.asyncio, "sleep", fake_sleep)

    monkeypatch.setenv("JOURNAL_PATH", "/test/journal")

    with caplog.at_level(logging.INFO):
        await mod.supervise(threshold=1, interval=1, procs=[])

    messages = [record.getMessage() for record in caplog.records]
    assert "hear heartbeat recovered" in messages
    assert messages.count("Heartbeat OK") == 1

    mod.shutdown_requested = False
