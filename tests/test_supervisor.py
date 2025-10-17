import importlib
import io
import logging
import os
import subprocess
import time

import pytest


def test_check_health(tmp_path, monkeypatch):
    mod = importlib.import_module("think.supervisor")
    health = tmp_path / "health"
    health.mkdir()
    for name in ("see.up", "hear.up"):
        (health / name).write_text("x")
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    assert mod.check_health(threshold=90) == []

    old = time.time() - 100
    for hb in health.iterdir():
        os.utime(hb, (old, old))
    stale = mod.check_health(threshold=90)
    assert sorted(stale) == ["hear", "see"]


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
    assert called[0]["title"] == "Sunstone Supervisor"
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


def test_start_runners(tmp_path, monkeypatch):
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

    procs = mod.start_observers()
    assert len(procs) == 2
    assert any(cmd == ["observe-gnome", "-v"] for cmd, _, _ in started)
    assert any(cmd == ["observe-sense", "-v"] for cmd, _, _ in started)
    # Check that stdout and stderr capture pipes
    for cmd, stdout, stderr in started:
        assert stdout == subprocess.PIPE
        assert stderr == subprocess.PIPE


def test_run_dream(tmp_path, monkeypatch):
    mod = importlib.import_module("think.supervisor")

    launch_calls = {}

    class DummyProcess:
        def __init__(self):
            self.pid = 12345

        def wait(self):
            return 0

    class DummyLogger:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    def fake_launch(name, cmd, *, restart=False, log_name=None):
        dummy_logger = DummyLogger()
        launch_calls["args"] = (name, cmd, restart, log_name)
        launch_calls["logger"] = dummy_logger
        return mod.ManagedProcess(
            process=DummyProcess(),
            name=name,
            logger=dummy_logger,
            cmd=list(cmd),
            restart=restart,
            threads=[],
        )

    monkeypatch.setattr(mod, "_launch_process", fake_launch)
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))

    times = iter([0, 1])
    monkeypatch.setattr(mod.time, "time", lambda: next(times))

    messages = []
    monkeypatch.setattr(
        mod.logging, "info", lambda msg, *a: messages.append(msg % a if a else msg)
    )

    assert mod.run_dream() is True

    name, cmd, restart, log_name = launch_calls["args"]
    assert name == "dream"
    assert cmd == ["think-dream", "-v"]
    assert restart is False
    assert log_name is None
    assert os.environ["JOURNAL_PATH"] == str(tmp_path)
    assert launch_calls["logger"].closed is True
    assert any("seconds" in m for m in messages)


@pytest.mark.asyncio
async def test_supervise_logs_recovery(monkeypatch, caplog):
    mod = importlib.reload(importlib.import_module("think.supervisor"))
    mod.shutdown_requested = False

    health_states = [["hear"], []]
    time_counter = iter(
        [0.0, 1.0, 2.0, 3.0, 4.0]
    )  # Incrementing time for health check timing

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

    monkeypatch.setattr(mod, "check_runner_exits", lambda procs: [])
    monkeypatch.setattr(mod, "check_health", fake_check_health)
    monkeypatch.setattr(mod, "send_notification", fake_send_notification)
    monkeypatch.setattr(mod, "clear_notification", fake_clear_notification)
    monkeypatch.setattr(
        mod, "check_scheduled_agents", lambda: None
    )  # Mock scheduled agents
    monkeypatch.setattr(mod.time, "time", lambda: next(time_counter))
    monkeypatch.setattr(mod.asyncio, "sleep", fake_sleep)

    monkeypatch.setenv("JOURNAL_PATH", "/test/journal")

    with caplog.at_level(logging.INFO):
        await mod.supervise(threshold=1, interval=1, procs=[])

    messages = [record.getMessage() for record in caplog.records]
    assert "hear heartbeat recovered" in messages
    assert messages.count("Heartbeat OK") == 1

    mod.shutdown_requested = False
