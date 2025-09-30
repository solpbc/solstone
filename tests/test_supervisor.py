import importlib
import io
import logging
import os
import subprocess
import time


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


def test_send_notification(monkeypatch):
    mod = importlib.import_module("think.supervisor")
    called = []

    def fake_run(cmd, check=False):
        called.append(cmd)

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    mod.send_notification("msg", command="notify-send")
    assert called


def test_start_runners(tmp_path, monkeypatch):
    mod = importlib.import_module("think.supervisor")

    started = []

    class DummyProc:
        def __init__(self):
            self.stdout = io.StringIO()
            self.stderr = io.StringIO()

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

    procs = mod.start_runners()
    assert len(procs) == 2
    assert any(cmd == ["hear-runner", "-v"] for cmd, _, _ in started)
    assert any(cmd == ["see-runner", "-v"] for cmd, _, _ in started)
    # Check that stdout and stderr capture pipes
    for cmd, stdout, stderr in started:
        assert stdout == subprocess.PIPE
        assert stderr == subprocess.PIPE


def test_main_no_runners(tmp_path, monkeypatch):
    mod = importlib.import_module("think.supervisor")
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    (tmp_path / "health").mkdir()

    called = []

    def fake_supervise(*args, **kwargs):
        called.append(True)

    monkeypatch.setattr(mod, "supervise", fake_supervise)
    monkeypatch.setattr(mod, "start_runners", lambda: called.append(False))
    monkeypatch.setattr("sys.argv", ["think-supervisor", "--no-runners"])

    mod.main()
    assert True in called
    assert False not in called


def test_main_no_daily(tmp_path, monkeypatch):
    mod = importlib.reload(importlib.import_module("think.supervisor"))
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    (tmp_path / "health").mkdir()

    called = {}

    def fake_supervise(*args, **kwargs):
        called.update(kwargs)

    monkeypatch.setattr(mod, "supervise", fake_supervise)
    monkeypatch.setattr(mod, "start_runners", lambda: None)
    monkeypatch.setattr("sys.argv", ["think-supervisor", "--no-daily", "--no-runners"])

    mod.main()
    assert called.get("daily") is False


def test_run_process_day(tmp_path, monkeypatch):
    mod = importlib.import_module("think.supervisor")

    launch_calls = {}

    class DummyProcess:
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

    assert mod.run_process_day() is True

    name, cmd, restart, log_name = launch_calls["args"]
    assert name == "process_day"
    assert cmd == ["think-process-day", "-v"]
    assert restart is False
    assert log_name is None
    assert os.environ["JOURNAL_PATH"] == str(tmp_path)
    assert launch_calls["logger"].closed is True
    assert any("seconds" in m for m in messages)


def test_supervise_logs_recovery(monkeypatch, caplog):
    mod = importlib.reload(importlib.import_module("think.supervisor"))
    mod.shutdown_requested = False

    health_states = [["hear"], []]
    time_counter = iter([0.0, 1.0, 2.0, 3.0, 4.0])  # Incrementing time for health check timing

    def fake_check_health(threshold):
        state = health_states.pop(0)
        if not health_states:
            mod.shutdown_requested = True
        return state

    monkeypatch.setattr(mod, "check_runner_exits", lambda procs: [])
    monkeypatch.setattr(mod, "check_health", fake_check_health)
    monkeypatch.setattr(mod, "send_notification", lambda *args, **kwargs: None)
    monkeypatch.setattr(mod, "check_scheduled_agents", lambda: None)  # Mock scheduled agents
    monkeypatch.setattr(mod.time, "time", lambda: next(time_counter))
    monkeypatch.setattr(mod.time, "sleep", lambda _: None)

    monkeypatch.setenv("JOURNAL_PATH", "/test/journal")

    with caplog.at_level(logging.INFO):
        mod.supervise(threshold=1, interval=1, procs=[])

    messages = [record.getMessage() for record in caplog.records]
    assert "hear heartbeat recovered" in messages
    assert messages.count("Heartbeat OK") == 1

    mod.shutdown_requested = False
