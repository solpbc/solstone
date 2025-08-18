import importlib
import os
import time


def test_check_health(tmp_path):
    mod = importlib.import_module("think.supervisor")
    health = tmp_path / "health"
    health.mkdir()
    for name in ("see.up", "hear.up"):
        (health / name).write_text("x")
    assert mod.check_health(str(tmp_path), threshold=90) == []

    old = time.time() - 100
    for hb in health.iterdir():
        os.utime(hb, (old, old))
    stale = mod.check_health(str(tmp_path), threshold=90)
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
        def terminate(self):
            pass

        def wait(self, timeout=None):
            pass

    def fake_popen(cmd, stdout=None, stderr=None, start_new_session=False):
        started.append((cmd, stdout, stderr))
        return DummyProc()

    monkeypatch.setattr(mod.subprocess, "Popen", fake_popen)

    procs = mod.start_runners(str(tmp_path))
    # No longer creating log file, output goes to console (None)
    assert len(procs) == 2
    assert any("hear.runner" in c[0] for c in started)
    assert any("see.runner" in c[0] for c in started)
    # Check that stdout and stderr are None (inherit from parent)
    for cmd, stdout, stderr in started:
        assert stdout is None
        assert stderr is None


def test_main_no_runners(tmp_path, monkeypatch):
    mod = importlib.import_module("think.supervisor")
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    (tmp_path / "health").mkdir()

    called = []

    def fake_supervise(*args, **kwargs):
        called.append(True)

    monkeypatch.setattr(mod, "supervise", fake_supervise)
    monkeypatch.setattr(mod, "start_runners", lambda journal: called.append(False))
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
    monkeypatch.setattr(mod, "start_runners", lambda journal: None)
    monkeypatch.setattr("sys.argv", ["think-supervisor", "--no-daily", "--no-runners"])

    mod.main()
    assert called.get("daily") is False


def test_run_process_day(tmp_path, monkeypatch):
    mod = importlib.import_module("think.supervisor")

    commands = []

    class DummyResult:
        returncode = 0

    def fake_run(cmd, stdout=None, stderr=None):
        commands.append((cmd, stdout, stderr))
        return DummyResult()

    monkeypatch.setattr(mod.subprocess, "run", fake_run)

    times = iter([0, 1])
    monkeypatch.setattr(mod.time, "time", lambda: next(times))

    messages = []
    monkeypatch.setattr(
        mod.logging, "info", lambda msg, *a: messages.append(msg % a if a else msg)
    )

    mod.run_process_day()  # No longer takes log_path parameter

    assert commands[0][0][2] == "think.process_day"
    # Check that stdout and stderr are None (inherit from parent)
    assert commands[0][1] is None
    assert commands[0][2] is None
    assert any("seconds" in m for m in messages)
