# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

import argparse
import os

import pytest


@pytest.fixture
def journal_path(tmp_path, monkeypatch):
    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))
    (tmp_path / "health").mkdir()
    return tmp_path


@pytest.fixture
def heartbeat_mocks(monkeypatch):
    monkeypatch.setattr(
        "think.heartbeat.setup_cli",
        lambda parser: argparse.Namespace(force=False),
    )
    monkeypatch.setattr("think.heartbeat.ensure_sol_directory", lambda: None)
    monkeypatch.setattr(
        "think.heartbeat.cortex_request", lambda *args, **kwargs: "agent-123"
    )
    monkeypatch.setattr(
        "think.heartbeat.wait_for_agents",
        lambda *args, **kwargs: ({"agent-123": "finish"}, []),
    )


def test_heartbeat_command_mapping():
    """heartbeat key in COMMANDS maps to think.heartbeat module."""
    from sol import COMMANDS

    assert COMMANDS["heartbeat"] == "think.heartbeat"


def test_heartbeat_main_is_callable():
    """think.heartbeat.main is a callable function."""
    from think.heartbeat import main

    assert callable(main)


def test_pid_guard_live_process_exits_zero(journal_path, heartbeat_mocks):
    """When PID file contains current process PID, main() exits 0 without cortex."""
    import think.heartbeat as mod

    pid_file = journal_path / "health" / "heartbeat.pid"
    pid_file.write_text(str(os.getpid()))

    mod.cortex_request = lambda *a, **kw: pytest.fail(
        "cortex_request should not be called"
    )

    with pytest.raises(SystemExit) as exc_info:
        mod.main()
    assert exc_info.value.code == 0


def test_pid_guard_dead_process_removes_stale_pid(journal_path, heartbeat_mocks):
    """When PID file contains a dead PID, main() removes it and proceeds to cortex."""
    import think.heartbeat as mod

    pid_file = journal_path / "health" / "heartbeat.pid"
    dead_pid = 99999999
    try:
        os.kill(dead_pid, 0)
        pytest.skip("PID 99999999 is unexpectedly alive")
    except ProcessLookupError:
        pass

    pid_file.write_text(str(dead_pid))

    cortex_called = []

    def fake_cortex(*args, **kwargs):
        cortex_called.append(True)
        return "agent-123"

    mod.cortex_request = fake_cortex

    with pytest.raises(SystemExit) as exc_info:
        mod.main()
    assert exc_info.value.code == 0
    assert len(cortex_called) == 1


def test_pid_file_created_and_removed_on_success(journal_path, heartbeat_mocks):
    """PID file exists during execution and is removed after main() completes."""
    import think.heartbeat as mod

    pid_file = journal_path / "health" / "heartbeat.pid"
    pid_during_run = []

    def capture_pid_cortex(*args, **kwargs):
        pid_during_run.append(pid_file.exists())
        if pid_file.exists():
            pid_during_run.append(pid_file.read_text().strip())
        return "agent-123"

    mod.cortex_request = capture_pid_cortex

    with pytest.raises(SystemExit):
        mod.main()

    assert pid_during_run[0] is True
    assert pid_during_run[1] == str(os.getpid())
    assert not pid_file.exists()


def test_pid_file_removed_on_error(journal_path, heartbeat_mocks):
    """PID file is removed even when cortex_request returns None (error path)."""
    import think.heartbeat as mod

    pid_file = journal_path / "health" / "heartbeat.pid"
    mod.cortex_request = lambda *a, **kw: None

    with pytest.raises(SystemExit) as exc_info:
        mod.main()
    assert exc_info.value.code == 1
    assert not pid_file.exists()


def test_pid_file_removed_on_timeout(journal_path, heartbeat_mocks):
    """PID file is removed on timeout path."""
    import think.heartbeat as mod

    pid_file = journal_path / "health" / "heartbeat.pid"
    mod.wait_for_agents = lambda *a, **kw: ({}, ["agent-123"])

    with pytest.raises(SystemExit) as exc_info:
        mod.main()
    assert exc_info.value.code == 2
    assert not pid_file.exists()


def test_log_run_appends_line(journal_path):
    """_log_run appends a correctly formatted line to heartbeat.log."""
    import time

    from think.heartbeat import _log_run

    health_dir = journal_path / "health"
    start_time = time.monotonic() - 5

    _log_run(health_dir, start_time, "success")

    log_file = health_dir / "heartbeat.log"
    assert log_file.exists()
    content = log_file.read_text()
    assert content.endswith("\n")
    line = content.strip()
    assert "duration=" in line
    assert "outcome=success" in line


def test_log_written_after_successful_run(journal_path, heartbeat_mocks):
    """After a successful main() run, heartbeat.log has a success entry."""
    import think.heartbeat as mod

    with pytest.raises(SystemExit) as exc_info:
        mod.main()
    assert exc_info.value.code == 0

    log_file = journal_path / "health" / "heartbeat.log"
    assert log_file.exists()
    content = log_file.read_text()
    assert "outcome=success" in content


def test_cortex_prompt_does_not_contain_journal_path(journal_path, heartbeat_mocks):
    """cortex_request prompt must not leak filesystem paths."""
    import think.heartbeat as mod

    captured_kwargs = {}

    def capture_cortex(*args, **kwargs):
        captured_kwargs.update(kwargs)
        if args:
            captured_kwargs["_positional"] = args
        return "agent-123"

    mod.cortex_request = capture_cortex

    with pytest.raises(SystemExit):
        mod.main()

    prompt = captured_kwargs.get("prompt", "")
    assert str(journal_path) not in prompt, "prompt must not leak filesystem paths"


def test_recency_check_skips_recent_heartbeat(journal_path, heartbeat_mocks):
    """When heartbeat.log has a recent success, main() exits 0 without cortex."""
    from datetime import datetime

    import think.heartbeat as mod

    # Write a recent success entry
    log_file = journal_path / "health" / "heartbeat.log"
    recent_ts = datetime.now().isoformat(timespec="seconds")
    log_file.write_text(f"{recent_ts} duration=5s outcome=success\n")

    mod.cortex_request = lambda *a, **kw: pytest.fail(
        "cortex_request should not be called"
    )

    with pytest.raises(SystemExit) as exc_info:
        mod.main()
    assert exc_info.value.code == 0


def test_recency_check_runs_after_old_heartbeat(journal_path, heartbeat_mocks):
    """When heartbeat.log success is older than the window, main() runs cortex."""
    from datetime import datetime, timedelta

    import think.heartbeat as mod

    # Write an old success entry (24 hours ago)
    log_file = journal_path / "health" / "heartbeat.log"
    old_ts = (datetime.now() - timedelta(hours=24)).isoformat(timespec="seconds")
    log_file.write_text(f"{old_ts} duration=5s outcome=success\n")

    cortex_called = []

    def fake_cortex(*args, **kwargs):
        cortex_called.append(True)
        return "agent-123"

    mod.cortex_request = fake_cortex

    with pytest.raises(SystemExit):
        mod.main()
    assert len(cortex_called) == 1


def test_force_flag_bypasses_recency_check(journal_path, monkeypatch):
    """--force runs full check even with a recent success."""
    import think.heartbeat as mod

    monkeypatch.setattr(
        "think.heartbeat.setup_cli",
        lambda parser: argparse.Namespace(force=True),
    )
    monkeypatch.setattr("think.heartbeat.ensure_sol_directory", lambda: None)
    monkeypatch.setattr(
        "think.heartbeat.wait_for_agents",
        lambda *args, **kwargs: ({"agent-123": "finish"}, []),
    )

    # Write a recent success entry
    from datetime import datetime

    log_file = journal_path / "health" / "heartbeat.log"
    recent_ts = datetime.now().isoformat(timespec="seconds")
    log_file.write_text(f"{recent_ts} duration=5s outcome=success\n")

    cortex_called = []

    def fake_cortex(*args, **kwargs):
        cortex_called.append(True)
        return "agent-123"

    mod.cortex_request = fake_cortex

    with pytest.raises(SystemExit):
        mod.main()
    assert len(cortex_called) == 1


def test_last_success_time_parses_log(journal_path):
    """_last_success_time returns the timestamp of the most recent success."""
    from think.heartbeat import _last_success_time

    health_dir = journal_path / "health"
    log_file = health_dir / "heartbeat.log"
    log_file.write_text(
        "2026-03-19T08:00:00 duration=120s outcome=success\n"
        "2026-03-19T12:00:00 duration=5s outcome=error\n"
        "2026-03-19T14:00:00 duration=90s outcome=success\n"
    )

    result = _last_success_time(health_dir)
    assert result is not None
    assert result.hour == 14
    assert result.day == 19


def test_last_success_time_returns_none_for_no_log(journal_path):
    """_last_success_time returns None when no log file exists."""
    from think.heartbeat import _last_success_time

    result = _last_success_time(journal_path / "health")
    assert result is None


def test_last_success_time_returns_none_for_no_successes(journal_path):
    """_last_success_time returns None when log has no success entries."""
    from think.heartbeat import _last_success_time

    health_dir = journal_path / "health"
    log_file = health_dir / "heartbeat.log"
    log_file.write_text(
        "2026-03-19T08:00:00 duration=5s outcome=error\n"
        "2026-03-19T12:00:00 duration=5s outcome=timeout\n"
    )

    result = _last_success_time(health_dir)
    assert result is None


def test_dream_emit_daily_complete_shape(monkeypatch):
    """dream.emit('daily_complete', ...) calls _callosum.emit with correct tract and fields."""
    from unittest.mock import Mock

    import think.dream as dream_mod

    mock_conn = Mock()
    monkeypatch.setattr(dream_mod, "_callosum", mock_conn)

    dream_mod.emit(
        "daily_complete", day="20260318", success=3, failed=0, duration_ms=5000
    )

    mock_conn.emit.assert_called_once_with(
        "dream",
        "daily_complete",
        day="20260318",
        success=3,
        failed=0,
        duration_ms=5000,
    )


def test_dream_emit_noop_without_callosum(monkeypatch):
    """dream.emit() does nothing when _callosum is None."""
    import think.dream as dream_mod

    monkeypatch.setattr(dream_mod, "_callosum", None)
    dream_mod.emit("daily_complete", day="20260318")
