# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for think.scheduler — clock-aligned task scheduler."""

import json
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock

import pytest

import think.scheduler


@contextmanager
def _fake_now(dt: datetime):
    """Temporarily replace think.scheduler.datetime with a fake that returns dt."""

    class _FakeDatetime:
        min = datetime.min

        @staticmethod
        def now():
            return dt

        @staticmethod
        def fromtimestamp(ts):
            return datetime.fromtimestamp(ts)

        @staticmethod
        def combine(*a, **k):
            return datetime.combine(*a, **k)

    think.scheduler.datetime = _FakeDatetime
    try:
        yield
    finally:
        think.scheduler.datetime = datetime


@pytest.fixture(autouse=True)
def reset_scheduler_state():
    """Reset scheduler module state between tests."""
    import think.scheduler as mod

    mod._entries = {}
    mod._state = {}
    mod._callosum = None
    mod._last_hour = None
    mod._last_day = None
    yield
    mod._entries = {}
    mod._state = {}
    mod._callosum = None
    mod._last_hour = None
    mod._last_day = None


@pytest.fixture
def journal_path(tmp_path, monkeypatch):
    """Create a temp journal with config/ and health/ dirs."""
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    (tmp_path / "config").mkdir()
    (tmp_path / "health").mkdir()
    return tmp_path


def _write_config(journal: Path, config: dict) -> None:
    with open(journal / "config" / "schedules.json", "w") as f:
        json.dump(config, f)


def _write_state(journal: Path, state: dict) -> None:
    with open(journal / "health" / "scheduler.json", "w") as f:
        json.dump(state, f)


def _read_state(journal: Path) -> dict:
    with open(journal / "health" / "scheduler.json") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------


class TestLoadConfig:
    def test_valid_config(self, journal_path):
        _write_config(
            journal_path,
            {
                "sync:plaud": {
                    "cmd": ["sol", "import", "--sync", "plaud"],
                    "every": "hourly",
                },
            },
        )
        from think.scheduler import load_config

        entries = load_config()
        assert "sync:plaud" in entries
        assert entries["sync:plaud"]["every"] == "hourly"
        assert entries["sync:plaud"]["cmd"] == ["sol", "import", "--sync", "plaud"]

    def test_missing_file_returns_empty(self, journal_path):
        from think.scheduler import load_config

        assert load_config() == {}

    def test_invalid_json_returns_empty(self, journal_path):
        (journal_path / "config" / "schedules.json").write_text("not json{")
        from think.scheduler import load_config

        assert load_config() == {}

    def test_unknown_every_skipped(self, journal_path):
        _write_config(
            journal_path,
            {
                "bad": {"cmd": ["sol", "noop"], "every": "weekly"},
            },
        )
        from think.scheduler import load_config

        assert load_config() == {}

    def test_missing_cmd_skipped(self, journal_path):
        _write_config(
            journal_path,
            {
                "bad": {"every": "hourly"},
            },
        )
        from think.scheduler import load_config

        assert load_config() == {}

    def test_disabled_entry_excluded(self, journal_path):
        _write_config(
            journal_path,
            {
                "off": {"cmd": ["sol", "noop"], "every": "hourly", "enabled": False},
            },
        )
        from think.scheduler import load_config

        assert load_config() == {}


# ---------------------------------------------------------------------------
# load_state / save_state
# ---------------------------------------------------------------------------


class TestState:
    def test_round_trip(self, journal_path):
        import think.scheduler as mod

        mod._state = {"sync:plaud": {"last_run": 1700000000.0}}
        mod.save_state()

        loaded = mod.load_state()
        assert loaded["sync:plaud"]["last_run"] == 1700000000.0

    def test_missing_file_returns_empty(self, journal_path):
        from think.scheduler import load_state

        assert load_state() == {}

    def test_atomic_write_no_partial(self, journal_path):
        """State file shouldn't have leftover tmp files on success."""
        import think.scheduler as mod

        mod._state = {"a": {"last_run": 1.0}}
        mod.save_state()

        tmps = list((journal_path / "health").glob(".scheduler_*"))
        assert tmps == []


# ---------------------------------------------------------------------------
# _is_due
# ---------------------------------------------------------------------------


class TestIsDue:
    def test_no_state_is_due(self):
        from think.scheduler import _is_due

        entry = {"cmd": ["sol", "x"], "every": "hourly"}
        assert _is_due(entry, None, datetime(2026, 2, 17, 14, 30)) is True

    def test_hourly_same_hour_not_due(self):
        from think.scheduler import _is_due

        entry = {"cmd": ["sol", "x"], "every": "hourly"}
        # Last run at 14:05, now is 14:30 — same hour
        state = {"last_run": datetime(2026, 2, 17, 14, 5).timestamp()}
        assert _is_due(entry, state, datetime(2026, 2, 17, 14, 30)) is False

    def test_hourly_new_hour_is_due(self):
        from think.scheduler import _is_due

        entry = {"cmd": ["sol", "x"], "every": "hourly"}
        # Last run at 13:45, now is 14:01 — new hour
        state = {"last_run": datetime(2026, 2, 17, 13, 45).timestamp()}
        assert _is_due(entry, state, datetime(2026, 2, 17, 14, 1)) is True

    def test_daily_same_day_not_due(self):
        from think.scheduler import _is_due

        entry = {"cmd": ["sol", "x"], "every": "daily"}
        # Last run today at 00:05, now is 14:00
        state = {"last_run": datetime(2026, 2, 17, 0, 5).timestamp()}
        assert _is_due(entry, state, datetime(2026, 2, 17, 14, 0)) is False

    def test_daily_new_day_is_due(self):
        from think.scheduler import _is_due

        entry = {"cmd": ["sol", "x"], "every": "daily"}
        # Last run yesterday at 23:50, now is 00:01
        state = {"last_run": datetime(2026, 2, 16, 23, 50).timestamp()}
        assert _is_due(entry, state, datetime(2026, 2, 17, 0, 1)) is True


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------


class TestInit:
    def test_loads_config_and_state(self, journal_path):
        _write_config(
            journal_path,
            {
                "a": {"cmd": ["sol", "x"], "every": "hourly"},
            },
        )
        _write_state(journal_path, {"a": {"last_run": 1700000000.0}})

        import think.scheduler as mod

        callosum = Mock()
        mod.init(callosum)

        assert "a" in mod._entries
        assert mod._state["a"]["last_run"] == 1700000000.0
        assert mod._callosum is callosum
        assert mod._last_hour is not None
        assert mod._last_day is not None

    def test_no_config_file(self, journal_path):
        import think.scheduler as mod

        mod.init(Mock())
        assert mod._entries == {}


# ---------------------------------------------------------------------------
# check
# ---------------------------------------------------------------------------


class TestCheck:
    def test_pre_init_returns_immediately(self, journal_path):
        """check() does nothing when init() hasn't been called."""
        import think.scheduler as mod

        callosum = Mock()
        callosum.emit = Mock(return_value=True)
        mod._callosum = callosum

        mod.check()
        callosum.emit.assert_not_called()

    def test_no_boundary_no_io(self, journal_path):
        """When no boundary has crossed, check() does nothing."""
        import think.scheduler as mod

        callosum = Mock()
        callosum.emit = Mock(return_value=True)
        now = datetime(2026, 2, 17, 14, 30)

        _write_config(
            journal_path,
            {
                "a": {"cmd": ["sol", "x"], "every": "hourly"},
            },
        )

        mod.init(callosum)
        # Set boundaries to current — no crossing
        mod._last_hour = mod._hour_mark(now)
        mod._last_day = now.date()

        with _fake_now(now):
            mod.check()

        callosum.emit.assert_not_called()

    def test_hourly_boundary_submits(self, journal_path):
        """Crossing an hour boundary submits due hourly tasks."""
        import think.scheduler as mod

        callosum = Mock()
        callosum.emit = Mock(return_value=True)

        _write_config(
            journal_path,
            {
                "a": {"cmd": ["sol", "test-task", "-v"], "every": "hourly"},
            },
        )

        mod.init(callosum)

        # Simulate: last check was at 13:59, now it's 14:01
        mod._last_hour = datetime(2026, 2, 17, 13, 0)
        mod._last_day = datetime(2026, 2, 17).date()
        # No prior state → task is due

        with _fake_now(datetime(2026, 2, 17, 14, 1)):
            mod.check()

        callosum.emit.assert_called_once()
        call_kwargs = callosum.emit.call_args
        assert call_kwargs[0][0] == "supervisor"
        assert call_kwargs[0][1] == "request"
        assert call_kwargs[1]["cmd"] == ["sol", "test-task", "-v"]
        assert call_kwargs[1]["ref"].startswith("sched:a:")

        # State should be updated
        assert "a" in mod._state
        assert mod._state["a"]["last_run"] > 0

        # State file should be written
        saved = _read_state(journal_path)
        assert "a" in saved

    def test_daily_boundary_submits(self, journal_path):
        """Crossing a day boundary submits due daily tasks."""
        import think.scheduler as mod

        callosum = Mock()
        callosum.emit = Mock(return_value=True)

        _write_config(
            journal_path,
            {
                "d": {"cmd": ["sol", "daily-thing"], "every": "daily"},
            },
        )

        mod.init(callosum)

        # Simulate: last check was yesterday 23:59, now it's 00:01
        mod._last_hour = datetime(2026, 2, 16, 23, 0)
        mod._last_day = datetime(2026, 2, 16).date()

        with _fake_now(datetime(2026, 2, 17, 0, 1)):
            mod.check()

        callosum.emit.assert_called_once()
        assert callosum.emit.call_args[1]["cmd"] == ["sol", "daily-thing"]

    def test_submits_on_new_hour_after_previous_run(self, journal_path):
        """Task ran in hour 14; crossing to hour 15 triggers resubmission."""
        import think.scheduler as mod

        callosum = Mock()
        callosum.emit = Mock(return_value=True)

        _write_config(
            journal_path,
            {
                "a": {"cmd": ["sol", "x"], "every": "hourly"},
            },
        )
        # Already ran at 14:02
        _write_state(
            journal_path,
            {
                "a": {"last_run": datetime(2026, 2, 17, 14, 2).timestamp()},
            },
        )

        mod.init(callosum)
        mod._last_hour = datetime(2026, 2, 17, 14, 0)
        mod._last_day = datetime(2026, 2, 17).date()

        # Cross to hour 15
        with _fake_now(datetime(2026, 2, 17, 15, 0, 1)):
            mod.check()

        # Should submit because we crossed to hour 15 and last_run was in hour 14
        callosum.emit.assert_called_once()

    def test_config_reloaded_on_boundary(self, journal_path):
        """Config file changes are picked up when a boundary is crossed."""
        import think.scheduler as mod

        callosum = Mock()
        callosum.emit = Mock(return_value=True)

        # Start with empty config
        _write_config(journal_path, {})
        mod.init(callosum)
        mod._last_hour = datetime(2026, 2, 17, 13, 0)
        mod._last_day = datetime(2026, 2, 17).date()

        # Now write a real config
        _write_config(
            journal_path,
            {
                "new": {"cmd": ["sol", "new-task"], "every": "hourly"},
            },
        )

        with _fake_now(datetime(2026, 2, 17, 14, 1)):
            mod.check()

        callosum.emit.assert_called_once()
        assert callosum.emit.call_args[1]["cmd"] == ["sol", "new-task"]

    def test_emit_failure_no_state_update(self, journal_path):
        """If emit fails, last_run should not be updated."""
        import think.scheduler as mod

        callosum = Mock()
        callosum.emit = Mock(return_value=False)

        _write_config(
            journal_path,
            {
                "a": {"cmd": ["sol", "x"], "every": "hourly"},
            },
        )

        mod.init(callosum)
        mod._last_hour = datetime(2026, 2, 17, 13, 0)
        mod._last_day = datetime(2026, 2, 17).date()

        with _fake_now(datetime(2026, 2, 17, 14, 1)):
            mod.check()

        assert mod._state.get("a") is None


# ---------------------------------------------------------------------------
# collect_status
# ---------------------------------------------------------------------------


class TestCollectStatus:
    def test_returns_entries(self, journal_path):
        import think.scheduler as mod

        mod._entries = {
            "a": {"cmd": ["sol", "x"], "every": "hourly"},
        }
        mod._state = {"a": {"last_run": time.time()}}

        status = mod.collect_status()
        assert len(status) == 1
        assert status[0]["name"] == "a"
        assert status[0]["every"] == "hourly"
        assert "last_run" in status[0]
        assert "due" in status[0]


# ---------------------------------------------------------------------------
# CLI main()
# ---------------------------------------------------------------------------


class TestCLI:
    def test_no_config_prints_message(self, journal_path, capsys, monkeypatch):
        monkeypatch.setattr("sys.argv", ["sol schedule"])
        from think.scheduler import main

        main()
        out = capsys.readouterr().out
        assert "No schedules configured" in out

    def test_with_config_prints_table(self, journal_path, capsys, monkeypatch):
        monkeypatch.setattr("sys.argv", ["sol schedule"])
        _write_config(
            journal_path,
            {
                "sync:plaud": {
                    "cmd": ["sol", "import", "--sync", "plaud"],
                    "every": "hourly",
                },
            },
        )

        from think.scheduler import main

        main()
        out = capsys.readouterr().out
        assert "sync:plaud" in out
        assert "hourly" in out
        assert "NAME" in out
