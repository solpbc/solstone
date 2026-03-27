# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for think.routines — user-defined routines engine."""

from contextlib import contextmanager
from datetime import datetime, timezone
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

import think.routines
from think.call import call_app
from think.routines import cron_matches, get_config, save_config

runner = CliRunner()


@contextmanager
def _fake_now(dt: datetime):
    """Temporarily replace think.routines.datetime with a fake that returns dt."""

    class _FakeDatetime:
        @staticmethod
        def now(tz=None):
            if tz is None:
                return dt
            if dt.tzinfo is None:
                return dt.replace(tzinfo=tz)
            return dt.astimezone(tz)

    think.routines.datetime = _FakeDatetime
    try:
        yield
    finally:
        think.routines.datetime = datetime


@pytest.fixture(autouse=True)
def reset_routines_state():
    """Reset routines module state between tests."""
    import think.routines as mod

    mod._config = {}
    mod._callosum = None
    mod._last_fired = {}
    yield
    mod._config = {}
    mod._callosum = None
    mod._last_fired = {}


@pytest.fixture
def journal_path(tmp_path, monkeypatch):
    """Create a temp journal with routines/ and health/ dirs."""
    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))
    (tmp_path / "routines").mkdir()
    (tmp_path / "health").mkdir()
    return tmp_path


class TestCronMatches:
    def test_wildcard_all(self):
        dt = datetime(2026, 3, 15, 9, 30)
        assert cron_matches("* * * * *", dt) is True

    def test_specific_values(self):
        assert cron_matches("30 9 15 3 *", datetime(2026, 3, 15, 9, 30)) is True
        assert cron_matches("30 9 15 3 *", datetime(2026, 3, 15, 9, 31)) is False

    def test_comma_list(self):
        assert cron_matches("0,15,30,45 * * * *", datetime(2026, 3, 15, 9, 15)) is True
        assert cron_matches("0,15,30,45 * * * *", datetime(2026, 3, 15, 9, 10)) is False

    def test_range(self):
        assert cron_matches("0 9-17 * * *", datetime(2026, 3, 15, 9, 0)) is True
        assert cron_matches("0 9-17 * * *", datetime(2026, 3, 15, 18, 0)) is False

    def test_step(self):
        assert cron_matches("*/15 * * * *", datetime(2026, 3, 15, 9, 45)) is True
        assert cron_matches("*/15 * * * *", datetime(2026, 3, 15, 9, 44)) is False

    def test_range_with_step(self):
        assert cron_matches("0 1-23/2 * * *", datetime(2026, 3, 15, 9, 0)) is True
        assert cron_matches("0 1-23/2 * * *", datetime(2026, 3, 15, 10, 0)) is False

    def test_dow_sunday_zero(self):
        sunday = datetime(2026, 3, 29, 0, 0)
        assert sunday.isoweekday() == 7
        assert cron_matches("0 0 * * 0", sunday) is True

    def test_dow_sunday_seven(self):
        sunday = datetime(2026, 3, 29, 0, 0)
        assert cron_matches("0 0 * * 7", sunday) is True

    def test_dow_monday(self):
        monday = datetime(2026, 3, 30, 0, 0)
        assert monday.isoweekday() == 1
        assert cron_matches("0 0 * * 1", monday) is True

    def test_invalid_field_count(self):
        with pytest.raises(ValueError):
            cron_matches("* * * *", datetime(2026, 3, 15, 9, 0))

    def test_step_zero(self):
        with pytest.raises(ValueError):
            cron_matches("*/0 * * * *", datetime(2026, 3, 15, 9, 0))

    def test_out_of_range(self):
        with pytest.raises(ValueError):
            cron_matches("60 * * * *", datetime(2026, 3, 15, 9, 0))


class TestConfigIO:
    def test_get_config_empty(self, journal_path):
        assert get_config() == {}

    def test_save_and_get_config(self, journal_path):
        routine = {
            "abc123": {
                "id": "abc123",
                "name": "Morning",
                "instruction": "Summarize today",
                "cadence": "0 9 * * *",
                "timezone": "UTC",
                "facets": ["work"],
                "enabled": True,
                "created": "2026-03-27T00:00:00+00:00",
                "last_run": None,
                "template": None,
                "notify": False,
            }
        }
        save_config(routine)
        loaded = get_config()
        assert loaded == routine

    def test_save_config_creates_directory(self, journal_path):
        (journal_path / "routines").rmdir()
        save_config({"abc123": {"id": "abc123"}})
        assert (journal_path / "routines").exists()
        assert (journal_path / "routines" / "config.json").exists()

    def test_get_config_corrupt_json(self, journal_path):
        (journal_path / "routines" / "config.json").write_text("not json{")
        assert get_config() == {}


class TestCheck:
    def test_fires_due_routine(self, journal_path):
        import think.routines as mod

        save_config(
            {
                "routine-1": {
                    "id": "routine-1",
                    "name": "Morning",
                    "instruction": "Do the thing",
                    "cadence": "0 9 * * *",
                    "timezone": "UTC",
                    "enabled": True,
                    "facets": [],
                    "template": None,
                    "notify": False,
                    "last_run": None,
                }
            }
        )

        dt = datetime(2026, 3, 27, 9, 0, tzinfo=timezone.utc)
        with (
            patch("think.routines.cortex_request", return_value="fake_agent_id") as mock_req,
            patch(
                "think.routines.wait_for_agents",
                return_value=({"fake_agent_id": "finish"}, []),
            ),
            patch("think.routines.callosum_send", return_value=True),
            _fake_now(dt),
        ):
            mod.check()

        mock_req.assert_called_once()

    def test_skips_disabled_routine(self, journal_path):
        import think.routines as mod

        save_config(
            {
                "routine-1": {
                    "id": "routine-1",
                    "name": "Morning",
                    "instruction": "Do the thing",
                    "cadence": "0 9 * * *",
                    "timezone": "UTC",
                    "enabled": False,
                    "facets": [],
                    "template": None,
                    "notify": False,
                    "last_run": None,
                }
            }
        )

        dt = datetime(2026, 3, 27, 9, 0, tzinfo=timezone.utc)
        with (
            patch("think.routines.cortex_request", return_value="fake_agent_id") as mock_req,
            patch(
                "think.routines.wait_for_agents",
                return_value=({"fake_agent_id": "finish"}, []),
            ),
            patch("think.routines.callosum_send", return_value=True),
            _fake_now(dt),
        ):
            mod.check()

        mock_req.assert_not_called()

    def test_idempotent_same_minute(self, journal_path):
        import think.routines as mod

        save_config(
            {
                "routine-1": {
                    "id": "routine-1",
                    "name": "Morning",
                    "instruction": "Do the thing",
                    "cadence": "0 9 * * *",
                    "timezone": "UTC",
                    "enabled": True,
                    "facets": [],
                    "template": None,
                    "notify": False,
                    "last_run": None,
                }
            }
        )

        dt = datetime(2026, 3, 27, 9, 0, tzinfo=timezone.utc)
        with (
            patch("think.routines.cortex_request", return_value="fake_agent_id") as mock_req,
            patch(
                "think.routines.wait_for_agents",
                return_value=({"fake_agent_id": "finish"}, []),
            ),
            patch("think.routines.callosum_send", return_value=True),
            _fake_now(dt),
        ):
            mod.check()
            mod.check()

        assert mock_req.call_count == 1

    def test_fires_again_next_minute(self, journal_path):
        import think.routines as mod

        save_config(
            {
                "routine-1": {
                    "id": "routine-1",
                    "name": "Hourly",
                    "instruction": "Do the thing",
                    "cadence": "0 * * * *",
                    "timezone": "UTC",
                    "enabled": True,
                    "facets": [],
                    "template": None,
                    "notify": False,
                    "last_run": None,
                }
            }
        )

        with (
            patch("think.routines.cortex_request", return_value="fake_agent_id") as mock_req,
            patch(
                "think.routines.wait_for_agents",
                return_value=({"fake_agent_id": "finish"}, []),
            ),
            patch("think.routines.callosum_send", return_value=True),
        ):
            with _fake_now(datetime(2026, 3, 27, 9, 0, tzinfo=timezone.utc)):
                mod.check()
            with _fake_now(datetime(2026, 3, 27, 10, 0, tzinfo=timezone.utc)):
                mod.check()

        assert mock_req.call_count == 2


class TestCLI:
    def test_create_routine(self, journal_path):
        result = runner.invoke(
            call_app,
            [
                "routines",
                "create",
                "--name",
                "Morning review",
                "--instruction",
                "Review the day",
                "--cadence",
                "0 9 * * *",
            ],
        )
        assert result.exit_code == 0
        config = get_config()
        assert len(config) == 1
        routine = next(iter(config.values()))
        assert routine["name"] == "Morning review"

    def test_list_routines(self, journal_path):
        save_config(
            {
                "routine-1": {
                    "id": "routine-1",
                    "name": "Morning review",
                    "instruction": "Review the day",
                    "cadence": "0 9 * * *",
                    "timezone": "UTC",
                    "enabled": True,
                    "facets": [],
                    "template": None,
                    "notify": False,
                    "last_run": None,
                }
            }
        )
        result = runner.invoke(call_app, ["routines", "list"])
        assert result.exit_code == 0
        assert "Morning review" in result.stdout

    def test_list_empty(self, journal_path):
        result = runner.invoke(call_app, ["routines", "list"])
        assert result.exit_code == 0
        assert "No routines configured." in result.stdout
