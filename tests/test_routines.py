# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for think.routines — user-defined routines engine."""

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import frontmatter
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
    mod._events_fired = {}
    yield
    mod._config = {}
    mod._callosum = None
    mod._last_fired = {}
    mod._events_fired = {}


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
            patch(
                "think.routines.cortex_request", return_value="fake_agent_id"
            ) as mock_req,
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
            patch(
                "think.routines.cortex_request", return_value="fake_agent_id"
            ) as mock_req,
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
            patch(
                "think.routines.cortex_request", return_value="fake_agent_id"
            ) as mock_req,
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
            patch(
                "think.routines.cortex_request", return_value="fake_agent_id"
            ) as mock_req,
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


class TestTemplates:
    def test_templates_command_lists_all(self):
        result = runner.invoke(call_app, ["routines", "templates"])
        assert result.exit_code == 0
        for template_name in (
            "morning-briefing",
            "weekly-review",
            "domain-watch",
            "relationship-pulse",
            "commitment-audit",
            "monthly-patterns",
            "meeting-prep",
        ):
            assert template_name in result.stdout

    def test_template_frontmatter_valid(self):
        templates_dir = Path(__file__).resolve().parents[1] / "routines" / "templates"
        for path in sorted(templates_dir.glob("*.md")):
            post = frontmatter.load(path)
            assert post.metadata["name"]
            assert post.metadata["description"]
            assert "default_cadence" in post.metadata
            assert post.content.strip()


class TestTemplateCreate:
    def test_create_from_template(self, journal_path):
        result = runner.invoke(
            call_app,
            ["routines", "create", "--template", "morning-briefing"],
        )
        assert result.exit_code == 0
        config = get_config()
        assert len(config) == 1
        routine = next(iter(config.values()))
        assert routine["name"] == "morning-briefing"
        assert routine["cadence"] == "0 7 * * *"
        assert routine["template"] == "morning-briefing"
        assert "daily morning briefing" in routine["instruction"].lower()

    def test_create_template_with_overrides(self, journal_path):
        result = runner.invoke(
            call_app,
            [
                "routines",
                "create",
                "--template",
                "morning-briefing",
                "--cadence",
                "0 8 * * *",
                "--name",
                "My Briefing",
            ],
        )
        assert result.exit_code == 0
        config = get_config()
        routine = next(iter(config.values()))
        assert routine["name"] == "My Briefing"
        assert routine["cadence"] == "0 8 * * *"
        assert routine["template"] == "morning-briefing"

    def test_create_template_not_found(self, journal_path):
        result = runner.invoke(
            call_app,
            ["routines", "create", "--template", "nonexistent"],
        )
        assert result.exit_code == 1
        assert "template 'nonexistent' not found" in result.stderr

    def test_create_invalid_event_template_cadence(self, journal_path, monkeypatch):
        import think.tools.routines as routines_cli

        def _fake_template(name: str):
            return (
                {
                    "name": name,
                    "description": "bad template",
                    "default_cadence": {
                        "type": "event",
                        "trigger": "wrong",
                        "offset_minutes": -30,
                    },
                    "default_timezone": "UTC",
                    "default_facets": [],
                },
                "Instruction body",
            )

        monkeypatch.setattr(routines_cli, "_load_template", _fake_template)
        result = runner.invoke(
            call_app,
            ["routines", "create", "--template", "bad-template"],
        )
        assert result.exit_code == 1
        assert "trigger must be 'calendar'" in result.stderr


class TestEventTrigger:
    def _write_calendar_event(self, journal_path, day="20260327"):
        facet_cal_dir = journal_path / "facets" / "work" / "calendar"
        facet_cal_dir.mkdir(parents=True)
        (facet_cal_dir / f"{day}.jsonl").write_text(
            '{"title":"Standup","start":"10:00","end":"10:30","participants":["Alice","Bob"],"cancelled":false}\n',
            encoding="utf-8",
        )

    def _event_routine(self):
        return {
            "routine-1": {
                "id": "routine-1",
                "name": "Meeting prep",
                "instruction": "Prepare for the meeting",
                "cadence": {
                    "type": "event",
                    "trigger": "calendar",
                    "offset_minutes": -30,
                },
                "timezone": "UTC",
                "enabled": True,
                "facets": ["work"],
                "template": "meeting-prep",
                "notify": False,
                "last_run": None,
            }
        }

    def test_event_cadence_fires(self, journal_path):
        import think.routines as mod

        self._write_calendar_event(journal_path)
        save_config(self._event_routine())

        dt = datetime(2026, 3, 27, 9, 35, tzinfo=timezone.utc)
        with (
            patch(
                "think.routines.cortex_request", return_value="fake_agent_id"
            ) as mock_req,
            patch(
                "think.routines.wait_for_agents",
                return_value=({"fake_agent_id": "finish"}, []),
            ),
            patch("think.routines.callosum_send", return_value=True),
            _fake_now(dt),
        ):
            mod.check()

        mock_req.assert_called_once()

    def test_event_cadence_dedup(self, journal_path):
        import think.routines as mod

        self._write_calendar_event(journal_path)
        save_config(self._event_routine())

        dt = datetime(2026, 3, 27, 9, 35, tzinfo=timezone.utc)
        with (
            patch(
                "think.routines.cortex_request", return_value="fake_agent_id"
            ) as mock_req,
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

    def test_event_cadence_no_events(self, journal_path):
        import think.routines as mod

        save_config(self._event_routine())

        dt = datetime(2026, 3, 27, 9, 35, tzinfo=timezone.utc)
        with (
            patch(
                "think.routines.cortex_request", return_value="fake_agent_id"
            ) as mock_req,
            patch(
                "think.routines.wait_for_agents",
                return_value=({"fake_agent_id": "finish"}, []),
            ),
            patch("think.routines.callosum_send", return_value=True),
            _fake_now(dt),
        ):
            mod.check()

        mock_req.assert_not_called()

    def test_event_cadence_past_event(self, journal_path):
        import think.routines as mod

        self._write_calendar_event(journal_path)
        save_config(self._event_routine())

        dt = datetime(2026, 3, 27, 10, 30, tzinfo=timezone.utc)
        with (
            patch(
                "think.routines.cortex_request", return_value="fake_agent_id"
            ) as mock_req,
            patch(
                "think.routines.wait_for_agents",
                return_value=({"fake_agent_id": "finish"}, []),
            ),
            patch("think.routines.callosum_send", return_value=True),
            _fake_now(dt),
        ):
            mod.check()

        mock_req.assert_not_called()


class TestEventState:
    def test_events_state_persistence(self, journal_path):
        from think.routines import _load_events_state, _save_events_state

        state = {"routine-1": {"20260327:work:1", "20260327:work:2"}}
        _save_events_state(state)
        loaded = _load_events_state()
        assert loaded == state


class TestNameResolution:
    def test_resolve_by_name(self, journal_path):
        save_config(
            {
                "abc-123-def": {
                    "id": "abc-123-def",
                    "name": "Morning Briefing",
                    "instruction": "Brief me",
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
        result = runner.invoke(
            call_app, ["routines", "edit", "Morning Briefing", "--name", "Updated"]
        )
        assert result.exit_code == 0
        config = get_config()
        assert config["abc-123-def"]["name"] == "Updated"

    def test_resolve_by_name_case_insensitive(self, journal_path):
        save_config(
            {
                "abc-123-def": {
                    "id": "abc-123-def",
                    "name": "Morning Briefing",
                    "instruction": "Brief me",
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
        result = runner.invoke(
            call_app, ["routines", "edit", "morning briefing", "--name", "Updated"]
        )
        assert result.exit_code == 0
        config = get_config()
        assert config["abc-123-def"]["name"] == "Updated"

    def test_resolve_name_ambiguous(self, journal_path):
        save_config(
            {
                "abc-123": {
                    "id": "abc-123",
                    "name": "Daily",
                    "instruction": "a",
                    "cadence": "0 9 * * *",
                    "timezone": "UTC",
                    "enabled": True,
                    "facets": [],
                    "template": None,
                    "notify": False,
                    "last_run": None,
                },
                "def-456": {
                    "id": "def-456",
                    "name": "Daily",
                    "instruction": "b",
                    "cadence": "0 10 * * *",
                    "timezone": "UTC",
                    "enabled": True,
                    "facets": [],
                    "template": None,
                    "notify": False,
                    "last_run": None,
                },
            }
        )
        result = runner.invoke(call_app, ["routines", "edit", "Daily", "--name", "X"])
        assert result.exit_code == 1
        assert "ambiguous" in result.stderr.lower()

    def test_meta_excluded_from_resolve(self, journal_path):
        save_config(
            {
                "_meta": {"suggestions_enabled": True},
                "abc-123": {
                    "id": "abc-123",
                    "name": "Test",
                    "instruction": "test",
                    "cadence": "0 9 * * *",
                    "timezone": "UTC",
                    "enabled": True,
                    "facets": [],
                    "template": None,
                    "notify": False,
                    "last_run": None,
                },
            }
        )
        result = runner.invoke(
            call_app, ["routines", "edit", "abc", "--name", "Updated"]
        )
        assert result.exit_code == 0


class TestResumeDate:
    def test_edit_resume_date(self, journal_path):
        save_config(
            {
                "routine-1": {
                    "id": "routine-1",
                    "name": "Test",
                    "instruction": "test",
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
        result = runner.invoke(
            call_app,
            [
                "routines",
                "edit",
                "routine-1",
                "--enabled",
                "false",
                "--resume-date",
                "2026-04-01",
            ],
        )
        assert result.exit_code == 0
        config = get_config()
        assert config["routine-1"]["resume_date"] == "2026-04-01"

    def test_enable_clears_resume_date(self, journal_path):
        save_config(
            {
                "routine-1": {
                    "id": "routine-1",
                    "name": "Test",
                    "instruction": "test",
                    "cadence": "0 9 * * *",
                    "timezone": "UTC",
                    "enabled": False,
                    "facets": [],
                    "template": None,
                    "notify": False,
                    "last_run": None,
                    "resume_date": "2026-04-01",
                }
            }
        )
        result = runner.invoke(
            call_app, ["routines", "edit", "routine-1", "--enabled", "true"]
        )
        assert result.exit_code == 0
        config = get_config()
        assert config["routine-1"]["enabled"] is True
        assert "resume_date" not in config["routine-1"]

    def test_auto_resume(self, journal_path):
        import think.routines as mod

        save_config(
            {
                "routine-1": {
                    "id": "routine-1",
                    "name": "Test",
                    "instruction": "test",
                    "cadence": "0 9 * * *",
                    "timezone": "UTC",
                    "enabled": False,
                    "facets": [],
                    "template": None,
                    "notify": False,
                    "last_run": None,
                    "resume_date": "2026-03-27",
                }
            }
        )

        dt = datetime(2026, 3, 27, 10, 0, tzinfo=timezone.utc)
        with (
            patch("think.routines.cortex_request", return_value="fake_agent_id"),
            patch(
                "think.routines.wait_for_agents",
                return_value=({"fake_agent_id": "finish"}, []),
            ),
            patch("think.routines.callosum_send", return_value=True),
            _fake_now(dt),
        ):
            mod.check()

        config = get_config()
        assert config["routine-1"]["enabled"] is True
        assert "resume_date" not in config["routine-1"]
        health_log = (journal_path / "health" / "routines.log").read_text()
        assert "auto-resumed" in health_log

    def test_auto_resume_future_date_not_resumed(self, journal_path):
        import think.routines as mod

        save_config(
            {
                "routine-1": {
                    "id": "routine-1",
                    "name": "Test",
                    "instruction": "test",
                    "cadence": "0 9 * * *",
                    "timezone": "UTC",
                    "enabled": False,
                    "facets": [],
                    "template": None,
                    "notify": False,
                    "last_run": None,
                    "resume_date": "2026-04-01",
                }
            }
        )

        dt = datetime(2026, 3, 27, 10, 0, tzinfo=timezone.utc)
        with (
            patch("think.routines.cortex_request", return_value="fake_agent_id"),
            patch(
                "think.routines.wait_for_agents",
                return_value=({"fake_agent_id": "finish"}, []),
            ),
            patch("think.routines.callosum_send", return_value=True),
            _fake_now(dt),
        ):
            mod.check()

        config = get_config()
        assert config["routine-1"]["enabled"] is False
        assert config["routine-1"]["resume_date"] == "2026-04-01"

    def test_resume_date_invalid_format(self, journal_path):
        save_config(
            {
                "routine-1": {
                    "id": "routine-1",
                    "name": "Test",
                    "instruction": "test",
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
        result = runner.invoke(
            call_app, ["routines", "edit", "routine-1", "--resume-date", "not-a-date"]
        )
        assert result.exit_code == 1
        assert "YYYY-MM-DD" in result.stderr


class TestOutputByDate:
    def test_output_specific_date(self, journal_path):
        output_dir = journal_path / "routines" / "routine-1"
        output_dir.mkdir(parents=True)
        (output_dir / "20260325.md").write_text("March 25 output", encoding="utf-8")
        (output_dir / "20260326.md").write_text("March 26 output", encoding="utf-8")

        save_config(
            {
                "routine-1": {
                    "id": "routine-1",
                    "name": "Test",
                    "instruction": "test",
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
        result = runner.invoke(
            call_app, ["routines", "output", "routine-1", "--date", "2026-03-25"]
        )
        assert result.exit_code == 0
        assert "March 25 output" in result.stdout

    def test_output_date_missing(self, journal_path):
        output_dir = journal_path / "routines" / "routine-1"
        output_dir.mkdir(parents=True)
        (output_dir / "20260325.md").write_text("content", encoding="utf-8")

        save_config(
            {
                "routine-1": {
                    "id": "routine-1",
                    "name": "Test",
                    "instruction": "test",
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
        result = runner.invoke(
            call_app, ["routines", "output", "routine-1", "--date", "2026-03-27"]
        )
        assert result.exit_code == 0
        assert "No output for that date" in result.stdout

    def test_output_date_collision_file(self, journal_path):
        output_dir = journal_path / "routines" / "routine-1"
        output_dir.mkdir(parents=True)
        (output_dir / "20260325.md").write_text("first run", encoding="utf-8")
        (output_dir / "20260325-093000.md").write_text("second run", encoding="utf-8")

        save_config(
            {
                "routine-1": {
                    "id": "routine-1",
                    "name": "Test",
                    "instruction": "test",
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
        result = runner.invoke(
            call_app, ["routines", "output", "routine-1", "--date", "2026-03-25"]
        )
        assert result.exit_code == 0
        assert "second run" in result.stdout

    def test_output_default_no_date(self, journal_path):
        output_dir = journal_path / "routines" / "routine-1"
        output_dir.mkdir(parents=True)
        (output_dir / "20260325.md").write_text("old", encoding="utf-8")
        (output_dir / "20260326.md").write_text("latest", encoding="utf-8")

        save_config(
            {
                "routine-1": {
                    "id": "routine-1",
                    "name": "Test",
                    "instruction": "test",
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
        result = runner.invoke(call_app, ["routines", "output", "routine-1"])
        assert result.exit_code == 0
        assert "latest" in result.stdout


class TestSuggestions:
    def test_suggestions_read_default(self, journal_path):
        save_config({})
        result = runner.invoke(call_app, ["routines", "suggestions"])
        assert result.exit_code == 0
        assert "enabled" in result.stdout

    def test_suggestions_disable(self, journal_path):
        save_config({})
        result = runner.invoke(call_app, ["routines", "suggestions", "--disable"])
        assert result.exit_code == 0
        assert "disabled" in result.stdout
        config = get_config()
        assert config["_meta"]["suggestions_enabled"] is False

    def test_suggestions_enable(self, journal_path):
        save_config({"_meta": {"suggestions_enabled": False}})
        result = runner.invoke(call_app, ["routines", "suggestions", "--enable"])
        assert result.exit_code == 0
        assert "enabled" in result.stdout
        config = get_config()
        assert config["_meta"]["suggestions_enabled"] is True


class TestGetRoutineState:
    def test_basic_structure(self, journal_path):
        from think.routines import get_routine_state

        save_config(
            {
                "routine-1": {
                    "id": "routine-1",
                    "name": "Morning",
                    "cadence": "0 9 * * *",
                    "timezone": "UTC",
                    "enabled": True,
                    "facets": [],
                    "last_run": None,
                }
            }
        )
        state = get_routine_state()
        assert len(state) == 1
        assert state[0]["name"] == "Morning"
        assert state[0]["cadence"] == "0 9 * * *"
        assert state[0]["enabled"] is True
        assert state[0]["output_summary"] is None

    def test_recent_output_summary(self, journal_path):
        from think.routines import get_routine_state

        last_run = datetime(2026, 3, 27, 9, 0, tzinfo=timezone.utc).isoformat()
        save_config(
            {
                "routine-1": {
                    "id": "routine-1",
                    "name": "Morning",
                    "cadence": "0 9 * * *",
                    "timezone": "UTC",
                    "enabled": True,
                    "facets": [],
                    "last_run": last_run,
                }
            }
        )
        output_dir = journal_path / "routines" / "routine-1"
        output_dir.mkdir(parents=True)
        (output_dir / "20260327.md").write_text(
            "Here is the morning briefing summary for today.", encoding="utf-8"
        )

        dt = datetime(2026, 3, 27, 10, 0, tzinfo=timezone.utc)
        with _fake_now(dt):
            state = get_routine_state()

        assert len(state) == 1
        assert state[0]["output_summary"] is not None
        assert "morning briefing" in state[0]["output_summary"]

    def test_meta_excluded(self, journal_path):
        from think.routines import get_routine_state

        save_config(
            {
                "_meta": {"suggestions_enabled": True},
                "routine-1": {
                    "id": "routine-1",
                    "name": "Morning",
                    "cadence": "0 9 * * *",
                    "timezone": "UTC",
                    "enabled": True,
                    "facets": [],
                    "last_run": None,
                },
            }
        )
        state = get_routine_state()
        assert len(state) == 1
        assert state[0]["name"] == "Morning"

    def test_paused_until(self, journal_path):
        from think.routines import get_routine_state

        save_config(
            {
                "routine-1": {
                    "id": "routine-1",
                    "name": "Morning",
                    "cadence": "0 9 * * *",
                    "timezone": "UTC",
                    "enabled": False,
                    "facets": [],
                    "last_run": None,
                    "resume_date": "2026-04-01",
                }
            }
        )
        state = get_routine_state()
        assert state[0]["paused_until"] == "2026-04-01"
        assert state[0]["enabled"] is False


class TestMetaFiltering:
    def test_list_excludes_meta(self, journal_path):
        save_config(
            {
                "_meta": {"suggestions_enabled": True},
                "routine-1": {
                    "id": "routine-1",
                    "name": "Test",
                    "instruction": "test",
                    "cadence": "0 9 * * *",
                    "timezone": "UTC",
                    "enabled": True,
                    "facets": [],
                    "template": None,
                    "notify": False,
                    "last_run": None,
                },
            }
        )
        result = runner.invoke(call_app, ["routines", "list"])
        assert result.exit_code == 0
        assert "Test" in result.stdout
        assert "_meta" not in result.stdout
        assert "suggestions" not in result.stdout

    def test_check_skips_meta(self, journal_path):
        import think.routines as mod

        save_config(
            {
                "_meta": {"suggestions_enabled": True},
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
                },
            }
        )

        dt = datetime(2026, 3, 27, 9, 0, tzinfo=timezone.utc)
        with (
            patch(
                "think.routines.cortex_request", return_value="fake_agent_id"
            ) as mock_req,
            patch(
                "think.routines.wait_for_agents",
                return_value=({"fake_agent_id": "finish"}, []),
            ),
            patch("think.routines.callosum_send", return_value=True),
            _fake_now(dt),
        ):
            mod.check()

        mock_req.assert_called_once()
