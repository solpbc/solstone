# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for calendar CLI commands (``sol call calendar ...``)."""

from __future__ import annotations

from typer.testing import CliRunner

from think.call import call_app

runner = CliRunner()


class TestCalendarList:
    """Tests for ``sol call calendar list`` command."""

    def test_list_with_facet(self, calendar_env):
        """List events for a single day with --facet."""
        calendar_env(
            [{"title": "Team standup", "start": "09:00", "end": "09:30"}],
            day="20240101",
            facet="work",
        )

        result = runner.invoke(
            call_app,
            ["calendar", "list", "20240101", "--facet", "work"],
        )

        assert result.exit_code == 0
        assert "1: 09:00-09:30 Team standup" in result.output

    def test_list_empty(self, calendar_env):
        """Empty day shows no-events message."""
        calendar_env([], day="20240101", facet="work")

        result = runner.invoke(
            call_app,
            ["calendar", "list", "20240101", "--facet", "work"],
        )

        assert result.exit_code == 0
        assert "No events found" in result.output

    def test_list_missing_file(self, calendar_env):
        """Missing day file (no JSONL) shows no-events message."""
        calendar_env(None, day="20240101", facet="work")

        result = runner.invoke(
            call_app,
            ["calendar", "list", "20240101", "--facet", "work"],
        )

        assert result.exit_code == 0
        assert "No events found" in result.output

    def test_list_all_facets(self, calendar_env, monkeypatch):
        """List events across all facets when --facet is omitted."""
        calendar_env(
            [{"title": "Work sync", "start": "09:00"}],
            day="20240101",
            facet="work",
        )
        calendar_env(
            [{"title": "Gym", "start": "18:00"}],
            day="20240101",
            facet="personal",
        )
        monkeypatch.delenv("SOL_FACET", raising=False)

        result = runner.invoke(call_app, ["calendar", "list", "20240101"])

        assert result.exit_code == 0
        assert "Work sync" in result.output
        assert "Gym" in result.output

    def test_list_shows_cancelled(self, calendar_env):
        """List includes cancelled events in strikethrough format."""
        calendar_env(
            [{"title": "Cancelled meeting", "start": "14:00", "cancelled": True}],
            day="20240101",
            facet="work",
        )

        result = runner.invoke(
            call_app,
            ["calendar", "list", "20240101", "--facet", "work"],
        )

        assert result.exit_code == 0
        assert "~~14:00 Cancelled meeting~~" in result.output


class TestCalendarCreate:
    """Tests for ``sol call calendar create`` command."""

    def test_create_basic(self, calendar_env):
        """Create an event with title and start."""
        calendar_env([], day="20240101", facet="work")

        result = runner.invoke(
            call_app,
            [
                "calendar",
                "create",
                "Team standup",
                "--start",
                "09:00",
                "--day",
                "20240101",
                "--facet",
                "work",
            ],
        )

        assert result.exit_code == 0
        assert "09:00 Team standup" in result.output

    def test_create_with_all_options(self, calendar_env):
        """Create an event with all optional fields."""
        calendar_env([], day="20240101", facet="work")

        result = runner.invoke(
            call_app,
            [
                "calendar",
                "create",
                "Planning",
                "--start",
                "10:00",
                "--end",
                "11:00",
                "--summary",
                "Sprint planning",
                "--participants",
                "Alice, Bob",
                "--day",
                "20240101",
                "--facet",
                "work",
            ],
        )

        assert result.exit_code == 0
        assert "10:00-11:00 Planning" in result.output

    def test_create_invalid_time(self, calendar_env):
        """Invalid start time format fails."""
        calendar_env([], day="20240101", facet="work")

        result = runner.invoke(
            call_app,
            [
                "calendar",
                "create",
                "Bad event",
                "--start",
                "25:00",
                "--day",
                "20240101",
                "--facet",
                "work",
            ],
        )

        assert result.exit_code == 1
        assert "invalid time format" in result.output

    def test_create_end_before_start(self, calendar_env):
        """End time before start fails validation."""
        calendar_env([], day="20240101", facet="work")

        result = runner.invoke(
            call_app,
            [
                "calendar",
                "create",
                "Backwards event",
                "--start",
                "11:00",
                "--end",
                "10:00",
                "--day",
                "20240101",
                "--facet",
                "work",
            ],
        )

        assert result.exit_code == 1
        assert "end time must be greater than or equal to start time" in result.output

    def test_create_empty_title(self, calendar_env):
        """Creating with empty title fails."""
        calendar_env([], day="20240101", facet="work")

        result = runner.invoke(
            call_app,
            [
                "calendar",
                "create",
                "   ",
                "--start",
                "09:00",
                "--day",
                "20240101",
                "--facet",
                "work",
            ],
        )

        assert result.exit_code == 1
        assert "event title cannot be empty" in result.output


class TestCalendarUpdate:
    """Tests for ``sol call calendar update`` command."""

    def test_update_title(self, calendar_env):
        """Update event title."""
        calendar_env(
            [{"title": "Old title", "start": "09:00"}],
            day="20240101",
            facet="work",
        )

        result = runner.invoke(
            call_app,
            [
                "calendar",
                "update",
                "1",
                "--title",
                "New title",
                "--day",
                "20240101",
                "--facet",
                "work",
            ],
        )

        assert result.exit_code == 0
        assert "New title" in result.output

    def test_update_start_time(self, calendar_env):
        """Update event start time."""
        calendar_env(
            [{"title": "Standup", "start": "09:00"}],
            day="20240101",
            facet="work",
        )

        result = runner.invoke(
            call_app,
            [
                "calendar",
                "update",
                "1",
                "--start",
                "10:00",
                "--day",
                "20240101",
                "--facet",
                "work",
            ],
        )

        assert result.exit_code == 0
        assert "10:00 Standup" in result.output

    def test_update_nonexistent(self, calendar_env):
        """Updating a missing entry fails."""
        calendar_env([], day="20240101", facet="work")

        result = runner.invoke(
            call_app,
            [
                "calendar",
                "update",
                "1",
                "--title",
                "Nope",
                "--day",
                "20240101",
                "--facet",
                "work",
            ],
        )

        assert result.exit_code == 1
        assert "out of range" in result.output

    def test_update_without_fields_updates_timestamp_only(self, calendar_env):
        """Update with no options still succeeds and preserves event content."""
        calendar_env(
            [{"title": "Standup", "start": "09:00"}],
            day="20240101",
            facet="work",
        )

        result = runner.invoke(
            call_app,
            ["calendar", "update", "1", "--day", "20240101", "--facet", "work"],
        )

        assert result.exit_code == 0
        assert "1: 09:00 Standup" in result.output


class TestCalendarCancel:
    """Tests for ``sol call calendar cancel`` command."""

    def test_cancel_event(self, calendar_env):
        """Cancel an event."""
        calendar_env(
            [{"title": "Standup", "start": "09:00"}],
            day="20240101",
            facet="work",
        )

        result = runner.invoke(
            call_app,
            ["calendar", "cancel", "1", "--day", "20240101", "--facet", "work"],
        )

        assert result.exit_code == 0
        assert "~~09:00 Standup~~" in result.output

    def test_cancel_nonexistent(self, calendar_env):
        """Cancelling a missing entry fails."""
        calendar_env([], day="20240101", facet="work")

        result = runner.invoke(
            call_app,
            ["calendar", "cancel", "1", "--day", "20240101", "--facet", "work"],
        )

        assert result.exit_code == 1
        assert "out of range" in result.output


class TestCalendarEnvResolution:
    """Tests SOL_* env var resolution in calendar commands."""

    def test_uses_sol_day_env(self, calendar_env):
        """Create without --day uses SOL_DAY."""
        day, facet, calendar_path = calendar_env([], day="20250101", facet="work")

        result = runner.invoke(
            call_app,
            [
                "calendar",
                "create",
                "Env day event",
                "--start",
                "09:00",
                "--facet",
                facet,
            ],
        )

        assert result.exit_code == 0
        assert calendar_path.is_file()
        assert day in str(calendar_path)

    def test_uses_sol_facet_env(self, calendar_env, monkeypatch):
        """Create without --facet uses SOL_FACET."""
        day, _facet, _path = calendar_env([], day="20250102", facet="work")
        _, _, personal_path = calendar_env([], day=day, facet="personal")
        monkeypatch.setenv("SOL_FACET", "personal")

        result = runner.invoke(
            call_app,
            ["calendar", "create", "Env facet event", "--start", "10:00", "--day", day],
        )

        assert result.exit_code == 0
        assert personal_path.is_file()
        assert "Env facet event" in personal_path.read_text(encoding="utf-8")
