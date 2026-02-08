# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for think/call.py CLI dispatcher and app discovery."""

from typer.testing import CliRunner

from think.call import call_app

runner = CliRunner()


class TestDiscovery:
    """Tests for app CLI discovery."""

    def test_no_args_shows_help(self):
        """Running 'sol call' with no args shows help."""
        result = runner.invoke(call_app, [])
        assert "Call app functions" in result.output

    def test_todos_app_discovered(self):
        """The todos app should be auto-discovered."""
        result = runner.invoke(call_app, ["todos", "--help"])
        assert result.exit_code == 0
        assert "list" in result.output

    def test_unknown_app_fails(self):
        """Unknown app name should produce an error."""
        result = runner.invoke(call_app, ["nonexistent"])
        assert result.exit_code != 0


class TestJournal:
    """Tests for 'sol call journal' commands."""

    def test_journal_app_discovered(self):
        """Journal sub-app is registered and shows help."""
        result = runner.invoke(call_app, ["journal", "--help"])
        assert result.exit_code == 0
        assert "search" in result.output
        assert "events" in result.output
        assert "facet" in result.output
        assert "news" in result.output

    def test_journal_search(self):
        """Search command runs without error."""
        result = runner.invoke(call_app, ["journal", "search", "test", "--limit", "5"])
        assert result.exit_code == 0
        assert "results" in result.output

    def test_journal_events(self):
        """Events command returns fixture data."""
        result = runner.invoke(call_app, ["journal", "events", "20240101"])
        assert result.exit_code == 0
        # Fixture has work + personal events for this day
        assert "Team standup" in result.output

    def test_journal_events_with_facet(self):
        """Events command filters by facet."""
        result = runner.invoke(
            call_app, ["journal", "events", "20240101", "--facet", "work"]
        )
        assert result.exit_code == 0
        assert "Team standup" in result.output

    def test_journal_facet(self):
        """Facet command shows summary for test-facet."""
        result = runner.invoke(call_app, ["journal", "facet", "test-facet"])
        assert result.exit_code == 0
        assert "Test Facet" in result.output

    def test_journal_news(self):
        """News command reads fixture news."""
        result = runner.invoke(
            call_app, ["journal", "news", "work", "--day", "20240101"]
        )
        assert result.exit_code == 0
        assert "Authentication" in result.output
