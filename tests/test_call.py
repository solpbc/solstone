# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for think/call.py CLI dispatcher and app discovery."""

import typer
from typer.testing import CliRunner

from think.call import call_app
from think.utils import resolve_sol_day, resolve_sol_facet, resolve_sol_segment

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
        for cmd in ("search", "events", "facet", "facets", "news", "topics", "read"):
            assert cmd in result.output

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

    def test_journal_search_shows_counts(self):
        """Search output includes facet/topic/day counts."""
        result = runner.invoke(call_app, ["journal", "search", ""])
        assert result.exit_code == 0
        assert "results" in result.output
        # Counts lines should appear when there are results
        output = result.output
        if "0 results" not in output:
            assert "Facets:" in output or "Topics:" in output

    def test_journal_events_shows_details(self):
        """Events output includes participants and details."""
        result = runner.invoke(call_app, ["journal", "events", "20240101"])
        assert result.exit_code == 0
        assert "Participants:" in result.output
        assert "Alice" in result.output
        assert "Details:" in result.output

    def test_journal_facets(self):
        """Facets command lists available facets."""
        result = runner.invoke(call_app, ["journal", "facets"])
        assert result.exit_code == 0
        assert "test-facet" in result.output

    def test_journal_topics(self):
        """Topics command lists agent outputs for a day."""
        result = runner.invoke(call_app, ["journal", "topics", "20240101"])
        assert result.exit_code == 0
        assert "flow.md" in result.output

    def test_journal_topics_no_data(self):
        """Topics command reports no data for missing day."""
        result = runner.invoke(call_app, ["journal", "topics", "19990101"])
        assert result.exit_code == 0
        assert "No data" in result.output

    def test_journal_read(self):
        """Read command returns full agent output content."""
        result = runner.invoke(
            call_app, ["journal", "read", "flow", "--day", "20240101"]
        )
        assert result.exit_code == 0
        assert len(result.output.strip()) > 0

    def test_journal_read_max_truncates(self):
        """Read command truncates output when --max is exceeded."""
        # flow.md is ~422 bytes; --max 50 should truncate
        result = runner.invoke(
            call_app, ["journal", "read", "flow", "--day", "20240101", "--max", "50"]
        )
        assert result.exit_code == 0
        # Output should be much shorter than the full file
        assert len(result.output.encode("utf-8")) < 200

    def test_journal_read_max_zero_unlimited(self):
        """Read command with --max 0 returns full content."""
        result = runner.invoke(
            call_app, ["journal", "read", "flow", "--day", "20240101", "--max", "0"]
        )
        assert result.exit_code == 0
        assert len(result.output.strip()) > 100

    def test_journal_read_not_found(self):
        """Read command reports missing topic."""
        result = runner.invoke(
            call_app, ["journal", "read", "nonexistent", "--day", "20240101"]
        )
        assert result.exit_code == 1
        assert "not found" in result.output.lower()

    def test_journal_news_write(self, tmp_path, monkeypatch):
        """News --write saves content from stdin."""
        import shutil

        # Copy fixtures to tmp so we can write
        journal = tmp_path / "journal"
        shutil.copytree(
            "tests/fixtures/journal/facets/work", journal / "facets" / "work"
        )
        monkeypatch.setenv("JOURNAL_PATH", str(journal))
        # Clear cached journal path
        import think.utils

        think.utils._journal_path_cache = None

        content = "# Test News\nSome content here."
        result = runner.invoke(
            call_app,
            ["journal", "news", "work", "--day", "20260208", "--write"],
            input=content,
        )
        assert result.exit_code == 0
        assert "saved" in result.output.lower()

        # Verify file was written
        news_file = journal / "facets" / "work" / "news" / "20260208.md"
        assert news_file.exists()
        assert news_file.read_text() == content

        # Reset cache
        think.utils._journal_path_cache = None

    def test_journal_news_write_requires_day(self):
        """News --write fails without --day."""
        result = runner.invoke(
            call_app,
            ["journal", "news", "work", "--write"],
            input="content",
        )
        assert result.exit_code == 1
        assert "day is required" in result.output


class TestResolveHelpers:
    """Unit tests for SOL_* resolve helpers in think/utils.py."""

    def test_resolve_sol_day_from_env(self, monkeypatch):
        """resolve_sol_day(None) with SOL_DAY set returns env value."""
        monkeypatch.setenv("SOL_DAY", "20240101")
        assert resolve_sol_day(None) == "20240101"

    def test_resolve_sol_day_arg_wins(self, monkeypatch):
        """resolve_sol_day with explicit arg ignores env."""
        monkeypatch.setenv("SOL_DAY", "20240101")
        assert resolve_sol_day("20260115") == "20260115"

    def test_resolve_sol_day_missing_exits(self, monkeypatch):
        """resolve_sol_day(None) with no env raises SystemExit."""
        monkeypatch.delenv("SOL_DAY", raising=False)
        try:
            resolve_sol_day(None)
            assert False, "Expected typer.Exit"
        except (typer.Exit, SystemExit):
            pass

    def test_resolve_sol_facet_from_env(self, monkeypatch):
        """resolve_sol_facet(None) with SOL_FACET set returns env value."""
        monkeypatch.setenv("SOL_FACET", "work")
        assert resolve_sol_facet(None) == "work"

    def test_resolve_sol_facet_arg_wins(self, monkeypatch):
        """resolve_sol_facet with explicit arg ignores env."""
        monkeypatch.setenv("SOL_FACET", "work")
        assert resolve_sol_facet("personal") == "personal"

    def test_resolve_sol_facet_missing_exits(self, monkeypatch):
        """resolve_sol_facet(None) with no env raises SystemExit."""
        monkeypatch.delenv("SOL_FACET", raising=False)
        try:
            resolve_sol_facet(None)
            assert False, "Expected typer.Exit"
        except (typer.Exit, SystemExit):
            pass

    def test_resolve_sol_segment_from_env(self, monkeypatch):
        """resolve_sol_segment(None) with SOL_SEGMENT returns env value."""
        monkeypatch.setenv("SOL_SEGMENT", "123456_300")
        assert resolve_sol_segment(None) == "123456_300"

    def test_resolve_sol_segment_arg_wins(self, monkeypatch):
        """resolve_sol_segment with explicit arg ignores env."""
        monkeypatch.setenv("SOL_SEGMENT", "123456_300")
        assert resolve_sol_segment("654321_600") == "654321_600"

    def test_resolve_sol_segment_missing_returns_none(self, monkeypatch):
        """resolve_sol_segment(None) with no env returns None."""
        monkeypatch.delenv("SOL_SEGMENT", raising=False)
        assert resolve_sol_segment(None) is None


class TestJournalSolEnv:
    """Tests for journal commands resolving SOL_* env vars."""

    def test_events_from_sol_day(self, monkeypatch):
        """events with SOL_DAY env and no arg works."""
        monkeypatch.setenv("SOL_DAY", "20240101")
        result = runner.invoke(call_app, ["journal", "events"])
        assert result.exit_code == 0
        assert "Team standup" in result.output

    def test_events_arg_overrides_env(self, monkeypatch):
        """events with both env and arg — arg wins."""
        monkeypatch.setenv("SOL_DAY", "19990101")
        result = runner.invoke(call_app, ["journal", "events", "20240101"])
        assert result.exit_code == 0
        assert "Team standup" in result.output

    def test_events_no_day_exits(self, monkeypatch):
        """events with neither arg nor env exits with error."""
        monkeypatch.delenv("SOL_DAY", raising=False)
        result = runner.invoke(call_app, ["journal", "events"])
        assert result.exit_code != 0

    def test_topics_from_sol_day(self, monkeypatch):
        """topics with SOL_DAY env and no arg works."""
        monkeypatch.setenv("SOL_DAY", "20240101")
        result = runner.invoke(call_app, ["journal", "topics"])
        assert result.exit_code == 0
        assert "flow.md" in result.output

    def test_read_from_sol_day(self, monkeypatch):
        """read with SOL_DAY env and no --day works."""
        monkeypatch.setenv("SOL_DAY", "20240101")
        result = runner.invoke(call_app, ["journal", "read", "flow"])
        assert result.exit_code == 0
        assert len(result.output.strip()) > 0

    def test_read_arg_overrides_sol_day(self, monkeypatch):
        """read with explicit --day works even with SOL_DAY set."""
        monkeypatch.setenv("SOL_DAY", "19990101")
        result = runner.invoke(
            call_app, ["journal", "read", "flow", "--day", "20240101"]
        )
        assert result.exit_code == 0
        assert len(result.output.strip()) > 0
