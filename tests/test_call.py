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
        result = runner.invoke(call_app, ["journal", "read", "20240101", "flow"])
        assert result.exit_code == 0
        assert len(result.output.strip()) > 0

    def test_journal_read_not_found(self):
        """Read command reports missing topic."""
        result = runner.invoke(call_app, ["journal", "read", "20240101", "nonexistent"])
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
        assert "--day" in result.output
