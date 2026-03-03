# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for think/call.py CLI dispatcher and app discovery."""

import json

import pytest
import typer
from typer.testing import CliRunner

from think.call import call_app
from think.utils import resolve_sol_day, resolve_sol_facet, resolve_sol_segment

runner = CliRunner()


@pytest.fixture
def facet_journal(tmp_path, monkeypatch):
    """Create a journal with test facets for CRUD testing."""
    journal = tmp_path / "journal"
    # Create a facet with full metadata
    facet_dir = journal / "facets" / "test-facet"
    facet_dir.mkdir(parents=True)
    (facet_dir / "facet.json").write_text(
        json.dumps(
            {
                "title": "Test Facet",
                "description": "A test",
                "emoji": "🧪",
                "color": "#007bff",
            }
        ),
        encoding="utf-8",
    )
    # Create a muted facet
    muted_dir = journal / "facets" / "muted-one"
    muted_dir.mkdir(parents=True)
    (muted_dir / "facet.json").write_text(
        json.dumps({"title": "Muted Facet", "muted": True}),
        encoding="utf-8",
    )
    monkeypatch.setenv("JOURNAL_PATH", str(journal))
    import think.utils

    think.utils._journal_path_cache = None
    yield journal
    think.utils._journal_path_cache = None


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
        for cmd in ("search", "events", "facet", "facets", "news", "agents", "read"):
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
        result = runner.invoke(call_app, ["journal", "facet", "show", "test-facet"])
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
        """Search output includes facet/agent/day counts."""
        result = runner.invoke(call_app, ["journal", "search", ""])
        assert result.exit_code == 0
        assert "results" in result.output
        # Counts lines should appear when there are results
        output = result.output
        if "0 results" not in output:
            assert "Facets:" in output or "Agents:" in output

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

    def test_journal_agents(self):
        """Agents command lists agent outputs for a day."""
        result = runner.invoke(call_app, ["journal", "agents", "20240101"])
        assert result.exit_code == 0
        assert "flow.md" in result.output

    def test_journal_agents_no_data(self):
        """Agents command reports no data for missing day."""
        result = runner.invoke(call_app, ["journal", "agents", "19990101"])
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
        """Read command reports missing agent output."""
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


class TestFacetCRUD:
    """Tests for journal facet CRUD and listing commands."""

    def test_facet_show(self, facet_journal):
        """Facet show displays the requested facet."""
        result = runner.invoke(call_app, ["journal", "facet", "show", "test-facet"])
        assert result.exit_code == 0
        assert "Test Facet" in result.output

    def test_facet_show_not_found(self, facet_journal):
        """Facet show returns error for missing facet."""
        result = runner.invoke(call_app, ["journal", "facet", "show", "nonexistent"])
        assert result.exit_code == 1

    def test_facet_create(self, facet_journal):
        """Create creates a new facet with defaults."""
        result = runner.invoke(call_app, ["journal", "facet", "create", "My Project"])
        assert result.exit_code == 0
        assert "my-project" in result.output
        facet_file = facet_journal / "facets" / "my-project" / "facet.json"
        assert facet_file.exists()
        payload = json.loads(facet_file.read_text(encoding="utf-8"))
        assert payload["title"] == "My Project"
        assert payload["emoji"] == "📦"
        assert payload["color"] == "#667eea"

    def test_facet_create_with_options(self, facet_journal):
        """Create respects provided emoji, color, and description."""
        result = runner.invoke(
            call_app,
            [
                "journal",
                "facet",
                "create",
                "Cool Project",
                "--emoji",
                "🎯",
                "--color",
                "#ff0000",
                "--description",
                "A cool project",
            ],
        )
        assert result.exit_code == 0
        facet_file = facet_journal / "facets" / "cool-project" / "facet.json"
        payload = json.loads(facet_file.read_text(encoding="utf-8"))
        assert payload["title"] == "Cool Project"
        assert payload["emoji"] == "🎯"
        assert payload["color"] == "#ff0000"
        assert payload["description"] == "A cool project"

    def test_facet_create_duplicate(self, facet_journal):
        """Create rejects duplicate slugs."""
        result = runner.invoke(call_app, ["journal", "facet", "create", "Test Facet"])
        assert result.exit_code == 1
        assert "already exists" in result.output

    def test_facet_update(self, facet_journal):
        """Update modifies allowed fields and reports changed fields."""
        result = runner.invoke(
            call_app,
            [
                "journal",
                "facet",
                "update",
                "test-facet",
                "--description",
                "New desc",
                "--emoji",
                "🎯",
            ],
        )
        assert result.exit_code == 0
        assert "Updated" in result.output
        facet_file = facet_journal / "facets" / "test-facet" / "facet.json"
        payload = json.loads(facet_file.read_text(encoding="utf-8"))
        assert payload["description"] == "New desc"
        assert payload["emoji"] == "🎯"

    def test_facet_update_not_found(self, facet_journal):
        """Update fails when facet does not exist."""
        result = runner.invoke(
            call_app,
            [
                "journal",
                "facet",
                "update",
                "nonexistent",
                "--title",
                "X",
            ],
        )
        assert result.exit_code == 1

    def test_facet_update_no_fields(self, facet_journal):
        """Update requires at least one updatable field."""
        result = runner.invoke(call_app, ["journal", "facet", "update", "test-facet"])
        assert result.exit_code == 1
        assert "No fields" in result.output

    def test_facet_rename(self, facet_journal):
        """Rename moves facet path."""
        result = runner.invoke(
            call_app,
            ["journal", "facet", "rename", "test-facet", "renamed-facet"],
        )
        assert result.exit_code == 0
        assert not (facet_journal / "facets" / "test-facet").exists()
        assert (facet_journal / "facets" / "renamed-facet").is_dir()

    def test_facet_rename_invalid(self, facet_journal):
        """Rename fails with invalid new facet name."""
        result = runner.invoke(
            call_app,
            ["journal", "facet", "rename", "test-facet", "INVALID"],
        )
        assert result.exit_code == 1

    def test_facet_mute(self, facet_journal):
        """Mute sets muted=true in facet metadata."""
        result = runner.invoke(call_app, ["journal", "facet", "mute", "test-facet"])
        assert result.exit_code == 0
        assert "muted" in result.output.lower()
        payload = json.loads(
            (facet_journal / "facets" / "test-facet" / "facet.json").read_text(
                encoding="utf-8"
            )
        )
        assert payload.get("muted") is True

    def test_facet_unmute(self, facet_journal):
        """Unmute removes muted field from metadata."""
        result = runner.invoke(call_app, ["journal", "facet", "unmute", "muted-one"])
        assert result.exit_code == 0
        assert "unmuted" in result.output.lower()
        payload = json.loads(
            (facet_journal / "facets" / "muted-one" / "facet.json").read_text(
                encoding="utf-8"
            )
        )
        assert payload.get("muted", False) is False

    def test_facet_mute_not_found(self, facet_journal):
        """Mute fails for missing facet."""
        result = runner.invoke(call_app, ["journal", "facet", "mute", "nonexistent"])
        assert result.exit_code == 1

    def test_facet_delete_no_confirm(self, facet_journal):
        """Delete requires --yes confirmation."""
        result = runner.invoke(call_app, ["journal", "facet", "delete", "test-facet"])
        assert result.exit_code == 1
        assert "permanently delete" in result.output

    def test_facet_delete_with_yes(self, facet_journal):
        """Delete removes facet directory."""
        result = runner.invoke(
            call_app,
            ["journal", "facet", "delete", "test-facet", "--yes"],
        )
        assert result.exit_code == 0
        assert "Deleted" in result.output
        assert not (facet_journal / "facets" / "test-facet").exists()

    def test_facet_delete_not_found(self, facet_journal):
        """Delete fails for nonexistent facet."""
        result = runner.invoke(
            call_app,
            ["journal", "facet", "delete", "nonexistent", "--yes"],
        )
        assert result.exit_code == 1

    def test_facets_list_shows_metadata(self, facet_journal):
        """facets lists unmuted facets with metadata."""
        result = runner.invoke(call_app, ["journal", "facets"])
        assert result.exit_code == 0
        assert "🧪" in result.output
        assert "test-facet" in result.output
        assert "muted-one" not in result.output

    def test_facets_list_all(self, facet_journal):
        """facets --all includes muted facets."""
        result = runner.invoke(call_app, ["journal", "facets", "--all"])
        assert result.exit_code == 0
        assert "muted-one" in result.output
        assert "[muted]" in result.output


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

    def test_agents_from_sol_day(self, monkeypatch):
        """agents with SOL_DAY env and no arg works."""
        monkeypatch.setenv("SOL_DAY", "20240101")
        result = runner.invoke(call_app, ["journal", "agents"])
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
