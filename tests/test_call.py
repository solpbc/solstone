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
    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(journal))
    import think.utils

    think.utils._journal_path_cache = None
    yield journal
    think.utils._journal_path_cache = None


@pytest.fixture
def merge_journal(tmp_path, monkeypatch):
    """Create a journal with source and destination facets for merge testing."""
    journal = tmp_path / "journal"
    src_dir = journal / "facets" / "src-facet"
    dst_dir = journal / "facets" / "dst-facet"
    src_dir.mkdir(parents=True)
    dst_dir.mkdir(parents=True)

    (src_dir / "facet.json").write_text(
        json.dumps({"title": "Source Facet"}, indent=2) + "\n",
        encoding="utf-8",
    )
    (dst_dir / "facet.json").write_text(
        json.dumps({"title": "Destination Facet"}, indent=2) + "\n",
        encoding="utf-8",
    )

    config_dir = journal / "config"
    config_dir.mkdir(parents=True)
    (config_dir / "facets.json").write_text(
        json.dumps({"facets": ["src-facet", "dst-facet"]}, indent=2) + "\n",
        encoding="utf-8",
    )

    src_entity_dir = src_dir / "entities" / "test_entity"
    src_entity_dir.mkdir(parents=True)
    (src_entity_dir / "entity.json").write_text(
        json.dumps(
            {
                "entity_id": "test_entity",
                "description": "Source relationship",
                "created_at": 1,
                "updated_at": 1,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (src_entity_dir / "observations.jsonl").write_text(
        json.dumps({"content": "Knows the migration plan", "observed_at": 101}) + "\n",
        encoding="utf-8",
    )

    src_todos_dir = src_dir / "todos"
    src_todos_dir.mkdir(parents=True)
    (src_todos_dir / "20260101.jsonl").write_text(
        json.dumps({"text": "Move the roadmap", "created_at": 1000}) + "\n",
        encoding="utf-8",
    )

    src_news_dir = src_dir / "news"
    dst_news_dir = dst_dir / "news"
    src_news_dir.mkdir(parents=True)
    dst_news_dir.mkdir(parents=True)
    (src_news_dir / "20260101.md").write_text(
        "# Source News\nThis should be skipped.\n",
        encoding="utf-8",
    )
    (src_news_dir / "20260102.md").write_text(
        "# Unique Source News\nThis should be copied.\n",
        encoding="utf-8",
    )
    (dst_news_dir / "20260101.md").write_text(
        "# Destination News\nKeep this version.\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(journal))
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

    def test_journal_search(self):
        """Search command runs without error."""
        result = runner.invoke(call_app, ["journal", "search", "test", "--limit", "5"])
        assert result.exit_code == 0
        assert "results" in result.output

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
        journal = tmp_path / "journal"
        (journal / "facets" / "work").mkdir(parents=True)
        monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(journal))
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

    def test_facet_create_with_consent(self, facet_journal):
        """Create with --consent records consent=True in log entry."""
        from datetime import datetime

        result = runner.invoke(
            call_app, ["journal", "facet", "create", "Consent Facet", "--consent"]
        )
        assert result.exit_code == 0
        today = datetime.now().strftime("%Y%m%d")
        log_path = (
            facet_journal / "facets" / "consent-facet" / "logs" / f"{today}.jsonl"
        )
        assert log_path.exists()
        import json as _json

        entries = [
            _json.loads(line)
            for line in log_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert len(entries) == 1
        assert entries[0]["action"] == "facet_create"
        assert entries[0]["params"].get("consent") is True

    def test_facet_create_without_consent(self, facet_journal):
        """Create without --consent omits consent key from log entry."""
        from datetime import datetime

        result = runner.invoke(
            call_app, ["journal", "facet", "create", "No Consent Facet"]
        )
        assert result.exit_code == 0
        today = datetime.now().strftime("%Y%m%d")
        log_path = (
            facet_journal / "facets" / "no-consent-facet" / "logs" / f"{today}.jsonl"
        )
        assert log_path.exists()
        import json as _json

        entries = [
            _json.loads(line)
            for line in log_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert len(entries) == 1
        assert entries[0]["action"] == "facet_create"
        assert "consent" not in entries[0]["params"]

    def test_facet_rename_with_consent(self, facet_journal):
        """Rename with --consent records consent=True in log entry."""
        from datetime import datetime

        result = runner.invoke(
            call_app,
            [
                "journal",
                "facet",
                "rename",
                "test-facet",
                "renamed-consent",
                "--consent",
            ],
        )
        assert result.exit_code == 0
        today = datetime.now().strftime("%Y%m%d")
        log_path = (
            facet_journal / "facets" / "renamed-consent" / "logs" / f"{today}.jsonl"
        )
        assert log_path.exists()
        import json as _json

        entries = [
            _json.loads(line)
            for line in log_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert len(entries) == 1
        assert entries[0]["action"] == "facet_rename"
        assert entries[0]["params"].get("consent") is True

    def test_facet_delete_with_consent(self, facet_journal):
        """Delete with --consent records consent=True in journal-level log."""
        from datetime import datetime

        result = runner.invoke(
            call_app,
            ["journal", "facet", "delete", "test-facet", "--yes", "--consent"],
        )
        assert result.exit_code == 0
        today = datetime.now().strftime("%Y%m%d")
        log_path = facet_journal / "config" / "actions" / f"{today}.jsonl"
        assert log_path.exists()
        import json as _json

        entries = [
            _json.loads(line)
            for line in log_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert any(
            e["action"] == "facet_delete" and e["params"].get("consent") is True
            for e in entries
        )

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


class TestFacetMerge:
    """Tests for journal facet merge."""

    @staticmethod
    def _mock_indexer(monkeypatch):
        import think.tools.call as call_module

        calls = []

        def _run(*args, **kwargs):
            calls.append((args, kwargs))
            return None

        monkeypatch.setattr(call_module.subprocess, "run", _run)
        return calls

    def test_merge_moves_entities(self, merge_journal, monkeypatch):
        """Merge moves entity directories into the destination facet."""
        self._mock_indexer(monkeypatch)

        result = runner.invoke(
            call_app,
            ["journal", "facet", "merge", "src-facet", "--into", "dst-facet"],
        )

        assert result.exit_code == 0
        assert not (merge_journal / "facets" / "src-facet" / "entities").exists()
        assert (
            merge_journal
            / "facets"
            / "dst-facet"
            / "entities"
            / "test_entity"
            / "entity.json"
        ).exists()
        observations_path = (
            merge_journal
            / "facets"
            / "dst-facet"
            / "entities"
            / "test_entity"
            / "observations.jsonl"
        )
        assert observations_path.exists()

    def test_merge_entity_conflict(self, merge_journal, monkeypatch):
        """Merge preserves destination relationship fields and appends observations."""
        self._mock_indexer(monkeypatch)

        src_entity_dir = (
            merge_journal / "facets" / "src-facet" / "entities" / "test_entity"
        )
        dst_entity_dir = (
            merge_journal / "facets" / "dst-facet" / "entities" / "test_entity"
        )
        dst_entity_dir.mkdir(parents=True)
        (src_entity_dir / "entity.json").write_text(
            json.dumps(
                {
                    "entity_id": "test_entity",
                    "description": "Source desc",
                    "source_only": "keep-if-missing",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        (dst_entity_dir / "entity.json").write_text(
            json.dumps(
                {
                    "entity_id": "test_entity",
                    "description": "Dest desc",
                    "dest_only": "wins",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        (src_entity_dir / "observations.jsonl").write_text(
            json.dumps({"content": "Source fact", "observed_at": 101}) + "\n",
            encoding="utf-8",
        )
        (dst_entity_dir / "observations.jsonl").write_text(
            json.dumps({"content": "Dest fact", "observed_at": 202}) + "\n",
            encoding="utf-8",
        )

        result = runner.invoke(
            call_app,
            ["journal", "facet", "merge", "src-facet", "--into", "dst-facet"],
        )

        assert result.exit_code == 0
        assert dst_entity_dir.exists()
        assert not (merge_journal / "facets" / "src-facet" / "entities").exists()
        merged_relationship = json.loads(
            (dst_entity_dir / "entity.json").read_text(encoding="utf-8")
        )
        assert merged_relationship["description"] == "Dest desc"
        assert merged_relationship["dest_only"] == "wins"
        assert merged_relationship["source_only"] == "keep-if-missing"
        merged_observations = [
            json.loads(line)
            for line in (dst_entity_dir / "observations.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert len(merged_observations) == 2
        assert {obs["content"] for obs in merged_observations} == {
            "Source fact",
            "Dest fact",
        }

    def test_merge_moves_open_todos(self, merge_journal, monkeypatch):
        """Merge appends open todos to destination and cancels them in source."""
        self._mock_indexer(monkeypatch)
        import think.tools.call as call_module

        monkeypatch.setattr(call_module, "delete_facet", lambda *args, **kwargs: None)

        result = runner.invoke(
            call_app,
            ["journal", "facet", "merge", "src-facet", "--into", "dst-facet"],
        )

        assert result.exit_code == 0
        dst_todos = (
            (merge_journal / "facets" / "dst-facet" / "todos" / "20260101.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        )
        assert any(json.loads(line)["text"] == "Move the roadmap" for line in dst_todos)
        src_payloads = [
            json.loads(line)
            for line in (
                merge_journal / "facets" / "src-facet" / "todos" / "20260101.jsonl"
            )
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert src_payloads[0]["cancelled"] is True
        assert src_payloads[0]["cancelled_reason"] == "moved_to_facet"
        assert src_payloads[0]["moved_to"] == "dst-facet"

    def test_merge_copies_news_skips_conflicts(self, merge_journal, monkeypatch):
        """Merge copies unique news files and preserves destination conflicts."""
        self._mock_indexer(monkeypatch)

        result = runner.invoke(
            call_app,
            ["journal", "facet", "merge", "src-facet", "--into", "dst-facet"],
        )

        assert result.exit_code == 0
        assert (
            merge_journal / "facets" / "dst-facet" / "news" / "20260102.md"
        ).read_text(
            encoding="utf-8"
        ) == "# Unique Source News\nThis should be copied.\n"
        assert (
            merge_journal / "facets" / "dst-facet" / "news" / "20260101.md"
        ).read_text(encoding="utf-8") == "# Destination News\nKeep this version.\n"

    def test_merge_deletes_source_facet(self, merge_journal, monkeypatch):
        """Merge deletes the source facet after moving data."""
        self._mock_indexer(monkeypatch)

        result = runner.invoke(
            call_app,
            ["journal", "facet", "merge", "src-facet", "--into", "dst-facet"],
        )

        assert result.exit_code == 0
        assert not (merge_journal / "facets" / "src-facet").exists()

    def test_merge_logs_action(self, merge_journal, monkeypatch):
        """Merge records a journal-level facet_merge action with counts."""
        self._mock_indexer(monkeypatch)
        from datetime import datetime

        result = runner.invoke(
            call_app,
            ["journal", "facet", "merge", "src-facet", "--into", "dst-facet"],
        )

        assert result.exit_code == 0
        today = datetime.now().strftime("%Y%m%d")
        log_path = merge_journal / "config" / "actions" / f"{today}.jsonl"
        entries = [
            json.loads(line)
            for line in log_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        merge_entry = next(
            entry for entry in entries if entry["action"] == "facet_merge"
        )
        assert merge_entry["params"]["source"] == "src-facet"
        assert merge_entry["params"]["dest"] == "dst-facet"
        assert merge_entry["params"]["entity_count"] == 1
        assert merge_entry["params"]["todo_count"] == 1
        assert merge_entry["params"]["news_count"] == 1

    def test_merge_same_facet_error(self, merge_journal):
        """Merge rejects using the same facet as source and destination."""
        result = runner.invoke(
            call_app,
            ["journal", "facet", "merge", "src-facet", "--into", "src-facet"],
        )

        assert result.exit_code == 1
        assert "Source and destination facets must be different" in result.output

    def test_merge_missing_source_error(self, merge_journal):
        """Merge rejects a missing source facet."""
        result = runner.invoke(
            call_app,
            ["journal", "facet", "merge", "missing-facet", "--into", "dst-facet"],
        )

        assert result.exit_code == 1
        assert "Error: Facet 'missing-facet' not found." in result.output

    def test_merge_missing_dest_error(self, merge_journal):
        """Merge rejects a missing destination facet."""
        result = runner.invoke(
            call_app,
            ["journal", "facet", "merge", "src-facet", "--into", "missing-facet"],
        )

        assert result.exit_code == 1
        assert "Error: Facet 'missing-facet' not found." in result.output

    def test_merge_consent_flag_logged(self, merge_journal, monkeypatch):
        """Merge records consent=True when requested."""
        self._mock_indexer(monkeypatch)
        from datetime import datetime

        result = runner.invoke(
            call_app,
            [
                "journal",
                "facet",
                "merge",
                "src-facet",
                "--into",
                "dst-facet",
                "--consent",
            ],
        )

        assert result.exit_code == 0
        today = datetime.now().strftime("%Y%m%d")
        log_path = merge_journal / "config" / "actions" / f"{today}.jsonl"
        entries = [
            json.loads(line)
            for line in log_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        merge_entry = next(
            entry for entry in entries if entry["action"] == "facet_merge"
        )
        assert merge_entry["params"]["consent"] is True


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
