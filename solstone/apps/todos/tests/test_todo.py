# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for the facet-scoped todo checklist system."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from solstone.apps.todos.todo import (
    TodoChecklist,
    TodoItem,
    find_cross_facet_matches,
    get_facets_with_todos,
    get_todos,
    upcoming,
)


@pytest.fixture
def journal_root(tmp_path):
    path = tmp_path / "journal"
    path.mkdir()
    return path


def _write_todos(root: Path, facet: str, day: str, items: list[dict]) -> Path:
    """Write todos to facets/{facet}/todos/{day}.jsonl"""
    todos_dir = root / "facets" / facet / "todos"
    todos_dir.mkdir(parents=True, exist_ok=True)
    todo_path = todos_dir / f"{day}.jsonl"
    lines = [json.dumps(item, ensure_ascii=False) for item in items]
    todo_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return todo_path


def test_get_todos_returns_none_when_missing(monkeypatch, journal_root):
    monkeypatch.setenv("SOLSTONE_JOURNAL", str(journal_root))
    assert get_todos("20240101", "personal") is None


def test_get_todos_parses_basic_fields(monkeypatch, journal_root):
    monkeypatch.setenv("SOLSTONE_JOURNAL", str(journal_root))
    _write_todos(
        journal_root,
        "personal",
        "20240102",
        [
            {"text": "Merge analytics PR", "nudge": "20240102T10:30"},
            {"text": "Project sync", "completed": True},  # no nudge
            {"text": "Write retrospective notes"},
        ],
    )

    todos = get_todos("20240102", "personal")
    assert todos is not None
    assert len(todos) == 3

    first = todos[0]
    assert first["index"] == 1
    assert first["nudge"] == "20240102T10:30"
    assert first["completed"] is False
    assert first["text"] == "Merge analytics PR"

    second = todos[1]
    assert second["completed"] is True
    assert second["nudge"] is None
    assert second["text"] == "Project sync"

    third = todos[2]
    assert third["text"] == "Write retrospective notes"
    assert third["index"] == 3


def test_get_todos_handles_cancelled(monkeypatch, journal_root):
    monkeypatch.setenv("SOLSTONE_JOURNAL", str(journal_root))
    _write_todos(
        journal_root,
        "work",
        "20240103",
        [
            {"text": "Optional experiment", "cancelled": True},
            {"text": "Address bug", "nudge": "20240103T14:45"},
            {"text": "Draft report", "completed": True},
        ],
    )

    todos = get_todos("20240103", "work")
    assert todos is not None
    assert len(todos) == 3

    cancelled = todos[0]
    assert cancelled["cancelled"] is True
    assert cancelled["text"] == "Optional experiment"

    second = todos[1]
    assert second["nudge"] == "20240103T14:45"
    assert second["index"] == 2

    third = todos[2]
    assert third["completed"] is True
    assert third["text"] == "Draft report"


def test_get_todos_ignores_blank_lines(monkeypatch, journal_root):
    monkeypatch.setenv("SOLSTONE_JOURNAL", str(journal_root))
    # Write with blank lines mixed in
    todos_dir = journal_root / "facets" / "personal" / "todos"
    todos_dir.mkdir(parents=True, exist_ok=True)
    todo_path = todos_dir / "20240104.jsonl"
    todo_path.write_text(
        '\n{"text": "First"}\n\n{"text": "Second"}\n',
        encoding="utf-8",
    )

    todos = get_todos("20240104", "personal")
    assert [item["index"] for item in todos] == [1, 2]


def test_todo_item_display_line():
    """Test TodoItem.display_line() formatting."""
    item = TodoItem(
        index=1, text="Simple task", nudge=None, completed=False, cancelled=False
    )
    assert item.display_line() == "[ ] Simple task"

    item_done = TodoItem(
        index=2, text="Done task", nudge=None, completed=True, cancelled=False
    )
    assert item_done.display_line() == "[x] Done task"

    item_cancelled = TodoItem(
        index=3, text="Cancelled task", nudge=None, completed=False, cancelled=True
    )
    assert item_cancelled.display_line() == "~~[cancelled] Cancelled task~~"

    item_with_nudge = TodoItem(
        index=4,
        text="Meeting",
        nudge="20240101T14:30",
        completed=False,
        cancelled=False,
    )
    assert "[ ] Meeting (" in item_with_nudge.display_line()

    item_cancelled_done = TodoItem(
        index=5, text="Was done", nudge=None, completed=True, cancelled=True
    )
    assert item_cancelled_done.display_line() == "~~[cancelled] Was done~~"


def test_todo_item_to_jsonl():
    """Test TodoItem.to_jsonl() serialization."""
    item = TodoItem(index=1, text="Task", nudge=None, completed=False, cancelled=False)
    assert item.to_jsonl() == {"text": "Task"}

    item_full = TodoItem(
        index=2,
        text="Full task",
        nudge="20240102T10:00",
        completed=True,
        cancelled=False,
    )
    assert item_full.to_jsonl() == {
        "text": "Full task",
        "nudge": "20240102T10:00",
        "completed": True,
    }

    item_cancelled = TodoItem(
        index=3, text="Cancelled", nudge=None, completed=False, cancelled=True
    )
    assert item_cancelled.to_jsonl() == {"text": "Cancelled", "cancelled": True}


def test_todo_item_from_jsonl():
    """Test TodoItem.from_jsonl() parsing."""
    item = TodoItem.from_jsonl({"text": "Simple"}, 1)
    assert item.index == 1
    assert item.text == "Simple"
    assert item.completed is False
    assert item.cancelled is False
    assert item.nudge is None

    item_full = TodoItem.from_jsonl(
        {
            "text": "Full",
            "nudge": "20240102T10:00",
            "completed": True,
            "cancelled": False,
        },
        2,
    )
    assert item_full.nudge == "20240102T10:00"
    assert item_full.completed is True


def test_todo_item_from_jsonl_legacy_time_with_day():
    """Legacy 'time' field should be converted when day context is provided."""
    item = TodoItem.from_jsonl({"text": "Legacy", "time": "10:00"}, 1, day="20240102")
    assert item.nudge == "20240102T10:00"


class TestParseNudge:
    """Tests for parse_nudge()."""

    def test_hhmm(self):
        from solstone.apps.todos.todo import parse_nudge

        assert parse_nudge("15:00", "20260301") == "20260301T15:00"

    def test_now(self):
        from solstone.apps.todos.todo import parse_nudge

        result = parse_nudge("now", "20260301")
        assert result.startswith("20")
        assert "T" in result

    def test_tomorrow(self):
        from solstone.apps.todos.todo import parse_nudge

        assert parse_nudge("tomorrow 09:00", "20260301") == "20260302T09:00"

    def test_full_datetime(self):
        from solstone.apps.todos.todo import parse_nudge

        assert parse_nudge("20260315T14:30", "20260301") == "20260315T14:30"

    def test_invalid(self):
        from solstone.apps.todos.todo import parse_nudge

        with pytest.raises(ValueError):
            parse_nudge("garbage", "20260301")


class TestFormatNudge:
    """Tests for format_nudge()."""

    def test_past_minutes(self):
        from datetime import datetime

        from solstone.apps.todos.todo import format_nudge

        now = datetime(2026, 3, 1, 15, 30)
        assert format_nudge("20260301T15:00", now) == "30m ago"

    def test_past_hours(self):
        from datetime import datetime

        from solstone.apps.todos.todo import format_nudge

        now = datetime(2026, 3, 1, 18, 0)
        assert format_nudge("20260301T15:00", now) == "3h ago"

    def test_future_today(self):
        from datetime import datetime

        from solstone.apps.todos.todo import format_nudge

        now = datetime(2026, 3, 1, 10, 0)
        assert format_nudge("20260301T15:00", now) == "nudge 15:00"

    def test_future_tomorrow(self):
        from datetime import datetime

        from solstone.apps.todos.todo import format_nudge

        now = datetime(2026, 3, 1, 10, 0)
        assert format_nudge("20260302T09:00", now) == "tomorrow 09:00"


def test_upcoming_groups_future_days(monkeypatch, journal_root):
    monkeypatch.setenv("SOLSTONE_JOURNAL", str(journal_root))
    # Create facet structure
    (journal_root / "facets" / "personal").mkdir(parents=True)
    (journal_root / "facets" / "personal" / "facet.json").write_text(
        '{"title": "Personal"}', encoding="utf-8"
    )

    _write_todos(
        journal_root,
        "personal",
        "20240105",
        [
            {"text": "First future task"},
            {"text": "Completed future task", "completed": True},
        ],
    )
    _write_todos(
        journal_root,
        "personal",
        "20240106",
        [
            {"text": "Another future task"},
        ],
    )
    _write_todos(
        journal_root,
        "personal",
        "20240103",
        [
            {"text": "Past task"},
        ],
    )

    result = upcoming(today="20240104")

    expected = (
        "### Personal: 20240105\n"
        "[ ] First future task\n"
        "[x] Completed future task\n\n"
        "### Personal: 20240106\n"
        "[ ] Another future task"
    )

    assert result == expected


def test_upcoming_respects_limit(monkeypatch, journal_root):
    monkeypatch.setenv("SOLSTONE_JOURNAL", str(journal_root))
    # Create facet structure
    (journal_root / "facets" / "work").mkdir(parents=True)
    (journal_root / "facets" / "work" / "facet.json").write_text(
        '{"title": "Work"}', encoding="utf-8"
    )

    _write_todos(
        journal_root,
        "work",
        "20240105",
        [
            {"text": "Task one"},
            {"text": "Task two"},
            {"text": "Task three"},
        ],
    )

    result = upcoming(limit=2, today="20240104")

    expected = "### Work: 20240105\n[ ] Task one\n[ ] Task two"

    assert result == expected


def test_upcoming_excludes_cancelled(monkeypatch, journal_root):
    """Cancelled todos should not appear in upcoming view."""
    monkeypatch.setenv("SOLSTONE_JOURNAL", str(journal_root))
    (journal_root / "facets" / "personal").mkdir(parents=True)
    (journal_root / "facets" / "personal" / "facet.json").write_text(
        '{"title": "Personal"}', encoding="utf-8"
    )

    _write_todos(
        journal_root,
        "personal",
        "20240105",
        [
            {"text": "Active task"},
            {"text": "Cancelled task", "cancelled": True},
            {"text": "Another active"},
        ],
    )

    result = upcoming(today="20240104")

    assert "Active task" in result
    assert "Another active" in result
    assert "Cancelled task" not in result


def test_upcoming_when_no_future_todos(monkeypatch, journal_root):
    monkeypatch.setenv("SOLSTONE_JOURNAL", str(journal_root))
    (journal_root / "facets" / "personal").mkdir(parents=True)

    _write_todos(
        journal_root,
        "personal",
        "20240102",
        [
            {"text": "Existing task"},
        ],
    )

    result = upcoming(today="20240102")

    assert result == "No upcoming todos."


def test_upcoming_filters_by_facet(monkeypatch, journal_root):
    monkeypatch.setenv("SOLSTONE_JOURNAL", str(journal_root))
    # Create multiple facets
    for facet_name in ["personal", "work"]:
        facet_dir = journal_root / "facets" / facet_name
        facet_dir.mkdir(parents=True)
        (facet_dir / "facet.json").write_text(
            f'{{"title": "{facet_name.title()}"}}', encoding="utf-8"
        )

    _write_todos(journal_root, "personal", "20240105", [{"text": "Personal task"}])
    _write_todos(journal_root, "work", "20240105", [{"text": "Work task"}])

    # Test filtering by facet
    result = upcoming(facet="personal", today="20240104")
    assert "Personal: 20240105" in result
    assert "Personal task" in result
    assert "Work task" not in result


def test_upcoming_aggregates_all_facets(monkeypatch, journal_root):
    monkeypatch.setenv("SOLSTONE_JOURNAL", str(journal_root))
    # Create multiple facets
    for facet_name in ["personal", "work"]:
        facet_dir = journal_root / "facets" / facet_name
        facet_dir.mkdir(parents=True)
        (facet_dir / "facet.json").write_text(
            f'{{"title": "{facet_name.title()}"}}', encoding="utf-8"
        )

    _write_todos(journal_root, "personal", "20240105", [{"text": "Personal task"}])
    _write_todos(journal_root, "work", "20240105", [{"text": "Work task"}])

    # Test aggregation (facet=None)
    result = upcoming(facet=None, today="20240104")
    assert "Personal: 20240105" in result
    assert "Work: 20240105" in result
    assert "Personal task" in result
    assert "Work task" in result


def test_checklist_append_entry(monkeypatch, journal_root):
    """Test TodoChecklist.append_entry() creates valid JSONL."""
    monkeypatch.setenv("SOLSTONE_JOURNAL", str(journal_root))

    # Create facet directory
    facets_dir = journal_root / "facets" / "work"
    facets_dir.mkdir(parents=True)

    checklist = TodoChecklist.load("20240105", "work")

    # Test normal entry works
    item = checklist.append_entry("Test task", nudge="20240105T10:30")
    assert len(checklist.items) == 1
    assert item.text == "Test task"
    assert item.nudge == "20240105T10:30"

    # Verify file contents
    content = checklist.path.read_text(encoding="utf-8")
    data = json.loads(content.strip())
    assert data["text"] == "Test task"
    assert data["nudge"] == "20240105T10:30"


def test_checklist_cancel_entry(monkeypatch, journal_root):
    """Test TodoChecklist.cancel_entry() soft-deletes items."""
    monkeypatch.setenv("SOLSTONE_JOURNAL", str(journal_root))

    _write_todos(
        journal_root,
        "personal",
        "20240105",
        [{"text": "Task to cancel"}],
    )

    checklist = TodoChecklist.load("20240105", "personal")
    item = checklist.cancel_entry(1)

    assert item.cancelled is True
    assert checklist.items[0].cancelled is True

    # Verify file was updated
    content = checklist.path.read_text(encoding="utf-8")
    data = json.loads(content.strip())
    assert data["cancelled"] is True


def test_checklist_display_includes_cancelled_with_strikethrough(
    monkeypatch, journal_root
):
    """Test TodoChecklist.display() always includes cancelled items with strikethrough."""
    monkeypatch.setenv("SOLSTONE_JOURNAL", str(journal_root))

    _write_todos(
        journal_root,
        "personal",
        "20240105",
        [
            {"text": "Active task"},
            {"text": "Cancelled task", "cancelled": True},
            {"text": "Another active"},
        ],
    )

    checklist = TodoChecklist.load("20240105", "personal")

    # All items included, cancelled ones have strikethrough
    display = checklist.display()
    assert "1: [ ] Active task" in display
    assert "2: ~~[cancelled] Cancelled task~~" in display
    assert "3: [ ] Another active" in display


def test_get_facets_with_todos(monkeypatch, journal_root):
    monkeypatch.setenv("SOLSTONE_JOURNAL", str(journal_root))

    # Create todos in multiple facets
    _write_todos(journal_root, "personal", "20240105", [{"text": "Personal task"}])
    _write_todos(journal_root, "work", "20240105", [{"text": "Work task"}])
    _write_todos(journal_root, "hobby", "20240106", [{"text": "Hobby task"}])

    # Test getting facets for a specific day
    facets_20240105 = get_facets_with_todos("20240105")
    assert sorted(facets_20240105) == ["personal", "work"]

    facets_20240106 = get_facets_with_todos("20240106")
    assert facets_20240106 == ["hobby"]

    facets_20240107 = get_facets_with_todos("20240107")
    assert facets_20240107 == []


# --- Timestamp tests ---


def test_todo_item_timestamps_on_creation(monkeypatch, journal_root):
    """Test that timestamps are set when creating a new todo."""
    import time

    monkeypatch.setenv("SOLSTONE_JOURNAL", str(journal_root))

    # Create facet directory
    (journal_root / "facets" / "personal" / "todos").mkdir(parents=True)

    before = int(time.time() * 1000)
    checklist = TodoChecklist.load("20240110", "personal")
    item = checklist.append_entry("New task")
    after = int(time.time() * 1000)

    assert item.created_at is not None
    assert item.updated_at is not None
    assert before <= item.created_at <= after
    assert before <= item.updated_at <= after
    assert item.created_at == item.updated_at  # Same on creation


def test_todo_item_updated_at_changes_on_mark_done(monkeypatch, journal_root):
    """Test that updated_at changes when marking todo complete."""
    import time

    monkeypatch.setenv("SOLSTONE_JOURNAL", str(journal_root))
    (journal_root / "facets" / "personal" / "todos").mkdir(parents=True)

    checklist = TodoChecklist.load("20240110", "personal")
    item = checklist.append_entry("Task to complete")
    original_created = item.created_at
    original_updated = item.updated_at

    time.sleep(0.01)  # Ensure time passes

    updated_item = checklist.mark_done(item.index)

    assert updated_item.created_at == original_created  # Unchanged
    assert updated_item.updated_at > original_updated


def test_todo_item_updated_at_changes_on_mark_undone(monkeypatch, journal_root):
    """Test that updated_at changes when marking todo incomplete."""
    import time

    monkeypatch.setenv("SOLSTONE_JOURNAL", str(journal_root))
    (journal_root / "facets" / "personal" / "todos").mkdir(parents=True)

    checklist = TodoChecklist.load("20240110", "personal")
    item = checklist.append_entry("Task to uncomplete")
    checklist.mark_done(item.index)

    time.sleep(0.01)

    reloaded = TodoChecklist.load("20240110", "personal")
    original_updated = reloaded.items[0].updated_at
    original_created = reloaded.items[0].created_at

    time.sleep(0.01)

    updated_item = reloaded.mark_undone(item.index)

    assert updated_item.created_at == original_created
    assert updated_item.updated_at > original_updated


def test_todo_item_updated_at_changes_on_cancel(monkeypatch, journal_root):
    """Test that updated_at changes when cancelling a todo."""
    import time

    monkeypatch.setenv("SOLSTONE_JOURNAL", str(journal_root))
    (journal_root / "facets" / "personal" / "todos").mkdir(parents=True)

    checklist = TodoChecklist.load("20240110", "personal")
    item = checklist.append_entry("Task to cancel")
    original_created = item.created_at
    original_updated = item.updated_at

    time.sleep(0.01)

    cancelled_item = checklist.cancel_entry(item.index)

    assert cancelled_item.created_at == original_created
    assert cancelled_item.updated_at > original_updated


def test_todo_item_updated_at_changes_on_text_update(monkeypatch, journal_root):
    """Test that updated_at changes when updating todo text."""
    import time

    monkeypatch.setenv("SOLSTONE_JOURNAL", str(journal_root))
    (journal_root / "facets" / "personal" / "todos").mkdir(parents=True)

    checklist = TodoChecklist.load("20240110", "personal")
    item = checklist.append_entry("Original text")
    original_created = item.created_at
    original_updated = item.updated_at

    time.sleep(0.01)

    updated_item = checklist.update_entry_text(item.index, "Updated text")

    assert updated_item.created_at == original_created
    assert updated_item.updated_at > original_updated


def test_todo_item_timestamps_serialization(monkeypatch, journal_root):
    """Test that timestamps are properly serialized to and from JSONL."""
    monkeypatch.setenv("SOLSTONE_JOURNAL", str(journal_root))
    (journal_root / "facets" / "personal" / "todos").mkdir(parents=True)

    checklist = TodoChecklist.load("20240110", "personal")
    item = checklist.append_entry("Persistent task")

    # Verify timestamps in to_jsonl output
    jsonl = item.to_jsonl()
    assert "created_at" in jsonl
    assert "updated_at" in jsonl
    assert jsonl["created_at"] == item.created_at
    assert jsonl["updated_at"] == item.updated_at

    # Verify timestamps survive round-trip
    reloaded = TodoChecklist.load("20240110", "personal")
    assert len(reloaded.items) == 1
    assert reloaded.items[0].created_at == item.created_at
    assert reloaded.items[0].updated_at == item.updated_at


def test_todo_item_timestamps_in_as_dict(monkeypatch, journal_root):
    """Test that timestamps are included in as_dict() output."""
    monkeypatch.setenv("SOLSTONE_JOURNAL", str(journal_root))
    (journal_root / "facets" / "personal" / "todos").mkdir(parents=True)

    checklist = TodoChecklist.load("20240110", "personal")
    item = checklist.append_entry("API task")

    d = item.as_dict()
    assert "created_at" in d
    assert "updated_at" in d
    assert d["created_at"] == item.created_at
    assert d["updated_at"] == item.updated_at


def test_todo_item_backward_compatibility_no_timestamps(monkeypatch, journal_root):
    """Test loading files without timestamps (backward compatibility)."""
    monkeypatch.setenv("SOLSTONE_JOURNAL", str(journal_root))

    # Write old-format todos without timestamps
    _write_todos(
        journal_root,
        "personal",
        "20240110",
        [
            {"text": "Old task"},
            {"text": "Old completed", "completed": True},
        ],
    )

    checklist = TodoChecklist.load("20240110", "personal")
    assert len(checklist.items) == 2

    # Old items have None timestamps
    assert checklist.items[0].created_at is None
    assert checklist.items[0].updated_at is None
    assert checklist.items[1].created_at is None
    assert checklist.items[1].updated_at is None


def test_append_entry_preserves_created_at(monkeypatch, journal_root):
    """Test that append_entry can preserve a provided created_at timestamp."""
    import time

    monkeypatch.setenv("SOLSTONE_JOURNAL", str(journal_root))
    (journal_root / "facets" / "personal" / "todos").mkdir(parents=True)

    checklist = TodoChecklist.load("20240110", "personal")
    original_created = 1700000000000  # Fixed timestamp in the past

    before = int(time.time() * 1000)
    item = checklist.append_entry("Moved task", created_at=original_created)
    after = int(time.time() * 1000)

    # created_at preserved from argument
    assert item.created_at == original_created
    # updated_at is current time
    assert before <= item.updated_at <= after


class TestFindCrossFacetMatches:
    """Tests for find_cross_facet_matches() cross-facet duplicate detection."""

    def test_detects_duplicate_in_other_facet(self, monkeypatch, journal_root):
        """Exact duplicate in another facet is detected."""
        monkeypatch.setenv("SOLSTONE_JOURNAL", str(journal_root))
        _write_todos(journal_root, "work", "20240102", [{"text": "Draft Q1 plan"}])
        matches = find_cross_facet_matches("Draft Q1 plan", "20240102", "personal")
        assert len(matches) == 1
        assert matches[0]["score"] == 100.0
        assert matches[0]["facet"] == "work"
        assert matches[0]["day"] == "20240102"
        assert matches[0]["line"] == 1

    def test_detects_fuzzy_match(self, monkeypatch, journal_root):
        """Fuzzy match above threshold is detected."""
        monkeypatch.setenv("SOLSTONE_JOURNAL", str(journal_root))
        _write_todos(
            journal_root,
            "work",
            "20240102",
            [{"text": "Draft Q1 plan doc"}],
        )
        matches = find_cross_facet_matches("Draft Q1 plan", "20240102", "personal")
        assert len(matches) >= 1
        assert matches[0]["score"] >= 70.0

    def test_no_false_positives(self, monkeypatch, journal_root):
        """Unrelated todos in other facets are not flagged."""
        monkeypatch.setenv("SOLSTONE_JOURNAL", str(journal_root))
        _write_todos(journal_root, "work", "20240102", [{"text": "Buy groceries"}])
        matches = find_cross_facet_matches("Draft Q1 plan", "20240102", "personal")
        assert len(matches) == 0

    def test_excludes_own_facet(self, monkeypatch, journal_root):
        """Todos in the requesting facet are excluded."""
        monkeypatch.setenv("SOLSTONE_JOURNAL", str(journal_root))
        _write_todos(journal_root, "personal", "20240102", [{"text": "Draft Q1 plan"}])
        matches = find_cross_facet_matches("Draft Q1 plan", "20240102", "personal")
        assert len(matches) == 0

    def test_excludes_cancelled(self, monkeypatch, journal_root):
        """Cancelled todos are not matched."""
        monkeypatch.setenv("SOLSTONE_JOURNAL", str(journal_root))
        _write_todos(
            journal_root,
            "work",
            "20240102",
            [{"text": "Draft Q1 plan", "cancelled": True}],
        )
        matches = find_cross_facet_matches("Draft Q1 plan", "20240102", "personal")
        assert len(matches) == 0

    def test_excludes_completed(self, monkeypatch, journal_root):
        """Completed todos are not matched."""
        monkeypatch.setenv("SOLSTONE_JOURNAL", str(journal_root))
        _write_todos(
            journal_root,
            "work",
            "20240102",
            [{"text": "Draft Q1 plan", "completed": True}],
        )
        matches = find_cross_facet_matches("Draft Q1 plan", "20240102", "personal")
        assert len(matches) == 0

    def test_day_range_covers_adjacent_days(self, monkeypatch, journal_root):
        """Matches within ±1 day window are detected."""
        monkeypatch.setenv("SOLSTONE_JOURNAL", str(journal_root))
        _write_todos(journal_root, "work", "20240101", [{"text": "Draft Q1 plan"}])
        _write_todos(journal_root, "work", "20240103", [{"text": "Draft Q1 plan"}])
        matches = find_cross_facet_matches("Draft Q1 plan", "20240102", "personal")
        assert len(matches) == 2

    def test_empty_journal_returns_empty(self, monkeypatch, journal_root):
        """No facets returns empty list."""
        monkeypatch.setenv("SOLSTONE_JOURNAL", str(journal_root))
        matches = find_cross_facet_matches("Draft Q1 plan", "20240102", "personal")
        assert matches == []
