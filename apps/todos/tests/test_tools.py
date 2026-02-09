# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for todos app tool functions."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from apps.todos import tools as todo_tools


def call_tool(tool, *args, **kwargs):
    """Invoke a tool function."""
    return tool(*args, **kwargs)


FIXTURES_JOURNAL = (
    Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "journal"
)


def test_todo_list_returns_numbered_view(todo_env):
    """todo_list should number checklist entries starting at 1."""
    day, facet, _ = todo_env([{"text": "First item"}, {"text": "Second item"}])

    result = call_tool(todo_tools.todo_list, day, facet)

    assert result == {
        "day": day,
        "facet": facet,
        "markdown": "1: [ ] First item\n2: [ ] Second item",
    }


def test_todo_list_includes_cancelled(todo_env):
    """todo_list should include cancelled items with strikethrough for line number continuity."""
    day, facet, _ = todo_env(
        [
            {"text": "Active item"},
            {"text": "Cancelled item", "cancelled": True},
            {"text": "Another active"},
        ]
    )

    result = call_tool(todo_tools.todo_list, day, facet)

    assert "1: [ ] Active item" in result["markdown"]
    assert "2: ~~[cancelled] Cancelled item~~" in result["markdown"]
    assert "3: [ ] Another active" in result["markdown"]


def test_todo_add_requires_next_line(todo_env):
    """todo_add should reject mismatched next line numbers."""
    day, facet, _ = todo_env([{"text": "First item"}])

    result = call_tool(
        todo_tools.todo_add, day, facet, line_number=1, text="Second item"
    )

    assert result["error"] == "line number 1 must match the next available line"


def test_todo_add_appends_entry(todo_env):
    """todo_add should append using the provided text and update storage."""
    day, facet, todo_path = todo_env([{"text": "First item"}])

    result = call_tool(
        todo_tools.todo_add, day, facet, line_number=2, text="Second item"
    )

    assert result == {
        "day": day,
        "facet": facet,
        "markdown": "1: [ ] First item\n2: [ ] Second item",
    }

    # Verify file contents
    lines = todo_path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[0])["text"] == "First item"
    assert json.loads(lines[1])["text"] == "Second item"


def test_todo_add_creates_missing_day(tmp_path, monkeypatch):
    """todo_add should create future day folders when needed."""
    day = "20991231"
    facet = "personal"
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))

    # Create facet directory structure
    facets_dir = tmp_path / "facets" / facet
    facets_dir.mkdir(parents=True)

    result = call_tool(
        todo_tools.todo_add,
        day,
        facet,
        line_number=1,
        text="Plan end-of-year celebration",
    )

    todo_path = tmp_path / "facets" / facet / "todos" / f"{day}.jsonl"

    assert result == {
        "day": day,
        "facet": facet,
        "markdown": "1: [ ] Plan end-of-year celebration",
    }

    content = todo_path.read_text(encoding="utf-8")
    data = json.loads(content.strip())
    assert data["text"] == "Plan end-of-year celebration"


def test_todo_cancel_sets_cancelled_flag(todo_env):
    """todo_cancel should mark the entry as cancelled."""
    day, facet, todo_path = todo_env([{"text": "First item"}, {"text": "Second item"}])

    result = call_tool(todo_tools.todo_cancel, day, facet, line_number=2)

    # Cancelled items shown with strikethrough
    assert result == {
        "day": day,
        "facet": facet,
        "markdown": "1: [ ] First item\n2: ~~[cancelled] Second item~~",
    }

    # Verify file still has both items, second one cancelled
    lines = todo_path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[1])["cancelled"] is True


def test_todo_done_marks_complete(todo_env):
    """todo_done should mark the entry as completed in storage."""
    day, facet, todo_path = todo_env([{"text": "First item"}, {"text": "Second item"}])

    result = call_tool(todo_tools.todo_done, day, facet, line_number=2)

    assert result == {
        "day": day,
        "facet": facet,
        "markdown": "1: [ ] First item\n2: [x] Second item",
    }

    lines = todo_path.read_text(encoding="utf-8").strip().split("\n")
    assert json.loads(lines[1])["completed"] is True


def test_todo_add_validates_empty_text(todo_env):
    """todo_add should reject empty todo text."""
    day, facet, _ = todo_env([])

    result = call_tool(todo_tools.todo_add, day, facet, 1, "")

    assert "error" in result
    assert "todo text cannot be empty" in result["error"]
    assert "suggestion" in result


def test_todo_add_extracts_time(todo_env):
    """todo_add should extract time from text."""
    day, facet, todo_path = todo_env([])

    result = call_tool(
        todo_tools.todo_add, day, facet, line_number=1, text="Meeting (14:30)"
    )

    assert "error" not in result
    assert "14:30" in result["markdown"]

    # Verify file stores time separately
    content = todo_path.read_text(encoding="utf-8")
    data = json.loads(content.strip())
    assert data["text"] == "Meeting"
    assert data["time"] == "14:30"


@pytest.mark.integration
def test_todo_tool_pack_round_trip(tmp_path, monkeypatch):
    """Exercise add/done/cancel flow against a copied fixture journal."""
    if not FIXTURES_JOURNAL.exists():
        pytest.skip("tests/fixtures/journal not found")

    journal_copy = tmp_path / "journal"
    shutil.copytree(FIXTURES_JOURNAL, journal_copy)

    day = "20991231"  # Use future date to avoid date validation
    facet = "personal"
    todos_dir = journal_copy / "facets" / facet / "todos"
    todos_dir.mkdir(parents=True, exist_ok=True)
    todo_path = todos_dir / f"{day}.jsonl"
    todo_path.write_text('{"text": "Fixture task"}\n', encoding="utf-8")

    monkeypatch.setenv("JOURNAL_PATH", str(journal_copy))

    list_result = call_tool(todo_tools.todo_list, day, facet)
    assert list_result == {
        "day": day,
        "facet": facet,
        "markdown": "1: [ ] Fixture task",
    }

    add_result = call_tool(
        todo_tools.todo_add, day, facet, line_number=2, text="Follow up task"
    )
    if "error" in add_result:
        raise AssertionError(f"todo_add failed: {add_result}")
    assert add_result["markdown"].splitlines() == [
        "1: [ ] Fixture task",
        "2: [ ] Follow up task",
    ]

    done_result = call_tool(todo_tools.todo_done, day, facet, line_number=2)
    assert done_result["markdown"].splitlines() == [
        "1: [ ] Fixture task",
        "2: [x] Follow up task",
    ]

    # Cancel the completed item
    cancel_result = call_tool(todo_tools.todo_cancel, day, facet, line_number=2)
    # Cancelled items shown with strikethrough
    assert cancel_result["markdown"].splitlines() == [
        "1: [ ] Fixture task",
        "2: ~~[cancelled] Follow up task~~",
    ]

    # Cancel remaining item
    cancel_last = call_tool(todo_tools.todo_cancel, day, facet, line_number=1)
    assert cancel_last["markdown"].splitlines() == [
        "1: ~~[cancelled] Fixture task~~",
        "2: ~~[cancelled] Follow up task~~",
    ]

    # File should still have both items (soft delete)
    lines = todo_path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[0])["cancelled"] is True
    assert json.loads(lines[1])["cancelled"] is True


# -----------------------------------------------------------------------------
# todo_list range tests
# -----------------------------------------------------------------------------


def test_todo_list_range_multiple_days(tmp_path, monkeypatch):
    """todo_list with day_to should return todos grouped by day."""
    facet = "work"
    todos_dir = tmp_path / "facets" / facet / "todos"
    todos_dir.mkdir(parents=True)
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))

    # Create todos for multiple days
    (todos_dir / "20250101.jsonl").write_text('{"text": "New Year task"}\n')
    (todos_dir / "20250103.jsonl").write_text(
        '{"text": "Task A"}\n{"text": "Task B", "completed": true}\n'
    )
    (todos_dir / "20250105.jsonl").write_text('{"text": "Weekend task"}\n')

    result = call_tool(todo_tools.todo_list, "20250101", facet, "20250105")

    assert result["day"] == "20250101"
    assert result["day_to"] == "20250105"
    assert result["facet"] == facet
    assert "### 20250101" in result["markdown"]
    assert "New Year task" in result["markdown"]
    assert "### 20250103" in result["markdown"]
    assert "Task A" in result["markdown"]
    assert "Task B" in result["markdown"]
    assert "### 20250105" in result["markdown"]
    assert "Weekend task" in result["markdown"]


def test_todo_list_range_empty(tmp_path, monkeypatch):
    """todo_list with day_to should return empty message when no todos in range."""
    facet = "work"
    todos_dir = tmp_path / "facets" / facet / "todos"
    todos_dir.mkdir(parents=True)
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))

    result = call_tool(todo_tools.todo_list, "20250101", facet, "20250107")

    assert result["day"] == "20250101"
    assert result["day_to"] == "20250107"
    assert result["markdown"] == "No todos in range."


def test_todo_list_range_swapped_error(tmp_path, monkeypatch):
    """todo_list should error when day > day_to with helpful suggestion."""
    facet = "work"
    todos_dir = tmp_path / "facets" / facet / "todos"
    todos_dir.mkdir(parents=True)
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))

    result = call_tool(todo_tools.todo_list, "20250107", facet, "20250101")

    assert "error" in result
    assert "must be before or equal to" in result["error"]
    assert "suggestion" in result
    assert "swap" in result["suggestion"]


def test_todo_list_range_same_day_no_headers(todo_env):
    """todo_list with day == day_to should behave like single day (no headers)."""
    day, facet, _ = todo_env([{"text": "Single day task"}])

    result = call_tool(todo_tools.todo_list, day, facet, day)

    # Should be same as single-day format (no day_to in response, no headers)
    assert result == {
        "day": day,
        "facet": facet,
        "markdown": "1: [ ] Single day task",
    }
    assert "day_to" not in result
    assert "###" not in result["markdown"]


def test_todo_list_range_includes_cancelled(tmp_path, monkeypatch):
    """todo_list range should include cancelled items with strikethrough."""
    facet = "work"
    todos_dir = tmp_path / "facets" / facet / "todos"
    todos_dir.mkdir(parents=True)
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))

    (todos_dir / "20250101.jsonl").write_text(
        '{"text": "Active"}\n{"text": "Cancelled", "cancelled": true}\n'
    )

    result = call_tool(todo_tools.todo_list, "20250101", facet, "20250103")

    assert "1: [ ] Active" in result["markdown"]
    assert "2: ~~[cancelled] Cancelled~~" in result["markdown"]


def test_todo_list_invalid_day_format(tmp_path, monkeypatch):
    """todo_list should error on invalid day format."""
    facet = "work"
    todos_dir = tmp_path / "facets" / facet / "todos"
    todos_dir.mkdir(parents=True)
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))

    result = call_tool(todo_tools.todo_list, "not-a-date", facet)

    assert "error" in result
    assert "Invalid day format" in result["error"]
    assert "suggestion" in result


def test_todo_list_invalid_day_to_format(tmp_path, monkeypatch):
    """todo_list should error on invalid day_to format."""
    facet = "work"
    todos_dir = tmp_path / "facets" / facet / "todos"
    todos_dir.mkdir(parents=True)
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))

    result = call_tool(todo_tools.todo_list, "20250101", facet, "bad-date")

    assert "error" in result
    assert "Invalid day_to format" in result["error"]
    assert "suggestion" in result
