"""Tests for todos app MCP tool helpers and handlers."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from apps.todos import tools as todo_tools


def call_tool(tool, *args, **kwargs):
    """Invoke the underlying callable for a FastMCP tool."""
    return tool.fn(*args, **kwargs)


FIXTURES_JOURNAL = Path(__file__).resolve().parents[3] / "fixtures" / "journal"


def test_todo_list_returns_numbered_view(todo_env):
    """todo_list should number checklist entries starting at 1."""
    day, facet, _ = todo_env(["- [ ] First item", "- [ ] Second item"])

    result = call_tool(todo_tools.todo_list, day, facet)

    assert result == {
        "day": day,
        "facet": facet,
        "markdown": "1: - [ ] First item\n2: - [ ] Second item",
    }


def test_todo_add_requires_next_line(todo_env):
    """todo_add should reject mismatched next line numbers."""
    day, facet, _ = todo_env(["- [ ] First item"])

    result = call_tool(
        todo_tools.todo_add, day, facet, line_number=1, text="Second item"
    )

    assert result["error"] == "line number 1 must match the next available line"


def test_todo_add_appends_entry(todo_env):
    """todo_add should append using the provided text and update storage."""
    day, facet, todo_path = todo_env(["- [ ] First item"])

    result = call_tool(
        todo_tools.todo_add, day, facet, line_number=2, text="Second item"
    )

    assert result == {
        "day": day,
        "facet": facet,
        "markdown": "1: - [ ] First item\n2: - [ ] Second item",
    }
    assert todo_path.read_text(encoding="utf-8") == (
        "- [ ] First item\n- [ ] Second item\n"
    )


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

    todo_path = tmp_path / "facets" / facet / "todos" / f"{day}.md"

    assert result == {
        "day": day,
        "facet": facet,
        "markdown": "1: - [ ] Plan end-of-year celebration",
    }
    assert todo_path.read_text(encoding="utf-8") == (
        "- [ ] Plan end-of-year celebration\n"
    )


def test_todo_remove_validates_guard(todo_env):
    """todo_remove should validate the guard string before deleting."""
    day, facet, _ = todo_env(["- [ ] First item", "- [ ] Second item"])

    result = call_tool(
        todo_tools.todo_remove, day, facet, line_number=2, guard="- [ ] Other"
    )

    assert result["error"] == "guard text does not match current todo"


def test_todo_remove_updates_file(todo_env):
    """todo_remove should delete the requested entry and renumber output."""
    day, facet, todo_path = todo_env(["- [ ] First item", "- [ ] Second item"])

    result = call_tool(
        todo_tools.todo_remove, day, facet, line_number=2, guard="- [ ] Second item"
    )

    assert result == {"day": day, "facet": facet, "markdown": "1: - [ ] First item"}
    assert todo_path.read_text(encoding="utf-8") == "- [ ] First item\n"


def test_todo_done_marks_complete(todo_env):
    """todo_done should mark the entry as completed in storage."""
    day, facet, todo_path = todo_env(["- [ ] First item", "- [ ] Second item"])

    result = call_tool(
        todo_tools.todo_done, day, facet, line_number=2, guard="- [ ] Second item"
    )

    assert result == {
        "day": day,
        "facet": facet,
        "markdown": "1: - [ ] First item\n2: - [x] Second item",
    }
    assert todo_path.read_text(encoding="utf-8") == (
        "- [ ] First item\n- [x] Second item\n"
    )


def test_todo_add_validates_empty_text(todo_env):
    """todo_add should reject empty todo text."""
    day, facet, _ = todo_env([])

    result = call_tool(todo_tools.todo_add, day, facet, 1, "")

    assert "error" in result
    assert "todo text cannot be empty" in result["error"]
    assert "suggestion" in result


@pytest.mark.integration
def test_todo_tool_pack_round_trip(tmp_path, monkeypatch):
    """Exercise add/done/remove flow against a copied fixture journal."""
    if not FIXTURES_JOURNAL.exists():
        pytest.skip("fixtures/journal not found")

    journal_copy = tmp_path / "journal"
    shutil.copytree(FIXTURES_JOURNAL, journal_copy)

    day = "20991231"  # Use future date to avoid date validation
    facet = "personal"
    todos_dir = journal_copy / "facets" / facet / "todos"
    todos_dir.mkdir(parents=True, exist_ok=True)
    todo_path = todos_dir / f"{day}.md"
    todo_path.write_text("- [ ] Fixture task\n", encoding="utf-8")

    monkeypatch.setenv("JOURNAL_PATH", str(journal_copy))

    list_result = call_tool(todo_tools.todo_list, day, facet)
    assert list_result == {
        "day": day,
        "facet": facet,
        "markdown": "1: - [ ] Fixture task",
    }

    add_result = call_tool(
        todo_tools.todo_add, day, facet, line_number=2, text="Follow up task"
    )
    if "error" in add_result:
        raise AssertionError(f"todo_add failed: {add_result}")
    assert add_result["markdown"].splitlines() == [
        "1: - [ ] Fixture task",
        "2: - [ ] Follow up task",
    ]

    done_result = call_tool(
        todo_tools.todo_done, day, facet, line_number=2, guard="- [ ] Follow up task"
    )
    assert done_result["markdown"].splitlines() == [
        "1: - [ ] Fixture task",
        "2: - [x] Follow up task",
    ]

    removed_done = call_tool(
        todo_tools.todo_remove, day, facet, line_number=2, guard="- [x] Follow up task"
    )
    assert removed_done == {
        "day": day,
        "facet": facet,
        "markdown": "1: - [ ] Fixture task",
    }

    empty_result = call_tool(
        todo_tools.todo_remove, day, facet, line_number=1, guard="- [ ] Fixture task"
    )
    assert empty_result == {
        "day": day,
        "facet": facet,
        "markdown": "0: (no todos)",
    }

    assert todo_path.read_text(encoding="utf-8") == ""
