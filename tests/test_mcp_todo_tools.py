"""Unit tests for MCP todo tool helpers and handlers."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from muse import mcp_tools


def call_tool(tool, *args, **kwargs):
    """Invoke the underlying callable for a FastMCP tool."""

    return tool.fn(*args, **kwargs)


@pytest.fixture
def todo_env(tmp_path, monkeypatch):
    """Create a temporary journal domain with optional todo entries."""

    def _create(
        entries: list[str] | None = None,
        day: str | None = None,
        domain: str = "personal",
    ):
        if day is None:
            day = datetime.now().strftime("%Y%m%d")
        todos_dir = tmp_path / "domains" / domain / "todos"
        todos_dir.mkdir(parents=True, exist_ok=True)
        todo_path = todos_dir / f"{day}.md"
        if entries is not None:
            todo_path.write_text("\n".join(entries) + "\n", encoding="utf-8")
        monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
        return day, domain, todo_path

    return _create


def test_todo_list_returns_numbered_view(todo_env):
    """todo_list should number checklist entries starting at 1."""

    day, domain, _ = todo_env(["- [ ] First item", "- [ ] Second item"])

    result = call_tool(mcp_tools.todo_list, day, domain)

    assert result == {
        "day": day,
        "domain": domain,
        "markdown": "1: - [ ] First item\n2: - [ ] Second item",
    }


def test_todo_add_requires_next_line(todo_env):
    """todo_add should reject mismatched next line numbers."""

    day, domain, _ = todo_env(["- [ ] First item"])

    result = call_tool(
        mcp_tools.todo_add, day, domain, line_number=1, text="Second item"
    )

    assert result["error"] == "line number 1 must match the next available line"


def test_todo_add_appends_entry(todo_env):
    """todo_add should append using the provided text and update storage."""

    day, domain, todo_path = todo_env(["- [ ] First item"])

    result = call_tool(
        mcp_tools.todo_add, day, domain, line_number=2, text="Second item"
    )

    assert result == {
        "day": day,
        "domain": domain,
        "markdown": "1: - [ ] First item\n2: - [ ] Second item",
    }
    assert todo_path.read_text(encoding="utf-8") == (
        "- [ ] First item\n- [ ] Second item\n"
    )


def test_todo_add_creates_missing_day(tmp_path, monkeypatch):
    """todo_add should create future day folders when needed."""

    day = "20991231"
    domain = "personal"
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))

    # Create domain directory structure
    domains_dir = tmp_path / "domains" / domain
    domains_dir.mkdir(parents=True)

    result = call_tool(
        mcp_tools.todo_add,
        day,
        domain,
        line_number=1,
        text="Plan end-of-year celebration",
    )

    todo_path = tmp_path / "domains" / domain / "todos" / f"{day}.md"

    assert result == {
        "day": day,
        "domain": domain,
        "markdown": "1: - [ ] Plan end-of-year celebration",
    }
    assert todo_path.read_text(encoding="utf-8") == (
        "- [ ] Plan end-of-year celebration\n"
    )


def test_todo_remove_validates_guard(todo_env):
    """todo_remove should validate the guard string before deleting."""

    day, domain, _ = todo_env(["- [ ] First item", "- [ ] Second item"])

    result = call_tool(
        mcp_tools.todo_remove, day, domain, line_number=2, guard="- [ ] Other"
    )

    assert result["error"] == "guard text does not match current todo"


def test_todo_remove_updates_file(todo_env):
    """todo_remove should delete the requested entry and renumber output."""

    day, domain, todo_path = todo_env(["- [ ] First item", "- [ ] Second item"])

    result = call_tool(
        mcp_tools.todo_remove, day, domain, line_number=2, guard="- [ ] Second item"
    )

    assert result == {"day": day, "domain": domain, "markdown": "1: - [ ] First item"}
    assert todo_path.read_text(encoding="utf-8") == "- [ ] First item\n"


def test_todo_done_marks_complete(todo_env):
    """todo_done should mark the entry as completed in storage."""

    day, domain, todo_path = todo_env(["- [ ] First item", "- [ ] Second item"])

    result = call_tool(
        mcp_tools.todo_done, day, domain, line_number=2, guard="- [ ] Second item"
    )

    assert result == {
        "day": day,
        "domain": domain,
        "markdown": "1: - [ ] First item\n2: - [x] Second item",
    }
    assert todo_path.read_text(encoding="utf-8") == (
        "- [ ] First item\n- [x] Second item\n"
    )


def test_todo_add_validates_empty_text(todo_env):
    """todo_add should reject empty todo text."""

    day, domain, _ = todo_env([])

    result = call_tool(mcp_tools.todo_add, day, domain, 1, "")

    assert "error" in result
    assert "todo text cannot be empty" in result["error"]
    assert "suggestion" in result
