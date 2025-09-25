"""Unit tests for MCP todo tool helpers and handlers."""

from __future__ import annotations

from pathlib import Path

import pytest

from think import mcp_tools


def call_tool(tool, *args, **kwargs):
    """Invoke the underlying callable for a FastMCP tool."""

    return tool.fn(*args, **kwargs)


@pytest.fixture
def todo_env(tmp_path, monkeypatch):
    """Create a temporary journal day with optional todo entries."""

    def _create(entries: list[str] | None = None, day: str = "20240202"):
        day_dir = tmp_path / day
        (day_dir / "todos").mkdir(parents=True, exist_ok=True)
        if entries is not None:
            todo_path = day_dir / "todos" / "today.md"
            todo_path.write_text("\n".join(entries) + "\n", encoding="utf-8")
        else:
            todo_path = day_dir / "todos" / "today.md"
        monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
        return day, todo_path

    return _create


def test_todo_list_returns_numbered_view(todo_env):
    """todo_list should number checklist entries starting at 1."""

    day, _ = todo_env(["- [ ] First item", "- [ ] Second item"])

    result = call_tool(mcp_tools.todo_list, day)

    assert result == {
        "day": day,
        "markdown": "1: - [ ] First item\n2: - [ ] Second item",
    }


def test_todo_add_requires_next_line(todo_env):
    """todo_add should reject mismatched next line numbers."""

    day, _ = todo_env(["- [ ] First item"])

    result = call_tool(mcp_tools.todo_add, day, line_number=1, text="Second item")

    assert result["error"] == "line number 1 must match the next available line"


def test_todo_add_appends_entry(todo_env):
    """todo_add should append using the provided text and update storage."""

    day, todo_path = todo_env(["- [ ] First item"])

    result = call_tool(mcp_tools.todo_add, day, line_number=2, text="Second item")

    assert result == {
        "day": day,
        "markdown": "1: - [ ] First item\n2: - [ ] Second item",
    }
    assert todo_path.read_text(encoding="utf-8") == (
        "- [ ] First item\n- [ ] Second item\n"
    )


def test_todo_remove_validates_guard(todo_env):
    """todo_remove should validate the guard string before deleting."""

    day, _ = todo_env(["- [ ] First item", "- [ ] Second item"])

    result = call_tool(mcp_tools.todo_remove, day, line_number=2, guard="- [ ] Other")

    assert result["error"] == "guard text does not match current todo"


def test_todo_remove_updates_file(todo_env):
    """todo_remove should delete the requested entry and renumber output."""

    day, todo_path = todo_env(["- [ ] First item", "- [ ] Second item"])

    result = call_tool(
        mcp_tools.todo_remove, day, line_number=2, guard="- [ ] Second item"
    )

    assert result == {"day": day, "markdown": "1: - [ ] First item"}
    assert todo_path.read_text(encoding="utf-8") == "- [ ] First item\n"


def test_todo_done_marks_complete(todo_env):
    """todo_done should mark the entry as completed in storage."""

    day, todo_path = todo_env(["- [ ] First item", "- [ ] Second item"])

    result = call_tool(
        mcp_tools.todo_done, day, line_number=2, guard="- [ ] Second item"
    )

    assert result == {
        "day": day,
        "markdown": "1: - [ ] First item\n2: - [x] Second item",
    }
    assert todo_path.read_text(encoding="utf-8") == (
        "- [ ] First item\n- [x] Second item\n"
    )


def test_todo_add_validates_domain(todo_env, tmp_path):
    """todo_add should validate domain and provide helpful error for unknown domains."""

    # Create domains with valid domain files
    domains_dir = tmp_path / "domains"
    domains_dir.mkdir(parents=True)
    for domain in ["work", "personal", "hobby"]:
        domain_path = domains_dir / domain
        domain_path.mkdir(parents=True)
        domain_json = domain_path / "domain.json"
        domain_json.write_text(f'{{"title": "{domain.title()}"}}', encoding="utf-8")

    day, _ = todo_env([])

    result = call_tool(mcp_tools.todo_add, day, 1, "Test task #invalid")

    assert "error" in result
    assert "Unknown domain: invalid" in result["error"]
    assert "suggestion" in result
    # The domains might be returned in any order, so check for each one
    assert "work" in result["suggestion"]
    assert "personal" in result["suggestion"]
    assert "hobby" in result["suggestion"]
