"""Integration tests for MCP todo tools using fixture journal data."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from muse import mcp_tools


def call_tool(tool, *args, **kwargs):
    """Invoke MCP tool callable without transport."""

    return tool.fn(*args, **kwargs)


FIXTURES_JOURNAL = Path(__file__).resolve().parents[2] / "fixtures" / "journal"


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

    list_result = call_tool(mcp_tools.todo_list, day, facet)
    assert list_result == {
        "day": day,
        "facet": facet,
        "markdown": "1: - [ ] Fixture task",
    }

    add_result = call_tool(
        mcp_tools.todo_add, day, facet, line_number=2, text="Follow up task"
    )
    # Check for error first
    if "error" in add_result:
        raise AssertionError(f"todo_add failed: {add_result}")
    assert add_result["markdown"].splitlines() == [
        "1: - [ ] Fixture task",
        "2: - [ ] Follow up task",
    ]

    done_result = call_tool(
        mcp_tools.todo_done, day, facet, line_number=2, guard="- [ ] Follow up task"
    )
    assert done_result["markdown"].splitlines() == [
        "1: - [ ] Fixture task",
        "2: - [x] Follow up task",
    ]

    removed_done = call_tool(
        mcp_tools.todo_remove, day, facet, line_number=2, guard="- [x] Follow up task"
    )
    assert removed_done == {
        "day": day,
        "facet": facet,
        "markdown": "1: - [ ] Fixture task",
    }

    empty_result = call_tool(
        mcp_tools.todo_remove, day, facet, line_number=1, guard="- [ ] Fixture task"
    )
    assert empty_result == {
        "day": day,
        "facet": facet,
        "markdown": "0: (no todos)",
    }

    assert todo_path.read_text(encoding="utf-8") == ""
