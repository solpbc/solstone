# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Test audit logging for todos app MCP tool actions."""

import json
from datetime import datetime
from pathlib import Path

from apps.todos.tools import todo_add, todo_cancel, todo_done


def read_log_entries(journal_path: Path, facet: str, day: str) -> list[dict]:
    """Read all log entries from a facet's log file."""
    log_path = journal_path / "facets" / facet / "logs" / f"{day}.jsonl"
    if not log_path.exists():
        return []

    entries = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    return entries


def test_todo_add_logging(facet_env):
    """Test that todo_add creates an audit log entry."""
    journal, facet = facet_env()
    day = datetime.now().strftime("%Y%m%d")

    # Add a todo
    result = todo_add(day, facet, 1, "Test task")
    assert "error" not in result

    # Check log entry was created
    entries = read_log_entries(journal, facet, day)
    assert len(entries) == 1
    assert entries[0]["action"] == "todo_add"
    assert entries[0]["params"]["line_number"] == 1
    assert entries[0]["params"]["text"] == "Test task"
    assert "timestamp" in entries[0]
    # Actor should be present (defaults to "mcp" in tests)
    assert entries[0]["actor"] == "mcp"
    # agent_id should be omitted when not available
    assert "agent_id" not in entries[0]


def test_todo_cancel_logging(facet_env):
    """Test that todo_cancel creates an audit log entry."""
    journal, facet = facet_env()
    day = datetime.now().strftime("%Y%m%d")

    # Add and cancel a todo
    todo_add(day, facet, 1, "Test task")
    result = todo_cancel(day, facet, 1)
    assert "error" not in result

    # Check both log entries exist
    entries = read_log_entries(journal, facet, day)
    assert len(entries) == 2
    assert entries[1]["action"] == "todo_cancel"
    assert entries[1]["params"]["line_number"] == 1
    assert entries[1]["params"]["text"] == "Test task"


def test_todo_done_logging(facet_env):
    """Test that todo_done creates an audit log entry."""
    journal, facet = facet_env()
    day = datetime.now().strftime("%Y%m%d")

    # Add and complete a todo
    todo_add(day, facet, 1, "Test task")
    result = todo_done(day, facet, 1)
    assert "error" not in result

    # Check both log entries exist
    entries = read_log_entries(journal, facet, day)
    assert len(entries) == 2
    assert entries[1]["action"] == "todo_done"
    assert entries[1]["params"]["line_number"] == 1
    assert entries[1]["params"]["text"] == "Test task"


def test_multiple_todo_actions_same_day(facet_env):
    """Test that multiple todo actions create separate log entries."""
    journal, facet = facet_env()
    day = datetime.now().strftime("%Y%m%d")

    # Perform multiple actions
    todo_add(day, facet, 1, "Task 1")
    todo_add(day, facet, 2, "Task 2")
    todo_done(day, facet, 1)

    # Check all log entries exist
    entries = read_log_entries(journal, facet, day)
    assert len(entries) == 3
    assert entries[0]["action"] == "todo_add"
    assert entries[1]["action"] == "todo_add"
    assert entries[2]["action"] == "todo_done"


def test_log_directory_created_automatically(facet_env):
    """Test that log directory is created if it doesn't exist."""
    journal, facet = facet_env()
    day = datetime.now().strftime("%Y%m%d")

    # Verify logs directory doesn't exist yet
    logs_dir = journal / "facets" / facet / "logs"
    assert not logs_dir.exists()

    # Add a todo (should create logs directory)
    todo_add(day, facet, 1, "Test task")

    # Verify logs directory was created
    assert logs_dir.exists()
    assert logs_dir.is_dir()


def test_log_timestamp_format(facet_env):
    """Test that log timestamp is in ISO format with timezone."""
    journal, facet = facet_env()
    day = datetime.now().strftime("%Y%m%d")

    # Add a todo
    todo_add(day, facet, 1, "Test task")

    # Check timestamp format
    entries = read_log_entries(journal, facet, day)
    timestamp = entries[0]["timestamp"]

    # Should be able to parse as ISO format
    parsed = datetime.fromisoformat(timestamp)
    assert parsed is not None
    # Should include timezone info
    assert "+" in timestamp or "Z" in timestamp
