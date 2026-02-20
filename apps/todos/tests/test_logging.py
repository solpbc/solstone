# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Test audit logging for todos call commands."""

import json
from datetime import datetime
from pathlib import Path

from typer.testing import CliRunner

from apps.todos.call import app

runner = CliRunner()


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
    """Test that todo add creates an audit log entry."""
    journal, facet = facet_env()
    day = datetime.now().strftime("%Y%m%d")

    result = runner.invoke(app, ["add", "Test task", "--day", day, "--facet", facet])
    assert result.exit_code == 0

    entries = read_log_entries(journal, facet, day)
    assert len(entries) == 1
    assert entries[0]["action"] == "todo_add"
    assert entries[0]["source"] == "call"
    assert entries[0]["actor"] == "agent"
    assert entries[0]["params"]["line_number"] == 1
    assert entries[0]["params"]["text"] == "Test task"
    assert "timestamp" in entries[0]


def test_todo_cancel_logging(facet_env):
    """Test that todo cancel creates an audit log entry."""
    journal, facet = facet_env()
    day = datetime.now().strftime("%Y%m%d")

    # Add and cancel a todo
    runner.invoke(app, ["add", "Test task", "--day", day, "--facet", facet])
    result = runner.invoke(app, ["cancel", "1", "--day", day, "--facet", facet])
    assert result.exit_code == 0

    entries = read_log_entries(journal, facet, day)
    assert len(entries) == 2
    assert entries[1]["action"] == "todo_cancel"
    assert entries[1]["params"]["line_number"] == 1
    assert entries[1]["params"]["text"] == "Test task"


def test_todo_done_logging(facet_env):
    """Test that todo done creates an audit log entry."""
    journal, facet = facet_env()
    day = datetime.now().strftime("%Y%m%d")

    # Add and complete a todo
    runner.invoke(app, ["add", "Test task", "--day", day, "--facet", facet])
    result = runner.invoke(app, ["done", "1", "--day", day, "--facet", facet])
    assert result.exit_code == 0

    entries = read_log_entries(journal, facet, day)
    assert len(entries) == 2
    assert entries[1]["action"] == "todo_done"
    assert entries[1]["params"]["line_number"] == 1
    assert entries[1]["params"]["text"] == "Test task"


def test_multiple_todo_actions_same_day(facet_env):
    """Test that multiple todo actions create separate log entries."""
    journal, facet = facet_env()
    day = datetime.now().strftime("%Y%m%d")

    runner.invoke(app, ["add", "Task 1", "--day", day, "--facet", facet])
    runner.invoke(app, ["add", "Task 2", "--day", day, "--facet", facet])
    runner.invoke(app, ["done", "1", "--day", day, "--facet", facet])

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

    runner.invoke(app, ["add", "Test task", "--day", day, "--facet", facet])

    # Verify logs directory was created
    assert logs_dir.exists()
    assert logs_dir.is_dir()


def test_log_timestamp_format(facet_env):
    """Test that log timestamp is in ISO format with timezone."""
    journal, facet = facet_env()
    day = datetime.now().strftime("%Y%m%d")

    runner.invoke(app, ["add", "Test task", "--day", day, "--facet", facet])

    entries = read_log_entries(journal, facet, day)
    timestamp = entries[0]["timestamp"]

    # Should be able to parse as ISO format
    parsed = datetime.fromisoformat(timestamp)
    assert parsed is not None
    # Should include timezone info
    assert "+" in timestamp or "Z" in timestamp
