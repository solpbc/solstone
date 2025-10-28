"""Test audit logging for MCP tool actions."""

import json
import os
from datetime import datetime
from pathlib import Path

import pytest

from muse.mcp_tools import (
    entity_attach,
    entity_detect,
    todo_add,
    todo_done,
    todo_remove,
)


@pytest.fixture
def test_domain(tmp_path, monkeypatch):
    """Set up a test domain with JOURNAL_PATH."""
    journal = tmp_path / "journal"
    journal.mkdir()
    domain_path = journal / "domains" / "test_domain"
    domain_path.mkdir(parents=True)

    # Create domain.json
    domain_json = domain_path / "domain.json"
    domain_json.write_text(
        json.dumps({"title": "Test Domain", "description": "Test"}), encoding="utf-8"
    )

    # Create todos directory
    (domain_path / "todos").mkdir()

    monkeypatch.setenv("JOURNAL_PATH", str(journal))
    return journal, "test_domain"


def read_log_entries(journal_path: Path, domain: str, day: str) -> list[dict]:
    """Read all log entries from a domain's log file."""
    log_path = journal_path / "domains" / domain / "logs" / f"{day}.jsonl"
    if not log_path.exists():
        return []

    entries = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    return entries


def test_todo_add_logging(test_domain):
    """Test that todo_add creates an audit log entry."""
    journal, domain = test_domain
    day = datetime.now().strftime("%Y%m%d")

    # Add a todo
    result = todo_add(day, domain, 1, "Test task")
    assert "error" not in result

    # Check log entry was created
    entries = read_log_entries(journal, domain, day)
    assert len(entries) == 1
    assert entries[0]["action"] == "todo_add"
    assert entries[0]["params"]["line_number"] == 1
    assert entries[0]["params"]["text"] == "Test task"
    assert "timestamp" in entries[0]
    # Actor should be present (defaults to "mcp" in tests)
    assert entries[0]["actor"] == "mcp"
    # agent_id should be omitted when not available
    assert "agent_id" not in entries[0]


def test_todo_remove_logging(test_domain):
    """Test that todo_remove creates an audit log entry."""
    journal, domain = test_domain
    day = datetime.now().strftime("%Y%m%d")

    # Add and remove a todo
    todo_add(day, domain, 1, "Test task")
    result = todo_remove(day, domain, 1, "- [ ] Test task")
    assert "error" not in result

    # Check both log entries exist
    entries = read_log_entries(journal, domain, day)
    assert len(entries) == 2
    assert entries[1]["action"] == "todo_remove"
    assert entries[1]["params"]["line_number"] == 1
    assert entries[1]["params"]["text"] == "- [ ] Test task"


def test_todo_done_logging(test_domain):
    """Test that todo_done creates an audit log entry."""
    journal, domain = test_domain
    day = datetime.now().strftime("%Y%m%d")

    # Add and complete a todo
    todo_add(day, domain, 1, "Test task")
    result = todo_done(day, domain, 1, "- [ ] Test task")
    assert "error" not in result

    # Check both log entries exist
    entries = read_log_entries(journal, domain, day)
    assert len(entries) == 2
    assert entries[1]["action"] == "todo_done"
    assert entries[1]["params"]["line_number"] == 1
    assert entries[1]["params"]["text"] == "- [ ] Test task"


def test_entity_detect_logging(test_domain):
    """Test that entity_detect creates an audit log entry."""
    journal, domain = test_domain
    day = "20250101"

    # Detect an entity
    result = entity_detect(day, domain, "Person", "John Doe", "Test person")
    assert "error" not in result

    # Check log entry was created
    entries = read_log_entries(journal, domain, day)
    assert len(entries) == 1
    assert entries[0]["action"] == "entity_detect"
    assert entries[0]["params"]["type"] == "Person"
    assert entries[0]["params"]["name"] == "John Doe"
    assert entries[0]["params"]["description"] == "Test person"


def test_entity_attach_logging(test_domain):
    """Test that entity_attach creates an audit log entry."""
    journal, domain = test_domain
    today = datetime.now().strftime("%Y%m%d")

    # Attach an entity
    result = entity_attach(domain, "Company", "Acme Corp", "Test company")
    assert "error" not in result

    # Check log entry was created (uses today's date)
    entries = read_log_entries(journal, domain, today)
    assert len(entries) == 1
    assert entries[0]["action"] == "entity_attach"
    assert entries[0]["params"]["type"] == "Company"
    assert entries[0]["params"]["name"] == "Acme Corp"
    assert entries[0]["params"]["description"] == "Test company"


def test_multiple_actions_same_day(test_domain):
    """Test that multiple actions create separate log entries."""
    journal, domain = test_domain
    day = datetime.now().strftime("%Y%m%d")

    # Perform multiple actions
    todo_add(day, domain, 1, "Task 1")
    todo_add(day, domain, 2, "Task 2")
    entity_attach(domain, "Person", "Alice", "Test person")

    # Check all log entries exist
    entries = read_log_entries(journal, domain, day)
    assert len(entries) == 3
    assert entries[0]["action"] == "todo_add"
    assert entries[1]["action"] == "todo_add"
    assert entries[2]["action"] == "entity_attach"


def test_log_directory_created_automatically(test_domain):
    """Test that log directory is created if it doesn't exist."""
    journal, domain = test_domain
    day = datetime.now().strftime("%Y%m%d")

    # Verify logs directory doesn't exist yet
    logs_dir = journal / "domains" / domain / "logs"
    assert not logs_dir.exists()

    # Add a todo (should create logs directory)
    todo_add(day, domain, 1, "Test task")

    # Verify logs directory was created
    assert logs_dir.exists()
    assert logs_dir.is_dir()


def test_log_timestamp_format(test_domain):
    """Test that log timestamp is in ISO format with timezone."""
    journal, domain = test_domain
    day = datetime.now().strftime("%Y%m%d")

    # Add a todo
    todo_add(day, domain, 1, "Test task")

    # Check timestamp format
    entries = read_log_entries(journal, domain, day)
    timestamp = entries[0]["timestamp"]

    # Should be able to parse as ISO format
    parsed = datetime.fromisoformat(timestamp)
    assert parsed is not None
    # Should include timezone info
    assert "+" in timestamp or "Z" in timestamp
