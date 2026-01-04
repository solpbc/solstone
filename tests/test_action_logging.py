# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Test audit logging for MCP entity tool actions."""

import json
from datetime import datetime
from pathlib import Path

import pytest

from apps.entities.tools import entity_attach, entity_detect
from apps.utils import log_app_action


@pytest.fixture
def test_facet(tmp_path, monkeypatch):
    """Set up a test facet with JOURNAL_PATH."""
    journal = tmp_path / "journal"
    journal.mkdir()
    facet_path = journal / "facets" / "test_facet"
    facet_path.mkdir(parents=True)

    # Create facet.json
    facet_json = facet_path / "facet.json"
    facet_json.write_text(
        json.dumps({"title": "Test Facet", "description": "Test"}), encoding="utf-8"
    )

    monkeypatch.setenv("JOURNAL_PATH", str(journal))
    return journal, "test_facet"


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


def read_journal_log_entries(journal_path: Path, day: str) -> list[dict]:
    """Read all log entries from the journal-level log file."""
    log_path = journal_path / "config" / "actions" / f"{day}.jsonl"
    if not log_path.exists():
        return []

    entries = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    return entries


def test_entity_detect_logging(test_facet):
    """Test that entity_detect creates an audit log entry."""
    journal, facet = test_facet
    day = "20250101"

    # Detect an entity
    result = entity_detect(day, facet, "Person", "John Doe", "Test person")
    assert "error" not in result

    # Check log entry was created
    entries = read_log_entries(journal, facet, day)
    assert len(entries) == 1
    assert entries[0]["action"] == "entity_detect"
    assert entries[0]["params"]["type"] == "Person"
    assert entries[0]["params"]["name"] == "John Doe"
    assert entries[0]["params"]["description"] == "Test person"


def test_entity_attach_logging(test_facet):
    """Test that entity_attach creates an audit log entry."""
    journal, facet = test_facet
    today = datetime.now().strftime("%Y%m%d")

    # Attach an entity
    result = entity_attach(facet, "Company", "Acme Corp", "Test company")
    assert "error" not in result

    # Check log entry was created (uses today's date)
    entries = read_log_entries(journal, facet, today)
    assert len(entries) == 1
    assert entries[0]["action"] == "entity_attach"
    assert entries[0]["params"]["type"] == "Company"
    assert entries[0]["params"]["name"] == "Acme Corp"
    assert entries[0]["params"]["description"] == "Test company"


def test_journal_level_logging(test_facet):
    """Test that log_app_action with facet=None writes to config/actions/."""
    journal, _ = test_facet
    today = datetime.now().strftime("%Y%m%d")

    # Log a journal-level action (no facet)
    log_app_action(
        app="settings",
        facet=None,
        action="identity_update",
        params={"name": "Test User"},
    )

    # Check log entry was created in config/actions/
    entries = read_journal_log_entries(journal, today)
    assert len(entries) == 1
    assert entries[0]["action"] == "identity_update"
    assert entries[0]["source"] == "app"
    assert entries[0]["actor"] == "settings"
    assert entries[0]["params"]["name"] == "Test User"
    # Should NOT have a facet field
    assert "facet" not in entries[0]


def test_journal_level_log_directory_created(test_facet):
    """Test that config/actions/ directory is created automatically."""
    journal, _ = test_facet

    # Verify config/actions doesn't exist yet
    actions_dir = journal / "config" / "actions"
    assert not actions_dir.exists()

    # Log a journal-level action
    log_app_action(
        app="remote",
        facet=None,
        action="observer_create",
        params={"name": "test-observer"},
    )

    # Verify directory was created
    assert actions_dir.exists()
    assert actions_dir.is_dir()


def test_facet_action_includes_facet_field(test_facet):
    """Test that facet-scoped actions include the facet field in the entry."""
    journal, facet = test_facet
    today = datetime.now().strftime("%Y%m%d")

    # Log a facet-scoped action
    log_app_action(
        app="todos",
        facet=facet,
        action="todo_add",
        params={"text": "Test todo"},
    )

    # Check that the facet field is included
    entries = read_log_entries(journal, facet, today)
    assert len(entries) == 1
    assert entries[0]["facet"] == facet
