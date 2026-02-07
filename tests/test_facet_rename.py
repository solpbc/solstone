# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for facet rename functionality."""

import json

import pytest

from think.facets import rename_facet


@pytest.fixture
def journal(tmp_path, monkeypatch):
    """Create a minimal journal with a facet for rename tests."""
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))

    # Create facet directory with facet.json
    facet_dir = tmp_path / "facets" / "old-name"
    facet_dir.mkdir(parents=True)
    (facet_dir / "facet.json").write_text(
        json.dumps({"title": "Old Facet", "description": "Test facet"})
    )

    # Create some facet content to verify it moves
    (facet_dir / "todos").mkdir()
    (facet_dir / "todos" / "20260101.jsonl").write_text('{"text": "Buy groceries"}\n')

    return tmp_path


def test_rename_moves_directory(journal):
    """Rename moves the facet directory."""
    rename_facet("old-name", "new-name")

    assert not (journal / "facets" / "old-name").exists()
    assert (journal / "facets" / "new-name").is_dir()
    assert (journal / "facets" / "new-name" / "facet.json").exists()

    # Content preserved
    todos = journal / "facets" / "new-name" / "todos" / "20260101.jsonl"
    assert todos.read_text().strip() == '{"text": "Buy groceries"}'


def test_rename_updates_convey_config_selected(journal):
    """Rename updates facets.selected in convey config."""
    config_dir = journal / "config"
    config_dir.mkdir(parents=True)
    (config_dir / "convey.json").write_text(
        json.dumps({"facets": {"selected": "old-name", "order": ["other"]}})
    )

    rename_facet("old-name", "new-name")

    config = json.loads((config_dir / "convey.json").read_text())
    assert config["facets"]["selected"] == "new-name"
    assert config["facets"]["order"] == ["other"]  # unchanged


def test_rename_updates_convey_config_order(journal):
    """Rename replaces old name in facets.order."""
    config_dir = journal / "config"
    config_dir.mkdir(parents=True)
    (config_dir / "convey.json").write_text(
        json.dumps(
            {"facets": {"selected": "other", "order": ["work", "old-name", "personal"]}}
        )
    )

    rename_facet("old-name", "new-name")

    config = json.loads((config_dir / "convey.json").read_text())
    assert config["facets"]["selected"] == "other"  # unchanged
    assert config["facets"]["order"] == ["work", "new-name", "personal"]


def test_rename_updates_chat_metadata(journal):
    """Rename updates facet field in matching chat metadata files."""
    chats_dir = journal / "apps" / "chat" / "chats"
    chats_dir.mkdir(parents=True)

    # Chat with matching facet
    (chats_dir / "111.json").write_text(
        json.dumps({"ts": 111, "title": "Chat 1", "facet": "old-name"})
    )
    # Chat with different facet
    (chats_dir / "222.json").write_text(
        json.dumps({"ts": 222, "title": "Chat 2", "facet": "work"})
    )
    # Chat with no facet
    (chats_dir / "333.json").write_text(json.dumps({"ts": 333, "title": "Chat 3"}))

    rename_facet("old-name", "new-name")

    chat1 = json.loads((chats_dir / "111.json").read_text())
    assert chat1["facet"] == "new-name"

    chat2 = json.loads((chats_dir / "222.json").read_text())
    assert chat2["facet"] == "work"  # unchanged

    chat3 = json.loads((chats_dir / "333.json").read_text())
    assert "facet" not in chat3  # unchanged


def test_rename_old_not_found(journal):
    """Rename fails if old facet doesn't exist."""
    with pytest.raises(ValueError, match="does not exist"):
        rename_facet("nonexistent", "new-name")


def test_rename_new_already_exists(journal):
    """Rename fails if new facet name already exists."""
    (journal / "facets" / "new-name").mkdir(parents=True)

    with pytest.raises(ValueError, match="already exists"):
        rename_facet("old-name", "new-name")


def test_rename_invalid_new_name(journal):
    """Rename fails with invalid new name."""
    with pytest.raises(ValueError, match="Invalid facet name"):
        rename_facet("old-name", "Bad Name!")

    with pytest.raises(ValueError, match="Invalid facet name"):
        rename_facet("old-name", "123start")

    with pytest.raises(ValueError, match="Invalid facet name"):
        rename_facet("old-name", "UPPER")


def test_rename_no_convey_config(journal):
    """Rename succeeds when convey config doesn't exist."""
    rename_facet("old-name", "new-name")

    assert (journal / "facets" / "new-name").is_dir()


def test_rename_no_chat_dir(journal):
    """Rename succeeds when chat directory doesn't exist."""
    rename_facet("old-name", "new-name")

    assert (journal / "facets" / "new-name").is_dir()


def test_rename_rebuilds_index(journal):
    """Rename creates a fresh search index."""
    # Create an old index file
    index_dir = journal / "indexer"
    index_dir.mkdir()
    (index_dir / "journal.sqlite").write_text("old data")

    rename_facet("old-name", "new-name")

    # Old index should be gone (reset removes it)
    # New index may or may not exist depending on content,
    # but the old one should not remain as-is
    old_content = None
    index_file = index_dir / "journal.sqlite"
    if index_file.exists():
        old_content = index_file.read_text(errors="replace")
    assert old_content != "old data"
