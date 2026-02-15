# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

import importlib
import sqlite3
from unittest.mock import patch

import pytest

# Module name starts with a digit — use importlib
_mod = importlib.import_module("apps.search.maint.003_migrate_index_stream")
migrate = _mod.migrate


@pytest.fixture(autouse=True)
def _set_journal(tmp_path, monkeypatch):
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))


def _create_old_schema(tmp_path):
    """Create a database with the pre-stream schema."""
    db_dir = tmp_path / "indexer"
    db_dir.mkdir()
    db_path = db_dir / "journal.sqlite"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE files(path TEXT PRIMARY KEY, mtime INTEGER)")
    conn.execute("""
        CREATE VIRTUAL TABLE chunks USING fts5(
            content,
            path UNINDEXED,
            day UNINDEXED,
            facet UNINDEXED,
            topic UNINDEXED,
            idx UNINDEXED
        )
    """)
    conn.execute(
        "INSERT INTO chunks(content, path, day, facet, topic, idx) "
        "VALUES ('test', 'test.md', '20240101', 'work', 'flow', 0)"
    )
    conn.commit()
    conn.close()
    return str(db_path)


def _create_current_schema(tmp_path):
    """Create a database with the current schema (has stream)."""
    db_dir = tmp_path / "indexer"
    db_dir.mkdir()
    db_path = db_dir / "journal.sqlite"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE files(path TEXT PRIMARY KEY, mtime INTEGER)")
    conn.execute("""
        CREATE VIRTUAL TABLE chunks USING fts5(
            content,
            path UNINDEXED,
            day UNINDEXED,
            facet UNINDEXED,
            topic UNINDEXED,
            stream UNINDEXED,
            idx UNINDEXED
        )
    """)
    conn.commit()
    conn.close()
    return str(db_path)


def test_migrate_old_schema_rebuilds(tmp_path):
    """Old schema (missing stream) is detected and rebuilt."""
    db_path = _create_old_schema(tmp_path)

    with patch.object(_mod, "_request_full_rescan") as mock_rescan:
        mock_rescan.return_value = True
        result = migrate(str(tmp_path))

    assert result is True
    mock_rescan.assert_called_once()

    # Verify new schema accepts stream column
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO chunks(content, path, day, facet, topic, stream, idx) "
        "VALUES ('test', 'test.md', '20240101', 'work', 'flow', 'archon', 0)"
    )
    row = conn.execute("SELECT stream FROM chunks").fetchone()
    assert row[0] == "archon"

    # Old data should be gone
    count = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
    assert count == 1  # only the one we just inserted
    conn.close()


def test_migrate_current_schema_noop(tmp_path):
    """Current schema (has stream) is detected as no-op."""
    _create_current_schema(tmp_path)

    with patch.object(_mod, "_request_full_rescan") as mock_rescan:
        result = migrate(str(tmp_path))

    assert result is False
    mock_rescan.assert_not_called()


def test_migrate_no_database_noop(tmp_path):
    """No existing database is detected as no-op."""
    result = migrate(str(tmp_path))
    assert result is False


def test_migrate_rescan_failure_still_succeeds(tmp_path):
    """Migration succeeds even if supervisor is unreachable."""
    _create_old_schema(tmp_path)

    with patch.object(_mod, "_request_full_rescan") as mock_rescan:
        mock_rescan.return_value = False
        result = migrate(str(tmp_path))

    assert result is True
    mock_rescan.assert_called_once()
