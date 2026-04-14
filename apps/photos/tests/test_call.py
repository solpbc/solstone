# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

import json
import sqlite3
import sys
from pathlib import Path

from typer.testing import CliRunner

from think.call import call_app
from think.indexer.journal import scan_journal

runner = CliRunner()


def _create_photos_db(
    db_path: Path, people: list[tuple[int, str | None]], faces: list[tuple[int, int, int]]
) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "CREATE TABLE ZPERSON (Z_PK INTEGER PRIMARY KEY, ZFULLNAME TEXT, ZMERGEDINTO INTEGER)"
        )
        conn.execute(
            "CREATE TABLE ZASSET (Z_PK INTEGER PRIMARY KEY, ZDATECREATED REAL)"
        )
        conn.execute(
            "CREATE TABLE ZDETECTEDFACE (Z_PK INTEGER PRIMARY KEY, ZPERSON INTEGER, ZASSET INTEGER)"
        )
        conn.executemany(
            "INSERT INTO ZPERSON (Z_PK, ZFULLNAME, ZMERGEDINTO) VALUES (?, ?, NULL)",
            people,
        )
        conn.executemany(
            "INSERT INTO ZASSET (Z_PK, ZDATECREATED) VALUES (?, ?)",
            [
                (1, 730000000),
                (2, 730086400),
            ],
        )
        conn.executemany(
            "INSERT INTO ZDETECTEDFACE (Z_PK, ZPERSON, ZASSET) VALUES (?, ?, ?)",
            faces,
        )
        conn.commit()
    finally:
        conn.close()


def _create_journal(journal_dir: Path, entities: list[dict]) -> None:
    for entity in entities:
        entity_dir = journal_dir / "entities" / entity["id"]
        entity_dir.mkdir(parents=True, exist_ok=True)
        (entity_dir / "entity.json").write_text(json.dumps(entity), encoding="utf-8")
    scan_journal(str(journal_dir), full=True)


def _photo_signal_count(journal_dir: Path) -> int:
    conn = sqlite3.connect(journal_dir / "indexer" / "journal.sqlite")
    try:
        row = conn.execute(
            "SELECT COUNT(*) FROM entity_signals WHERE signal_type='photo_cooccurrence'"
        ).fetchone()
        return int(row[0] or 0) if row else 0
    finally:
        conn.close()


class TestPhotosSync:
    def test_non_macos_exits(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "linux")

        result = runner.invoke(call_app, ["photos", "sync"])

        assert result.exit_code != 0
        assert "macOS" in result.output

    def test_missing_db_exits(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sys, "platform", "darwin")

        result = runner.invoke(
            call_app,
            ["photos", "sync", "--library", str(tmp_path / "missing.sqlite")],
        )

        assert result.exit_code != 0
        assert "not found" in result.output

    def test_sync_with_mock_photos_db(self, tmp_path, monkeypatch):
        photos_db = tmp_path / "Photos.sqlite"
        journal_dir = tmp_path / "journal"
        journal_dir.mkdir()
        _create_photos_db(
            photos_db,
            [(1, "Alice Johnson")],
            [(1, 1, 1), (2, 1, 2)],
        )
        _create_journal(
            journal_dir,
            [{"id": "alice_johnson", "name": "Alice Johnson", "type": "Person"}],
        )

        monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(journal_dir))
        monkeypatch.setattr(sys, "platform", "darwin")

        result = runner.invoke(
            call_app,
            ["photos", "sync", "--library", str(photos_db)],
        )

        assert result.exit_code == 0
        assert "Found 1 named face clusters." in result.output
        assert "Matched 1 to entities." in result.output
        assert "Created 2 photo signals." in result.output

        conn = sqlite3.connect(journal_dir / "indexer" / "journal.sqlite")
        try:
            rows = conn.execute(
                """
                SELECT entity_name, day, path
                FROM entity_signals
                WHERE signal_type='photo_cooccurrence'
                ORDER BY day
                """
            ).fetchall()
        finally:
            conn.close()

        assert rows == [
            ("Alice Johnson", "20240218", "photos/1/20240218"),
            ("Alice Johnson", "20240219", "photos/1/20240219"),
        ]

    def test_idempotent_sync(self, tmp_path, monkeypatch):
        photos_db = tmp_path / "Photos.sqlite"
        journal_dir = tmp_path / "journal"
        journal_dir.mkdir()
        _create_photos_db(
            photos_db,
            [(1, "Alice Johnson")],
            [(1, 1, 1), (2, 1, 2)],
        )
        _create_journal(
            journal_dir,
            [{"id": "alice_johnson", "name": "Alice Johnson", "type": "Person"}],
        )

        monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(journal_dir))
        monkeypatch.setattr(sys, "platform", "darwin")

        first = runner.invoke(call_app, ["photos", "sync", "--library", str(photos_db)])
        second = runner.invoke(
            call_app,
            ["photos", "sync", "--library", str(photos_db)],
        )

        assert first.exit_code == 0
        assert second.exit_code == 0
        assert _photo_signal_count(journal_dir) == 2

    def test_zero_faces(self, tmp_path, monkeypatch):
        photos_db = tmp_path / "Photos.sqlite"
        _create_photos_db(photos_db, [], [])

        monkeypatch.setattr(sys, "platform", "darwin")

        result = runner.invoke(
            call_app,
            ["photos", "sync", "--library", str(photos_db)],
        )

        assert result.exit_code == 0
        assert "Found 0 named face clusters." in result.output

    def test_zero_matches(self, tmp_path, monkeypatch):
        photos_db = tmp_path / "Photos.sqlite"
        journal_dir = tmp_path / "journal"
        journal_dir.mkdir()
        _create_photos_db(
            photos_db,
            [(1, "Unmatched Person")],
            [(1, 1, 1)],
        )

        monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(journal_dir))
        monkeypatch.setattr(sys, "platform", "darwin")

        result = runner.invoke(
            call_app,
            ["photos", "sync", "--library", str(photos_db)],
        )

        assert result.exit_code == 0
        assert "Found 1 named face clusters." in result.output
        assert "Matched 0 to entities." in result.output

    def test_strength_includes_photo_count(self, tmp_path, monkeypatch):
        photos_db = tmp_path / "Photos.sqlite"
        journal_dir = tmp_path / "journal"
        journal_dir.mkdir()
        _create_photos_db(
            photos_db,
            [(1, "Alice Johnson")],
            [(1, 1, 1), (2, 1, 2)],
        )
        _create_journal(
            journal_dir,
            [{"id": "alice_johnson", "name": "Alice Johnson", "type": "Person"}],
        )

        monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(journal_dir))
        monkeypatch.setattr(sys, "platform", "darwin")

        runner.invoke(call_app, ["photos", "sync", "--library", str(photos_db)])

        from think.indexer.journal import get_entity_strength

        results = get_entity_strength()
        alice = next((r for r in results if r.get("entity_id") == "alice_johnson"), None)
        assert alice is not None
        assert "photo_count" in alice
        assert alice["photo_count"] == 2
        assert alice["score"] > 0

    def test_fallback_tables(self, tmp_path, monkeypatch):
        photos_db = tmp_path / "Photos.sqlite"
        journal_dir = tmp_path / "journal"
        journal_dir.mkdir()
        conn = sqlite3.connect(photos_db)
        try:
            conn.execute(
                "CREATE TABLE ZGENERICPERSON (Z_PK INTEGER PRIMARY KEY, ZFULLNAME TEXT, ZMERGEDINTO INTEGER)"
            )
            conn.execute(
                "CREATE TABLE ZGENERICASSET (Z_PK INTEGER PRIMARY KEY, ZDATECREATED REAL)"
            )
            conn.execute(
                "CREATE TABLE ZDETECTEDFACE (Z_PK INTEGER PRIMARY KEY, ZPERSON INTEGER, ZASSET INTEGER)"
            )
            conn.execute(
                "INSERT INTO ZGENERICPERSON (Z_PK, ZFULLNAME, ZMERGEDINTO) VALUES (1, 'Alice Johnson', NULL)"
            )
            conn.execute(
                "INSERT INTO ZGENERICASSET (Z_PK, ZDATECREATED) VALUES (1, 730000000)"
            )
            conn.execute(
                "INSERT INTO ZDETECTEDFACE (Z_PK, ZPERSON, ZASSET) VALUES (1, 1, 1)"
            )
            conn.commit()
        finally:
            conn.close()
        _create_journal(
            journal_dir,
            [{"id": "alice_johnson", "name": "Alice Johnson", "type": "Person"}],
        )

        monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(journal_dir))
        monkeypatch.setattr(sys, "platform", "darwin")

        result = runner.invoke(call_app, ["photos", "sync", "--library", str(photos_db)])
        assert result.exit_code == 0
        assert "Found 1 named face clusters." in result.output
