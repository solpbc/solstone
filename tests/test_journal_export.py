# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for journal export CLI and helper."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

from typer.testing import CliRunner

from think.call import call_app

runner = CliRunner()


def _read_manifest(archive_path: Path) -> dict:
    with zipfile.ZipFile(archive_path, "r") as archive:
        return json.loads(archive.read("_export.json"))


def test_journal_export_writes_zip_and_manifest(journal_copy):
    archive_path = journal_copy.parent / "journal-export.zip"

    result = runner.invoke(call_app, ["journal", "export", "--out", str(archive_path)])

    assert result.exit_code == 0
    assert result.stdout.strip() == str(archive_path.resolve())
    assert archive_path.exists()
    manifest = _read_manifest(archive_path)
    assert manifest["solstone_version"] == "0.1.0"
    assert manifest["source_journal"] == str(journal_copy.resolve())
    assert manifest["day_count"] > 0
    assert manifest["entity_count"] > 0
    assert manifest["facet_count"] > 0
    with zipfile.ZipFile(archive_path, "r") as archive:
        names = set(archive.namelist())
    assert "chronicle/" in names
    assert "entities/" in names
    assert "facets/" in names
    assert "imports/" in names


def test_journal_export_default_path_and_quiet(tmp_path, monkeypatch):
    journal_root = tmp_path / "journal"
    journal_root.mkdir()
    monkeypatch.setenv("SOLSTONE_JOURNAL", str(journal_root))

    result = runner.invoke(call_app, ["journal", "export", "--quiet"])

    assert result.exit_code == 0
    assert result.stdout == ""
    export_dir = journal_root.parent / f"{journal_root.name}.exports"
    archives = sorted(export_dir.glob("*.zip"))
    assert len(archives) == 1
    assert archives[0].exists()


def test_journal_export_service_down_bypasses_require_up(journal_copy, monkeypatch):
    import think.tools.call as call_module

    archive_path = journal_copy.parent / "journal-export.zip"

    def should_not_run():
        raise AssertionError("require_solstone should not be called for export")

    monkeypatch.setattr(call_module, "require_solstone", should_not_run)

    result = runner.invoke(call_app, ["journal", "export", "--out", str(archive_path)])

    assert result.exit_code == 0
    assert archive_path.exists()


def test_journal_export_atomic_write_failure(journal_copy, monkeypatch):
    import think.journal_export as export_module

    archive_path = journal_copy.parent / "journal-export.zip"

    def fail_replace(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(export_module.os, "replace", fail_replace)

    result = runner.invoke(call_app, ["journal", "export", "--out", str(archive_path)])

    assert result.exit_code == 1
    assert "error: failed to write archive; try:" in result.stderr
    assert not archive_path.exists()


def test_journal_export_service_up_advisory(journal_copy, monkeypatch):
    import think.tools.call as call_module

    archive_path = journal_copy.parent / "journal-export.zip"
    monkeypatch.setattr(call_module, "is_solstone_up", lambda: True)

    result = runner.invoke(call_app, ["journal", "export", "--out", str(archive_path)])

    assert result.exit_code == 0
    assert (
        "warning: solstone supervisor is running; export reflects a live snapshot"
        in result.stderr
    )


def test_journal_export_top_level_skipped_advisory(tmp_path, monkeypatch):
    journal_root = tmp_path / "journal"
    journal_root.mkdir()
    (journal_root / "misc").mkdir()
    monkeypatch.setenv("SOLSTONE_JOURNAL", str(journal_root))
    archive_path = tmp_path / "journal-export.zip"

    result = runner.invoke(call_app, ["journal", "export", "--out", str(archive_path)])

    assert result.exit_code == 0
    assert "advisory: skipped non-export entries: misc" in result.stderr


def test_journal_export_empty_journal(tmp_path, monkeypatch):
    journal_root = tmp_path / "journal"
    journal_root.mkdir()
    monkeypatch.setenv("SOLSTONE_JOURNAL", str(journal_root))
    archive_path = tmp_path / "journal-export.zip"

    result = runner.invoke(call_app, ["journal", "export", "--out", str(archive_path)])

    assert result.exit_code == 0
    manifest = _read_manifest(archive_path)
    assert manifest["day_count"] == 0
    assert manifest["entity_count"] == 0
    assert manifest["facet_count"] == 0
    with zipfile.ZipFile(archive_path, "r") as archive:
        assert {"chronicle/", "entities/", "facets/", "imports/"} <= set(
            archive.namelist()
        )
