# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for journal archive validation."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import solstone.think.importers.journal_archive as journal_archive


def _write_zip(path: Path, members: dict[str, str]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)


def _write_zip_infos(path: Path, members: list[tuple[zipfile.ZipInfo, str]]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for info, payload in members:
            archive.writestr(info, payload)


def test_validate_journal_archive_rejects_missing_file(tmp_path):
    archive_path = tmp_path / "missing.zip"

    result = journal_archive.validate_journal_archive(archive_path)

    assert result.ok is False
    assert result.warnings[-1].code == "archive-not-found"


def test_validate_journal_archive_accepts_flat_layout(tmp_path):
    archive_path = tmp_path / "flat.zip"
    _write_zip(
        archive_path,
        {
            "chronicle/": "",
            "chronicle/20260101/default/090000_300/audio.jsonl": "{}\n",
            "entities/alice/entity.json": "{}\n",
            "facets/work/facet.json": "{}\n",
            "imports/20260101_090000/manifest.json": "{}\n",
            "_export.json": json.dumps(
                {
                    "solstone_version": "0.1.0",
                    "exported_at": "2026-04-26T20:00:00Z",
                    "source_journal": "/tmp/source",
                    "day_count": 1,
                    "entity_count": 1,
                    "facet_count": 1,
                }
            ),
        },
    )

    result = journal_archive.validate_journal_archive(archive_path)

    assert result.ok is True
    assert result.root_prefix == ""
    assert result.day_count == 1
    assert result.entity_count == 1
    assert result.facet_count == 1
    assert result.warnings == []


def test_validate_journal_archive_accepts_wrapper_layout(tmp_path):
    archive_path = tmp_path / "wrapped.zip"
    _write_zip(
        archive_path,
        {
            "snapshot/chronicle/20260101/default/090000_300/audio.jsonl": "{}\n",
            "snapshot/entities/alice/entity.json": "{}\n",
            "snapshot/facets/work/facet.json": "{}\n",
            "snapshot/imports/20260101_090000/manifest.json": "{}\n",
            "snapshot/_export.json": json.dumps(
                {
                    "solstone_version": "0.1.0",
                    "exported_at": "2026-04-26T20:00:00Z",
                    "source_journal": "/tmp/source",
                    "day_count": 1,
                    "entity_count": 1,
                    "facet_count": 1,
                }
            ),
        },
    )

    result = journal_archive.validate_journal_archive(archive_path)

    assert result.ok is True
    assert result.root_prefix == "snapshot/"


def test_validate_journal_archive_rejects_invalid_structure(tmp_path):
    archive_path = tmp_path / "structure.zip"
    _write_zip(archive_path, {"notes/readme.txt": "hello\n"})

    result = journal_archive.validate_journal_archive(archive_path)

    assert result.ok is False
    assert result.warnings[-1].code == "archive-structure-invalid"


def test_validate_journal_archive_rejects_invalid_and_encrypted_zip(
    tmp_path, monkeypatch
):
    invalid_path = tmp_path / "invalid.zip"
    invalid_path.write_text("not a zip", encoding="utf-8")
    invalid_result = journal_archive.validate_journal_archive(invalid_path)
    assert invalid_result.ok is False
    assert invalid_result.warnings[-1].code == "archive-invalid-zip"

    archive_path = tmp_path / "encrypted.zip"
    _write_zip(
        archive_path, {"chronicle/20260101/default/090000_300/audio.jsonl": "{}\n"}
    )

    class EncryptedInfo:
        filename = "chronicle/20260101/default/090000_300/audio.jsonl"
        flag_bits = 0x1

    monkeypatch.setattr(
        journal_archive.zipfile.ZipFile,
        "infolist",
        lambda self: [EncryptedInfo()],
    )

    encrypted_result = journal_archive.validate_journal_archive(archive_path)
    assert encrypted_result.ok is False
    assert encrypted_result.warnings[-1].code == "archive-encrypted"


def test_validate_journal_archive_rejects_too_large_archive(tmp_path):
    archive_path = tmp_path / "large.zip"
    _write_zip(
        archive_path, {"chronicle/20260101/default/090000_300/audio.jsonl": "{}"}
    )

    result = journal_archive.validate_journal_archive(archive_path, max_size_bytes=1)

    assert result.ok is False
    assert result.warnings[-1].code == "archive-too-large"


def test_validate_journal_archive_manifest_warnings(tmp_path):
    archive_path = tmp_path / "warnings.zip"
    _write_zip(
        archive_path,
        {
            "__MACOSX/ignored.txt": "ignored\n",
            "snapshot/entities/alice/entity.json": "{}\n",
            "snapshot/facets/work/facet.json": "{}\n",
            "snapshot/_export.json": json.dumps(
                {
                    "solstone_version": "0.1.0",
                    "day_count": 99,
                    "entity_count": 3,
                }
            ),
            "snapshot/.DS_Store": "ignored\n",
        },
    )

    result = journal_archive.validate_journal_archive(archive_path)

    assert result.ok is True
    codes = [warning.code for warning in result.warnings]
    assert "manifest-fields-missing" in codes
    assert "manifest-count-mismatch" in codes
    assert "chronicle-missing" in codes


def test_validate_journal_archive_warns_for_missing_and_unparseable_manifest(tmp_path):
    missing_manifest_path = tmp_path / "missing-manifest.zip"
    _write_zip(
        missing_manifest_path,
        {
            "__MACOSX/ignored.txt": "ignored\n",
            "chronicle/20260101/default/090000_300/audio.jsonl": "{}\n",
            ".DS_Store": "ignored\n",
        },
    )

    missing_result = journal_archive.validate_journal_archive(missing_manifest_path)

    assert missing_result.ok is True
    assert [warning.code for warning in missing_result.warnings] == ["manifest-missing"]

    unparseable_manifest_path = tmp_path / "unparseable-manifest.zip"
    _write_zip(
        unparseable_manifest_path,
        {
            "chronicle/20260101/default/090000_300/audio.jsonl": "{}\n",
            "_export.json": "{not-json",
        },
    )

    unparseable_result = journal_archive.validate_journal_archive(
        unparseable_manifest_path
    )

    assert unparseable_result.ok is True
    assert [warning.code for warning in unparseable_result.warnings] == [
        "manifest-unparseable"
    ]


def test_validate_journal_archive_rejects_absolute_member(tmp_path):
    archive_path = tmp_path / "absolute.zip"
    _write_zip(
        archive_path,
        {
            "chronicle/20260101/default/090000_300/audio.jsonl": "{}\n",
            "/etc/passwd": "unsafe\n",
        },
    )

    result = journal_archive.validate_journal_archive(archive_path)

    assert result.ok is False
    assert result.warnings[-1].code == "archive-unsafe-path"


def test_validate_journal_archive_rejects_parent_traversal_member(tmp_path):
    archive_path = tmp_path / "traversal.zip"
    _write_zip(
        archive_path,
        {
            "chronicle/20260101/default/090000_300/audio.jsonl": "{}\n",
            "../escape.txt": "unsafe\n",
        },
    )

    result = journal_archive.validate_journal_archive(archive_path)

    assert result.ok is False
    assert result.warnings[-1].code == "archive-unsafe-path"


def test_validate_journal_archive_rejects_symlink_member(tmp_path):
    archive_path = tmp_path / "symlink.zip"
    symlink_info = zipfile.ZipInfo("chronicle/20260101/default/link")
    symlink_info.external_attr = 0xA1ED << 16
    safe_info = zipfile.ZipInfo("chronicle/20260101/default/090000_300/audio.jsonl")

    _write_zip_infos(
        archive_path,
        [
            (safe_info, "{}\n"),
            (symlink_info, "target\n"),
        ],
    )

    result = journal_archive.validate_journal_archive(archive_path)

    assert result.ok is False
    assert result.warnings[-1].code == "archive-unsafe-path"
