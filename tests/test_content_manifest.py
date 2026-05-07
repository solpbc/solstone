# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

import datetime as dt
import json
import zipfile

from solstone.think.importers.chatgpt import ChatGPTImporter
from solstone.think.importers.ics import ICSImporter
from solstone.think.importers.shared import (
    map_items_to_segments,
    write_content_manifest,
)
from solstone.think.importers.utils import generate_content_manifest


def test_write_content_manifest(tmp_path, monkeypatch):
    monkeypatch.setenv("SOLSTONE_JOURNAL", str(tmp_path))
    entries = [
        {
            "id": "conv-0",
            "title": "Conversation 1",
            "date": "20260101",
            "type": "conversation",
            "preview": "hello",
            "segments": [{"day": "20260101", "key": "100000_300"}],
        }
    ]

    manifest_path = write_content_manifest("20260101_100000", entries)

    assert (
        manifest_path
        == tmp_path / "imports" / "20260101_100000" / "content_manifest.jsonl"
    )
    lines = manifest_path.read_text(encoding="utf-8").strip().splitlines()
    assert [json.loads(line) for line in lines] == entries


def test_map_items_to_segments():
    timestamps = [
        dt.datetime(2026, 1, 1, 10, 0, 0).timestamp(),
        dt.datetime(2026, 1, 1, 10, 1, 0).timestamp(),
        dt.datetime(2026, 1, 1, 10, 10, 0).timestamp(),
    ]

    assert map_items_to_segments(timestamps, tz=None) == [
        ("20260101", "100000_300"),
        ("20260101", "100000_300"),
        ("20260101", "101000_300"),
    ]


def test_generate_content_manifest_from_segments(tmp_path):
    journal_root = tmp_path
    import_dir = journal_root / "imports" / "20260101_090000"
    segment_dir = journal_root / "chronicle" / "20260101" / "import.ics" / "090000_300"
    segment_dir.mkdir(parents=True)
    import_dir.mkdir(parents=True)

    (segment_dir / "event_transcript.md").write_text(
        "## Event One\n\nBody one.\n\n## Event Two\n\nBody two.\n",
        encoding="utf-8",
    )
    (import_dir / "imported.json").write_text(
        json.dumps(
            {
                "source_type": "ics",
                "all_created_files": [
                    "20260101/import.ics/090000_300/event_transcript.md",
                ],
            }
        ),
        encoding="utf-8",
    )

    manifest_path = generate_content_manifest(journal_root, "20260101_090000")

    assert manifest_path == import_dir / "content_manifest.jsonl"
    entries = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [entry["title"] for entry in entries] == ["Event One", "Event Two"]
    assert all(entry["type"] == "event" for entry in entries)
    assert all(
        entry["segments"] == [{"day": "20260101", "key": "090000_300"}]
        for entry in entries
    )


def test_chatgpt_importer_writes_content_manifest(tmp_path, monkeypatch):
    monkeypatch.setenv("SOLSTONE_JOURNAL", str(tmp_path))
    archive = tmp_path / "chatgpt.zip"
    conversations = [
        {
            "title": "Async help",
            "create_time": dt.datetime(2026, 1, 1, 10, 0, 0).timestamp(),
            "current_node": "assistant-node",
            "mapping": {
                "user-node": {
                    "parent": None,
                    "message": {
                        "author": {"role": "user"},
                        "content": {"parts": ["How do I debug asyncio cancellation?"]},
                        "create_time": dt.datetime(2026, 1, 1, 10, 0, 0).timestamp(),
                    },
                },
                "assistant-node": {
                    "parent": "user-node",
                    "message": {
                        "author": {"role": "assistant"},
                        "content": {"parts": ["Trace CancelledError propagation."]},
                        "create_time": dt.datetime(2026, 1, 1, 10, 1, 0).timestamp(),
                        "metadata": {"model_slug": "gpt-4o"},
                    },
                },
            },
        }
    ]
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("conversations.json", json.dumps(conversations))

    result = ChatGPTImporter().process(
        archive,
        tmp_path,
        import_id="20260101_100000",
    )

    assert result.entries_written == 2
    manifest_path = tmp_path / "imports" / "20260101_100000" / "content_manifest.jsonl"
    assert manifest_path.exists()
    entries = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert entries[0]["title"] == "Async help"
    assert entries[0]["segments"] == [{"day": "20260101", "key": "100000_300"}]


def test_ics_importer_writes_content_manifest(tmp_path, monkeypatch):
    monkeypatch.setenv("SOLSTONE_JOURNAL", str(tmp_path))
    ics_path = tmp_path / "calendar.ics"
    ics_path.write_bytes(
        b"""BEGIN:VCALENDAR
VERSION:2.0
BEGIN:VEVENT
DTSTART:20260101T170000Z
DTEND:20260101T173000Z
SUMMARY:Design Review
DESCRIPTION:Review the roadmap.
CREATED:20260101T120000Z
END:VEVENT
END:VCALENDAR"""
    )

    result = ICSImporter().process(
        ics_path,
        tmp_path,
        import_id="20260101_090000",
    )

    assert result.entries_written == 1
    manifest_path = tmp_path / "imports" / "20260101_090000" / "content_manifest.jsonl"
    assert manifest_path.exists()
    entries = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert entries[0]["title"] == "Design Review"
    assert entries[0]["type"] == "event"
    assert entries[0]["segments"] == [{"day": "20260101", "key": "120000_300"}]
