# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for think.importers.shared JSONL format writing."""

import json
import tempfile
from pathlib import Path

from think.importers.shared import _write_import_jsonl


def test_write_import_jsonl_with_entries():
    """Test writing imported JSONL with entries."""
    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = Path(tmpdir) / "test_audio.jsonl"

        entries = [
            {"start": "00:00:01", "text": "Hello world"},
            {"start": "00:00:05", "text": "How are you?"},
        ]

        _write_import_jsonl(
            str(json_path),
            entries,
            import_id="20240101_120000",
            facet="test",
            setting="personal",
        )

        # Read and verify JSONL format
        with open(json_path, "r") as f:
            lines = f.read().strip().split("\n")

        assert len(lines) == 3

        # First line should be metadata
        metadata = json.loads(lines[0])
        assert metadata == {
            "imported": {
                "id": "20240101_120000",
                "facet": "test",
                "setting": "personal",
            }
        }

        # Second line should be first entry with source="import"
        entry1 = json.loads(lines[1])
        assert entry1["start"] == "00:00:01"
        assert entry1["text"] == "Hello world"
        assert entry1["source"] == "import"

        # Third line should be second entry with source="import"
        entry2 = json.loads(lines[2])
        assert entry2["start"] == "00:00:05"
        assert entry2["text"] == "How are you?"
        assert entry2["source"] == "import"


def test_write_import_jsonl_no_entries():
    """Test writing imported JSONL with no entries (only metadata)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = Path(tmpdir) / "test_audio.jsonl"

        _write_import_jsonl(
            str(json_path), [], import_id="20240101_120000", facet="test"
        )

        # Read and verify JSONL format
        with open(json_path, "r") as f:
            lines = f.read().strip().split("\n")

        assert len(lines) == 1

        # Only line should be metadata
        metadata = json.loads(lines[0])
        assert metadata == {"imported": {"id": "20240101_120000", "facet": "test"}}


def test_write_import_jsonl_minimal():
    """Test writing imported JSONL with minimal metadata."""
    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = Path(tmpdir) / "test_audio.jsonl"

        entries = [{"start": "00:00:01", "text": "Test"}]

        _write_import_jsonl(str(json_path), entries, import_id="20240101_120000")

        # Read and verify JSONL format
        with open(json_path, "r") as f:
            lines = f.read().strip().split("\n")

        assert len(lines) == 2

        # First line should be metadata with only id
        metadata = json.loads(lines[0])
        assert metadata == {"imported": {"id": "20240101_120000"}}

        # Second line should be entry with source="import"
        entry = json.loads(lines[1])
        assert entry["start"] == "00:00:01"
        assert entry["text"] == "Test"
        assert entry["source"] == "import"
