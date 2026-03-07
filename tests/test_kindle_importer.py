# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for think.importers.kindle — Kindle My Clippings.txt importer."""

import os
import tempfile
from pathlib import Path

from think.importers.kindle import KindleImporter, _parse_block, _parse_date

importer = KindleImporter()


def _make_clipping(
    title: str = "Test Book (Author Name)",
    meta: str = "- Your Highlight on page 42 | location 100-101 | Added on Saturday, March 15, 2025 10:30:00 AM",
    content: str = "This is a highlighted passage.",
) -> str:
    return f"{title}\n{meta}\n\n{content}\n"


def _make_clippings_file(clippings: list[str]) -> str:
    return "==========\n".join(clippings) + "==========\n"


# --- Unit tests for helpers ---


def test_parse_date():
    result = _parse_date("Saturday, March 15, 2025 10:30:00 AM")
    assert result is not None
    assert result.month == 3
    assert result.day == 15


def test_parse_block_basic():
    block = _make_clipping()
    entry = _parse_block(block)
    assert entry is not None
    assert entry["type"] == "highlight"
    assert entry["book_title"] == "Test Book"
    assert entry["author"] == "Author Name"
    assert entry["content"] == "This is a highlighted passage."
    assert entry["page"] == 42
    assert entry["location"] == "100-101"


def test_parse_block_note():
    block = _make_clipping(
        meta="- Your Note on page 10 | Added on Saturday, March 15, 2025 10:30:00 AM",
        content="My personal note",
    )
    entry = _parse_block(block)
    assert entry is not None
    assert entry["clip_type"] == "note"


def test_parse_block_no_author():
    block = _make_clipping(title="Title Without Author")
    entry = _parse_block(block)
    assert entry is not None
    assert entry["book_title"] == "Title Without Author"
    assert entry["author"] == ""


# --- Detection tests ---


def test_detect_valid():
    content = _make_clippings_file([_make_clipping()])
    with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
        f.write(content)
        f.flush()
        try:
            assert importer.detect(Path(f.name)) is True
        finally:
            os.unlink(f.name)


def test_detect_wrong_format():
    with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
        f.write("Just some random text file.\n")
        f.flush()
        try:
            assert importer.detect(Path(f.name)) is False
        finally:
            os.unlink(f.name)


# --- Preview tests ---


def test_preview():
    content = _make_clippings_file(
        [
            _make_clipping(),
            _make_clipping(
                title="Another Book (Jane Doe)",
                meta="- Your Note on page 5 | Added on Sunday, March 16, 2025 02:00:00 PM",
                content="A note",
            ),
        ]
    )
    with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
        f.write(content)
        f.flush()
        try:
            preview = importer.preview(Path(f.name))
            assert preview.item_count == 2
            assert preview.entity_count > 0
            assert "2 books" in preview.summary
        finally:
            os.unlink(f.name)


# --- Process tests ---


def test_process_basic():
    content = _make_clippings_file(
        [
            _make_clipping(),
            _make_clipping(
                meta="- Your Highlight on page 43 | Added on Saturday, March 15, 2025 10:31:00 AM",
                content="Another highlight from same session.",
            ),
        ]
    )
    with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
        f.write(content)
        f.flush()
        try:
            with tempfile.TemporaryDirectory() as journal:
                os.environ["JOURNAL_PATH"] = journal
                result = importer.process(Path(f.name), Path(journal))
                assert result.entries_written == 2
                assert result.errors == []
                assert result.segments is not None
                assert len(result.segments) >= 1

                md_path = Path(result.files_created[0])
                assert md_path.exists()
                assert md_path.name == "highlights_transcript.md"
                md = md_path.read_text()
                assert "Test Book" in md
                assert "> This is a highlighted passage." in md
                assert "Page 42" in md
        finally:
            os.unlink(f.name)
            os.environ.pop("JOURNAL_PATH", None)


def test_process_multiple_windows():
    """Highlights more than 5 minutes apart land in different segments."""
    content = _make_clippings_file(
        [
            _make_clipping(
                meta="- Your Highlight on page 1 | Added on Saturday, March 15, 2025 10:00:00 AM",
                content="First highlight",
            ),
            _make_clipping(
                meta="- Your Highlight on page 2 | Added on Saturday, March 15, 2025 10:10:00 AM",
                content="Second highlight, 10 min later",
            ),
        ]
    )
    with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
        f.write(content)
        f.flush()
        try:
            with tempfile.TemporaryDirectory() as journal:
                os.environ["JOURNAL_PATH"] = journal
                result = importer.process(Path(f.name), Path(journal))
                assert result.entries_written == 2
                assert result.segments is not None
                assert len(result.segments) == 2
                assert len(result.files_created) == 2
        finally:
            os.unlink(f.name)
            os.environ.pop("JOURNAL_PATH", None)


def test_process_note_markdown():
    """Notes render with Note: prefix instead of blockquote."""
    content = _make_clippings_file(
        [
            _make_clipping(
                meta="- Your Note on page 10 | Added on Saturday, March 15, 2025 10:30:00 AM",
                content="My personal note",
            ),
        ]
    )
    with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
        f.write(content)
        f.flush()
        try:
            with tempfile.TemporaryDirectory() as journal:
                os.environ["JOURNAL_PATH"] = journal
                result = importer.process(Path(f.name), Path(journal))
                md = Path(result.files_created[0]).read_text()
                assert "Note: My personal note" in md
        finally:
            os.unlink(f.name)
            os.environ.pop("JOURNAL_PATH", None)


# --- Registry test ---


def test_registered_in_registry():
    from think.importers.file_importer import FILE_IMPORTER_REGISTRY, get_file_importer

    assert "kindle" in FILE_IMPORTER_REGISTRY
    imp = get_file_importer("kindle")
    assert imp is not None
    assert imp.name == "kindle"
