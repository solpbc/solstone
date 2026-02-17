# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for think.importers.utils module."""

import json
import tempfile
from pathlib import Path

import pytest

from think.importers.utils import (
    build_import_info,
    calculate_duration_from_files,
    get_import_details,
    list_import_timestamps,
    load_import_segments,
    read_import_metadata,
    read_imported_results,
    save_import_file,
    save_import_segments,
    save_import_text,
    update_import_metadata_fields,
    write_import_metadata,
)


@pytest.fixture
def temp_journal():
    """Create a temporary journal directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        journal_path = Path(tmpdir) / "journal"
        journal_path.mkdir()
        yield journal_path


def test_save_import_file(temp_journal):
    """Test saving an import file."""
    # Create a source file
    source = temp_journal / "source.txt"
    source.write_text("test content", encoding="utf-8")

    # Save to import
    result_path = save_import_file(
        journal_root=temp_journal,
        timestamp="20250101_120000",
        source_path=source,
        filename="imported.txt",
    )

    # Verify it was saved correctly
    assert result_path.exists()
    assert result_path.read_text(encoding="utf-8") == "test content"
    assert result_path.parent.name == "20250101_120000"
    assert result_path.name == "imported.txt"


def test_save_import_text(temp_journal):
    """Test saving text content as import."""
    result_path = save_import_text(
        journal_root=temp_journal,
        timestamp="20250101_130000",
        content="Hello world",
        filename="paste.txt",
    )

    assert result_path.exists()
    assert result_path.read_text(encoding="utf-8") == "Hello world"
    assert result_path.parent.name == "20250101_130000"


def test_write_and_read_import_metadata(temp_journal):
    """Test writing and reading import metadata."""
    timestamp = "20250101_140000"
    metadata = {
        "original_filename": "test.txt",
        "upload_timestamp": 1234567890000,
        "file_size": 1024,
        "facet": "work",
    }

    # Write metadata
    write_import_metadata(
        journal_root=temp_journal,
        timestamp=timestamp,
        metadata=metadata,
    )

    # Read it back
    read_metadata = read_import_metadata(
        journal_root=temp_journal,
        timestamp=timestamp,
    )

    assert read_metadata == metadata


def test_read_import_metadata_not_found(temp_journal):
    """Test reading metadata when it doesn't exist."""
    with pytest.raises(FileNotFoundError):
        read_import_metadata(
            journal_root=temp_journal,
            timestamp="20250101_999999",
        )


def test_update_import_metadata_fields(temp_journal):
    """Test updating specific metadata fields."""
    timestamp = "20250101_150000"

    # Create initial metadata
    initial = {"original_filename": "test.txt", "facet": None}
    write_import_metadata(temp_journal, timestamp, initial)

    # Update fields
    updated_metadata, was_modified = update_import_metadata_fields(
        journal_root=temp_journal,
        timestamp=timestamp,
        updates={"facet": "personal", "setting": "home office"},
    )

    assert was_modified is True
    assert updated_metadata["facet"] == "personal"
    assert updated_metadata["setting"] == "home office"
    assert updated_metadata["original_filename"] == "test.txt"

    # Update with same values
    updated_metadata2, was_modified2 = update_import_metadata_fields(
        journal_root=temp_journal,
        timestamp=timestamp,
        updates={"facet": "personal", "setting": "home office"},
    )

    assert was_modified2 is False


def test_read_imported_results(temp_journal):
    """Test reading imported.json results."""
    timestamp = "20250101_160000"
    import_dir = temp_journal / "imports" / timestamp
    import_dir.mkdir(parents=True)

    # Create imported.json
    results = {
        "processed_timestamp": timestamp,
        "total_files_created": 5,
        "target_day": "20250101",
    }
    (import_dir / "imported.json").write_text(json.dumps(results), encoding="utf-8")

    # Read it
    read_results = read_imported_results(temp_journal, timestamp)
    assert read_results == results

    # Test when it doesn't exist
    read_results_none = read_imported_results(temp_journal, "20250101_999999")
    assert read_results_none is None



def test_list_import_timestamps(temp_journal):
    """Test listing all import timestamps."""
    # Create some import folders
    (temp_journal / "imports" / "20250101_120000").mkdir(parents=True)
    (temp_journal / "imports" / "20250101_130000").mkdir(parents=True)
    (temp_journal / "imports" / "20250101_140000").mkdir(parents=True)

    # Create invalid folder (should be ignored)
    (temp_journal / "imports" / "invalid").mkdir(parents=True)

    timestamps = list_import_timestamps(temp_journal)

    assert len(timestamps) == 3
    assert "20250101_120000" in timestamps
    assert "20250101_130000" in timestamps
    assert "20250101_140000" in timestamps
    assert "invalid" not in timestamps


def test_list_import_timestamps_empty(temp_journal):
    """Test listing when no imports exist."""
    timestamps = list_import_timestamps(temp_journal)
    assert timestamps == []

    # Create imports dir but leave it empty
    (temp_journal / "imports").mkdir()
    timestamps = list_import_timestamps(temp_journal)
    assert timestamps == []


def test_calculate_duration_from_files():
    """Test calculating duration from imported file timestamps."""
    files = [
        "/path/to/120000_imported_audio.jsonl",
        "/path/to/120500_imported_audio.jsonl",
        "/path/to/121000_imported_audio.jsonl",
        "/path/to/123000_imported_audio.jsonl",
    ]

    duration = calculate_duration_from_files(files)
    assert duration == 30  # 12:00 to 12:30 = 30 minutes

    # Test with single file
    duration_single = calculate_duration_from_files([files[0]])
    assert duration_single is None

    # Test with empty list
    duration_empty = calculate_duration_from_files([])
    assert duration_empty is None

    # Test with files without timestamps
    duration_no_ts = calculate_duration_from_files(["file.txt", "other.jsonl"])
    assert duration_no_ts is None


def test_build_import_info(temp_journal):
    """Test building complete import info."""
    timestamp = "20250101_190000"
    import_dir = temp_journal / "imports" / timestamp
    import_dir.mkdir(parents=True)

    # Create import.json
    import_metadata = {
        "original_filename": "recording.m4a",
        "file_size": 2048000,
        "mime_type": "audio/m4a",
        "upload_timestamp": 1704124800000,
        "facet": "work",
        "setting": "meeting",
    }
    (import_dir / "import.json").write_text(
        json.dumps(import_metadata), encoding="utf-8"
    )

    # Create imported.json
    imported_results = {
        "total_files_created": 3,
        "target_day": "20250101",
        "all_created_files": [
            "/path/190000_imported_audio.jsonl",
            "/path/190500_imported_audio.jsonl",
            "/path/191000_imported_audio.jsonl",
        ],
    }
    (import_dir / "imported.json").write_text(
        json.dumps(imported_results), encoding="utf-8"
    )

    # Build info
    info = build_import_info(temp_journal, timestamp)

    assert info["timestamp"] == timestamp
    assert info["original_filename"] == "recording.m4a"
    assert info["file_size"] == 2048000
    assert info["mime_type"] == "audio/m4a"
    assert info["facet"] == "work"
    assert info["setting"] == "meeting"
    assert info["processed"] is True
    assert info["total_files_created"] == 3
    assert info["target_day"] == "20250101"
    assert info["duration_minutes"] == 10  # 19:00 to 19:10


def test_get_import_details(temp_journal):
    """Test getting all details for an import."""
    timestamp = "20250101_200000"
    import_dir = temp_journal / "imports" / timestamp
    import_dir.mkdir(parents=True)

    # Create all metadata files
    (import_dir / "import.json").write_text('{"file": "test.m4a"}', encoding="utf-8")
    (import_dir / "imported.json").write_text(
        '{"total_files_created": 2}', encoding="utf-8"
    )
    details = get_import_details(temp_journal, timestamp)

    assert details["timestamp"] == timestamp
    assert details["import_json"]["file"] == "test.m4a"
    assert details["imported_json"]["total_files_created"] == 2


def test_get_import_details_not_found(temp_journal):
    """Test getting details for non-existent import."""
    with pytest.raises(FileNotFoundError):
        get_import_details(temp_journal, "20250101_999999")


def test_save_and_load_import_segments(temp_journal):
    """Test saving and loading segment list for an import."""
    timestamp = "20250101_210000"
    segments = ["120000_300", "120500_300", "121000_300"]
    day = "20250101"

    # Save segments
    save_import_segments(temp_journal, timestamp, segments, day)

    # Load them back
    result = load_import_segments(temp_journal, timestamp)
    assert result is not None
    loaded_segments, loaded_day = result
    assert loaded_segments == segments
    assert loaded_day == day


def test_load_import_segments_not_found(temp_journal):
    """Test loading segments when file doesn't exist."""
    result = load_import_segments(temp_journal, "20250101_999999")
    assert result is None


def test_get_import_details_includes_segments(temp_journal):
    """Test that get_import_details includes segments.json."""
    timestamp = "20250101_220000"
    import_dir = temp_journal / "imports" / timestamp
    import_dir.mkdir(parents=True)

    # Create segments.json
    segments_data = {
        "segments": ["120000_300", "120500_300"],
        "day": "20250101",
    }
    (import_dir / "segments.json").write_text(
        json.dumps(segments_data), encoding="utf-8"
    )

    details = get_import_details(temp_journal, timestamp)
    assert details["segments_json"] == segments_data
