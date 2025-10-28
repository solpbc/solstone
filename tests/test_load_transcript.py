"""Tests for observe.hear.load_transcript() function."""

import json
import tempfile
from pathlib import Path

from observe.hear import load_transcript


def test_load_transcript_native_with_metadata():
    """Test loading native transcript with topics/setting metadata."""
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "120000_audio.jsonl"

        lines = [
            json.dumps({"topics": "meeting, standup", "setting": "work"}),
            json.dumps({"start": "12:00:01", "source": "mic", "text": "Hello"}),
            json.dumps({"start": "12:00:05", "source": "sys", "text": "Hi there"}),
        ]
        file_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        metadata, entries, formatted_text = load_transcript(file_path)

        assert entries is not None
        assert metadata["topics"] == "meeting, standup"
        assert metadata["setting"] == "work"
        assert len(entries) == 2
        assert entries[0]["start"] == "12:00:01"
        assert entries[0]["text"] == "Hello"
        assert entries[1]["start"] == "12:00:05"
        assert entries[1]["text"] == "Hi there"


def test_load_transcript_native_empty_metadata():
    """Test loading native transcript with empty metadata."""
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "120000_audio.jsonl"

        lines = [
            json.dumps({}),
            json.dumps({"start": "12:00:01", "source": "mic", "text": "Test"}),
        ]
        file_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        metadata, entries, formatted_text = load_transcript(file_path)

        assert entries is not None
        assert metadata == {}
        assert len(entries) == 1
        assert entries[0]["text"] == "Test"


def test_load_transcript_imported():
    """Test loading imported transcript with imported metadata."""
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "120000_imported_audio.jsonl"

        lines = [
            json.dumps({"imported": {"id": "20240101_120000", "domain": "personal"}}),
            json.dumps({"start": "12:00:01", "text": "Imported entry"}),
        ]
        file_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        metadata, entries, formatted_text = load_transcript(file_path)

        assert entries is not None
        assert "imported" in metadata
        assert metadata["imported"]["id"] == "20240101_120000"
        assert metadata["imported"]["domain"] == "personal"
        assert len(entries) == 1
        assert entries[0]["text"] == "Imported entry"


def test_load_transcript_empty_file():
    """Test loading an empty file returns error."""
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "empty.jsonl"
        file_path.write_text("", encoding="utf-8")

        metadata, entries, formatted_text = load_transcript(file_path)

        assert entries is None
        assert "error" in metadata
        assert "empty" in metadata["error"].lower()


def test_load_transcript_file_not_found():
    """Test loading non-existent file returns error."""
    metadata, entries, formatted_text = load_transcript("/nonexistent/file.jsonl")

    assert entries is None
    assert "error" in metadata
    assert "not found" in metadata["error"].lower()


def test_load_transcript_invalid_metadata_json():
    """Test loading file with invalid JSON in metadata line."""
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "bad_metadata.jsonl"
        file_path.write_text("not valid json\n", encoding="utf-8")

        metadata, entries, formatted_text = load_transcript(file_path)

        assert entries is None
        assert "error" in metadata
        assert "metadata" in metadata["error"].lower()


def test_load_transcript_invalid_entry_json():
    """Test loading file with invalid JSON in entry line."""
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "bad_entry.jsonl"

        lines = [
            json.dumps({}),
            "not valid json",
        ]
        file_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        metadata, entries, formatted_text = load_transcript(file_path)

        assert entries is None
        assert "error" in metadata
        assert "line 2" in metadata["error"].lower()


def test_load_transcript_metadata_not_dict():
    """Test loading file where metadata is not a dict."""
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "bad_metadata_type.jsonl"
        file_path.write_text('["not", "a", "dict"]\n', encoding="utf-8")

        metadata, entries, formatted_text = load_transcript(file_path)

        assert entries is None
        assert "error" in metadata
        assert "object" in metadata["error"].lower()


def test_load_transcript_entry_not_dict():
    """Test loading file where entry is not a dict."""
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "bad_entry_type.jsonl"

        lines = [
            json.dumps({}),
            '"string entry"',
        ]
        file_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        metadata, entries, formatted_text = load_transcript(file_path)

        assert entries is None
        assert "error" in metadata
        assert "line 2" in metadata["error"].lower()


def test_load_transcript_blank_lines_ignored():
    """Test that blank lines between entries are ignored."""
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "with_blanks.jsonl"

        lines = [
            json.dumps({}),
            "",
            json.dumps({"start": "12:00:01", "text": "First"}),
            "",
            "",
            json.dumps({"start": "12:00:02", "text": "Second"}),
            "",
        ]
        file_path.write_text("\n".join(lines), encoding="utf-8")

        metadata, entries, formatted_text = load_transcript(file_path)

        assert entries is not None
        assert len(entries) == 2
        assert entries[0]["text"] == "First"
        assert entries[1]["text"] == "Second"


def test_load_transcript_only_metadata_no_entries():
    """Test loading file with only metadata line and no entries."""
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "only_metadata.jsonl"
        file_path.write_text(json.dumps({"topics": "test"}) + "\n", encoding="utf-8")

        metadata, entries, formatted_text = load_transcript(file_path)

        assert entries is not None
        assert metadata["topics"] == "test"
        assert entries == []


def test_load_transcript_with_path_object():
    """Test that function accepts Path objects."""
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "test.jsonl"

        lines = [
            json.dumps({}),
            json.dumps({"start": "12:00:01", "text": "Test"}),
        ]
        file_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        # Pass as Path object
        metadata, entries, formatted_text = load_transcript(file_path)

        assert entries is not None
        assert len(entries) == 1


def test_load_transcript_with_string_path():
    """Test that function accepts string paths."""
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "test.jsonl"

        lines = [
            json.dumps({}),
            json.dumps({"start": "12:00:01", "text": "Test"}),
        ]
        file_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        # Pass as string
        metadata, entries, formatted_text = load_transcript(str(file_path))

        assert entries is not None
        assert len(entries) == 1


def test_load_transcript_all_fields():
    """Test that all entry fields are preserved."""
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "complete.jsonl"

        lines = [
            json.dumps({}),
            json.dumps(
                {
                    "start": "12:00:01",
                    "source": "mic",
                    "speaker": 1,
                    "text": "Complete entry",
                    "description": "confident",
                }
            ),
        ]
        file_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        metadata, entries, formatted_text = load_transcript(file_path)

        assert entries is not None
        assert len(entries) == 1
        entry = entries[0]
        assert entry["start"] == "12:00:01"
        assert entry["source"] == "mic"
        assert entry["speaker"] == 1
        assert entry["text"] == "Complete entry"
        assert entry["description"] == "confident"


def test_load_transcript_formatted_text_basic():
    """Test formatted text output with metadata and entries."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test transcript file in YYYYMMDD directory
        day_dir = Path(tmpdir) / "20250615"
        day_dir.mkdir()
        file_path = day_dir / "100500_audio.jsonl"

        # Write JSONL with metadata and entries
        metadata = {"topics": ["meeting", "planning"], "setting": "work"}
        entries = [
            {"start": "00:01:23", "source": "mic", "speaker": 1, "text": "Hello world"},
            {"start": "00:01:25", "source": "sys", "speaker": 2, "text": "Hi there"},
            {
                "start": "00:01:30",
                "source": "mic",
                "speaker": 1,
                "text": "How are you?",
            },
        ]

        lines = [json.dumps(metadata)]
        lines.extend(json.dumps(entry) for entry in entries)
        file_path.write_text("\n".join(lines) + "\n")

        # Load transcript
        metadata_out, entries_out, formatted_text = load_transcript(file_path)

        # Verify formatted text output
        assert "Start: 2025-06-15 10:05am" in formatted_text
        assert "Topics: meeting, planning" in formatted_text
        assert "Setting: work" in formatted_text
        assert "[00:01:23] (mic) Speaker 1: Hello world" in formatted_text
        assert "[00:01:25] (sys) Speaker 2: Hi there" in formatted_text
        assert "[00:01:30] (mic) Speaker 1: How are you?" in formatted_text


def test_load_transcript_formatted_text_minimal():
    """Test formatted text with minimal metadata."""
    with tempfile.TemporaryDirectory() as tmpdir:
        day_dir = Path(tmpdir) / "20250615"
        day_dir.mkdir()
        file_path = day_dir / "100500_audio.jsonl"

        # Minimal metadata (empty dict)
        metadata = {}
        entries = [
            {"start": "00:00:01", "text": "Simple text"},
            {"start": "00:00:05", "source": "mic", "text": "With source"},
        ]

        lines = [json.dumps(metadata)]
        lines.extend(json.dumps(entry) for entry in entries)
        file_path.write_text("\n".join(lines) + "\n")

        metadata_out, entries_out, formatted_text = load_transcript(file_path)

        assert "Start: 2025-06-15 10:05am" in formatted_text
        assert "[00:00:01] Simple text" in formatted_text
        assert "[00:00:05] (mic) With source" in formatted_text


def test_load_transcript_formatted_text_imported():
    """Test formatted text of imported transcript with domain metadata."""
    with tempfile.TemporaryDirectory() as tmpdir:
        day_dir = Path(tmpdir) / "20250615"
        day_dir.mkdir()
        file_path = day_dir / "100500_imported_audio.jsonl"

        metadata = {
            "imported": {"id": "abc123", "domain": "uavionix"},
            "topics": ["discussion"],
        }
        entries = [
            {
                "start": "00:00:10",
                "source": "mic",
                "speaker": 1,
                "text": "Imported content",
            }
        ]

        lines = [json.dumps(metadata)]
        lines.extend(json.dumps(entry) for entry in entries)
        file_path.write_text("\n".join(lines) + "\n")

        metadata_out, entries_out, formatted_text = load_transcript(file_path)

        assert "Start: 2025-06-15 10:05am" in formatted_text
        assert "Topics: discussion" in formatted_text
        assert "Domain: uavionix" in formatted_text
        assert "Import ID: abc123" in formatted_text
        assert "[00:00:10] (mic) Speaker 1: Imported content" in formatted_text


def test_load_transcript_formatted_text_no_speaker():
    """Test formatted text for entries without speaker information."""
    with tempfile.TemporaryDirectory() as tmpdir:
        day_dir = Path(tmpdir) / "20250615"
        day_dir.mkdir()
        file_path = day_dir / "100500_audio.jsonl"

        metadata = {"setting": "personal"}
        entries = [
            {"start": "00:00:01", "source": "mic", "text": "No speaker here"},
            {"start": "00:00:05", "text": "No source or speaker"},
        ]

        lines = [json.dumps(metadata)]
        lines.extend(json.dumps(entry) for entry in entries)
        file_path.write_text("\n".join(lines) + "\n")

        metadata_out, entries_out, formatted_text = load_transcript(file_path)

        assert "[00:00:01] (mic) No speaker here" in formatted_text
        assert "[00:00:05] No source or speaker" in formatted_text


def test_load_transcript_formatted_text_error():
    """Test formatted text for error cases."""
    metadata, entries, formatted_text = load_transcript("/nonexistent/file.jsonl")

    assert entries is None
    assert "Error loading transcript:" in formatted_text
    assert "not found" in formatted_text.lower()


def test_load_transcript_formatted_text_no_day_in_path():
    """Test formatted text when day can't be parsed from path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # File not in a YYYYMMDD directory
        file_path = Path(tmpdir) / "100500_audio.jsonl"

        metadata = {"setting": "test"}
        entries = [{"start": "00:00:01", "text": "Test"}]

        lines = [json.dumps(metadata)]
        lines.extend(json.dumps(entry) for entry in entries)
        file_path.write_text("\n".join(lines) + "\n")

        metadata_out, entries_out, formatted_text = load_transcript(file_path)

        # Should still work, just without "Start:" header
        assert "Setting: test" in formatted_text
        assert "[00:00:01] Test" in formatted_text
        # Start header should not be present since we couldn't parse the day
        assert "Start:" not in formatted_text
