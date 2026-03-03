# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

import datetime as dt
import importlib
import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

from think.importers.file_importer import ImportPreview, ImportResult
from think.utils import day_path


def _make_mock_file_importer(name="ics", display_name="ICS Calendar"):
    """Create a mock FileImporter for testing."""
    mock_imp = MagicMock()
    mock_imp.name = name
    mock_imp.display_name = display_name
    mock_imp.file_patterns = ["*.ics"]
    mock_imp.description = "Import calendar events from ICS files"

    mock_imp.preview.return_value = ImportPreview(
        date_range=("20250101", "20250301"),
        item_count=42,
        entity_count=5,
        summary="42 calendar events from 5 calendars",
    )
    mock_imp.process.return_value = ImportResult(
        entries_written=42,
        entities_seeded=5,
        files_created=["/journal/20250101/import.ics/imported.jsonl"],
        errors=[],
        summary="Imported 42 events",
    )
    return mock_imp


def test_slice_audio_segment(tmp_path):
    """Test slice_audio_segment extracts audio with stream copy."""
    mod = importlib.import_module("think.importers.audio")

    source = tmp_path / "source.mp3"
    source.write_bytes(b"fake audio")
    output = tmp_path / "segment.mp3"

    # Mock subprocess.run to simulate successful ffmpeg
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = None

        result = mod.slice_audio_segment(str(source), str(output), 0, 300)

        assert result == str(output)
        # First call should use -c:a copy
        call_args = mock_run.call_args_list[0][0][0]
        assert "-c:a" in call_args
        assert "copy" in call_args


def test_slice_audio_segment_fallback(tmp_path):
    """Test slice_audio_segment falls back to re-encode on copy failure."""
    mod = importlib.import_module("think.importers.audio")

    source = tmp_path / "source.mp3"
    source.write_bytes(b"fake audio")
    output = tmp_path / "segment.mp3"

    # First call (copy) fails, second call (re-encode) succeeds
    call_count = [0]

    def mock_run(cmd, *args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            # First call (stream copy) fails
            raise subprocess.CalledProcessError(1, cmd)
        # Second call (re-encode) succeeds
        return None

    with patch("subprocess.run", side_effect=mock_run):
        result = mod.slice_audio_segment(str(source), str(output), 0, 300)

        assert result == str(output)
        assert call_count[0] == 2  # Both attempts were made


def test_importer_text(tmp_path, monkeypatch):
    """Test importing a text transcript file."""
    mod = importlib.import_module("think.importers.cli")
    text_mod = importlib.import_module("think.importers.text")

    transcript = "hello\nworld"
    txt = tmp_path / "sample.txt"
    txt.write_text(transcript)

    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    monkeypatch.setattr(
        mod, "detect_created", lambda p, **kw: {"day": "20240101", "time": "120000"}
    )

    # Mock segment detection: returns (start_at, text) tuples with absolute times
    def mock_detect_segment(text, start_time):
        return [("12:00:00", "seg1"), ("12:05:00", "seg2")]

    monkeypatch.setattr(text_mod, "detect_transcript_segment", mock_detect_segment)

    # Mock JSON conversion: returns entries with absolute timestamps
    def mock_detect_json(text, segment_start):
        return [{"start": segment_start, "speaker": "Unknown", "text": text}]

    monkeypatch.setattr(text_mod, "detect_transcript_json", mock_detect_json)

    # Mock CallosumConnection and status emitter to avoid real sockets/threads
    monkeypatch.setattr(mod, "CallosumConnection", lambda **kwargs: MagicMock())
    monkeypatch.setattr(mod, "_status_emitter", lambda: None)

    monkeypatch.setattr(
        "sys.argv",
        ["sol import", str(txt), "--timestamp", "20240101_120000"],
    )
    mod.main()

    day_dir = day_path("20240101")
    # Duration: seg1 starts at 12:00:00, seg2 at 12:05:00 = 300s duration
    # Last segment (seg2) defaults to 5s since no audio duration
    # Segments are under stream directory (import.text for .txt files)
    f1 = day_dir / "import.text" / "120000_300" / "imported_audio.jsonl"
    f2 = day_dir / "import.text" / "120500_5" / "imported_audio.jsonl"

    # Read JSONL format: first line is metadata, subsequent lines are entries
    lines1 = f1.read_text().strip().split("\n")
    metadata1 = json.loads(lines1[0])
    entries1 = [json.loads(line) for line in lines1[1:]]

    lines2 = f2.read_text().strip().split("\n")
    metadata2 = json.loads(lines2[0])
    entries2 = [json.loads(line) for line in lines2[1:]]

    # Timestamps are relative offsets from segment start (not absolute time-of-day)
    assert entries1 == [
        {"start": "00:00:00", "speaker": "Unknown", "text": "seg1", "source": "import"}
    ]
    assert metadata1["imported"]["id"] == "20240101_120000"
    assert "facet" not in metadata1["imported"]
    # raw path should resolve from segment dir (3 levels deep) to imports/
    assert metadata1["raw"] == "../../../imports/20240101_120000/sample.txt"

    assert entries2 == [
        {"start": "00:00:00", "speaker": "Unknown", "text": "seg2", "source": "import"}
    ]
    assert metadata2["imported"]["id"] == "20240101_120000"
    assert "facet" not in metadata2["imported"]

    # segments.json should be written in the import directory
    segments_json = tmp_path / "imports" / "20240101_120000" / "segments.json"
    assert segments_json.exists()
    seg_data = json.loads(segments_json.read_text())
    assert seg_data["day"] == "20240101"
    assert "120000_300" in seg_data["segments"]
    assert "120500_5" in seg_data["segments"]

    # stream.json should be written in each segment directory
    stream1 = day_dir / "import.text" / "120000_300" / "stream.json"
    assert stream1.exists()
    stream1_data = json.loads(stream1.read_text())
    assert stream1_data["stream"] == "import.text"


def test_importer_pdf(tmp_path, monkeypatch):
    """Test importing a PDF transcript file."""
    mod = importlib.import_module("think.importers.cli")
    text_mod = importlib.import_module("think.importers.text")

    # Create a fake PDF file (content doesn't matter — pypdf is mocked)
    pdf = tmp_path / "meeting.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")

    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    monkeypatch.setattr(
        mod, "detect_created", lambda p, **kw: {"day": "20251205", "time": "163000"}
    )

    # Mock _read_transcript to return extracted text (bypasses pypdf)
    monkeypatch.setattr(
        text_mod, "_read_transcript", lambda path: "Board meeting notes\nAction items"
    )

    # Mock segment detection: single segment for short text
    def mock_detect_segment(text, start_time):
        return [("16:30:00", text)]

    monkeypatch.setattr(text_mod, "detect_transcript_segment", mock_detect_segment)

    # Mock JSON conversion
    def mock_detect_json(text, segment_start):
        return [
            {"start": segment_start, "speaker": "Jack", "text": "Board meeting notes"},
            {"start": "16:30:30", "speaker": "Ramon", "text": "Action items"},
            {"topics": "board meeting, action items", "setting": "workplace"},
        ]

    monkeypatch.setattr(text_mod, "detect_transcript_json", mock_detect_json)

    # Mock CallosumConnection and status emitter
    monkeypatch.setattr(mod, "CallosumConnection", lambda **kwargs: MagicMock())
    monkeypatch.setattr(mod, "_status_emitter", lambda: None)

    monkeypatch.setattr(
        "sys.argv",
        [
            "sol import",
            str(pdf),
            "--timestamp",
            "20251205_163000",
            "--facet",
            "work",
            "--setting",
            "board meeting",
        ],
    )
    mod.main()

    day_dir = day_path("20251205")
    # Single segment, last segment defaults to 5s
    f1 = day_dir / "import.text" / "163000_5" / "imported_audio.jsonl"
    assert f1.exists()

    lines = f1.read_text().strip().split("\n")
    metadata = json.loads(lines[0])
    entries = [json.loads(line) for line in lines[1:]]

    # Verify metadata
    assert metadata["imported"]["id"] == "20251205_163000"
    assert metadata["imported"]["facet"] == "work"
    assert metadata["imported"]["setting"] == "board meeting"
    assert metadata["raw"] == "../../../imports/20251205_163000/meeting.pdf"

    # Verify entries — timestamps are relative offsets, topics/setting in metadata
    assert entries[0] == {
        "start": "00:00:00",
        "speaker": "Jack",
        "text": "Board meeting notes",
        "source": "import",
    }
    assert entries[1] == {
        "start": "00:00:30",
        "speaker": "Ramon",
        "text": "Action items",
        "source": "import",
    }
    # Topics/setting extracted to metadata (not written as entry)
    assert len(entries) == 2
    assert metadata["topics"] == "board meeting, action items"
    assert metadata["setting"] == "workplace"

    # Verify .pdf auto-detected as text import (stream = import.text)
    stream_json = day_dir / "import.text" / "163000_5" / "stream.json"
    assert stream_json.exists()
    stream_data = json.loads(stream_json.read_text())
    assert stream_data["stream"] == "import.text"

    # Verify segments.json written
    segments_json = tmp_path / "imports" / "20251205_163000" / "segments.json"
    assert segments_json.exists()


def test_format_audio_stream_path():
    """Test format_audio correctly parses timestamps from stream-based paths."""
    from observe.hear import format_audio

    entries = [
        {"imported": {"id": "20240101_120000"}, "raw": "test.txt"},
        {"start": "12:00:00", "speaker": "Alice", "text": "Hello"},
        {"start": "12:00:30", "speaker": "Bob", "text": "Hi there"},
    ]

    # Stream-based path: day/stream/segment/imported_audio.jsonl
    context = {
        "file_path": Path(
            "/journal/20240101/import.text/120000_300/imported_audio.jsonl"
        )
    }
    chunks, meta = format_audio(entries, context)

    assert len(chunks) == 2
    # Verify timestamps are non-zero (base_timestamp correctly parsed from path)
    assert chunks[0]["timestamp"] > 0
    assert chunks[1]["timestamp"] > chunks[0]["timestamp"]
    # Verify header includes start time
    assert meta.get("header") and "12:00" in meta["header"]


def test_format_audio_legacy_path():
    """Test format_audio still works with legacy day/segment/ paths."""
    from observe.hear import format_audio

    entries = [
        {"raw": "raw.flac", "model": "whisper-1"},
        {"start": "12:34:56", "source": "mic", "text": "Test"},
    ]

    # Legacy path: day/segment/audio.jsonl (no stream directory)
    context = {"file_path": Path("/journal/20240101/123456_300/audio.jsonl")}
    chunks, meta = format_audio(entries, context)

    assert len(chunks) == 1
    assert chunks[0]["timestamp"] > 0


def test_get_audio_duration(tmp_path):
    """Test _get_audio_duration calls ffprobe correctly."""
    mod = importlib.import_module("think.importers.audio")

    audio_file = tmp_path / "test.mp3"
    audio_file.write_bytes(b"fake audio")

    # Mock ffprobe returning duration
    mock_result = MagicMock()
    mock_result.stdout = "123.456\n"

    with patch("subprocess.run", return_value=mock_result) as mock_run:
        duration = mod._get_audio_duration(str(audio_file))

        assert duration == 123.456
        # Verify ffprobe was called with correct args
        call_args = mock_run.call_args[0][0]
        assert "ffprobe" in call_args
        assert str(audio_file) in call_args


def test_get_audio_duration_failure(tmp_path):
    """Test _get_audio_duration returns None on error."""
    mod = importlib.import_module("think.importers.audio")

    audio_file = tmp_path / "test.mp3"
    audio_file.write_bytes(b"fake audio")

    with patch(
        "subprocess.run", side_effect=subprocess.CalledProcessError(1, "ffprobe")
    ):
        duration = mod._get_audio_duration(str(audio_file))
        assert duration is None


def test_prepare_audio_segments(tmp_path, monkeypatch):
    """Test prepare_audio_segments creates segment directories with audio slices."""
    mod = importlib.import_module("think.importers.audio")

    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))

    audio_file = tmp_path / "test.mp3"
    audio_file.write_bytes(b"fake audio content")

    day_dir = tmp_path / "20240101"
    day_dir.mkdir()

    base_dt = dt.datetime(2024, 1, 1, 12, 0, 0)

    # Mock _get_audio_duration to return 7 minutes (2.33 segments)
    monkeypatch.setattr(mod, "_get_audio_duration", lambda p: 420.0)

    # Mock slice_audio_segment to create the file
    def mock_slice(src, dst, start, duration):
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        Path(dst).write_bytes(b"sliced audio")
        return dst

    monkeypatch.setattr(mod, "slice_audio_segment", mock_slice)

    # Mock find_available_segment to return segment as-is (no collision)
    monkeypatch.setattr(mod, "find_available_segment", lambda day, seg: seg)

    segments = mod.prepare_audio_segments(
        str(audio_file),
        str(day_dir),
        base_dt,
        "20240101_120000",
        "import.audio",
    )

    # Should create 2 segments (0-5 min, 5-7 min)
    assert len(segments) == 2

    seg1_key, seg1_dir, seg1_files = segments[0]
    assert seg1_key == "120000_300"
    assert seg1_files == ["imported_audio.mp3"]
    assert (seg1_dir / "imported_audio.mp3").exists()
    # Segment should be under stream directory
    assert seg1_dir == day_dir / "import.audio" / "120000_300"

    seg2_key, seg2_dir, seg2_files = segments[1]
    assert seg2_key == "120500_300"
    assert seg2_files == ["imported_audio.mp3"]
    assert (seg2_dir / "imported_audio.mp3").exists()
    assert seg2_dir == day_dir / "import.audio" / "120500_300"


def test_prepare_audio_segments_with_collision(tmp_path, monkeypatch):
    """Test prepare_audio_segments handles segment key collisions."""
    mod = importlib.import_module("think.importers.audio")

    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))

    audio_file = tmp_path / "test.mp3"
    audio_file.write_bytes(b"fake audio content")

    day_dir = tmp_path / "20240101"
    day_dir.mkdir()

    base_dt = dt.datetime(2024, 1, 1, 12, 0, 0)

    monkeypatch.setattr(mod, "_get_audio_duration", lambda p: 300.0)

    def mock_slice(src, dst, start, duration):
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        Path(dst).write_bytes(b"sliced audio")
        return dst

    monkeypatch.setattr(mod, "slice_audio_segment", mock_slice)

    # Simulate collision - return modified segment key
    def mock_find_available(day, seg):
        if seg == "120000_300":
            return "120001_300"  # Deconflicted
        return seg

    monkeypatch.setattr(mod, "find_available_segment", mock_find_available)

    segments = mod.prepare_audio_segments(
        str(audio_file),
        str(day_dir),
        base_dt,
        "20240101_120000",
        "import.audio",
    )

    assert len(segments) == 1
    seg_key, seg_dir, seg_files = segments[0]
    assert seg_key == "120001_300"  # Deconflicted key
    assert seg_dir == day_dir / "import.audio" / "120001_300"


def test_importer_dry_run_text(tmp_path, monkeypatch, capsys):
    """Test --dry-run for text import prints plan without writing files."""
    mod = importlib.import_module("think.importers.cli")

    txt = tmp_path / "sample.txt"
    txt.write_text("hello\nworld\n")

    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    monkeypatch.setattr(
        "sys.argv",
        ["sol import", str(txt), "--timestamp", "20240101_120000", "--dry-run"],
    )

    mod.main()

    captured = capsys.readouterr()
    assert "File:" in captured.out
    assert "Size:" in captured.out
    assert "Timestamp:" in captured.out
    assert "Source:" in captured.out
    assert "Stream:" in captured.out
    assert "Target day:" in captured.out
    assert "Content:" in captured.out
    assert "characters" in captured.out
    assert "lines" in captured.out
    assert "import.text" in captured.out
    assert "20240101" in captured.out
    assert "12 characters" in captured.out
    assert "2 lines" in captured.out

    assert not (tmp_path / "imports").exists()
    assert not (tmp_path / "20240101").exists()


def test_importer_dry_run_audio(tmp_path, monkeypatch, capsys):
    """Test --dry-run for audio import prints plan without writing files."""
    mod = importlib.import_module("think.importers.cli")

    mp3 = tmp_path / "sample.mp3"
    mp3.write_bytes(b"fake audio")

    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    monkeypatch.setattr(mod, "_get_audio_duration", lambda p: 420.0)
    callosum_cls = MagicMock()
    monkeypatch.setattr(mod, "CallosumConnection", callosum_cls)
    monkeypatch.setattr(
        "sys.argv",
        ["sol import", str(mp3), "--timestamp", "20240101_120000", "--dry-run"],
    )

    mod.main()

    captured = capsys.readouterr()
    assert "File:" in captured.out
    assert "Size:" in captured.out
    assert "Timestamp:" in captured.out
    assert "Source:" in captured.out
    assert "Stream:" in captured.out
    assert "Target day:" in captured.out
    assert "Duration:" in captured.out
    assert "Segments:" in captured.out
    assert "Keys:" in captured.out
    assert "import.audio" in captured.out
    assert "20240101" in captured.out
    assert "7.0 minutes" in captured.out
    assert "2 (5-minute chunks)" in captured.out
    assert "120000_300" in captured.out
    assert "120500_300" in captured.out

    assert not (tmp_path / "imports").exists()
    assert not (tmp_path / "20240101").exists()
    assert callosum_cls.call_count == 0


def test_importer_dry_run_auto(tmp_path, monkeypatch, capsys):
    """Test --dry-run with --auto detects timestamp and prints summary."""
    mod = importlib.import_module("think.importers.cli")

    txt = tmp_path / "notes.txt"
    txt.write_text("meeting notes")

    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    monkeypatch.setattr(
        mod, "detect_created", lambda p, **kw: {"day": "20240315", "time": "140000"}
    )
    monkeypatch.setattr(
        "sys.argv",
        ["sol import", str(txt), "--auto", "--dry-run"],
    )

    mod.main()

    captured = capsys.readouterr()
    assert "Detected timestamp: 20240315_140000" in captured.out
    assert "auto-importing" in captured.out
    assert "import.text" in captured.out
    assert "Target day: 20240315" in captured.out
    assert "Content:" in captured.out

    assert not (tmp_path / "imports").exists()
    assert not (tmp_path / "20240315").exists()


def test_file_importer_without_timestamp(tmp_path, monkeypatch, capsys):
    """File importers auto-generate timestamp and skip import setup."""
    mod = importlib.import_module("think.importers.cli")

    ics_file = tmp_path / "calendar.ics"
    ics_file.write_text("BEGIN:VCALENDAR\nEND:VCALENDAR")

    fixed_dt = dt.datetime(2026, 3, 3, 12, 34, 56)

    class FixedDateTime(dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed_dt

    monkeypatch.setattr(mod.dt, "datetime", FixedDateTime)

    mock_imp = _make_mock_file_importer()
    callosum = MagicMock()

    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    monkeypatch.setattr("sys.argv", ["sol import", str(ics_file), "--source", "ics"])
    monkeypatch.setattr(
        "think.importers.file_importer.get_file_importer", lambda name: mock_imp
    )
    monkeypatch.setattr(mod, "CallosumConnection", lambda **kwargs: callosum)
    monkeypatch.setattr(mod, "get_rev", lambda: "test-rev")
    monkeypatch.setattr(mod, "_status_emitter", lambda: None)

    mod.main()

    mock_imp.process.assert_called_once_with(Path(ics_file), Path(tmp_path), facet=None)
    mock_call = callosum.emit.call_args_list[0]
    assert mock_call.args[0] == "importer"
    assert mock_call.args[1] == "started"
    assert mock_call.kwargs["import_id"] == "20260303_123456"
    assert not (tmp_path / "imports").exists()


def test_file_importer_with_timestamp(tmp_path, monkeypatch):
    """File importer uses provided --timestamp and still skips import setup."""
    mod = importlib.import_module("think.importers.cli")

    ics_file = tmp_path / "calendar.ics"
    ics_file.write_text("BEGIN:VCALENDAR\nEND:VCALENDAR")

    mock_imp = _make_mock_file_importer()
    callosum = MagicMock()

    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    monkeypatch.setattr(
        "sys.argv",
        [
            "sol import",
            str(ics_file),
            "--source",
            "ics",
            "--timestamp",
            "20260303_120000",
        ],
    )
    monkeypatch.setattr(
        "think.importers.file_importer.get_file_importer", lambda name: mock_imp
    )
    monkeypatch.setattr(mod, "CallosumConnection", lambda **kwargs: callosum)
    monkeypatch.setattr(mod, "get_rev", lambda: "test-rev")
    monkeypatch.setattr(mod, "_status_emitter", lambda: None)

    mod.main()

    mock_imp.process.assert_called_once_with(Path(ics_file), Path(tmp_path), facet=None)
    mock_call = callosum.emit.call_args_list[0]
    assert mock_call.args[0] == "importer"
    assert mock_call.args[1] == "started"
    assert mock_call.kwargs["import_id"] == "20260303_120000"
    assert not (tmp_path / "imports").exists()


def test_list_importers_json(capsys, monkeypatch):
    """--list-importers --json returns machine-readable output."""
    mod = importlib.import_module("think.importers.cli")

    mock_imp = _make_mock_file_importer()
    monkeypatch.setattr("sys.argv", ["sol import", "--list-importers", "--json"])
    with patch(
        "think.importers.file_importer.get_file_importers", return_value=[mock_imp]
    ):
        mod.main()

    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["name"] == "ics"
    assert "display_name" in data[0]
    assert "file_patterns" in data[0]
    assert "description" in data[0]


def test_dry_run_file_importer_json(tmp_path, monkeypatch, capsys):
    """--dry-run --json for file importer returns JSON metadata."""
    mod = importlib.import_module("think.importers.cli")

    ics_file = tmp_path / "calendar.ics"
    ics_file.write_text("BEGIN:VCALENDAR\nEND:VCALENDAR")

    mock_imp = _make_mock_file_importer()

    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    monkeypatch.setattr(
        "sys.argv",
        ["sol import", str(ics_file), "--source", "ics", "--dry-run", "--json"],
    )
    monkeypatch.setattr(
        "think.importers.file_importer.get_file_importer", lambda name: mock_imp
    )

    mod.main()

    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["importer"] == "ics"
    assert data["source"] == str(ics_file)
    assert data["item_count"] == 42
    assert data["entity_count"] == 5
    assert data["summary"] == "42 calendar events from 5 calendars"
    assert isinstance(data["date_range"], list)
    mock_imp.process.assert_not_called()


def test_file_import_json(tmp_path, monkeypatch, capsys):
    """File importer prints machine-readable completion output."""
    mod = importlib.import_module("think.importers.cli")

    ics_file = tmp_path / "calendar.ics"
    ics_file.write_text("BEGIN:VCALENDAR\nEND:VCALENDAR")
    fixed_dt = dt.datetime(2026, 3, 3, 12, 34, 56)

    class FixedDateTime(dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed_dt

    monkeypatch.setattr(mod.dt, "datetime", FixedDateTime)

    mock_imp = _make_mock_file_importer()
    callosum = MagicMock()

    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    monkeypatch.setattr(
        "sys.argv",
        ["sol import", str(ics_file), "--source", "ics", "--json"],
    )
    monkeypatch.setattr(
        "think.importers.file_importer.get_file_importer", lambda name: mock_imp
    )
    monkeypatch.setattr(mod, "CallosumConnection", lambda **kwargs: callosum)
    monkeypatch.setattr(mod, "get_rev", lambda: "test-rev")
    monkeypatch.setattr(mod, "_status_emitter", lambda: None)

    mod.main()

    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["importer"] == "ics"
    assert data["entries_written"] == 42
    assert data["entities_seeded"] == 5
    assert data["files_created"] == ["/journal/20250101/import.ics/imported.jsonl"]
    assert data["errors"] == []
    assert data["summary"] == "Imported 42 events"
    assert not (tmp_path / "imports").exists()


def test_file_importer_no_imports_dir(tmp_path, monkeypatch):
    """File importers should never create the legacy imports/ folder."""
    mod = importlib.import_module("think.importers.cli")

    ics_file = tmp_path / "calendar.ics"
    ics_file.write_text("BEGIN:VCALENDAR\nEND:VCALENDAR")

    mock_imp = _make_mock_file_importer()

    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    monkeypatch.setattr(
        "sys.argv",
        ["sol import", str(ics_file), "--source", "ics"],
    )
    monkeypatch.setattr(
        "think.importers.file_importer.get_file_importer", lambda name: mock_imp
    )
    monkeypatch.setattr(mod, "CallosumConnection", lambda **kwargs: MagicMock())
    monkeypatch.setattr(mod, "get_rev", lambda: "test-rev")
    monkeypatch.setattr(mod, "_status_emitter", lambda: None)

    mod.main()

    assert not (tmp_path / "imports").exists()
