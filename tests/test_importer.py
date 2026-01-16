# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

import datetime as dt
import importlib
import json
import subprocess
from pathlib import Path
from unittest.mock import patch

from think.utils import day_path


def test_slice_audio_segment(tmp_path):
    """Test slice_audio_segment extracts audio with stream copy."""
    mod = importlib.import_module("think.importer")

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
    mod = importlib.import_module("think.importer")

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
    mod = importlib.import_module("think.importer")

    transcript = "hello\nworld"
    txt = tmp_path / "sample.txt"
    txt.write_text(transcript)

    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    monkeypatch.setattr(
        mod, "detect_created", lambda p: {"day": "20240101", "time": "120000"}
    )

    # Mock segment detection: returns (start_at, text) tuples with absolute times
    def mock_detect_segment(text, start_time):
        return [("12:00:00", "seg1"), ("12:05:00", "seg2")]

    monkeypatch.setattr(mod, "detect_transcript_segment", mock_detect_segment)

    # Mock JSON conversion: returns entries with absolute timestamps
    def mock_detect_json(text, segment_start):
        return [{"start": segment_start, "speaker": "Unknown", "text": text}]

    monkeypatch.setattr(mod, "detect_transcript_json", mock_detect_json)

    # Mock generate to prevent real API calls during summarization
    monkeypatch.setattr(mod, "generate", lambda **kwargs: "Mocked summary")

    monkeypatch.setattr(
        "sys.argv",
        ["think-importer", str(txt), "--timestamp", "20240101_120000"],
    )
    mod.main()

    day_dir = day_path("20240101")
    # Duration: seg1 starts at 12:00:00, seg2 at 12:05:00 = 300s duration
    # Last segment (seg2) defaults to 5s since no audio duration
    f1 = day_dir / "120000_300" / "imported_audio.jsonl"
    f2 = day_dir / "120500_5" / "imported_audio.jsonl"

    # Read JSONL format: first line is metadata, subsequent lines are entries
    lines1 = f1.read_text().strip().split("\n")
    metadata1 = json.loads(lines1[0])
    entries1 = [json.loads(line) for line in lines1[1:]]

    lines2 = f2.read_text().strip().split("\n")
    metadata2 = json.loads(lines2[0])
    entries2 = [json.loads(line) for line in lines2[1:]]

    # Output has absolute timestamps from segment detection and source="import"
    assert entries1 == [
        {"start": "12:00:00", "speaker": "Unknown", "text": "seg1", "source": "import"}
    ]
    assert metadata1["imported"]["id"] == "20240101_120000"
    assert "facet" not in metadata1["imported"]

    assert entries2 == [
        {"start": "12:05:00", "speaker": "Unknown", "text": "seg2", "source": "import"}
    ]
    assert metadata2["imported"]["id"] == "20240101_120000"
    assert "facet" not in metadata2["imported"]


def test_importer_audio_transcribe(tmp_path, monkeypatch):
    """Test the new audio_transcribe functionality with Rev AI."""
    mod = importlib.import_module("think.importer")

    # Create a test audio file
    audio_file = tmp_path / "test_audio.mp3"
    audio_file.write_bytes(b"fake audio content")

    # Mock Rev AI response
    mock_revai_response = {
        "monologues": [
            {
                "speaker": 0,
                "elements": [
                    {"type": "text", "value": "Hello", "ts": 0.0, "confidence": 0.95},
                    {"type": "text", "value": " ", "ts": 0.5},
                    {"type": "text", "value": "world", "ts": 0.6, "confidence": 0.98},
                    {"type": "punct", "value": "."},
                    {"type": "text", "value": " ", "ts": 1.0},
                    {"type": "text", "value": "This", "ts": 1.1, "confidence": 0.9},
                    {"type": "text", "value": " ", "ts": 1.5},
                    {"type": "text", "value": "is", "ts": 1.6, "confidence": 0.92},
                    {"type": "text", "value": " ", "ts": 1.8},
                    {"type": "text", "value": "a", "ts": 1.9, "confidence": 0.93},
                    {"type": "text", "value": " ", "ts": 2.0},
                    {"type": "text", "value": "test", "ts": 2.1, "confidence": 0.91},
                    {"type": "punct", "value": "."},
                ],
            },
            {
                "speaker": 1,
                "elements": [
                    {
                        "type": "text",
                        "value": "Second",
                        "ts": 310.0,
                        "confidence": 0.88,
                    },  # After 5 minutes
                    {"type": "text", "value": " ", "ts": 310.5},
                    {"type": "text", "value": "chunk", "ts": 310.6, "confidence": 0.89},
                    {"type": "punct", "value": "."},
                ],
            },
        ]
    }

    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))

    # Mock generate to prevent real API calls during summarization
    monkeypatch.setattr(mod, "generate", lambda **kwargs: "Mocked summary")

    # Mock the transcribe_file function
    with patch("think.importer.transcribe_file") as mock_transcribe:
        mock_transcribe.return_value = mock_revai_response

        # Mock slice_audio_segment to avoid needing real ffmpeg
        with patch("think.importer.slice_audio_segment") as mock_slice:
            mock_slice.return_value = str(
                tmp_path / "120000_300" / "imported_audio.mp3"
            )

            # Run with --hear option
            monkeypatch.setattr(
                "sys.argv",
                [
                    "think-importer",
                    str(audio_file),
                    "--timestamp",
                    "20240101_120000",
                    "--hear",
                    "true",
                ],
            )
            mod.main()

    # Check that the files were created correctly
    day_dir = day_path("20240101")
    f1 = day_dir / "120000_300" / "imported_audio.jsonl"
    f2 = day_dir / "120500_300" / "imported_audio.jsonl"

    assert f1.exists()
    assert f2.exists()

    # Check first chunk (0-5 minutes) - JSONL format
    # With per_speaker=True, each monologue becomes one entry (not split by sentence)
    lines1 = f1.read_text().strip().split("\n")
    metadata1 = json.loads(lines1[0])
    entries1 = [json.loads(line) for line in lines1[1:]]

    assert metadata1["imported"]["id"] == "20240101_120000"
    assert metadata1["raw"] == "imported_audio.mp3"  # Local audio slice
    assert len(entries1) == 1  # One monologue = one entry (per-speaker mode)
    assert "Hello" in entries1[0]["text"]  # Full monologue text
    assert "test" in entries1[0]["text"]  # Contains both sentences
    assert entries1[0]["speaker"] == 1  # Rev uses 0-based, we use 1-based
    assert entries1[0]["source"] == "import"
    assert entries1[0]["start"] == "12:00:00"  # Absolute timestamp

    # Check second chunk (5-10 minutes) - JSONL format
    lines2 = f2.read_text().strip().split("\n")
    metadata2 = json.loads(lines2[0])
    entries2 = [json.loads(line) for line in lines2[1:]]

    assert metadata2["imported"]["id"] == "20240101_120000"
    assert metadata2["raw"] == "imported_audio.mp3"  # Local audio slice
    assert len(entries2) == 1
    assert entries2[0]["text"] == "Second chunk."
    assert entries2[0]["speaker"] == 2
    assert entries2[0]["start"] == "12:05:10"  # Absolute timestamp (5:10 after base)


def test_audio_transcribe_sanitizes_entities(tmp_path, monkeypatch):
    mod = importlib.import_module("think.importer")

    audio_file = tmp_path / "test_audio.mp3"
    audio_file.write_bytes(b"fake audio content")

    # Use fixtures journal for facet entities lookup
    from pathlib import Path

    fixtures_journal = Path(__file__).parent.parent / "fixtures" / "journal"
    monkeypatch.setenv("JOURNAL_PATH", str(fixtures_journal))

    captured: list[list[str]] = []

    def fake_transcribe_file(media_path, config=None):
        # Config is now a dict with entities key
        config = config or {}
        captured.append(config.get("entities", []))
        return {}

    monkeypatch.setattr(mod, "transcribe_file", fake_transcribe_file)
    monkeypatch.setattr(mod, "convert_to_statements", lambda _: [])

    mod.audio_transcribe(
        str(audio_file),
        str(tmp_path),
        dt.datetime(2024, 1, 1, 12, 0, 0),
        import_id="20240101_120000",
        facet="acme",
    )

    assert captured
    # Entities are sorted by type then name, so Organization comes before Person
    assert captured[0] == [
        "Test",  # First name from "Test Initiative (TI)" (Organization comes first)
        "TI",  # Nickname from "Test Initiative (TI)"
        "TP",  # Nickname from "Test Person (TP)"
    ]


def test_audio_transcribe_includes_import_metadata(tmp_path, monkeypatch):
    mod = importlib.import_module("think.importer")

    audio_file = tmp_path / "test_audio.mp3"
    audio_file.write_bytes(b"fake audio content")

    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))

    monkeypatch.setattr(
        "think.facets.get_facets",
        lambda: {
            "uavionix": {
                "entities": {"Person": ["Ryan Reed (R2)"]},
            }
        },
    )

    monkeypatch.setattr(mod, "transcribe_file", lambda *_, **__: {"ok": True})
    monkeypatch.setattr(
        mod,
        "convert_to_statements",
        lambda _: [
            {
                "id": 1,
                "start": 0.0,
                "end": 1.5,
                "text": "Test entry",
                "speaker": 1,
            }
        ],
    )
    # Mock slice_audio_segment to avoid needing real ffmpeg
    monkeypatch.setattr(
        mod,
        "slice_audio_segment",
        lambda *_: str(tmp_path / "120000_300" / "imported_audio.mp3"),
    )

    created_files, _ = mod.audio_transcribe(
        str(audio_file),
        str(tmp_path),
        dt.datetime(2024, 1, 1, 12, 0, 0),
        import_id="20240101_120000",
        facet="uavionix",
    )

    assert created_files

    # Find the JSONL file (filter out audio files)
    jsonl_files = [f for f in created_files if f.endswith(".jsonl")]
    assert jsonl_files

    # Read JSONL format: first line is metadata, subsequent lines are entries
    lines = Path(jsonl_files[0]).read_text().strip().split("\n")
    metadata = json.loads(lines[0])
    entries = [json.loads(line) for line in lines[1:]]

    assert entries[0]["text"] == "Test entry"
    assert metadata["imported"]["id"] == "20240101_120000"
    assert metadata["imported"]["facet"] == "uavionix"
    assert metadata["raw"] == "imported_audio.mp3"  # Local audio slice
