# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

import datetime as dt
import importlib
import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

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
    """Test importing a text transcript file."""
    mod = importlib.import_module("think.importer")

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

    monkeypatch.setattr(mod, "detect_transcript_segment", mock_detect_segment)

    # Mock JSON conversion: returns entries with absolute timestamps
    def mock_detect_json(text, segment_start):
        return [{"start": segment_start, "speaker": "Unknown", "text": text}]

    monkeypatch.setattr(mod, "detect_transcript_json", mock_detect_json)

    # Mock CallosumConnection and status emitter to avoid real sockets/threads
    monkeypatch.setattr(mod, "CallosumConnection", lambda: MagicMock())
    monkeypatch.setattr(mod, "_status_emitter", lambda: None)

    monkeypatch.setattr(
        "sys.argv",
        ["sol import", str(txt), "--timestamp", "20240101_120000", "--skip-summary"],
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


def test_get_audio_duration(tmp_path):
    """Test _get_audio_duration calls ffprobe correctly."""
    mod = importlib.import_module("think.importer")

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
    mod = importlib.import_module("think.importer")

    audio_file = tmp_path / "test.mp3"
    audio_file.write_bytes(b"fake audio")

    with patch(
        "subprocess.run", side_effect=subprocess.CalledProcessError(1, "ffprobe")
    ):
        duration = mod._get_audio_duration(str(audio_file))
        assert duration is None


def test_prepare_audio_segments(tmp_path, monkeypatch):
    """Test prepare_audio_segments creates segment directories with audio slices."""
    mod = importlib.import_module("think.importer")

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
    )

    # Should create 2 segments (0-5 min, 5-7 min)
    assert len(segments) == 2

    seg1_key, seg1_dir, seg1_files = segments[0]
    assert seg1_key == "120000_300"
    assert seg1_files == ["imported_audio.mp3"]
    assert (seg1_dir / "imported_audio.mp3").exists()

    seg2_key, seg2_dir, seg2_files = segments[1]
    assert seg2_key == "120500_300"
    assert seg2_files == ["imported_audio.mp3"]
    assert (seg2_dir / "imported_audio.mp3").exists()


def test_prepare_audio_segments_with_collision(tmp_path, monkeypatch):
    """Test prepare_audio_segments handles segment key collisions."""
    mod = importlib.import_module("think.importer")

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
    )

    assert len(segments) == 1
    seg_key, seg_dir, seg_files = segments[0]
    assert seg_key == "120001_300"  # Deconflicted key


def test_run_import_summary(tmp_path, monkeypatch):
    """Test _run_import_summary calls cortex_request correctly."""
    mod = importlib.import_module("think.importer")

    import_dir = tmp_path / "imports" / "20240101_120000"
    import_dir.mkdir(parents=True)

    captured_request = {}

    def mock_cortex_request(prompt, name, config):
        captured_request.update({"prompt": prompt, "name": name, "config": config})
        # Create the summary file like the generator would
        summary_path = import_dir / "summary.md"
        summary_path.write_text("# Test Summary\n\nContent here.")
        return "mock_agent_id"

    def mock_wait_for_agents(agent_ids, timeout):
        return (agent_ids, [])  # All completed, none timed out

    def mock_get_agent_end_state(agent_id):
        return "finish"

    with (
        patch("think.cortex_client.cortex_request", side_effect=mock_cortex_request),
        patch("think.cortex_client.wait_for_agents", side_effect=mock_wait_for_agents),
        patch(
            "think.cortex_client.get_agent_end_state",
            side_effect=mock_get_agent_end_state,
        ),
    ):
        result = mod._run_import_summary(
            import_dir,
            "20240101",
            ["120000_300", "120500_300"],
        )

        assert result is True
        assert (import_dir / "summary.md").exists()

        # Verify cortex_request was called with correct config
        assert captured_request["name"] == "importer"
        assert captured_request["config"]["day"] == "20240101"
        assert captured_request["config"]["span"] == ["120000_300", "120500_300"]
        assert captured_request["config"]["output"] == "md"
        assert (
            str(import_dir / "summary.md") in captured_request["config"]["output_path"]
        )


def test_run_import_summary_no_segments(tmp_path):
    """Test _run_import_summary returns False with no segments."""
    mod = importlib.import_module("think.importer")

    import_dir = tmp_path / "imports" / "20240101_120000"
    import_dir.mkdir(parents=True)

    result = mod._run_import_summary(import_dir, "20240101", [])
    assert result is False
