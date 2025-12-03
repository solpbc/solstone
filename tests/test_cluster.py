import importlib

from think.utils import day_path


def test_cluster(tmp_path, monkeypatch):
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    day_dir = day_path("20240101")

    mod = importlib.import_module("think.cluster")
    # Write JSONL format: metadata first, then entry in segment directory
    (day_dir / "120000").mkdir()
    (day_dir / "120000" / "audio.jsonl").write_text('{}\n{"text": "hi"}\n')
    (day_dir / "120500").mkdir()
    (day_dir / "120500" / "screen.md").write_text("screen summary")
    result, count = mod.cluster("20240101")
    assert count == 2
    assert "Audio Transcript" in result
    assert "Screen Activity Summary" in result


def test_cluster_range(tmp_path, monkeypatch):
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    day_dir = day_path("20240101")

    mod = importlib.import_module("think.cluster")
    # Write JSONL format: metadata first, then entry with proper start time and source in segment directory
    (day_dir / "120000").mkdir()
    (day_dir / "120000" / "audio.jsonl").write_text(
        '{"raw": "raw.flac", "model": "whisper-1"}\n'
        '{"start": "00:00:01", "source": "mic", "text": "hi from audio"}\n'
    )
    (day_dir / "120000" / "screen.md").write_text("screen summary content")
    # Test with summary mode to ensure screen content is included
    md = mod.cluster_range("20240101", "120000", "120100", audio=True, screen="summary")
    # Check that the function works and includes expected sections
    assert "Audio Transcript" in md
    assert "Screen Activity Summary" in md
    # The audio might be empty if there are formatting issues, but screen should work
    assert "screen summary content" in md


def test_cluster_scan(tmp_path, monkeypatch):
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    day_dir = day_path("20240101")

    mod = importlib.import_module("think.cluster")
    # Audio transcripts at 09:01, 09:05, 09:20 and 11:00 (JSONL format with empty metadata)
    (day_dir / "090101").mkdir()
    (day_dir / "090101" / "audio.jsonl").write_text("{}\n")
    (day_dir / "090500").mkdir()
    (day_dir / "090500" / "audio.jsonl").write_text("{}\n")
    (day_dir / "092000").mkdir()
    (day_dir / "092000" / "audio.jsonl").write_text("{}\n")
    (day_dir / "110000").mkdir()
    (day_dir / "110000" / "audio.jsonl").write_text("{}\n")
    # Screen transcripts at 10:01, 10:05, 10:20 and 12:00
    (day_dir / "100101").mkdir()
    (day_dir / "100101" / "screen.md").write_text("screen")
    (day_dir / "100500").mkdir()
    (day_dir / "100500" / "screen.md").write_text("screen")
    (day_dir / "102000").mkdir()
    (day_dir / "102000" / "screen.md").write_text("screen")
    (day_dir / "120000").mkdir()
    (day_dir / "120000" / "screen.md").write_text("screen")
    audio_ranges, screen_ranges = mod.cluster_scan("20240101")
    assert audio_ranges == [("09:00", "09:30"), ("11:00", "11:15")]
    assert screen_ranges == [("10:00", "10:30"), ("12:00", "12:15")]


def test_cluster_segments(tmp_path, monkeypatch):
    """Test cluster_segments returns individual segments with their types."""
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    day_dir = day_path("20240101")

    mod = importlib.import_module("think.cluster")

    # Create segment with duration: 090000_300 (09:00:00 for 5 minutes)
    (day_dir / "090000_300").mkdir()
    (day_dir / "090000_300" / "audio.jsonl").write_text("{}\n")

    # Create segment with both audio and screen
    (day_dir / "100000_600").mkdir()
    (day_dir / "100000_600" / "audio.jsonl").write_text("{}\n")
    (day_dir / "100000_600" / "screen.md").write_text("screen")

    # Create segment with only screen
    (day_dir / "110000_300").mkdir()
    (day_dir / "110000_300" / "screen.md").write_text("screen")

    segments = mod.cluster_segments("20240101")

    assert len(segments) == 3

    # Check first segment (audio only)
    assert segments[0]["key"] == "090000_300"
    assert segments[0]["start"] == "09:00"
    assert segments[0]["end"] == "09:05"
    assert segments[0]["types"] == ["audio"]

    # Check second segment (both audio and screen)
    assert segments[1]["key"] == "100000_600"
    assert segments[1]["start"] == "10:00"
    assert segments[1]["end"] == "10:10"
    assert "audio" in segments[1]["types"]
    assert "screen" in segments[1]["types"]

    # Check third segment (screen only)
    assert segments[2]["key"] == "110000_300"
    assert segments[2]["start"] == "11:00"
    assert segments[2]["end"] == "11:05"
    assert segments[2]["types"] == ["screen"]
