# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

import importlib

import pytest

from think.utils import day_path


def test_cluster(tmp_path, monkeypatch):
    """Test cluster() uses audio and agent output summaries (*.md files)."""
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    day_dir = day_path("20240101")

    mod = importlib.import_module("think.cluster")
    # Write JSONL format: metadata first, then entry in segment directory
    (day_dir / "120000_300").mkdir()
    (day_dir / "120000_300" / "audio.jsonl").write_text('{}\n{"text": "hi"}\n')
    (day_dir / "120500_300").mkdir()
    (day_dir / "120500_300" / "screen.md").write_text("screen summary")
    result, counts = mod.cluster(
        "20240101", sources={"audio": True, "screen": False, "agents": True}
    )
    assert counts["audio"] == 1
    assert counts["agents"] == 1
    assert "Audio Transcript" in result
    # Now uses insight rendering: "### {stem} summary"
    assert "screen summary" in result


def test_cluster_range(tmp_path, monkeypatch):
    """Test cluster_range with audio and agents sources."""
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    day_dir = day_path("20240101")

    mod = importlib.import_module("think.cluster")
    # Write JSONL format: metadata first, then entry with proper start time and source in segment directory
    (day_dir / "120000_300").mkdir()
    (day_dir / "120000_300" / "audio.jsonl").write_text(
        '{"raw": "raw.flac", "model": "whisper-1"}\n'
        '{"start": "00:00:01", "source": "mic", "text": "hi from audio"}\n'
    )
    (day_dir / "120000_300" / "screen.md").write_text("screen summary content")
    # Test with agents=True to include *.md files
    md = mod.cluster_range(
        "20240101",
        "120000",
        "120100",
        sources={"audio": True, "screen": False, "agents": True},
    )
    # Check that the function works and includes expected sections
    assert "Audio Transcript" in md
    # Now uses insight rendering: "### {stem} summary"
    assert "screen summary" in md
    assert "screen summary content" in md


def test_cluster_scan(tmp_path, monkeypatch):
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    day_dir = day_path("20240101")

    mod = importlib.import_module("think.cluster")
    # Audio transcripts at 09:01, 09:05, 09:20 and 11:00 (JSONL format with empty metadata)
    (day_dir / "090101_300").mkdir()
    (day_dir / "090101_300" / "audio.jsonl").write_text("{}\n")
    (day_dir / "090500_300").mkdir()
    (day_dir / "090500_300" / "audio.jsonl").write_text("{}\n")
    (day_dir / "092000_300").mkdir()
    (day_dir / "092000_300" / "audio.jsonl").write_text("{}\n")
    (day_dir / "110000_300").mkdir()
    (day_dir / "110000_300" / "audio.jsonl").write_text("{}\n")
    # Screen transcripts at 10:01, 10:05, 10:20 and 12:00
    (day_dir / "100101_300").mkdir()
    (day_dir / "100101_300" / "screen.jsonl").write_text('{"raw": "screen.webm"}\n')
    (day_dir / "100500_300").mkdir()
    (day_dir / "100500_300" / "screen.jsonl").write_text('{"raw": "screen.webm"}\n')
    (day_dir / "102000_300").mkdir()
    (day_dir / "102000_300" / "screen.jsonl").write_text('{"raw": "screen.webm"}\n')
    (day_dir / "120000_300").mkdir()
    (day_dir / "120000_300" / "screen.jsonl").write_text('{"raw": "screen.webm"}\n')
    audio_ranges, screen_ranges = mod.cluster_scan("20240101")
    # Expected ranges: 15-minute slot grouping (segments 09:01-09:05-09:20 group together)
    # Slots: 09:00, 09:00, 09:15 -> ranges: 09:00-09:30; 11:00 -> 11:00-11:15
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
    (day_dir / "100000_600" / "screen.jsonl").write_text('{"raw": "screen.webm"}\n')

    # Create segment with only screen
    (day_dir / "110000_300").mkdir()
    (day_dir / "110000_300" / "screen.jsonl").write_text('{"raw": "screen.webm"}\n')

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


def test_cluster_period_uses_raw_screen(tmp_path, monkeypatch):
    """Test cluster_period uses raw screen.jsonl, not insight *.md files."""
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    day_dir = day_path("20240101")

    mod = importlib.import_module("think.cluster")

    # Create segment with both audio and raw screen data
    segment = day_dir / "100000_300"
    segment.mkdir()
    (segment / "audio.jsonl").write_text(
        '{"raw": "audio.flac"}\n{"start": "00:00:01", "text": "hello"}\n'
    )
    # Raw screen.jsonl with frame analysis (what cluster_period should use)
    (segment / "screen.jsonl").write_text(
        '{"raw": "screen.webm"}\n'
        '{"timestamp": 10, "analysis": {"primary": "code_editor", '
        '"visual_description": "VS Code with Python file"}}\n'
    )
    # Also create screen.md (insight) to verify it's NOT used by cluster_period
    (segment / "screen.md").write_text("This insight should NOT appear")

    result, counts = mod.cluster_period(
        "20240101",
        "100000_300",
        sources={"audio": True, "screen": True, "agents": False},
    )

    # Should have both audio and screen entries
    assert counts["audio"] == 1
    assert counts["screen"] == 1
    assert "Audio Transcript" in result
    # Should use raw screen format header
    assert "Screen Activity" in result
    # Raw screen content should be present
    assert "VS Code with Python file" in result
    # Insight content should NOT be present (agents=False for cluster_period)
    assert "This insight should NOT appear" not in result


def test_cluster_range_with_agents(tmp_path, monkeypatch):
    """Test cluster_range with agents source loads all *.md files."""
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    day_dir = day_path("20240101")

    mod = importlib.import_module("think.cluster")

    # Create segment with multiple insight files
    segment = day_dir / "100000_300"
    segment.mkdir()
    (segment / "audio.jsonl").write_text(
        '{"raw": "audio.flac"}\n{"start": "00:00:01", "text": "hello"}\n'
    )
    (segment / "screen.md").write_text("Screen activity summary")
    (segment / "activity.md").write_text("Activity insight content")
    # Also create screen.jsonl to verify it's NOT used when agents=True, screen=False
    (segment / "screen.jsonl").write_text(
        '{"raw": "screen.webm"}\n'
        '{"timestamp": 10, "analysis": {"primary": "code_editor"}}\n'
    )

    # Test agents=True returns *.md summaries, not raw screen data
    result = mod.cluster_range(
        "20240101",
        "100000",
        "100500",
        sources={"audio": True, "screen": False, "agents": True},
    )

    assert "Audio Transcript" in result
    # Should include both .md files as agent outputs
    assert "### screen summary" in result
    assert "Screen activity summary" in result
    assert "### activity summary" in result
    assert "Activity insight content" in result
    # Should NOT include raw screen data
    assert "code_editor" not in result


def test_cluster_range_with_screen(tmp_path, monkeypatch):
    """Test cluster_range with screen source loads raw screen.jsonl data."""
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    day_dir = day_path("20240101")

    mod = importlib.import_module("think.cluster")

    # Create segment with raw screen data and insight file
    segment = day_dir / "100000_300"
    segment.mkdir()
    (segment / "screen.jsonl").write_text(
        '{"raw": "screen.webm"}\n'
        '{"timestamp": 10, "analysis": {"primary": "code_editor"}}\n'
    )
    (segment / "screen.md").write_text("Screen summary insight")

    # Test screen=True returns raw screen data, not agent outputs
    result = mod.cluster_range(
        "20240101",
        "100000",
        "100500",
        sources={"audio": False, "screen": True, "agents": False},
    )

    assert "Screen Activity" in result
    assert "code_editor" in result
    # Should NOT include insight content
    assert "Screen summary insight" not in result
    assert "### screen summary" not in result


def test_cluster_range_with_multiple_screen_files(tmp_path, monkeypatch):
    """Test cluster_range loads multiple *_screen.jsonl files per segment."""
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    day_dir = day_path("20240101")

    mod = importlib.import_module("think.cluster")

    # Create segment with multiple screen files (like multi-monitor setup)
    segment = day_dir / "100000_300"
    segment.mkdir()
    (segment / "screen.jsonl").write_text(
        '{"raw": "screen.webm"}\n'
        '{"timestamp": 10, "analysis": {"primary": "code_editor", '
        '"visual_description": "Primary monitor with VS Code"}}\n'
    )
    (segment / "monitor_2_screen.jsonl").write_text(
        '{"raw": "monitor_2.webm"}\n'
        '{"timestamp": 10, "analysis": {"primary": "browser", '
        '"visual_description": "Secondary monitor with documentation"}}\n'
    )

    # Test screen=True returns data from both screen files
    result = mod.cluster_range(
        "20240101",
        "100000",
        "100500",
        sources={"audio": False, "screen": True, "agents": False},
    )

    # Should include content from both screen files
    assert "Primary monitor with VS Code" in result
    assert "Secondary monitor with documentation" in result


def test_cluster_scan_with_split_screen(tmp_path, monkeypatch):
    """Test cluster_scan detects *_screen.jsonl files."""
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    day_dir = day_path("20240101")

    mod = importlib.import_module("think.cluster")

    # Create segment with only *_screen.jsonl (no screen.jsonl)
    (day_dir / "100000_300").mkdir()
    (day_dir / "100000_300" / "monitor_1_screen.jsonl").write_text(
        '{"raw": "m1.webm"}\n'
    )

    audio_ranges, screen_ranges = mod.cluster_scan("20240101")

    # Should detect the segment as having screen content (15-minute slot grouping)
    assert screen_ranges == [("10:00", "10:15")]


def test_cluster_segments_with_split_screen(tmp_path, monkeypatch):
    """Test cluster_segments detects *_screen.jsonl files."""
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    day_dir = day_path("20240101")

    mod = importlib.import_module("think.cluster")

    # Create segment with only *_screen.jsonl (no screen.jsonl)
    (day_dir / "100000_300").mkdir()
    (day_dir / "100000_300" / "wayland_screen.jsonl").write_text('{"raw": "w.webm"}\n')

    segments = mod.cluster_segments("20240101")

    assert len(segments) == 1
    assert segments[0]["key"] == "100000_300"
    assert "screen" in segments[0]["types"]


def test_cluster_span(tmp_path, monkeypatch):
    """Test cluster_span processes a span of segments."""
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    day_dir = day_path("20240101")

    mod = importlib.import_module("think.cluster")

    # Create three segments with different content
    (day_dir / "090000_300").mkdir()
    (day_dir / "090000_300" / "audio.jsonl").write_text(
        '{"raw": "audio.flac"}\n{"start": "00:00:01", "text": "morning segment"}\n'
    )

    (day_dir / "100000_300").mkdir()
    (day_dir / "100000_300" / "audio.jsonl").write_text(
        '{"raw": "audio.flac"}\n{"start": "00:00:01", "text": "mid-morning segment"}\n'
    )
    (day_dir / "100000_300" / "screen.jsonl").write_text(
        '{"raw": "screen.webm"}\n'
        '{"timestamp": 10, "analysis": {"primary": "code_editor"}}\n'
    )

    (day_dir / "110000_300").mkdir()
    (day_dir / "110000_300" / "audio.jsonl").write_text(
        '{"raw": "audio.flac"}\n{"start": "00:00:01", "text": "late morning segment"}\n'
    )

    # Process only first and third segments as a span (audio only, no screen)
    result, counts = mod.cluster_span(
        "20240101",
        ["090000_300", "110000_300"],
        sources={"audio": True, "screen": False, "agents": False},
    )

    # Should have 2 audio entries (one per segment)
    assert counts["audio"] == 2
    assert counts["screen"] == 0
    assert "morning segment" in result
    assert "late morning segment" in result
    # Should NOT include the skipped segment
    assert "mid-morning segment" not in result
    assert "code_editor" not in result


def test_cluster_span_missing_segment(tmp_path, monkeypatch):
    """Test cluster_span fails fast when segment is missing."""
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    day_dir = day_path("20240101")

    mod = importlib.import_module("think.cluster")

    # Create only one segment
    (day_dir / "090000_300").mkdir()
    (day_dir / "090000_300" / "audio.jsonl").write_text('{"raw": "audio.flac"}\n')

    # Try to process existing and non-existing segments
    with pytest.raises(ValueError) as exc_info:
        mod.cluster_span(
            "20240101",
            ["090000_300", "100000_300"],
            sources={"audio": True, "screen": False, "agents": False},
        )

    assert "100000_300" in str(exc_info.value)
    assert "not found" in str(exc_info.value)
