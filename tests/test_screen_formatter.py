"""Tests for observe.screen formatter module."""

from pathlib import Path

from observe.screen import format_screen, format_screen_text


def test_format_screen_extracts_segment_from_directory():
    """Test that format_screen correctly extracts base time from segment directory."""
    # Mock frames with relative timestamps (seconds from segment start)
    frames = [
        {
            "timestamp": 0,
            "monitor": "0",
            "analysis": {"visible": "code", "visual_description": "Editing Python"},
        },
        {
            "timestamp": 30,
            "monitor": "0",
            "analysis": {"visible": "terminal", "visual_description": "Running tests"},
        },
        {
            "timestamp": 120,
            "monitor": "0",
            "analysis": {"visible": "browser", "visual_description": "Reading docs"},
        },
    ]

    # Simulate path structure: YYYYMMDD/HHMMSS/screen.jsonl
    context = {
        "file_path": Path("20240101/143022/screen.jsonl"),
        "include_entity_context": False,
    }

    chunks, meta = format_screen(frames, context)
    markdown = "\n".join([meta.get("header", "")] + [c["markdown"] for c in chunks])

    # Verify absolute times are calculated correctly from segment (14:30:22)
    assert "14:30:22" in markdown  # Base time from segment
    assert "14:30:52" in markdown  # Base + 30 seconds
    assert "14:32:22" in markdown  # Base + 120 seconds (2 minutes)

    # Verify frame content is included
    assert "Editing Python" in markdown
    assert "Running tests" in markdown
    assert "Reading docs" in markdown


def test_format_screen_handles_segment_with_duration_suffix():
    """Test that format_screen handles HHMMSS_LEN segment format."""
    frames = [
        {
            "timestamp": 0,
            "monitor": "0",
            "analysis": {"visible": "code", "visual_description": "Code"},
        },
        {
            "timestamp": 60,
            "monitor": "0",
            "analysis": {"visible": "terminal", "visual_description": "Terminal"},
        },
    ]

    # Segment with duration suffix: 143022_300 (5 minutes)
    context = {
        "file_path": Path("20240101/143022_300/screen.jsonl"),
        "include_entity_context": False,
    }

    chunks, meta = format_screen(frames, context)
    markdown = "\n".join([meta.get("header", "")] + [c["markdown"] for c in chunks])

    # Should still extract base time correctly
    assert "14:30:22" in markdown  # Base time
    assert "14:31:22" in markdown  # Base + 60 seconds
    assert "Code" in markdown
    assert "Terminal" in markdown


def test_format_screen_handles_no_file_path():
    """Test that format_screen works when file_path is None (defaults to midnight)."""
    frames = [
        {
            "timestamp": 0,
            "monitor": "0",
            "analysis": {"visible": "code", "visual_description": "Code"},
        },
        {
            "timestamp": 3600,
            "monitor": "0",
            "analysis": {"visible": "browser", "visual_description": "Browser"},
        },
    ]

    chunks, meta = format_screen(frames, {"include_entity_context": False})
    markdown = "\n".join([meta.get("header", "")] + [c["markdown"] for c in chunks])

    # Should default to 00:00:00 base time
    assert "00:00:00" in markdown
    assert "01:00:00" in markdown  # 3600 seconds = 1 hour
    assert "Code" in markdown
    assert "Browser" in markdown


def test_format_screen_handles_multiple_monitors():
    """Test that monitor information is included when multiple monitors are present."""
    frames = [
        {
            "timestamp": 0,
            "monitor": "0",
            "monitor_position": "left",
            "analysis": {"visible": "code", "visual_description": "Editing code"},
        },
        {
            "timestamp": 0,
            "monitor": "1",
            "monitor_position": "right",
            "analysis": {"visible": "browser", "visual_description": "Documentation"},
        },
    ]

    context = {
        "file_path": Path("20240101/120000/screen.jsonl"),
        "include_entity_context": False,
    }

    chunks, meta = format_screen(frames, context)
    markdown = "\n".join([meta.get("header", "")] + [c["markdown"] for c in chunks])

    # Should include monitor info when multiple monitors present
    assert "Monitor 0 - left" in markdown
    assert "Monitor 1 - right" in markdown
    assert "Editing code" in markdown
    assert "Documentation" in markdown


def test_format_screen_includes_entity_context():
    """Test that entity context is included when requested."""
    frames = [
        {
            "timestamp": 0,
            "monitor": "0",
            "analysis": {"visible": "code", "visual_description": "Code"},
        },
    ]

    context = {
        "file_path": Path("20240101/120000/screen.jsonl"),
        "entity_names": "Alice, Bob, ProjectX",
        "include_entity_context": True,
    }

    chunks, meta = format_screen(frames, context)
    markdown = "\n".join([meta.get("header", "")] + [c["markdown"] for c in chunks])

    # Should include entity context header
    assert "# Entity Context" in markdown
    assert "Alice, Bob, ProjectX" in markdown


def test_format_screen_includes_extracted_text():
    """Test that extracted text is included in output."""
    frames = [
        {
            "timestamp": 0,
            "monitor": "0",
            "analysis": {
                "visible": "terminal",
                "visual_description": "Terminal window",
            },
            "extracted_text": "$ python test.py\nAll tests passed",
        },
    ]

    context = {
        "file_path": Path("20240101/120000/screen.jsonl"),
        "include_entity_context": False,
    }

    chunks, meta = format_screen(frames, context)
    markdown = "\n".join([meta.get("header", "")] + [c["markdown"] for c in chunks])

    # Should include extracted text in code block
    assert "**Extracted Text:**" in markdown
    assert "$ python test.py" in markdown
    assert "All tests passed" in markdown


def test_format_screen_returns_chunks_with_timestamps():
    """Test that format_screen returns chunks with timestamp metadata."""
    frames = [
        {
            "timestamp": 0,
            "monitor": "0",
            "analysis": {"visible": "code", "visual_description": "Frame 1"},
        },
        {
            "timestamp": 30,
            "monitor": "0",
            "analysis": {"visible": "terminal", "visual_description": "Frame 2"},
        },
    ]

    chunks, meta = format_screen(frames)

    assert len(chunks) == 2
    assert chunks[0]["timestamp"] == 0
    assert chunks[1]["timestamp"] == 30
    assert "Frame 1" in chunks[0]["markdown"]
    assert "Frame 2" in chunks[1]["markdown"]


def test_format_screen_returns_indexer_metadata():
    """Test that format_screen returns indexer metadata with topic."""
    frames = [
        {
            "timestamp": 0,
            "monitor": "0",
            "analysis": {"visible": "code", "visual_description": "Test"},
        },
    ]

    chunks, meta = format_screen(frames)

    assert "indexer" in meta
    assert meta["indexer"]["topic"] == "screen"
