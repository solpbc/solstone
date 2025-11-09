"""Tests for observe.reduce module."""

from pathlib import Path

from observe.reduce import assemble_markdown


def test_assemble_markdown_extracts_period_from_directory():
    """Test that assemble_markdown correctly extracts base time from period directory."""
    # Mock frames with relative timestamps (seconds from period start)
    frames = [
        {"timestamp": 0, "monitor": "0", "analysis": {"visible": "code", "visual_description": "Editing Python"}},
        {"timestamp": 30, "monitor": "0", "analysis": {"visible": "terminal", "visual_description": "Running tests"}},
        {"timestamp": 120, "monitor": "0", "analysis": {"visible": "browser", "visual_description": "Reading docs"}},
    ]

    # Simulate path structure: YYYYMMDD/HHMMSS/screen.jsonl
    jsonl_path = Path("20240101/143022/screen.jsonl")

    markdown = assemble_markdown(frames, entity_names="", video_path=jsonl_path, include_entity_context=False)

    # Verify absolute times are calculated correctly from period (14:30:22)
    assert "14:30:22" in markdown  # Base time from period
    assert "14:30:52" in markdown  # Base + 30 seconds
    assert "14:32:22" in markdown  # Base + 120 seconds (2 minutes)

    # Verify frame content is included
    assert "Editing Python" in markdown
    assert "Running tests" in markdown
    assert "Reading docs" in markdown


def test_assemble_markdown_handles_period_with_duration_suffix():
    """Test that assemble_markdown handles HHMMSS_LEN period format."""
    frames = [
        {"timestamp": 0, "monitor": "0", "analysis": {"visible": "code", "visual_description": "Code"}},
        {"timestamp": 60, "monitor": "0", "analysis": {"visible": "terminal", "visual_description": "Terminal"}},
    ]

    # Period with duration suffix: 143022_300 (5 minutes)
    jsonl_path = Path("20240101/143022_300/screen.jsonl")

    markdown = assemble_markdown(frames, entity_names="", video_path=jsonl_path, include_entity_context=False)

    # Should still extract base time correctly
    assert "14:30:22" in markdown  # Base time
    assert "14:31:22" in markdown  # Base + 60 seconds
    assert "Code" in markdown
    assert "Terminal" in markdown


def test_assemble_markdown_handles_no_video_path():
    """Test that assemble_markdown works when video_path is None (defaults to midnight)."""
    frames = [
        {"timestamp": 0, "monitor": "0", "analysis": {"visible": "code", "visual_description": "Code"}},
        {"timestamp": 3600, "monitor": "0", "analysis": {"visible": "browser", "visual_description": "Browser"}},
    ]

    markdown = assemble_markdown(frames, entity_names="", video_path=None, include_entity_context=False)

    # Should default to 00:00:00 base time
    assert "00:00:00" in markdown
    assert "01:00:00" in markdown  # 3600 seconds = 1 hour
    assert "Code" in markdown
    assert "Browser" in markdown


def test_assemble_markdown_handles_multiple_monitors():
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

    jsonl_path = Path("20240101/120000/screen.jsonl")

    markdown = assemble_markdown(frames, entity_names="", video_path=jsonl_path, include_entity_context=False)

    # Should include monitor info when multiple monitors present
    assert "Monitor 0 - left" in markdown
    assert "Monitor 1 - right" in markdown
    assert "Editing code" in markdown
    assert "Documentation" in markdown


def test_assemble_markdown_includes_entity_context():
    """Test that entity context is included when requested."""
    frames = [
        {"timestamp": 0, "monitor": "0", "analysis": {"visible": "code", "visual_description": "Code"}},
    ]

    jsonl_path = Path("20240101/120000/screen.jsonl")
    entity_names = "Alice, Bob, ProjectX"

    markdown = assemble_markdown(
        frames, entity_names=entity_names, video_path=jsonl_path, include_entity_context=True
    )

    # Should include entity context header
    assert "# Entity Context" in markdown
    assert "Alice, Bob, ProjectX" in markdown


def test_assemble_markdown_includes_extracted_text():
    """Test that extracted text is included in output."""
    frames = [
        {
            "timestamp": 0,
            "monitor": "0",
            "analysis": {"visible": "terminal", "visual_description": "Terminal window"},
            "extracted_text": "$ python test.py\nAll tests passed",
        },
    ]

    jsonl_path = Path("20240101/120000/screen.jsonl")

    markdown = assemble_markdown(frames, entity_names="", video_path=jsonl_path, include_entity_context=False)

    # Should include extracted text in code block
    assert "**Extracted Text:**" in markdown
    assert "$ python test.py" in markdown
    assert "All tests passed" in markdown
