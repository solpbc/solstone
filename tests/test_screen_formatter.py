# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for observe.screen formatter module."""

from pathlib import Path

from observe.screen import (
    CATEGORIES,
    _load_category_formatter,
    format_screen,
)


def test_format_screen_extracts_segment_from_directory():
    """Test that format_screen correctly extracts base time from segment directory."""
    # Mock frames with relative timestamps (seconds from segment start)
    frames = [
        {
            "timestamp": 0,
            "analysis": {
                "primary": "code",
                "visual_description": "Editing Python",
            },
        },
        {
            "timestamp": 30,
            "analysis": {
                "primary": "terminal",
                "visual_description": "Running tests",
            },
        },
        {
            "timestamp": 120,
            "analysis": {
                "primary": "browser",
                "visual_description": "Reading docs",
            },
        },
    ]

    # Simulate path structure: YYYYMMDD/HHMMSS_LEN/screen.jsonl
    context = {
        "file_path": Path("20240101/default/143022_300/screen.jsonl"),
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
            "analysis": {"primary": "code", "visual_description": "Code"},
        },
        {
            "timestamp": 60,
            "analysis": {
                "primary": "terminal",
                "visual_description": "Terminal",
            },
        },
    ]

    # Segment with duration suffix: 143022_300 (5 minutes)
    context = {
        "file_path": Path("20240101/default/143022_300/screen.jsonl"),
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
            "analysis": {"primary": "code", "visual_description": "Code"},
        },
        {
            "timestamp": 3600,
            "analysis": {
                "primary": "browser",
                "visual_description": "Browser",
            },
        },
    ]

    chunks, meta = format_screen(frames, {"include_entity_context": False})
    markdown = "\n".join([meta.get("header", "")] + [c["markdown"] for c in chunks])

    # Should default to 00:00:00 base time
    assert "00:00:00" in markdown
    assert "01:00:00" in markdown  # 3600 seconds = 1 hour
    assert "Code" in markdown
    assert "Browser" in markdown


def test_format_screen_header_includes_monitor_info():
    """Test that monitor info is included in header for per-monitor files."""
    frames = [
        {
            "timestamp": 0,
            "analysis": {
                "primary": "code",
                "visual_description": "Editing code",
            },
        },
        {
            "timestamp": 30,
            "analysis": {
                "primary": "browser",
                "visual_description": "Documentation",
            },
        },
    ]

    # Per-monitor file with position/connector in filename
    context = {
        "file_path": Path("20240101/default/120000_300/center_DP-3_screen.jsonl"),
        "include_entity_context": False,
    }

    chunks, meta = format_screen(frames, context)

    # Should include monitor info in header
    assert "(center - DP-3)" in meta.get("header", "")
    assert "Editing code" in chunks[0]["markdown"]
    assert "Documentation" in chunks[1]["markdown"]


def test_format_screen_plain_screen_no_monitor_info():
    """Test that plain screen.jsonl has no monitor info in header."""
    frames = [
        {
            "timestamp": 0,
            "analysis": {
                "primary": "code",
                "visual_description": "Editing code",
            },
        },
    ]

    context = {
        "file_path": Path("20240101/default/120000_300/screen.jsonl"),
        "include_entity_context": False,
    }

    chunks, meta = format_screen(frames, context)

    # Plain screen.jsonl should not have monitor info
    assert "(center" not in meta.get("header", "")
    assert "# Frame Analyses" in meta.get("header", "")


def test_format_screen_includes_entity_context():
    """Test that entity context is included when requested."""
    frames = [
        {
            "timestamp": 0,
            "analysis": {"primary": "code", "visual_description": "Code"},
        },
    ]

    context = {
        "file_path": Path("20240101/default/120000/screen.jsonl"),
        "entity_names": "Alice, Bob, ProjectX",
        "include_entity_context": True,
    }

    chunks, meta = format_screen(frames, context)
    markdown = "\n".join([meta.get("header", "")] + [c["markdown"] for c in chunks])

    # Should include entity context header
    assert "# Entity Context" in markdown
    assert "Alice, Bob, ProjectX" in markdown


def test_format_screen_includes_category_content():
    """Test that category-specific content is included in output."""
    frames = [
        {
            "timestamp": 0,
            "analysis": {
                "primary": "productivity",
                "visual_description": "Spreadsheet view",
            },
            "content": {
                "productivity": "| Name | Value |\n|------|-------|\n| Test | 123 |",
            },
        },
    ]

    context = {
        "file_path": Path("20240101/default/120000/screen.jsonl"),
        "include_entity_context": False,
    }

    chunks, meta = format_screen(frames, context)
    markdown = "\n".join([meta.get("header", "")] + [c["markdown"] for c in chunks])

    # Should include category content
    assert "**Productivity:**" in markdown
    assert "| Name | Value |" in markdown
    assert "| Test | 123 |" in markdown


def test_format_screen_returns_chunks_with_timestamps():
    """Test that format_screen returns chunks with timestamp metadata."""
    frames = [
        {
            "timestamp": 0,
            "analysis": {
                "primary": "code",
                "visual_description": "Frame 1",
            },
        },
        {
            "timestamp": 30,
            "analysis": {
                "primary": "terminal",
                "visual_description": "Frame 2",
            },
        },
    ]

    chunks, meta = format_screen(frames)

    assert len(chunks) == 2
    assert chunks[0]["timestamp"] == 0  # 0 seconds = 0ms
    assert chunks[1]["timestamp"] == 30000  # 30 seconds = 30000ms
    assert "Frame 1" in chunks[0]["markdown"]
    assert "Frame 2" in chunks[1]["markdown"]


def test_format_screen_returns_indexer_metadata():
    """Test that format_screen returns indexer metadata with topic."""
    frames = [
        {
            "timestamp": 0,
            "analysis": {"primary": "code", "visual_description": "Test"},
        },
    ]

    chunks, meta = format_screen(frames)

    assert "indexer" in meta
    assert meta["indexer"]["topic"] == "screen"


def test_load_category_formatter_finds_meeting():
    """Test that meeting formatter can be loaded from describe/."""
    formatter = _load_category_formatter("meeting")
    assert formatter is not None
    assert callable(formatter)


def test_load_category_formatter_returns_none_for_missing():
    """Test that missing formatter returns None without error."""
    formatter = _load_category_formatter("nonexistent_category")
    assert formatter is None


def test_load_category_formatter_caches_result():
    """Test that formatter loading is cached."""
    # Clear cache first
    from observe.screen import _formatter_cache

    _formatter_cache.clear()

    # First call loads
    formatter1 = _load_category_formatter("meeting")
    # Second call should return cached
    formatter2 = _load_category_formatter("meeting")

    assert formatter1 is formatter2
    assert "meeting" in _formatter_cache


def test_meeting_formatter_output():
    """Test that meeting formatter produces expected markdown."""
    from observe.categories.meeting import format as meeting_format

    content = {
        "platform": "zoom",
        "participants": [
            {"name": "Alice", "status": "speaking", "video": True},
            {"name": "Bob", "status": "muted", "video": False},
        ],
        "screen_share": {
            "presenter": "Alice",
            "description": "Showing slides",
            "formatted_text": "# Slide Title\n\nBullet points...",
        },
    }

    result = meeting_format(content, {})

    assert "**Meeting** (zoom)" in result
    assert "📹 Alice (speaking)" in result
    assert "🔇 Bob (muted)" in result
    assert "**Screen Share by Alice:**" in result
    assert "*Showing slides*" in result
    assert "# Slide Title" in result


def test_format_screen_uses_meeting_formatter():
    """Test that format_screen uses the meeting formatter for meeting content."""
    frames = [
        {
            "timestamp": 0,
            "analysis": {
                "primary": "meeting",
                "visual_description": "Video call",
            },
            "content": {
                "meeting": {
                    "platform": "meet",
                    "participants": [
                        {"name": "Test User", "status": "active", "video": True},
                    ],
                    "screen_share": None,
                },
            },
        },
    ]

    context = {
        "file_path": Path("20240101/default/120000/screen.jsonl"),
        "include_entity_context": False,
    }

    chunks, meta = format_screen(frames, context)
    markdown = chunks[0]["markdown"]

    # Should use meeting formatter, not JSON dump
    assert "**Meeting** (meet)" in markdown
    assert "📹 Test User (active)" in markdown
    # Should NOT have JSON code block (that was the old format)
    assert "```json" not in markdown


def test_format_screen_falls_back_for_missing_formatter():
    """Test that categories without .py formatter use default formatting."""
    frames = [
        {
            "timestamp": 0,
            "analysis": {
                "primary": "messaging",
                "visual_description": "Chat app",
            },
            "content": {
                "messaging": "**Alice**: Hello!\n**Bob**: Hi there!",
            },
        },
    ]

    context = {"include_entity_context": False}

    chunks, meta = format_screen(frames, context)
    markdown = chunks[0]["markdown"]

    # Should use default text formatting
    assert "**Messaging:**" in markdown
    assert "**Alice**: Hello!" in markdown


def test_format_screen_handles_multiple_categories():
    """Test that both primary and secondary categories are formatted."""
    frames = [
        {
            "timestamp": 0,
            "analysis": {
                "primary": "meeting",
                "secondary": "productivity",
                "overlap": False,
                "visual_description": "Meeting with shared doc",
            },
            "content": {
                "meeting": {
                    "platform": "teams",
                    "participants": [
                        {"name": "User", "status": "active", "video": True}
                    ],
                    "screen_share": None,
                },
                "productivity": "| Task | Status |\n|------|--------|\n| Review | Done |",
            },
        },
    ]

    context = {"include_entity_context": False}

    chunks, meta = format_screen(frames, context)
    markdown = chunks[0]["markdown"]

    # Both categories should be present
    assert "**Meeting** (teams)" in markdown
    assert "**Productivity:**" in markdown
    assert "| Task | Status |" in markdown


def test_categories_includes_all_expected():
    """Test that CATEGORIES includes all expected values.

    Note: tmux is NOT a describe category - it has a formatter (tmux.py)
    but no metadata (tmux.json) because tmux frames are generated directly
    by the observer, not by the describe process.
    """
    expected = [
        "terminal",
        "code",
        "messaging",
        "meeting",
        "browsing",
        "reading",
        "media",
        "gaming",
        "productivity",
    ]
    for cat in expected:
        assert cat in CATEGORIES, f"Expected category {cat} not found"
    assert len(CATEGORIES) == 9


def test_tmux_formatter_output():
    """Test that tmux formatter produces expected markdown."""
    from observe.categories.tmux import format as tmux_format

    content = {
        "session": "main",
        "window": {"id": "@1", "index": 0, "name": "bash"},
        "panes": [
            {
                "id": "%1",
                "index": 0,
                "active": True,
                "content": "$ ls -la\n\x1b[32mtotal 42\x1b[0m\ndrwxr-xr-x 2 user",
            },
        ],
    }

    result = tmux_format(content, {})

    assert "**Tmux** (main:bash)" in result
    assert "```" in result
    # ANSI codes should be stripped
    assert "\x1b[32m" not in result
    assert "total 42" in result
    assert "$ ls -la" in result


def test_tmux_formatter_multiple_panes():
    """Test tmux formatter labels multiple panes."""
    from observe.categories.tmux import format as tmux_format

    content = {
        "session": "dev",
        "window": {"name": "work"},
        "panes": [
            {"index": 0, "active": True, "content": "pane zero"},
            {"index": 1, "active": False, "content": "pane one"},
        ],
    }

    result = tmux_format(content, {})

    assert "**Pane 0 (active):**" in result
    assert "**Pane 1:**" in result
    assert "pane zero" in result
    assert "pane one" in result
