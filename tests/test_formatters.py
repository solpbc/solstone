# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for the formatters framework."""

import json
import os
import tempfile
from pathlib import Path

import pytest

# Set JOURNAL_PATH to fixtures for tests
os.environ["JOURNAL_PATH"] = str(Path(__file__).parent.parent / "fixtures" / "journal")


class TestRegistry:
    """Tests for the formatter registry."""

    def test_get_formatter_screen(self):
        """Test pattern matching for screen.jsonl."""
        from think.formatters import get_formatter

        formatter = get_formatter("20240102/234567_300/screen.jsonl")
        assert formatter is not None
        assert formatter.__name__ == "format_screen"

    def test_get_formatter_audio(self):
        """Test pattern matching for audio.jsonl."""
        from think.formatters import get_formatter

        formatter = get_formatter("20240101/123456_300/audio.jsonl")
        assert formatter is not None
        assert formatter.__name__ == "format_audio"

    def test_get_formatter_split_audio(self):
        """Test pattern matching for *_audio.jsonl files (split, imported, etc.)."""
        from think.formatters import get_formatter

        # Split audio
        formatter = get_formatter("20240101/123456_300/123456_300_audio.jsonl")
        assert formatter is not None
        assert formatter.__name__ == "format_audio"

        # Imported audio (matched by *_audio.jsonl pattern)
        formatter = get_formatter("20240101/123456_300/imported_audio.jsonl")
        assert formatter is not None
        assert formatter.__name__ == "format_audio"

    def test_get_formatter_split_screen(self):
        """Test pattern matching for *_screen.jsonl files."""
        from think.formatters import get_formatter

        # Monitor-specific screen
        formatter = get_formatter("20240101/123456_300/monitor_1_screen.jsonl")
        assert formatter is not None
        assert formatter.__name__ == "format_screen"

        # Other suffixed screen files
        formatter = get_formatter("20240101/123456_300/wayland_screen.jsonl")
        assert formatter is not None
        assert formatter.__name__ == "format_screen"

    def test_get_formatter_no_match(self):
        """Test that unmatched patterns return None."""
        from think.formatters import get_formatter

        formatter = get_formatter("random/path/unknown.jsonl")
        assert formatter is None


class TestLoadJsonl:
    """Tests for JSONL loading utility."""

    def test_load_jsonl_basic(self):
        """Test loading a basic JSONL file."""
        from think.formatters import load_jsonl

        path = Path(os.environ["JOURNAL_PATH"]) / "20240101/123456_300/audio.jsonl"
        entries = load_jsonl(path)

        assert len(entries) == 6  # 1 metadata + 5 transcript entries
        assert entries[0].get("raw") == "raw.flac"
        assert entries[1].get("start") == "00:00:01"

    def test_load_jsonl_empty_lines(self):
        """Test that empty lines are skipped."""
        from think.formatters import load_jsonl

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"a": 1}\n')
            f.write("\n")
            f.write('{"b": 2}\n')
            f.write("   \n")
            f.write('{"c": 3}\n')
            temp_path = f.name

        try:
            entries = load_jsonl(temp_path)
            assert len(entries) == 3
            assert entries[0]["a"] == 1
            assert entries[1]["b"] == 2
            assert entries[2]["c"] == 3
        finally:
            os.unlink(temp_path)

    def test_load_jsonl_malformed_skipped(self):
        """Test that malformed lines are skipped."""
        from think.formatters import load_jsonl

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"valid": 1}\n')
            f.write("not json\n")
            f.write('{"also_valid": 2}\n')
            temp_path = f.name

        try:
            entries = load_jsonl(temp_path)
            assert len(entries) == 2
        finally:
            os.unlink(temp_path)


class TestFormatFile:
    """Tests for the format_file end-to-end function."""

    def test_format_file_screen(self):
        """Test format_file with screen.jsonl."""
        from think.formatters import format_file

        path = Path(os.environ["JOURNAL_PATH"]) / "20240102/234567_300/screen.jsonl"
        chunks, meta = format_file(path)

        assert len(chunks) > 0
        # Header should contain Frame Analyses
        assert "header" in meta
        assert "Frame Analyses" in meta["header"]
        # Should have chunks for frames
        assert any("VSCode IDE" in c["markdown"] for c in chunks)

    def test_format_file_audio(self):
        """Test format_file with audio.jsonl."""
        from think.formatters import format_file

        path = Path(os.environ["JOURNAL_PATH"]) / "20240101/123456_300/audio.jsonl"
        chunks, meta = format_file(path)

        assert len(chunks) > 0
        # Should contain transcript entries
        assert any("authentication module" in c["markdown"] for c in chunks)

    def test_format_file_not_found(self):
        """Test format_file raises on missing file."""
        from think.formatters import format_file

        with pytest.raises(FileNotFoundError):
            format_file("/nonexistent/path/screen.jsonl")

    def test_format_file_no_formatter(self):
        """Test format_file raises when no formatter matches."""
        from think.formatters import format_file

        # Create a temp file that won't match any pattern
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, dir="/tmp"
        ) as f:
            f.write('{"test": 1}\n')
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="No formatter found"):
                format_file(temp_path)
        finally:
            os.unlink(temp_path)


class TestFormatScreen:
    """Tests for the screen formatter."""

    def test_format_screen_basic(self):
        """Test basic screen formatting."""
        from observe.screen import format_screen

        entries = [
            {
                "timestamp": 5,
                "analysis": {
                    "primary": "reading",
                    "visual_description": "Documentation page",
                },
                "content": {
                    "reading": "# API Reference\n\ndef hello():\n    pass",
                },
            }
        ]

        chunks, meta = format_screen(entries)

        assert len(chunks) == 1  # 1 frame chunk
        assert "header" in meta
        assert "Frame Analyses" in meta["header"]
        assert chunks[0]["timestamp"] == 5000  # 5 seconds = 5000ms
        assert "**Category:** reading" in chunks[0]["markdown"]
        assert "def hello()" in chunks[0]["markdown"]

    def test_format_screen_with_entity_context(self):
        """Test screen formatting with entity context."""
        from observe.screen import format_screen

        entries = [{"timestamp": 0, "analysis": {"primary": "browser"}}]
        context = {"entity_names": "Alice, Bob", "include_entity_context": True}

        chunks, meta = format_screen(entries, context)

        assert "header" in meta
        assert "Entity Context" in meta["header"]
        assert "Alice, Bob" in meta["header"]

    def test_format_screen_without_entity_context(self):
        """Test screen formatting without entity context."""
        from observe.screen import format_screen

        entries = [{"timestamp": 0, "analysis": {"primary": "browser"}}]
        context = {"include_entity_context": False}

        chunks, meta = format_screen(entries, context)

        assert "header" in meta
        assert "Entity Context" not in meta["header"]

    def test_format_screen_per_monitor_file(self):
        """Test screen formatting includes monitor info from filename."""
        from pathlib import Path

        from observe.screen import format_screen

        entries = [
            {"timestamp": 0, "analysis": {}},
            {"timestamp": 5, "analysis": {}},
        ]

        context = {"file_path": Path("20240101/120000_300/left_DP-1_screen.jsonl")}
        chunks, meta = format_screen(entries, context)

        # Monitor info should be in the header, not per-frame
        assert "(left - DP-1)" in meta["header"]

    def test_format_screen_meeting(self):
        """Test screen formatting with meeting data."""
        from observe.screen import format_screen

        entries = [
            {
                "timestamp": 0,
                "analysis": {"primary": "meeting"},
                "content": {
                    "meeting": {"participants": ["Alice", "Bob"]},
                },
            }
        ]

        chunks, meta = format_screen(entries)

        # New meeting formatter uses "**Meeting** (platform)" format
        assert "**Meeting**" in chunks[0]["markdown"]
        assert "Alice" in chunks[0]["markdown"]

    def test_format_screen_extracts_metadata(self):
        """Test that metadata line is extracted and not treated as a frame."""
        from observe.screen import format_screen

        entries = [
            {"raw": "screen.webm"},  # Metadata line
            {"timestamp": 5, "analysis": {"primary": "code"}},
        ]

        chunks, meta = format_screen(entries)

        # Should only have 1 frame chunk (metadata is extracted)
        assert len(chunks) == 1
        assert chunks[0]["timestamp"] == 5000  # 5 seconds = 5000ms

    def test_format_screen_skipped_entries_error(self):
        """Test that skipped entries are reported in meta.error."""
        from observe.screen import format_screen

        entries = [
            {"raw": "screen.webm"},  # Metadata
            {"timestamp": 5, "analysis": {}},  # Valid
            {"invalid": "no timestamp"},  # Skipped
            {"also_invalid": True},  # Skipped
        ]

        chunks, meta = format_screen(entries)

        assert len(chunks) == 1
        assert "error" in meta
        assert "Skipped 2 entries" in meta["error"]
        assert "timestamp" in meta["error"]


class TestFormatAudio:
    """Tests for the audio formatter."""

    def test_format_audio_basic(self):
        """Test basic audio formatting."""
        from observe.hear import format_audio

        entries = [
            {"start": "00:00:05", "source": "mic", "speaker": 1, "text": "Hello world"}
        ]

        chunks, meta = format_audio(entries)

        assert len(chunks) == 1
        assert "[00:00:05]" in chunks[0]["markdown"]
        assert "(mic)" in chunks[0]["markdown"]
        assert "Speaker 1:" in chunks[0]["markdown"]
        assert "Hello world" in chunks[0]["markdown"]

    def test_format_audio_with_metadata(self):
        """Test audio formatting extracts metadata from entries."""
        from observe.hear import format_audio

        # Metadata as first entry (like real JSONL)
        entries = [
            {"raw": "audio.flac", "setting": "office", "topics": ["work", "meeting"]},
            {"start": "00:00:01", "text": "Test"},
        ]

        chunks, meta = format_audio(entries)

        # Header should contain metadata
        assert "header" in meta
        assert "Setting: office" in meta["header"]
        assert "Topics: work, meeting" in meta["header"]
        # Should have 1 transcript chunk
        assert len(chunks) == 1
        assert "Test" in chunks[0]["markdown"]

    def test_format_audio_imported_metadata(self):
        """Test audio formatting with imported metadata."""
        from observe.hear import format_audio

        entries = [
            {"imported": {"facet": "work", "id": "20250115_103045"}},
            {"start": "00:00:01", "text": "Test"},
        ]

        chunks, meta = format_audio(entries)

        assert "header" in meta
        assert "Facet: work" in meta["header"]
        assert "Import ID: 20250115_103045" in meta["header"]

    def test_format_audio_empty_entries_skipped(self):
        """Test that entries missing 'start' field are skipped (after first line)."""
        from observe.hear import format_audio

        entries = [
            {"raw": "audio.flac"},  # Metadata (first entry without start)
            {"start": "00:00:01", "text": "Valid"},
            {"start": "00:00:05", "text": ""},  # Has start, keeps entry
            {},  # Missing start, skipped and logged
            {"start": "00:00:10", "text": "Also valid"},
        ]

        chunks, meta = format_audio(entries)

        # Should have 3 chunks (empty {} is skipped, but timestamp-only is kept)
        assert len(chunks) == 3
        assert "Valid" in chunks[0]["markdown"]
        assert "[00:00:05]" in chunks[1]["markdown"]  # Timestamp-only entry
        assert "Also valid" in chunks[2]["markdown"]
        # Should report skipped entry in error
        assert "error" in meta
        assert "Skipped 1 entries" in meta["error"]

    def test_format_audio_timestamp_ordering(self):
        """Test that timestamps are calculated correctly."""
        from observe.hear import format_audio

        entries = [
            {"start": "00:00:30", "text": "First"},
            {"start": "00:01:00", "text": "Second"},
        ]

        chunks, meta = format_audio(entries)

        # Second chunk should have higher timestamp
        assert chunks[0]["timestamp"] < chunks[1]["timestamp"]

    def test_format_audio_extracts_metadata(self):
        """Test that metadata line is extracted and not treated as transcript."""
        from observe.hear import format_audio

        entries = [
            {"raw": "audio.flac", "model": "whisper-1"},  # Metadata line
            {"start": "00:00:01", "text": "Hello"},
        ]

        chunks, meta = format_audio(entries)

        # Should only have 1 transcript chunk (metadata is extracted)
        assert len(chunks) == 1
        assert "Hello" in chunks[0]["markdown"]
        # Metadata raw/model shouldn't appear in chunks
        assert not any("audio.flac" in c["markdown"] for c in chunks)
        assert not any("whisper" in c["markdown"] for c in chunks)

    def test_format_audio_skipped_entries_error(self):
        """Test that skipped entries are reported in meta.error."""
        from observe.hear import format_audio

        entries = [
            {"raw": "audio.flac"},  # Metadata
            {"start": "00:00:01", "text": "Valid"},
            {"invalid": "no start"},  # Skipped
            {"also_invalid": True},  # Skipped
        ]

        chunks, meta = format_audio(entries)

        assert len(chunks) == 1
        assert "error" in meta
        assert "Skipped 2 entries" in meta["error"]
        assert "start" in meta["error"]


class TestLoadTranscriptBackwardCompat:
    """Tests for backward compatibility of load_transcript."""

    def test_load_transcript_returns_tuple(self):
        """Test that load_transcript still returns (metadata, entries, text) tuple."""
        from observe.hear import load_transcript

        path = Path(os.environ["JOURNAL_PATH"]) / "20240101/123456_300/audio.jsonl"
        metadata, entries, formatted_text = load_transcript(path)

        assert isinstance(metadata, dict)
        assert isinstance(entries, list)
        assert isinstance(formatted_text, str)
        assert "raw" in metadata
        assert len(entries) == 5  # 5 transcript entries (not counting metadata)
        assert "authentication module" in formatted_text


class TestFormatAgent:
    """Tests for the agent formatter."""

    def test_get_formatter_agent(self):
        """Test pattern matching for agents/*.jsonl."""
        from think.formatters import get_formatter

        formatter = get_formatter("agents/1700000000001.jsonl")
        assert formatter is not None
        assert formatter.__name__ == "format_agent"

    def test_get_formatter_active_agent_no_match(self):
        """Test that *_active.jsonl files don't match (running agents excluded)."""
        from think.formatters import get_formatter

        # Pattern agents/*.jsonl should NOT match agents/*_active.jsonl
        # because the pattern requires direct child, not a specific suffix
        formatter = get_formatter("agents/1700000000001_active.jsonl")
        # It will still match agents/*.jsonl - pattern just includes all .jsonl
        # The filtering of active vs completed is done by the caller
        assert formatter is not None

    def test_format_agent_basic(self):
        """Test basic agent formatting with fixture file."""
        from think.formatters import format_file

        path = Path(os.environ["JOURNAL_PATH"]) / "agents/1700000000001.jsonl"
        chunks, meta = format_file(path)

        assert len(chunks) > 0
        assert "header" in meta
        assert "Agent Run: 1700000000001" in meta["header"]
        assert "project updates" in meta["header"]

    def test_format_agent_direct(self):
        """Test format_agent function directly."""
        from think.cortex import format_agent

        entries = [
            {
                "event": "request",
                "ts": 1700000000000,
                "agent_id": "test123",
                "prompt": "Hello world",
                "name": "default",
                "provider": "openai",
            },
            {
                "event": "start",
                "ts": 1700000000100,
                "agent_id": "test123",
                "model": "gpt-4",
                "name": "default",
            },
            {"event": "finish", "result": "Hello!", "ts": 1700000000200},
        ]

        chunks, meta = format_agent(entries)

        assert "header" in meta
        assert "Agent Run: test123" in meta["header"]
        assert "Hello world" in meta["header"]
        assert "gpt-4" in meta["header"]
        assert len(chunks) == 1  # Just finish
        assert "Hello!" in chunks[0]["markdown"]

    def test_format_agent_with_thinking(self):
        """Test agent formatting with thinking event."""
        from think.cortex import format_agent

        entries = [
            {"event": "start", "ts": 1700000000000, "agent_id": "test"},
            {
                "event": "thinking",
                "summary": "I should analyze this...",
                "ts": 1700000000100,
            },
            {"event": "finish", "result": "Done", "ts": 1700000000200},
        ]

        chunks, meta = format_agent(entries)

        # Should have thinking chunk and finish chunk
        assert len(chunks) == 2
        thinking_chunk = chunks[0]
        assert "Thinking" in thinking_chunk["markdown"]
        assert "> I should analyze this" in thinking_chunk["markdown"]

    def test_format_agent_tool_pairing(self):
        """Test that tool_start/tool_end are paired by call_id."""
        from think.cortex import format_agent

        entries = [
            {"event": "start", "ts": 1700000000000, "agent_id": "test"},
            {
                "event": "tool_start",
                "tool": "search",
                "args": {"q": "test"},
                "call_id": "call_1",
                "ts": 1700000000100,
            },
            {
                "event": "tool_end",
                "result": '{"results": []}',
                "call_id": "call_1",
                "ts": 1700000000200,
            },
            {"event": "finish", "result": "Done", "ts": 1700000000300},
        ]

        chunks, meta = format_agent(entries)

        # Tool start/end should be combined into one chunk
        assert len(chunks) == 2  # tool + finish
        tool_chunk = chunks[0]
        assert "Tool: search" in tool_chunk["markdown"]
        assert '"q": "test"' in tool_chunk["markdown"]
        assert '{"results": []}' in tool_chunk["markdown"]

    def test_format_agent_truncates_long_results(self):
        """Test that tool results are truncated at 500 chars."""
        from think.cortex import format_agent

        long_result = "x" * 1000  # 1000 chars

        entries = [
            {"event": "start", "ts": 1700000000000, "agent_id": "test"},
            {
                "event": "tool_start",
                "tool": "fetch",
                "args": {},
                "call_id": "call_1",
                "ts": 1700000000100,
            },
            {
                "event": "tool_end",
                "result": long_result,
                "call_id": "call_1",
                "ts": 1700000000200,
            },
            {"event": "finish", "result": "Done", "ts": 1700000000300},
        ]

        chunks, meta = format_agent(entries)

        tool_chunk = chunks[0]
        # Should be truncated and mention how many chars truncated
        assert "500 chars truncated" in tool_chunk["markdown"]
        assert "x" * 500 in tool_chunk["markdown"]
        assert "x" * 600 not in tool_chunk["markdown"]

    def test_format_agent_unpaired_tool_start(self):
        """Test that unpaired tool_start shows 'did not complete'."""
        from think.cortex import format_agent

        entries = [
            {"event": "start", "ts": 1700000000000, "agent_id": "test"},
            {
                "event": "tool_start",
                "tool": "slow_task",
                "args": {"x": 1},
                "call_id": "call_orphan",
                "ts": 1700000000100,
            },
            # No tool_end - agent crashed
            {"event": "error", "error": "Agent timeout", "ts": 1700000000200},
        ]

        chunks, meta = format_agent(entries)

        # Should have error chunk and unpaired tool chunk
        assert len(chunks) == 2
        # Find the unpaired tool chunk
        tool_chunks = [c for c in chunks if "slow_task" in c["markdown"]]
        assert len(tool_chunks) == 1
        assert "did not complete" in tool_chunks[0]["markdown"]

    def test_format_agent_error_event(self):
        """Test formatting of error events."""
        from think.cortex import format_agent

        entries = [
            {"event": "start", "ts": 1700000000000, "agent_id": "test"},
            {
                "event": "error",
                "error": "Connection failed",
                "trace": "Traceback:\n  File...",
                "ts": 1700000000100,
            },
        ]

        chunks, meta = format_agent(entries)

        assert len(chunks) == 1
        error_chunk = chunks[0]
        assert "Error" in error_chunk["markdown"]
        assert "Connection failed" in error_chunk["markdown"]
        assert "Trace" in error_chunk["markdown"]

    def test_format_agent_skipped_entries(self):
        """Test that entries without 'event' field are skipped and reported."""
        from think.cortex import format_agent

        entries = [
            {"event": "start", "ts": 1700000000000, "agent_id": "test"},
            {"invalid": "no event field"},
            {"also_invalid": True},
            {"event": "finish", "result": "Done", "ts": 1700000000100},
        ]

        chunks, meta = format_agent(entries)

        assert "error" in meta
        assert "Skipped 2 entries" in meta["error"]

    def test_format_agent_agent_updated(self):
        """Test formatting of agent_updated events."""
        from think.cortex import format_agent

        entries = [
            {"event": "start", "ts": 1700000000000, "agent_id": "test"},
            {"event": "agent_updated", "agent": "NewAgent", "ts": 1700000000100},
            {"event": "finish", "result": "Done", "ts": 1700000000200},
        ]

        chunks, meta = format_agent(entries)

        assert len(chunks) == 2
        assert "Switched to agent: NewAgent" in chunks[0]["markdown"]

    def test_format_agent_continue_event(self):
        """Test formatting of continue events."""
        from think.cortex import format_agent

        entries = [
            {"event": "start", "ts": 1700000000000, "agent_id": "test"},
            {"event": "finish", "result": "Done", "ts": 1700000000100},
            {"event": "continue", "to": "agent_456", "ts": 1700000000200},
        ]

        chunks, meta = format_agent(entries)

        assert len(chunks) == 2
        assert "Continued in agent: agent_456" in chunks[1]["markdown"]

    def test_format_agent_handoff_from_header(self):
        """Test that handoff_from appears in header."""
        from think.cortex import format_agent

        entries = [
            {
                "event": "request",
                "ts": 1700000000000,
                "agent_id": "test",
                "prompt": "Continue work",
                "handoff_from": "parent_123",
            },
            {"event": "start", "ts": 1700000000100, "agent_id": "test"},
            {"event": "finish", "result": "Done", "ts": 1700000000200},
        ]

        chunks, meta = format_agent(entries)

        assert "Handoff from:" in meta["header"]
        assert "parent_123" in meta["header"]


class TestFormatEntities:
    """Tests for the entities formatter."""

    def test_get_formatter_detected_entities(self):
        """Test pattern matching for detected entities."""
        from think.formatters import get_formatter

        formatter = get_formatter("facets/personal/entities/20250101.jsonl")
        assert formatter is not None
        assert formatter.__name__ == "format_entities"

    def test_format_entities_detected_basic(self):
        """Test basic detected entities formatting with fixture file."""
        from think.formatters import format_file

        path = (
            Path(os.environ["JOURNAL_PATH"]) / "facets/personal/entities/20250101.jsonl"
        )
        chunks, meta = format_file(path)

        assert len(chunks) == 2  # 2 entities in fixture
        assert "header" in meta
        assert "Detected Entities: personal" in meta["header"]
        assert "2025-01-01" in meta["header"]
        assert "2 entities" in meta["header"]

    def test_format_entities_direct(self):
        """Test format_entities function directly."""
        from think.entities import format_entities

        entries = [
            {"type": "Person", "name": "Alice", "description": "Friend from work"},
            {"type": "Company", "name": "Acme Corp", "description": "Tech startup"},
        ]

        chunks, meta = format_entities(entries)

        assert len(chunks) == 2
        assert "Person: Alice" in chunks[0]["markdown"]
        assert "Friend from work" in chunks[0]["markdown"]
        assert "Company: Acme Corp" in chunks[1]["markdown"]

    def test_format_entities_no_description(self):
        """Test that missing description shows placeholder."""
        from think.entities import format_entities

        entries = [{"type": "Person", "name": "Bob", "description": ""}]

        chunks, meta = format_entities(entries)

        assert len(chunks) == 1
        assert "Person: Bob" in chunks[0]["markdown"]
        assert "No description available" in chunks[0]["markdown"]

    def test_format_entities_with_tags(self):
        """Test formatting with tags field."""
        from think.entities import format_entities

        entries = [
            {
                "type": "Company",
                "name": "Acme",
                "description": "A company",
                "tags": ["tech", "startup"],
            }
        ]

        chunks, meta = format_entities(entries)

        assert "**Tags:** tech, startup" in chunks[0]["markdown"]

    def test_format_entities_with_aka(self):
        """Test formatting with aka field."""
        from think.entities import format_entities

        entries = [
            {
                "type": "Person",
                "name": "Robert Smith",
                "description": "Colleague",
                "aka": ["Bob", "Bobby"],
            }
        ]

        chunks, meta = format_entities(entries)

        assert "**Also known as:** Bob, Bobby" in chunks[0]["markdown"]

    def test_format_entities_with_custom_fields(self):
        """Test formatting with custom fields."""
        from think.entities import format_entities

        entries = [
            {
                "type": "Person",
                "name": "Alice",
                "description": "Friend",
                "contact": "alice@example.com",
                "since": "2020",
            }
        ]

        chunks, meta = format_entities(entries)

        assert "**Contact:** alice@example.com" in chunks[0]["markdown"]
        assert "**Since:** 2020" in chunks[0]["markdown"]

    def test_format_entities_timestamp_updated_at(self):
        """Test that updated_at is used for timestamp."""
        from think.entities import format_entities

        entries = [
            {
                "type": "Person",
                "name": "Alice",
                "description": "Friend",
                "updated_at": 1700000000000,
            }
        ]

        chunks, meta = format_entities(entries)

        assert chunks[0]["timestamp"] == 1700000000000

    def test_format_entities_timestamp_attached_at_fallback(self):
        """Test that attached_at is used when updated_at is missing."""
        from think.entities import format_entities

        entries = [
            {
                "type": "Person",
                "name": "Alice",
                "description": "Friend",
                "attached_at": 1600000000000,
            }
        ]

        chunks, meta = format_entities(entries)

        assert chunks[0]["timestamp"] == 1600000000000

    def test_format_entities_timestamp_priority(self):
        """Test that updated_at takes priority over attached_at."""
        from think.entities import format_entities

        entries = [
            {
                "type": "Person",
                "name": "Alice",
                "description": "Friend",
                "updated_at": 1700000000000,
                "attached_at": 1600000000000,
            }
        ]

        chunks, meta = format_entities(entries)

        # updated_at should take priority
        assert chunks[0]["timestamp"] == 1700000000000

    def test_format_entities_timestamp_last_seen_priority(self):
        """Test that last_seen takes priority over updated_at and attached_at."""
        from datetime import datetime

        from think.entities import format_entities

        entries = [
            {
                "type": "Person",
                "name": "Alice",
                "description": "Friend",
                "last_seen": "20260115",  # Jan 15 2026
                "updated_at": 1700000000000,
                "attached_at": 1600000000000,
            }
        ]

        chunks, meta = format_entities(entries)

        # last_seen should take priority (converted to local midnight ms)
        expected = int(datetime(2026, 1, 15).timestamp() * 1000)
        assert chunks[0]["timestamp"] == expected

    def test_format_entities_header_facet_from_path(self):
        """Test that facet name is extracted from file path for detected entities."""
        from think.entities import format_entities

        entries = [{"type": "Person", "name": "Test", "description": ""}]
        # Use detected entities path pattern (facets/*/entities/YYYYMMDD.jsonl)
        context = {"file_path": "/journal/facets/work/entities/20260115.jsonl"}

        chunks, meta = format_entities(entries, context)

        assert "Detected Entities: work" in meta["header"]
        assert "2026-01-15" in meta["header"]

    def test_format_entities_detected_header_from_path(self):
        """Test that detected entities include day in header."""
        from think.entities import format_entities

        entries = [{"type": "Person", "name": "Test", "description": ""}]
        context = {"file_path": "/journal/facets/personal/entities/20251201.jsonl"}

        chunks, meta = format_entities(entries, context)

        assert "Detected Entities: personal" in meta["header"]
        assert "2025-12-01" in meta["header"]


class TestFormatTodos:
    """Tests for the todos formatter."""

    def test_get_formatter_todos(self):
        """Test pattern matching for todos/*.jsonl."""
        from think.formatters import get_formatter

        formatter = get_formatter("facets/personal/todos/20240101.jsonl")
        assert formatter is not None
        assert formatter.__name__ == "format_todos"

    def test_format_todos_basic(self):
        """Test basic todos formatting with fixture file."""
        from think.formatters import format_file

        path = Path(os.environ["JOURNAL_PATH"]) / "facets/personal/todos/20240101.jsonl"
        chunks, meta = format_file(path)

        assert len(chunks) == 4  # 4 items in fixture
        assert "header" in meta
        assert "Todos: personal" in meta["header"]
        assert "2024-01-01" in meta["header"]
        assert "4 items" in meta["header"]

    def test_format_todos_direct(self):
        """Test format_todos function directly."""
        from apps.todos.todo import format_todos

        entries = [
            {"text": "Do something", "completed": False},
            {"text": "Done task", "completed": True},
        ]

        chunks, meta = format_todos(entries)

        assert len(chunks) == 2
        assert "* [ ] Do something" in chunks[0]["markdown"]
        assert "* [x] Done task" in chunks[1]["markdown"]

    def test_format_todos_list_item_prefix(self):
        """Test that all items have * prefix for markdown list."""
        from apps.todos.todo import format_todos

        entries = [
            {"text": "Task one", "completed": False},
            {"text": "Task two", "completed": True},
            {"text": "Cancelled", "cancelled": True},
        ]

        chunks, meta = format_todos(entries)

        assert len(chunks) == 3
        for chunk in chunks:
            assert chunk["markdown"].startswith("* ")

    def test_format_todos_completed_cancelled_display(self):
        """Test checkbox and strikethrough rendering."""
        from apps.todos.todo import format_todos

        entries = [
            {"text": "Incomplete", "completed": False},
            {"text": "Complete", "completed": True},
            {"text": "Cancelled", "cancelled": True},
        ]

        chunks, meta = format_todos(entries)

        assert "[ ] Incomplete" in chunks[0]["markdown"]
        assert "[x] Complete" in chunks[1]["markdown"]
        assert "~~[cancelled] Cancelled~~" in chunks[2]["markdown"]

    def test_format_todos_with_time(self):
        """Test formatting with time annotation."""
        from apps.todos.todo import format_todos

        entries = [{"text": "Meeting", "time": "14:00", "completed": False}]

        chunks, meta = format_todos(entries)

        assert len(chunks) == 1
        assert "Meeting (14:00)" in chunks[0]["markdown"]

    def test_format_todos_header_facet_from_path(self):
        """Test that facet name and day are extracted from file path."""
        from apps.todos.todo import format_todos

        entries = [{"text": "Test", "completed": False}]
        context = {"file_path": "/journal/facets/work/todos/20251215.jsonl"}

        chunks, meta = format_todos(entries, context)

        assert "Todos: work" in meta["header"]
        assert "2025-12-15" in meta["header"]

    def test_format_todos_skipped_entries_error(self):
        """Test that entries without 'text' field are skipped and reported."""
        from apps.todos.todo import format_todos

        entries = [
            {"text": "Valid", "completed": False},
            {"invalid": "no text"},
            {"also_invalid": True},
        ]

        chunks, meta = format_todos(entries)

        assert len(chunks) == 1
        assert "error" in meta
        assert "Skipped 2 entries" in meta["error"]
        assert "text" in meta["error"]

    def test_format_todos_timestamp_fallback(self):
        """Test timestamp fallback: updated_at -> created_at -> file mtime."""
        from apps.todos.todo import format_todos

        # Entry with updated_at takes priority
        entries_updated = [
            {"text": "Test", "updated_at": 1700000000000, "created_at": 1600000000000}
        ]
        chunks, _ = format_todos(entries_updated)
        assert chunks[0]["timestamp"] == 1700000000000

        # Entry with only created_at
        entries_created = [{"text": "Test", "created_at": 1600000000000}]
        chunks, _ = format_todos(entries_created)
        assert chunks[0]["timestamp"] == 1600000000000

        # Entry with neither uses 0 (no file context)
        entries_none = [{"text": "Test"}]
        chunks, _ = format_todos(entries_none)
        assert chunks[0]["timestamp"] == 0


class TestFormatEvents:
    """Tests for the events formatter."""

    def test_get_formatter_events(self):
        """Test pattern matching for events/*.jsonl."""
        from think.formatters import get_formatter

        formatter = get_formatter("facets/work/events/20240101.jsonl")
        assert formatter is not None
        assert formatter.__name__ == "format_events"

    def test_format_events_basic(self):
        """Test basic events formatting with fixture file."""
        from think.formatters import format_file

        path = Path(os.environ["JOURNAL_PATH"]) / "facets/work/events/20240101.jsonl"
        chunks, meta = format_file(path)

        assert len(chunks) == 2  # 2 events in fixture
        assert "header" in meta
        assert meta["header"] == "# Events for 'work' facet on 2024-01-01"

    def test_format_events_direct(self):
        """Test format_events function directly."""
        from think.events import format_events

        entries = [
            {
                "type": "meeting",
                "title": "Team standup",
                "start": "09:00:00",
                "end": "09:30:00",
                "summary": "Daily sync",
                "participants": ["Alice", "Bob"],
                "occurred": True,
            },
            {
                "type": "task",
                "title": "Code review",
                "start": "10:00:00",
                "summary": "Review PR",
                "occurred": True,
            },
        ]

        chunks, meta = format_events(entries)

        assert len(chunks) == 2
        assert "Meeting: Team standup" in chunks[0]["markdown"]
        assert "**Time Occurred:** 09:00 - 09:30" in chunks[0]["markdown"]
        assert "**Participants:** Alice, Bob" in chunks[0]["markdown"]
        assert "Daily sync" in chunks[0]["markdown"]
        assert "Task: Code review" in chunks[1]["markdown"]

    def test_format_events_anticipation_labels(self):
        """Test that anticipations use 'Planned', 'Scheduled', 'Expected' labels."""
        from think.events import format_events

        entries = [
            {
                "type": "meeting",
                "title": "Project kickoff",
                "start": "14:00:00",
                "occurred": False,
                "source": "20240101/agents/schedule.md",
                "participants": ["Alice", "Bob"],
            }
        ]

        chunks, meta = format_events(entries)

        assert len(chunks) == 1
        assert "### Planned Meeting: Project kickoff" in chunks[0]["markdown"]
        assert "**Time Scheduled:** 14:00" in chunks[0]["markdown"]
        assert "**Expected Participants:** Alice, Bob" in chunks[0]["markdown"]
        assert "**Created on:** 2024-01-01" in chunks[0]["markdown"]

    def test_format_events_occurrence_no_created_on(self):
        """Test that occurrences do NOT show 'Created on' or 'Planned' prefix."""
        from think.events import format_events

        entries = [
            {
                "type": "meeting",
                "title": "Team standup",
                "start": "09:00:00",
                "occurred": True,
                "source": "20240101/agents/meetings.md",
                "participants": ["Alice"],
            }
        ]

        chunks, meta = format_events(entries)

        assert len(chunks) == 1
        assert "### Meeting: Team standup" in chunks[0]["markdown"]
        assert "Planned" not in chunks[0]["markdown"]
        assert "Created on" not in chunks[0]["markdown"]
        assert "**Participants:** Alice" in chunks[0]["markdown"]
        assert "Expected" not in chunks[0]["markdown"]

    def test_format_events_header_facet_from_path(self):
        """Test that facet name and day are extracted from file path."""
        from think.events import format_events

        entries = [{"type": "task", "title": "Test", "occurred": True}]
        context = {"file_path": "/journal/facets/personal/events/20251215.jsonl"}

        chunks, meta = format_events(entries, context)

        assert meta["header"] == "# Events for 'personal' facet on 2025-12-15"

    def test_format_events_timestamp_calculation(self):
        """Test that timestamp is calculated from day + start time."""
        from think.events import format_events

        entries = [
            {
                "type": "meeting",
                "title": "Morning",
                "start": "09:00:00",
                "occurred": True,
            },
            {
                "type": "meeting",
                "title": "Afternoon",
                "start": "14:30:00",
                "occurred": True,
            },
        ]
        context = {"file_path": "/journal/facets/work/events/20240101.jsonl"}

        chunks, meta = format_events(entries, context)

        # Afternoon should have higher timestamp than morning
        assert chunks[1]["timestamp"] > chunks[0]["timestamp"]

    def test_format_events_skipped_entries_error(self):
        """Test that entries without 'title' field are skipped and reported."""
        from think.events import format_events

        entries = [
            {"type": "meeting", "title": "Valid", "occurred": True},
            {"type": "task", "summary": "No title"},  # Missing title
            {"invalid": True},  # Missing title
        ]

        chunks, meta = format_events(entries)

        assert len(chunks) == 1
        assert "error" in meta
        assert "Skipped 2 entries" in meta["error"]
        assert "title" in meta["error"]

    def test_format_events_mixed_occurred_anticipated(self):
        """Test header counts for mixed occurred/anticipated events."""
        from think.events import format_events

        entries = [
            {"type": "meeting", "title": "Past event", "occurred": True},
            {"type": "meeting", "title": "Future event 1", "occurred": False},
            {"type": "meeting", "title": "Future event 2", "occurred": False},
        ]

        chunks, meta = format_events(entries)

        # Header doesn't include counts anymore
        assert "header" in meta

    def test_format_events_time_display_24h(self):
        """Test that times are displayed in 24-hour format without seconds."""
        from think.events import format_events

        entries = [
            {
                "type": "meeting",
                "title": "Late meeting",
                "start": "14:30:00",
                "end": "16:00:00",
                "occurred": True,
            }
        ]

        chunks, meta = format_events(entries)

        assert "**Time Occurred:** 14:30 - 16:00" in chunks[0]["markdown"]
        # Should NOT include seconds
        assert "14:30:00" not in chunks[0]["markdown"]

    def test_format_events_with_details(self):
        """Test that details field is included in output."""
        from think.events import format_events

        entries = [
            {
                "type": "meeting",
                "title": "Planning",
                "summary": "Sprint planning",
                "details": "Discussed Q1 roadmap and priorities",
                "occurred": True,
            }
        ]

        chunks, meta = format_events(entries)

        assert "Sprint planning" in chunks[0]["markdown"]
        assert "Discussed Q1 roadmap" in chunks[0]["markdown"]


class TestFormatMarkdown:
    """Tests for the markdown output formatter."""

    def test_get_formatter_markdown(self):
        """Test pattern matching for .md files."""
        from think.formatters import get_formatter

        formatter = get_formatter("20240101/agents/flow.md")
        assert formatter is not None
        assert formatter.__name__ == "format_markdown"

    def test_get_formatter_segment_screen_md(self):
        """Test pattern matching for segment screen.md files."""
        from think.formatters import get_formatter

        formatter = get_formatter("20240101/123456_300/screen.md")
        assert formatter is not None
        assert formatter.__name__ == "format_markdown"

    def test_get_formatter_nested_md(self):
        """Test pattern matching for deeply nested .md files."""
        from think.formatters import get_formatter

        formatter = get_formatter("facets/work/news/20240101.md")
        assert formatter is not None
        assert formatter.__name__ == "format_markdown"

    def test_format_markdown_basic(self):
        """Test basic markdown formatting."""
        from think.markdown import format_markdown

        text = "# Hello\n\nThis is a paragraph.\n"
        chunks, meta = format_markdown(text)

        assert len(chunks) == 1
        assert "# Hello" in chunks[0]["markdown"]
        assert "This is a paragraph" in chunks[0]["markdown"]
        assert meta == {}

    def test_format_markdown_multiple_chunks(self):
        """Test that lists are split into multiple chunks."""
        from think.markdown import format_markdown

        text = "# List\n\n- Item one\n- Item two\n- Item three\n"
        chunks, meta = format_markdown(text)

        assert len(chunks) == 3
        for chunk in chunks:
            assert "# List" in chunk["markdown"]

    def test_format_markdown_no_timestamp(self):
        """Test that markdown chunks don't have timestamp key."""
        from think.markdown import format_markdown

        text = "# Test\n\nSome content.\n"
        chunks, meta = format_markdown(text)

        assert len(chunks) == 1
        assert "markdown" in chunks[0]
        assert "timestamp" not in chunks[0]

    def test_format_markdown_preserves_headers(self):
        """Test that each chunk includes its header context."""
        from think.markdown import format_markdown

        text = "# Top\n\n## Section\n\nParagraph content.\n"
        chunks, meta = format_markdown(text)

        assert len(chunks) == 1
        assert "# Top" in chunks[0]["markdown"]
        assert "## Section" in chunks[0]["markdown"]
        assert "Paragraph content" in chunks[0]["markdown"]

    def test_format_markdown_definition_list(self):
        """Test that definition lists stay as single chunk."""
        from think.markdown import format_markdown

        text = "# Info\n\n- **Name:** Alice\n- **Role:** Engineer\n"
        chunks, meta = format_markdown(text)

        # Definition list stays together
        assert len(chunks) == 1
        assert "**Name:** Alice" in chunks[0]["markdown"]
        assert "**Role:** Engineer" in chunks[0]["markdown"]

    def test_format_markdown_table_rows(self):
        """Test that table rows become separate chunks."""
        from think.markdown import format_markdown

        text = """# Data

| Name | Value |
|------|-------|
| A    | 1     |
| B    | 2     |
"""
        chunks, meta = format_markdown(text)

        assert len(chunks) == 2
        # Each chunk should have the header
        for chunk in chunks:
            assert "# Data" in chunk["markdown"]
            assert "| Name | Value |" in chunk["markdown"]

    def test_format_markdown_code_block(self):
        """Test that code blocks become chunks."""
        from think.markdown import format_markdown

        text = "# Code\n\n```python\nprint('hello')\n```\n"
        chunks, meta = format_markdown(text)

        assert len(chunks) == 1
        assert "```python" in chunks[0]["markdown"]
        assert "print('hello')" in chunks[0]["markdown"]

    def test_format_file_markdown(self):
        """Test format_file with a markdown file."""
        from think.formatters import format_file

        path = Path(os.environ["JOURNAL_PATH"]) / "20240101/agents/flow.md"
        chunks, meta = format_file(path)

        assert len(chunks) > 0
        assert all("markdown" in c for c in chunks)
        assert meta == {}

    def test_load_markdown(self):
        """Test load_markdown utility."""
        from think.formatters import load_markdown

        path = Path(os.environ["JOURNAL_PATH"]) / "20240101/agents/flow.md"
        text = load_markdown(path)

        assert isinstance(text, str)
        assert len(text) > 0


class TestExtractPathMetadata:
    """Tests for extract_path_metadata helper."""

    def test_daily_output(self):
        """Test day extraction from daily agent output path."""
        from think.formatters import extract_path_metadata

        meta = extract_path_metadata("20240101/agents/flow.md")
        assert meta["day"] == "20240101"
        assert meta["facet"] == ""
        assert meta["topic"] == "flow"

    def test_segment_markdown(self):
        """Test day and topic extraction from segment markdown."""
        from think.formatters import extract_path_metadata

        meta = extract_path_metadata("20240101/100000/screen.md")
        assert meta["day"] == "20240101"
        assert meta["facet"] == ""
        assert meta["topic"] == "screen"

    def test_segment_jsonl_no_topic(self):
        """Test that JSONL files get empty topic (formatter provides it)."""
        from think.formatters import extract_path_metadata

        meta = extract_path_metadata("20240101/100000/audio.jsonl")
        assert meta["day"] == "20240101"
        assert meta["facet"] == ""
        assert meta["topic"] == ""  # Formatter provides topic for JSONL

    def test_facet_event(self):
        """Test facet and day extraction from event path."""
        from think.formatters import extract_path_metadata

        meta = extract_path_metadata("facets/work/events/20240101.jsonl")
        assert meta["day"] == "20240101"
        assert meta["facet"] == "work"
        assert meta["topic"] == ""  # Formatter provides topic

    def test_facet_entities_detected_personal(self):
        """Test facet and day extraction from detected entities path."""
        from think.formatters import extract_path_metadata

        meta = extract_path_metadata("facets/personal/entities/20260115.jsonl")
        assert meta["day"] == "20260115"
        assert meta["facet"] == "personal"
        assert meta["topic"] == ""  # Formatter provides topic

    def test_facet_entities_detected(self):
        """Test facet and day extraction from detected entities path."""
        from think.formatters import extract_path_metadata

        meta = extract_path_metadata("facets/work/entities/20250101.jsonl")
        assert meta["day"] == "20250101"
        assert meta["facet"] == "work"
        assert meta["topic"] == ""  # Formatter provides topic

    def test_facet_news(self):
        """Test facet news markdown gets topic from path."""
        from think.formatters import extract_path_metadata

        meta = extract_path_metadata("facets/work/news/20240101.md")
        assert meta["day"] == "20240101"
        assert meta["facet"] == "work"
        assert meta["topic"] == "news"

    def test_import_summary(self):
        """Test import summary path extraction."""
        from think.formatters import extract_path_metadata

        meta = extract_path_metadata("imports/20240101_093000/summary.md")
        assert meta["day"] == "20240101"
        assert meta["facet"] == ""
        assert meta["topic"] == "import"

    def test_app_output(self):
        """Test app output path extraction."""
        from think.formatters import extract_path_metadata

        meta = extract_path_metadata("apps/myapp/agents/custom.md")
        assert meta["day"] == ""
        assert meta["facet"] == ""
        assert meta["topic"] == "myapp:custom"

    def test_config_actions(self):
        """Test journal-level action log path extraction."""
        from think.formatters import extract_path_metadata

        meta = extract_path_metadata("config/actions/20240101.jsonl")
        assert meta["day"] == "20240101"
        assert meta["facet"] == ""
        assert meta["topic"] == ""


class TestFormatterIndexerMetadata:
    """Tests verifying formatters return indexer metadata."""

    def test_format_audio_returns_indexer(self):
        """Test format_audio returns indexer with topic."""
        from observe.hear import format_audio

        entries = [{"start": "00:00:01", "text": "Hello"}]
        chunks, meta = format_audio(entries)

        assert "indexer" in meta
        assert meta["indexer"]["topic"] == "audio"

    def test_format_screen_returns_indexer(self):
        """Test format_screen returns indexer with topic."""
        from observe.screen import format_screen

        entries = [{"timestamp": 0, "analysis": {"primary": "code"}}]
        chunks, meta = format_screen(entries)

        assert "indexer" in meta
        assert meta["indexer"]["topic"] == "screen"

    def test_format_events_returns_indexer(self):
        """Test format_events returns indexer with topic."""
        from think.events import format_events

        entries = [{"type": "meeting", "title": "Test", "occurred": True}]
        chunks, meta = format_events(entries)

        assert "indexer" in meta
        assert meta["indexer"]["topic"] == "event"

    def test_format_entities_attached_returns_indexer(self):
        """Test format_entities returns indexer with attached topic."""
        from think.entities import format_entities

        entries = [{"type": "Person", "name": "Alice", "description": "Test"}]
        # No file_path context means not detected
        chunks, meta = format_entities(entries)

        assert "indexer" in meta
        assert meta["indexer"]["topic"] == "entity:attached"

    def test_format_entities_detected_returns_indexer(self):
        """Test format_entities returns indexer with detected topic."""
        from think.entities import format_entities

        entries = [{"type": "Person", "name": "Alice", "description": "Test"}]
        context = {"file_path": "/journal/facets/work/entities/20240101.jsonl"}
        chunks, meta = format_entities(entries, context)

        assert "indexer" in meta
        assert meta["indexer"]["topic"] == "entity:detected"

    def test_format_todos_returns_indexer(self):
        """Test format_todos returns indexer with topic."""
        from apps.todos.todo import format_todos

        entries = [{"text": "Test task", "completed": False}]
        chunks, meta = format_todos(entries)

        assert "indexer" in meta
        assert meta["indexer"]["topic"] == "todo"


class TestFormatterSourceKey:
    """Tests verifying formatters return source key with original entry."""

    def test_format_audio_returns_source(self):
        """Test format_audio returns source with original entry."""
        from observe.hear import format_audio

        entry = {"start": "00:00:01", "source": "mic", "speaker": 1, "text": "Hello"}
        entries = [entry]
        chunks, meta = format_audio(entries)

        assert len(chunks) == 1
        assert "source" in chunks[0]
        assert chunks[0]["source"] is entry
        assert chunks[0]["source"]["text"] == "Hello"

    def test_format_audio_string_speaker(self):
        """Test format_audio handles string speaker labels from diarization."""
        from observe.hear import format_audio

        entries = [
            {"start": "00:00:01", "speaker": "Speaker 1", "text": "Hello"},
            {"start": "00:00:05", "speaker": "Speaker 2", "text": "Hi there"},
        ]
        chunks, meta = format_audio(entries)

        assert len(chunks) == 2
        # String speakers should be used directly (no "Speaker" prefix added)
        assert "Speaker 1:" in chunks[0]["markdown"]
        assert "Speaker 2:" in chunks[1]["markdown"]

    def test_format_audio_int_speaker(self):
        """Test format_audio handles integer speaker labels (legacy format)."""
        from observe.hear import format_audio

        entries = [
            {"start": "00:00:01", "speaker": 1, "text": "Hello"},
            {"start": "00:00:05", "speaker": 2, "text": "Hi there"},
        ]
        chunks, meta = format_audio(entries)

        assert len(chunks) == 2
        # Integer speakers should get "Speaker" prefix
        assert "Speaker 1:" in chunks[0]["markdown"]
        assert "Speaker 2:" in chunks[1]["markdown"]

    def test_format_screen_returns_source(self):
        """Test format_screen returns source with original frame."""
        from observe.screen import format_screen

        frame = {
            "timestamp": 5,
            "analysis": {"primary": "code"},
            "extra_field": "value",
        }
        entries = [frame]
        chunks, meta = format_screen(entries)

        assert len(chunks) == 1
        assert "source" in chunks[0]
        assert chunks[0]["source"] is frame
        assert chunks[0]["source"]["extra_field"] == "value"

    def test_format_events_returns_source(self):
        """Test format_events returns source with original event."""
        from think.events import format_events

        event = {"type": "meeting", "title": "Test", "occurred": True, "custom": "data"}
        entries = [event]
        chunks, meta = format_events(entries)

        assert len(chunks) == 1
        assert "source" in chunks[0]
        assert chunks[0]["source"] is event
        assert chunks[0]["source"]["custom"] == "data"

    def test_format_entities_returns_source(self):
        """Test format_entities returns source with original entity."""
        from think.entities import format_entities

        entity = {
            "type": "Person",
            "name": "Alice",
            "description": "Test",
            "custom": 123,
        }
        entries = [entity]
        chunks, meta = format_entities(entries)

        assert len(chunks) == 1
        assert "source" in chunks[0]
        assert chunks[0]["source"] is entity
        assert chunks[0]["source"]["custom"] == 123

    def test_format_todos_returns_source(self):
        """Test format_todos returns source with original entry."""
        from apps.todos.todo import format_todos

        entry = {"text": "Test task", "completed": False, "priority": "high"}
        entries = [entry]
        chunks, meta = format_todos(entries)

        assert len(chunks) == 1
        assert "source" in chunks[0]
        assert chunks[0]["source"] is entry
        assert chunks[0]["source"]["priority"] == "high"

    def test_format_audio_timestamp_in_milliseconds(self):
        """Test format_audio returns timestamp in milliseconds."""
        from observe.hear import format_audio

        entries = [
            {"start": "00:00:00", "text": "First"},
            {"start": "00:00:01", "text": "Second"},
        ]
        chunks, meta = format_audio(entries)

        # Without path context, base_timestamp is 0, so offsets are in ms
        assert chunks[0]["timestamp"] == 0
        assert chunks[1]["timestamp"] == 1000  # 1 second = 1000ms

    def test_format_screen_timestamp_in_milliseconds(self):
        """Test format_screen returns timestamp in milliseconds."""
        from observe.screen import format_screen

        entries = [
            {"timestamp": 0, "analysis": {}},
            {"timestamp": 1, "analysis": {}},
        ]
        chunks, meta = format_screen(entries)

        # Without path context, base_timestamp_ms is 0, so offsets are in ms
        assert chunks[0]["timestamp"] == 0
        assert chunks[1]["timestamp"] == 1000  # 1 second = 1000ms


class TestFormatLogs:
    """Tests for the action logs formatter."""

    def test_get_formatter_logs(self):
        """Test pattern matching for logs/*.jsonl."""
        from think.formatters import get_formatter

        formatter = get_formatter("facets/work/logs/20240101.jsonl")
        assert formatter is not None
        assert formatter.__name__ == "format_logs"

    def test_format_logs_basic(self):
        """Test basic action log formatting."""
        from think.facets import format_logs

        entries = [
            {
                "timestamp": "2025-12-16T07:33:05.135587+00:00",
                "source": "tool",
                "actor": "todos:todo",
                "action": "todo_add",
                "params": {"line_number": 1, "text": "Test task"},
            }
        ]

        chunks, meta = format_logs(entries)

        assert len(chunks) == 1
        assert "Todo Add by todos:todo" in chunks[0]["markdown"]
        assert "**Source:** tool" in chunks[0]["markdown"]
        assert "**Time:** 07:33:05" in chunks[0]["markdown"]
        assert "**Parameters:**" in chunks[0]["markdown"]
        assert "- line_number: 1" in chunks[0]["markdown"]
        assert "- text: Test task" in chunks[0]["markdown"]

    def test_format_logs_with_agent_id(self):
        """Test that agent_id renders as a link."""
        from think.facets import format_logs

        entries = [
            {
                "timestamp": "2025-12-16T07:33:05.135587+00:00",
                "source": "tool",
                "actor": "mcp",
                "action": "entity_attach",
                "params": {"type": "Person", "name": "Alice"},
                "agent_id": "1765870373972",
            }
        ]

        chunks, meta = format_logs(entries)

        assert len(chunks) == 1
        assert (
            "**Agent:** [1765870373972](/app/agents/1765870373972)"
            in chunks[0]["markdown"]
        )

    def test_format_logs_missing_action(self):
        """Test that entries without action are skipped."""
        from think.facets import format_logs

        entries = [
            {
                "timestamp": "2025-12-16T07:33:05.135587+00:00",
                "source": "tool",
                "actor": "mcp",
                "action": "todo_add",
                "params": {},
            },
            {
                "timestamp": "2025-12-16T07:34:00.000000+00:00",
                "source": "tool",
                "actor": "mcp",
                # Missing action
                "params": {},
            },
        ]

        chunks, meta = format_logs(entries)

        assert len(chunks) == 1
        assert "error" in meta
        assert "Skipped 1 entries" in meta["error"]
        assert "action" in meta["error"]

    def test_format_logs_returns_indexer(self):
        """Test format_logs returns indexer with topic 'action'."""
        from think.facets import format_logs

        entries = [
            {
                "timestamp": "2025-12-16T07:33:05.135587+00:00",
                "source": "tool",
                "actor": "mcp",
                "action": "todo_add",
                "params": {},
            }
        ]

        chunks, meta = format_logs(entries)

        assert "indexer" in meta
        assert meta["indexer"]["topic"] == "action"

    def test_format_logs_header_with_path(self):
        """Test that header includes facet name and day from path."""
        from think.facets import format_logs

        entries = [
            {
                "timestamp": "2025-12-16T07:33:05.135587+00:00",
                "source": "app",
                "actor": "todos",
                "action": "todo_complete",
                "params": {},
            }
        ]
        context = {"file_path": "/journal/facets/work/logs/20251216.jsonl"}

        chunks, meta = format_logs(entries, context)

        assert "header" in meta
        assert "Action Log: work" in meta["header"]
        assert "2025-12-16" in meta["header"]

    def test_format_logs_returns_source(self):
        """Test format_logs returns source with original entry."""
        from think.facets import format_logs

        entry = {
            "timestamp": "2025-12-16T07:33:05.135587+00:00",
            "source": "tool",
            "actor": "mcp",
            "action": "todo_add",
            "params": {"text": "Test"},
            "extra_field": "custom_value",
        }
        entries = [entry]

        chunks, meta = format_logs(entries)

        assert len(chunks) == 1
        assert "source" in chunks[0]
        assert chunks[0]["source"] is entry
        assert chunks[0]["source"]["extra_field"] == "custom_value"

    def test_format_logs_timestamp_parsing(self):
        """Test that ISO timestamps are converted to unix ms."""
        from think.facets import format_logs

        entries = [
            {
                "timestamp": "2025-12-16T07:33:05.135587+00:00",
                "source": "tool",
                "actor": "mcp",
                "action": "todo_add",
                "params": {},
            },
            {
                "timestamp": "2025-12-16T07:34:00.000000+00:00",
                "source": "tool",
                "actor": "mcp",
                "action": "todo_done",
                "params": {},
            },
        ]

        chunks, meta = format_logs(entries)

        assert len(chunks) == 2
        # Second entry should have higher timestamp
        assert chunks[1]["timestamp"] > chunks[0]["timestamp"]
        # First timestamp should be approximately 1734336785135 (for 2025-12-16T07:33:05)
        assert chunks[0]["timestamp"] > 1700000000000

    def test_format_logs_truncates_long_params(self):
        """Test that long param values are truncated."""
        from think.facets import format_logs

        long_text = "x" * 200

        entries = [
            {
                "timestamp": "2025-12-16T07:33:05.135587+00:00",
                "source": "tool",
                "actor": "mcp",
                "action": "todo_add",
                "params": {"text": long_text},
            }
        ]

        chunks, meta = format_logs(entries)

        assert len(chunks) == 1
        # Should truncate to 100 chars + "..."
        assert ("x" * 100 + "...") in chunks[0]["markdown"]
        assert ("x" * 150) not in chunks[0]["markdown"]

    def test_format_logs_action_display_formatting(self):
        """Test that action names are formatted nicely."""
        from think.facets import format_logs

        entries = [
            {
                "timestamp": "2025-12-16T07:33:05.135587+00:00",
                "source": "tool",
                "actor": "mcp",
                "action": "entity_update_description",
                "params": {},
            }
        ]

        chunks, meta = format_logs(entries)

        assert len(chunks) == 1
        # "entity_update_description" should become "Entity Update Description"
        assert "Entity Update Description by mcp" in chunks[0]["markdown"]

    def test_get_formatter_journal_level_logs(self):
        """Test pattern matching for config/actions/*.jsonl."""
        from think.formatters import get_formatter

        formatter = get_formatter("config/actions/20240101.jsonl")
        assert formatter is not None
        assert formatter.__name__ == "format_logs"

    def test_format_logs_journal_level_header(self):
        """Test that journal-level logs have appropriate header."""
        from think.facets import format_logs

        entries = [
            {
                "timestamp": "2025-12-16T07:33:05.135587+00:00",
                "source": "app",
                "actor": "settings",
                "action": "identity_update",
                "params": {"name": "Test User"},
            }
        ]
        context = {"file_path": "/journal/config/actions/20251216.jsonl"}

        chunks, meta = format_logs(entries, context)

        assert "header" in meta
        assert "Journal Action Log" in meta["header"]
        assert "2025-12-16" in meta["header"]
        # Should NOT contain a facet name
        assert ":" not in meta["header"] or "Journal" in meta["header"]
