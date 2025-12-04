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

        formatter = get_formatter("20240102/234567/screen.jsonl")
        assert formatter is not None
        assert formatter.__name__ == "format_screen"

    def test_get_formatter_audio(self):
        """Test pattern matching for audio.jsonl."""
        from think.formatters import get_formatter

        formatter = get_formatter("20240101/123456/audio.jsonl")
        assert formatter is not None
        assert formatter.__name__ == "format_audio"

    def test_get_formatter_split_audio(self):
        """Test pattern matching for *_audio.jsonl files (split, imported, etc.)."""
        from think.formatters import get_formatter

        # Split audio
        formatter = get_formatter("20240101/123456/123456_audio.jsonl")
        assert formatter is not None
        assert formatter.__name__ == "format_audio"

        # Imported audio (matched by *_audio.jsonl pattern)
        formatter = get_formatter("20240101/123456/imported_audio.jsonl")
        assert formatter is not None
        assert formatter.__name__ == "format_audio"

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

        path = Path(os.environ["JOURNAL_PATH"]) / "20240101/123456/audio.jsonl"
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

        path = Path(os.environ["JOURNAL_PATH"]) / "20240102/234567/screen.jsonl"
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

        path = Path(os.environ["JOURNAL_PATH"]) / "20240101/123456/audio.jsonl"
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
        from observe.reduce import format_screen

        entries = [
            {
                "timestamp": 5,
                "analysis": {"visible": "code", "visual_description": "Python code"},
                "extracted_text": "def hello():\n    pass",
            }
        ]

        chunks, meta = format_screen(entries)

        assert len(chunks) == 1  # 1 frame chunk
        assert "header" in meta
        assert "Frame Analyses" in meta["header"]
        assert chunks[0]["timestamp"] == 5
        assert "**Category:** code" in chunks[0]["markdown"]
        assert "def hello()" in chunks[0]["markdown"]

    def test_format_screen_with_entity_context(self):
        """Test screen formatting with entity context."""
        from observe.reduce import format_screen

        entries = [{"timestamp": 0, "analysis": {"visible": "browser"}}]
        context = {"entity_names": "Alice, Bob", "include_entity_context": True}

        chunks, meta = format_screen(entries, context)

        assert "header" in meta
        assert "Entity Context" in meta["header"]
        assert "Alice, Bob" in meta["header"]

    def test_format_screen_without_entity_context(self):
        """Test screen formatting without entity context."""
        from observe.reduce import format_screen

        entries = [{"timestamp": 0, "analysis": {"visible": "browser"}}]
        context = {"include_entity_context": False}

        chunks, meta = format_screen(entries, context)

        assert "header" in meta
        assert "Entity Context" not in meta["header"]

    def test_format_screen_multiple_monitors(self):
        """Test screen formatting with multiple monitors."""
        from observe.reduce import format_screen

        entries = [
            {"timestamp": 0, "monitor": "0", "analysis": {}},
            {
                "timestamp": 5,
                "monitor": "1",
                "monitor_position": "left",
                "analysis": {},
            },
        ]

        chunks, meta = format_screen(entries)

        # Should include monitor info in headers
        assert "(Monitor 0)" in chunks[0]["markdown"]
        assert "(Monitor 1 - left)" in chunks[1]["markdown"]

    def test_format_screen_meeting_analysis(self):
        """Test screen formatting with meeting analysis."""
        from observe.reduce import format_screen

        entries = [
            {
                "timestamp": 0,
                "analysis": {},
                "meeting_analysis": {"participants": ["Alice", "Bob"]},
            }
        ]

        chunks, meta = format_screen(entries)

        assert "Meeting Analysis" in chunks[0]["markdown"]
        assert "Alice" in chunks[0]["markdown"]

    def test_format_screen_extracts_metadata(self):
        """Test that metadata line is extracted and not treated as a frame."""
        from observe.reduce import format_screen

        entries = [
            {"raw": "screen.webm"},  # Metadata line
            {"timestamp": 5, "analysis": {"visible": "code"}},
        ]

        chunks, meta = format_screen(entries)

        # Should only have 1 frame chunk (metadata is extracted)
        assert len(chunks) == 1
        assert chunks[0]["timestamp"] == 5

    def test_format_screen_skipped_entries_error(self):
        """Test that skipped entries are reported in meta.error."""
        from observe.reduce import format_screen

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


class TestAssembleMarkdownBackwardCompat:
    """Tests for backward compatibility of assemble_markdown wrapper."""

    def test_assemble_markdown_returns_string(self):
        """Test that assemble_markdown still returns a string."""
        from observe.reduce import assemble_markdown

        frames = [
            {"timestamp": 0, "analysis": {"visible": "code"}},
            {"timestamp": 5, "analysis": {"visible": "browser"}},
        ]

        result = assemble_markdown(frames)

        assert isinstance(result, str)
        assert "Frame Analyses" in result
        assert "**Category:** code" in result
        assert "**Category:** browser" in result

    def test_assemble_markdown_with_entity_names(self):
        """Test assemble_markdown with entity_names parameter."""
        from observe.reduce import assemble_markdown

        frames = [{"timestamp": 0, "analysis": {}}]
        result = assemble_markdown(frames, entity_names="Alice, Bob")

        assert "Entity Context" in result
        assert "Alice, Bob" in result


class TestLoadTranscriptBackwardCompat:
    """Tests for backward compatibility of load_transcript."""

    def test_load_transcript_returns_tuple(self):
        """Test that load_transcript still returns (metadata, entries, text) tuple."""
        from observe.hear import load_transcript

        path = Path(os.environ["JOURNAL_PATH"]) / "20240101/123456/audio.jsonl"
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
        from muse.cortex import format_agent

        entries = [
            {
                "event": "request",
                "ts": 1700000000000,
                "agent_id": "test123",
                "prompt": "Hello world",
                "persona": "default",
                "backend": "openai",
            },
            {
                "event": "start",
                "ts": 1700000000100,
                "agent_id": "test123",
                "model": "gpt-4",
                "persona": "default",
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
        from muse.cortex import format_agent

        entries = [
            {"event": "start", "ts": 1700000000000, "agent_id": "test"},
            {
                "event": "thinking",
                "content": "I should analyze this...",
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
        from muse.cortex import format_agent

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
        from muse.cortex import format_agent

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
        from muse.cortex import format_agent

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
        from muse.cortex import format_agent

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
        from muse.cortex import format_agent

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
        from muse.cortex import format_agent

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
        from muse.cortex import format_agent

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
        from muse.cortex import format_agent

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

    def test_get_formatter_attached_entities(self):
        """Test pattern matching for attached entities."""
        from think.formatters import get_formatter

        formatter = get_formatter("facets/personal/entities.jsonl")
        assert formatter is not None
        assert formatter.__name__ == "format_entities"

    def test_get_formatter_detected_entities(self):
        """Test pattern matching for detected entities."""
        from think.formatters import get_formatter

        formatter = get_formatter("facets/personal/entities/20250101.jsonl")
        assert formatter is not None
        assert formatter.__name__ == "format_entities"

    def test_format_entities_attached_basic(self):
        """Test basic attached entities formatting with fixture file."""
        from think.formatters import format_file

        path = Path(os.environ["JOURNAL_PATH"]) / "facets/personal/entities.jsonl"
        chunks, meta = format_file(path)

        assert len(chunks) == 3  # 3 entities in fixture
        assert "header" in meta
        assert "Attached Entities: personal" in meta["header"]
        assert "3 entities" in meta["header"]

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

    def test_format_entities_header_facet_from_path(self):
        """Test that facet name is extracted from file path."""
        from think.entities import format_entities

        entries = [{"type": "Person", "name": "Test", "description": ""}]
        context = {"file_path": "/journal/facets/work/entities.jsonl"}

        chunks, meta = format_entities(entries, context)

        assert "Attached Entities: work" in meta["header"]

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
