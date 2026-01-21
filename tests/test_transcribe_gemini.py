# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for the Gemini STT backend."""

from observe.transcribe.gemini import (
    _normalize_segments,
    _parse_speaker,
    _parse_timestamp,
    get_model_info,
)


class TestParseTimestamp:
    """Tests for _parse_timestamp function."""

    def test_mm_ss_format(self):
        """Standard MM:SS format."""
        assert _parse_timestamp("1:23") == 83.0
        assert _parse_timestamp("0:05") == 5.0
        assert _parse_timestamp("10:30") == 630.0

    def test_hh_mm_ss_format(self):
        """HH:MM:SS format."""
        assert _parse_timestamp("1:05:30") == 3930.0
        assert _parse_timestamp("0:10:00") == 600.0
        assert _parse_timestamp("2:00:00") == 7200.0

    def test_just_seconds(self):
        """Just seconds (no colons)."""
        assert _parse_timestamp("5") == 5.0
        assert _parse_timestamp("0") == 0.0
        assert _parse_timestamp("123") == 123.0

    def test_zero_timestamp(self):
        """Zero timestamps."""
        assert _parse_timestamp("0:00") == 0.0
        assert _parse_timestamp("0:00:00") == 0.0
        assert _parse_timestamp("0") == 0.0

    def test_whitespace_handling(self):
        """Whitespace should be stripped."""
        assert _parse_timestamp(" 1:23 ") == 83.0
        assert _parse_timestamp("\t0:05\n") == 5.0

    def test_fractional_seconds(self):
        """Fractional seconds."""
        assert _parse_timestamp("1:23.5") == 83.5
        assert _parse_timestamp("0:05.25") == 5.25

    def test_invalid_returns_none(self):
        """Invalid timestamps return None."""
        assert _parse_timestamp("") is None
        assert _parse_timestamp(None) is None
        assert _parse_timestamp("invalid") is None
        assert _parse_timestamp("abc:def") is None
        assert _parse_timestamp("1:2:3:4") is None  # Too many parts

    def test_negative_clamped_to_zero(self):
        """Negative values are clamped to 0."""
        # This is an edge case - negative minutes/seconds shouldn't happen
        # but if somehow parsed, we clamp to 0
        assert _parse_timestamp("-5") == 0.0


class TestParseSpeaker:
    """Tests for _parse_speaker function."""

    def test_speaker_n_format(self):
        """Speaker N format."""
        assert _parse_speaker("Speaker 1") == 1
        assert _parse_speaker("Speaker 2") == 2
        assert _parse_speaker("speaker 3") == 3  # Case insensitive

    def test_just_number(self):
        """Just a number."""
        assert _parse_speaker("1") == 1
        assert _parse_speaker("2") == 2

    def test_integer_input(self):
        """Integer input."""
        assert _parse_speaker(1) == 1
        assert _parse_speaker(2) == 2

    def test_zero_and_negative_invalid(self):
        """Zero and negative speaker IDs are invalid."""
        assert _parse_speaker(0) is None
        assert _parse_speaker(-1) is None
        assert _parse_speaker("0") is None

    def test_none_returns_none(self):
        """None input returns None."""
        assert _parse_speaker(None) is None

    def test_unparseable_returns_none(self):
        """Unparseable strings return None."""
        assert _parse_speaker("John") is None
        assert _parse_speaker("unknown") is None
        assert _parse_speaker("") is None


class TestNormalizeSegments:
    """Tests for _normalize_segments function."""

    def test_basic_normalization(self):
        """Basic segment normalization."""
        segments = [
            {
                "start": "0:05",
                "end": "0:12",
                "speaker": "Speaker 1",
                "text": "Hello there",
                "emotion": "friendly",
            }
        ]

        statements, invalid_count = _normalize_segments(segments, 60.0)

        assert len(statements) == 1
        assert invalid_count == 0
        stmt = statements[0]
        assert stmt["id"] == 1
        assert stmt["start"] == 5.0
        assert stmt["end"] == 12.0
        assert stmt["text"] == "Hello there"
        assert stmt["speaker"] == 1
        assert stmt["emotion"] == "friendly"
        assert stmt["words"] is None

    def test_multiple_segments(self):
        """Multiple segments get sequential IDs."""
        segments = [
            {"start": "0:00", "end": "0:10", "text": "First"},
            {"start": "0:15", "end": "0:25", "text": "Second"},
            {"start": "0:30", "end": "0:40", "text": "Third"},
        ]

        statements, invalid_count = _normalize_segments(segments, 60.0)

        assert len(statements) == 3
        assert invalid_count == 0
        assert [s["id"] for s in statements] == [1, 2, 3]

    def test_clamps_to_audio_duration(self):
        """Timestamps beyond audio duration are clamped."""
        segments = [
            {"start": "0:00", "end": "2:00", "text": "Goes past end"}  # 120s > 60s
        ]

        statements, invalid_count = _normalize_segments(segments, 60.0)

        assert len(statements) == 1
        assert statements[0]["end"] == 60.0  # Clamped to duration

    def test_invalid_timestamps_counted(self):
        """Invalid timestamps are counted but segment is still included."""
        segments = [
            {"start": "invalid", "end": "0:10", "text": "Bad start"},
            {"start": "0:00", "end": "bad", "text": "Bad end"},
            {"start": "0:00", "end": "0:10", "text": "Good one"},
        ]

        statements, invalid_count = _normalize_segments(segments, 60.0)

        # All segments included, but 2 have invalid timestamps
        assert len(statements) == 3
        assert invalid_count == 2
        assert statements[0]["start"] is None
        assert statements[1]["end"] is None
        assert statements[2]["start"] == 0.0
        assert statements[2]["end"] == 10.0

    def test_missing_optional_fields(self):
        """Missing optional fields are handled gracefully."""
        segments = [{"start": "0:00", "end": "0:10", "text": "Minimal"}]

        statements, invalid_count = _normalize_segments(segments, 60.0)

        assert len(statements) == 1
        assert "speaker" not in statements[0]
        assert "emotion" not in statements[0]

    def test_empty_segments(self):
        """Empty segments list."""
        statements, invalid_count = _normalize_segments([], 60.0)
        assert statements == []
        assert invalid_count == 0

    def test_whitespace_text_stripped(self):
        """Text is stripped of whitespace."""
        segments = [{"start": "0:00", "end": "0:10", "text": "  Hello  "}]

        statements, invalid_count = _normalize_segments(segments, 60.0)

        assert statements[0]["text"] == "Hello"

    def test_empty_text_dropped(self):
        """Segments with empty text are dropped."""
        segments = [
            {"start": "0:00", "end": "0:10", "text": "First"},
            {"start": "0:15", "end": "0:20", "text": ""},
            {"start": "0:25", "end": "0:30", "text": "   "},  # whitespace only
            {"start": "0:35", "end": "0:40", "text": "Last"},
        ]

        statements, invalid_count = _normalize_segments(segments, 60.0)

        # Only non-empty segments kept, IDs renumbered
        assert len(statements) == 2
        assert statements[0]["text"] == "First"
        assert statements[0]["id"] == 1
        assert statements[1]["text"] == "Last"
        assert statements[1]["id"] == 2

    def test_empty_emotion_not_included(self):
        """Empty emotion is not included in statement."""
        segments = [{"start": "0:00", "end": "0:10", "text": "Test", "emotion": ""}]

        statements, invalid_count = _normalize_segments(segments, 60.0)

        assert "emotion" not in statements[0]


class TestGetModelInfo:
    """Tests for get_model_info function."""

    def test_returns_expected_format(self):
        """Returns expected metadata format."""
        info = get_model_info({})

        assert info["model"] == "gemini"
        assert info["device"] == "cloud"
        assert info["compute_type"] == "api"


class TestBackendRegistry:
    """Tests for backend registry integration."""

    def test_gemini_registered(self):
        """Gemini backend is registered."""
        from observe.transcribe import BACKEND_REGISTRY

        assert "gemini" in BACKEND_REGISTRY

    def test_get_backend(self):
        """Can get Gemini backend module."""
        from observe.transcribe import get_backend

        backend = get_backend("gemini")
        assert hasattr(backend, "transcribe")
        assert hasattr(backend, "get_model_info")
