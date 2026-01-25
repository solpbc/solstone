# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

import importlib
import json

import pytest


def test_number_lines_and_segments():
    mod = importlib.import_module("think.detect_transcript")
    numbered, lines = mod.number_lines("a\nb\nc\nd")
    assert numbered == "1: a\n2: b\n3: c\n4: d"
    assert lines == ["a", "b", "c", "d"]

    # Test parse_segment_boundaries with new format
    boundaries_json = json.dumps(
        [
            {"start_at": "12:00:00", "line": 2},
            {"start_at": "12:05:00", "line": 4},
        ]
    )
    boundaries = mod.parse_segment_boundaries(boundaries_json, len(lines))
    assert boundaries == [
        {"start_at": "12:00:00", "line": 2},
        {"start_at": "12:05:00", "line": 4},
    ]

    # Test segments_from_boundaries with new format
    segments = mod.segments_from_boundaries(lines, boundaries)
    assert segments == [("12:00:00", "b\nc"), ("12:05:00", "d")]


def test_parse_segment_boundaries_invalid():
    mod = importlib.import_module("think.detect_transcript")

    # Invalid JSON
    with pytest.raises(ValueError):
        mod.parse_segment_boundaries("not json", 3)

    # Empty list
    with pytest.raises(ValueError):
        mod.parse_segment_boundaries("[]", 3)

    # Not an object
    with pytest.raises(ValueError):
        mod.parse_segment_boundaries("[1]", 3)

    # Missing fields
    with pytest.raises(ValueError):
        mod.parse_segment_boundaries('[{"line": 1}]', 3)
    with pytest.raises(ValueError):
        mod.parse_segment_boundaries('[{"start_at": "12:00:00"}]', 3)

    # Invalid line number (0)
    with pytest.raises(ValueError):
        mod.parse_segment_boundaries('[{"start_at": "12:00:00", "line": 0}]', 3)

    # Line number exceeds max
    with pytest.raises(ValueError):
        mod.parse_segment_boundaries('[{"start_at": "12:00:00", "line": 5}]', 3)

    # Non-increasing line numbers
    with pytest.raises(ValueError):
        mod.parse_segment_boundaries(
            '[{"start_at": "12:00:00", "line": 2}, {"start_at": "12:05:00", "line": 1}]',
            3,
        )


def test_detect_transcript_segment(monkeypatch):
    mod = importlib.import_module("think.detect_transcript")

    # Mock returns new format with start_at and line
    def mock_generate(**kwargs):
        return (
            '[{"start_at": "14:30:00", "line": 1}, {"start_at": "14:35:00", "line": 3}]'
        )

    monkeypatch.setattr("think.models.generate", mock_generate)

    # Now requires start_time argument
    result = mod.detect_transcript_segment("a\nb\nc\nd", "14:30:00")

    # Returns list of (start_at, text) tuples
    assert result == [("14:30:00", "a\nb"), ("14:35:00", "c\nd")]
