# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for observe/extract.py frame selection logic."""

from observe.extract import (
    DEFAULT_MAX_EXTRACTIONS,
    _fallback_select_frames,
    select_frames_for_extraction,
)


def _make_frames(count: int, start_id: int = 1) -> list[dict]:
    """Create test frame data."""
    return [
        {"frame_id": i, "timestamp": float(i), "analysis": {"primary": "code"}}
        for i in range(start_id, start_id + count)
    ]


def test_default_max_extractions():
    """Test default max extractions value."""
    assert DEFAULT_MAX_EXTRACTIONS == 20


def test_empty_list():
    """Test empty input returns empty output."""
    result = select_frames_for_extraction([])
    assert result == []


def test_single_frame():
    """Test single frame is always selected."""
    frames = _make_frames(1)
    result = select_frames_for_extraction(frames, max_extractions=5)
    assert result == [1]


def test_fewer_than_max_returns_all():
    """Test that frames under the limit are all returned."""
    frames = _make_frames(5)
    result = select_frames_for_extraction(frames, max_extractions=10)
    assert result == [1, 2, 3, 4, 5]


def test_exactly_max_returns_all():
    """Test that exactly max frames returns all."""
    frames = _make_frames(10)
    result = select_frames_for_extraction(frames, max_extractions=10)
    assert result == list(range(1, 11))


def test_more_than_max_returns_max():
    """Test that more than max frames returns max count."""
    frames = _make_frames(30)
    result = select_frames_for_extraction(frames, max_extractions=5)
    assert len(result) == 5


def test_first_frame_always_included():
    """Test that first frame is always in selection."""
    frames = _make_frames(100)
    # Run multiple times to account for randomness
    for _ in range(10):
        result = select_frames_for_extraction(frames, max_extractions=10)
        assert 1 in result, "First frame must always be included"


def test_results_sorted():
    """Test that results are sorted by frame_id."""
    frames = _make_frames(50)
    result = select_frames_for_extraction(frames, max_extractions=10)
    assert result == sorted(result)


def test_fallback_deterministic_for_small_lists():
    """Test fallback is deterministic for lists under max."""
    frames = _make_frames(5)
    result1 = _fallback_select_frames(frames, max_extractions=10)
    result2 = _fallback_select_frames(frames, max_extractions=10)
    assert result1 == result2 == [1, 2, 3, 4, 5]


def test_max_extractions_of_one():
    """Test edge case of max_extractions=1."""
    frames = _make_frames(10)
    result = select_frames_for_extraction(frames, max_extractions=1)
    assert result == [1]  # Only first frame


def test_non_sequential_frame_ids():
    """Test with non-sequential frame IDs."""
    frames = [
        {"frame_id": 5, "timestamp": 1.0, "analysis": {}},
        {"frame_id": 10, "timestamp": 2.0, "analysis": {}},
        {"frame_id": 15, "timestamp": 3.0, "analysis": {}},
    ]
    result = select_frames_for_extraction(frames, max_extractions=10)
    assert result == [5, 10, 15]
