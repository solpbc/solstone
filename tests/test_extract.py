# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for observe/extract.py frame selection logic."""

from unittest.mock import patch

from observe.extract import (
    DEFAULT_MAX_EXTRACTIONS,
    _build_extraction_guidance,
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


def test_more_than_max_returns_around_max():
    """Test that more than max frames returns approximately max count."""
    frames = _make_frames(30)
    result = select_frames_for_extraction(frames, max_extractions=5)
    # May be max or max+1 if first frame wasn't in random selection
    assert 5 <= len(result) <= 6


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
    # First frame always included, plus possibly one random
    assert 1 in result
    assert 1 <= len(result) <= 2


def test_non_sequential_frame_ids():
    """Test with non-sequential frame IDs."""
    frames = [
        {"frame_id": 5, "timestamp": 1.0, "analysis": {}},
        {"frame_id": 10, "timestamp": 2.0, "analysis": {}},
        {"frame_id": 15, "timestamp": 3.0, "analysis": {}},
    ]
    result = select_frames_for_extraction(frames, max_extractions=10)
    assert result == [5, 10, 15]


# --- Tests for _build_extraction_guidance ---


def test_build_extraction_guidance_empty():
    """Test extraction guidance with empty categories."""
    result = _build_extraction_guidance({})
    assert result == "No category-specific rules."


def test_build_extraction_guidance_no_extraction_fields():
    """Test extraction guidance when no categories have extraction field."""
    categories = {
        "code": {"description": "Code editors"},
        "browsing": {"description": "Web browsing"},
    }
    result = _build_extraction_guidance(categories)
    assert result == "No category-specific rules."


def test_build_extraction_guidance_with_extraction():
    """Test extraction guidance with valid extraction fields."""
    categories = {
        "code": {
            "description": "Code editors",
            "extraction": "Extract on file changes",
        },
        "browsing": {
            "description": "Web browsing",
            "extraction": "Extract on site changes",
        },
        "meeting": {"description": "Video calls"},  # No extraction field
    }
    result = _build_extraction_guidance(categories)
    assert "- browsing: Extract on site changes" in result
    assert "- code: Extract on file changes" in result
    assert "meeting" not in result


def test_build_extraction_guidance_sorted():
    """Test extraction guidance is sorted alphabetically."""
    categories = {
        "zebra": {"extraction": "Z rule"},
        "apple": {"extraction": "A rule"},
        "mango": {"extraction": "M rule"},
    }
    result = _build_extraction_guidance(categories)
    lines = result.split("\n")
    assert lines[0].startswith("- apple:")
    assert lines[1].startswith("- mango:")
    assert lines[2].startswith("- zebra:")


# --- Tests for AI selection with mocked generate ---


def test_ai_selection_with_categories():
    """Test that AI selection is used when categories are provided."""
    frames = _make_frames(10)
    categories = {"code": {"description": "Code editors"}}

    # Mock generate to return specific frame IDs
    with patch("muse.models.generate") as mock_generate:
        mock_generate.return_value = "[1, 3, 5, 7]"
        result = select_frames_for_extraction(
            frames, max_extractions=5, categories=categories
        )

    assert result == [1, 3, 5, 7]
    mock_generate.assert_called_once()


def test_ai_selection_filters_invalid_ids():
    """Test that AI selection filters out invalid frame IDs."""
    frames = _make_frames(5)  # IDs 1-5
    categories = {"code": {"description": "Code editors"}}

    # Mock generate to return some invalid IDs
    with patch("muse.models.generate") as mock_generate:
        mock_generate.return_value = "[1, 3, 99, 100, 5]"  # 99, 100 are invalid
        result = select_frames_for_extraction(
            frames, max_extractions=10, categories=categories
        )

    assert result == [1, 3, 5]
    assert 99 not in result
    assert 100 not in result


def test_ai_selection_hard_cap():
    """Test that AI selection caps at 2x max_extractions."""
    frames = _make_frames(50)
    categories = {"code": {"description": "Code editors"}}

    # Mock generate to return way too many IDs
    many_ids = list(range(1, 51))  # 50 IDs
    with patch("muse.models.generate") as mock_generate:
        mock_generate.return_value = str(many_ids)
        result = select_frames_for_extraction(
            frames, max_extractions=5, categories=categories
        )

    # Hard cap is 2 * 5 = 10, plus first frame guarantee
    assert len(result) <= 11


def test_ai_selection_fallback_on_error():
    """Test that fallback is used when AI selection fails."""
    frames = _make_frames(10)
    categories = {"code": {"description": "Code editors"}}

    # Mock generate to raise an exception
    with patch("muse.models.generate") as mock_generate:
        mock_generate.side_effect = Exception("API error")
        result = select_frames_for_extraction(
            frames, max_extractions=5, categories=categories
        )

    # Should still get a valid result from fallback
    assert len(result) >= 5
    assert 1 in result  # First frame always included


def test_ai_selection_fallback_on_invalid_json():
    """Test that fallback is used when AI returns invalid JSON."""
    frames = _make_frames(10)
    categories = {"code": {"description": "Code editors"}}

    with patch("muse.models.generate") as mock_generate:
        mock_generate.return_value = "not valid json"
        result = select_frames_for_extraction(
            frames, max_extractions=5, categories=categories
        )

    # Should still get a valid result from fallback
    assert len(result) >= 5
    assert 1 in result


def test_no_ai_selection_without_categories():
    """Test that AI selection is skipped when categories is None."""
    frames = _make_frames(10)

    with patch("muse.models.generate") as mock_generate:
        result = select_frames_for_extraction(
            frames, max_extractions=5, categories=None
        )

    # generate should not be called
    mock_generate.assert_not_called()
    # Should get fallback result
    assert len(result) >= 5
    assert 1 in result
