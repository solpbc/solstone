# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Frame extraction selection for vision analysis pipeline.

Determines which categorized frames should receive detailed content extraction.
Provides AI-based selection (stub for future implementation) with random fallback.

The first qualified frame is always included regardless of selection results.
"""

from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

logger = logging.getLogger(__name__)

# Default maximum frames to extract content from
DEFAULT_MAX_EXTRACTIONS = 20


def select_frames_for_extraction(
    categorized_frames: list[dict[str, Any]],
    max_extractions: int = DEFAULT_MAX_EXTRACTIONS,
) -> list[int]:
    """Select which frames should receive detailed content extraction.

    The first qualified frame is always included, even if it exceeds max_extractions
    by one. This ensures we always have context from the start of the recording.

    Parameters
    ----------
    categorized_frames : list[dict]
        List of categorized frame data, each containing:
        - frame_id: int
        - timestamp: float
        - analysis: dict with primary, secondary, overlap, visual_description
    max_extractions : int
        Maximum number of frames to extract (default: 20).
        First frame may cause this to be exceeded by 1.

    Returns
    -------
    list[int]
        Frame IDs to extract, sorted in frame order.
    """
    if not categorized_frames:
        return []

    # Try AI selection first
    try:
        selected = _ai_select_frames(categorized_frames, max_extractions)
    except NotImplementedError:
        # Expected until AI selection is implemented
        selected = _fallback_select_frames(categorized_frames, max_extractions)
    except Exception as e:
        logger.warning(f"AI frame selection failed, using fallback: {e}")
        selected = _fallback_select_frames(categorized_frames, max_extractions)

    # Ensure first frame is always included
    first_frame_id = categorized_frames[0]["frame_id"]
    if first_frame_id not in selected:
        selected = [first_frame_id] + selected

    # Return sorted by frame_id for deterministic output order
    return sorted(selected)


def _ai_select_frames(
    categorized_frames: list[dict[str, Any]],
    max_extractions: int,
) -> list[int]:
    """AI-based frame selection.

    Analyzes all categorization results to determine which frames contain
    the most valuable content for detailed extraction.

    Parameters
    ----------
    categorized_frames : list[dict]
        List of categorized frame data.
    max_extractions : int
        Maximum number of frames to select.

    Returns
    -------
    list[int]
        Selected frame IDs.

    Raises
    ------
    NotImplementedError
        AI selection not yet implemented.
    """
    # TODO: Implement AI-based selection using Gemini
    # Input: List of {frame_id, timestamp, analysis} for all frames
    # Output: List of frame_ids ranked by extraction value
    raise NotImplementedError("AI frame selection not yet implemented")


def _fallback_select_frames(
    categorized_frames: list[dict[str, Any]],
    max_extractions: int,
) -> list[int]:
    """Fallback frame selection when AI selection is unavailable.

    If total frames <= max_extractions: returns all frames.
    Otherwise: returns random sample of max_extractions frames.

    Parameters
    ----------
    categorized_frames : list[dict]
        List of categorized frame data.
    max_extractions : int
        Maximum number of frames to select.

    Returns
    -------
    list[int]
        Selected frame IDs.
    """
    if not categorized_frames:
        return []

    all_ids = [f["frame_id"] for f in categorized_frames]

    if len(all_ids) <= max_extractions:
        return all_ids

    return random.sample(all_ids, max_extractions)


__all__ = [
    "DEFAULT_MAX_EXTRACTIONS",
    "select_frames_for_extraction",
]
