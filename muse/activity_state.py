# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Pre-hook for activity_state generator.

Builds context for activity detection by:
1. Loading the facet's configured activities
2. Finding and loading previous segment's activity state
3. Formatting both as context for the prompt
"""

import json
import logging
import os
import re
from datetime import datetime, timedelta

from think.utils import day_path, segment_parse

logger = logging.getLogger(__name__)

# Default timeout for activity continuity (1 hour in seconds)
DEFAULT_TIMEOUT_SECONDS = 3600


def _extract_facet_from_output_path(output_path: str) -> str | None:
    """Extract facet name from output path.

    Output paths for faceted generators follow the pattern:
    {day}/{segment}/activity_state_{facet}.json

    Returns None if facet cannot be extracted.
    """
    if not output_path:
        return None

    # Match pattern like activity_state_work.json
    filename = os.path.basename(output_path)
    match = re.match(r"activity_state_([^.]+)\.json$", filename)
    if match:
        return match.group(1)
    return None


def _parse_segment_time(segment: str) -> datetime | None:
    """Parse segment key to datetime (time only, date is placeholder)."""
    start, _ = segment_parse(segment)
    if start is None:
        return None
    # Convert time to datetime with placeholder date
    return datetime(2000, 1, 1, start.hour, start.minute, start.second)


def _get_segment_end_time(segment: str) -> datetime | None:
    """Get the end time of a segment."""
    _, end = segment_parse(segment)
    if end is None:
        return None
    return datetime(2000, 1, 1, end.hour, end.minute, end.second)


def find_previous_segment(day: str, current_segment: str) -> str | None:
    """Find the segment immediately before the current one.

    Scans the day directory for segment folders, sorts by time,
    and returns the one before current_segment.

    Returns None if current is the first segment or no segments found.
    """
    day_dir = day_path(day)
    if not day_dir.is_dir():
        return None

    # Collect valid segment folders
    segments = []
    for entry in os.listdir(day_dir):
        entry_path = day_dir / entry
        if not entry_path.is_dir():
            continue
        # Match segment pattern: HHMMSS_LEN or HHMMSS_LEN_suffix
        if re.match(r"^\d{6}_\d+", entry):
            segments.append(entry)

    if not segments:
        return None

    # Sort by time (segment keys sort lexicographically by time)
    segments.sort()

    # Find current segment's position
    try:
        current_idx = segments.index(current_segment)
    except ValueError:
        # Current segment not in list - might be new
        # Find where it would be inserted
        for i, seg in enumerate(segments):
            if seg > current_segment:
                current_idx = i
                break
        else:
            current_idx = len(segments)

    # Return previous if exists
    if current_idx > 0:
        return segments[current_idx - 1]
    return None


def load_previous_state(
    day: str, segment: str, facet: str
) -> tuple[dict | None, str | None]:
    """Load activity state from a previous segment.

    Returns tuple of (state, segment_key) where state is the parsed JSON
    or None if not found/invalid.
    """
    state_path = day_path(day) / segment / f"activity_state_{facet}.json"
    if not state_path.exists():
        return None, None

    try:
        content = state_path.read_text().strip()
        if not content:
            return None, segment

        data = json.loads(content)
        return data, segment
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to load previous state from %s: %s", state_path, e)
        return None, segment


def check_timeout(
    current_segment: str,
    previous_segment: str,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> bool:
    """Check if gap between segments exceeds timeout threshold.

    Returns True if timed out (gap too large), False if within threshold.
    """
    current_start = _parse_segment_time(current_segment)
    previous_end = _get_segment_end_time(previous_segment)

    if current_start is None or previous_end is None:
        # Can't determine times, assume no timeout
        return False

    gap = current_start - previous_end
    # Handle day wraparound (negative gap means previous was yesterday)
    if gap < timedelta(0):
        return True  # Cross-day, always timeout

    return gap.total_seconds() > timeout_seconds


def format_activities_context(facet: str) -> str:
    """Format the facet's configured activities as context.

    Returns markdown-formatted list of activities with descriptions.
    """
    from think.activities import get_facet_activities

    activities = get_facet_activities(facet)
    if not activities:
        return "No activities configured for this facet."

    lines = ["## Facet Activities (only detect these)", ""]
    for activity in activities:
        activity_id = activity.get("id", "")
        name = activity.get("name", activity_id)
        description = activity.get("description", "")
        priority = activity.get("priority", "normal")

        if priority == "high":
            priority_note = " [high priority]"
        elif priority == "low":
            priority_note = " [low priority]"
        else:
            priority_note = ""

        if description:
            lines.append(f"- **{activity_id}** ({name}){priority_note}: {description}")
        else:
            lines.append(f"- **{activity_id}** ({name}){priority_note}")

    return "\n".join(lines)


def format_previous_state(
    state: dict | None,
    segment: str | None,
    current_segment: str,
    timed_out: bool,
) -> str:
    """Format previous state as context for the prompt.

    Args:
        state: Previous segment's activity state dict
        segment: Previous segment key
        current_segment: Current segment key
        timed_out: Whether the gap exceeded timeout threshold
    """
    if state is None or segment is None:
        return "## Previous State\n\nNo previous segment state available (first segment or fresh start)."

    if timed_out:
        return "## Previous State\n\nPrevious segment too long ago (>1 hour gap). Starting fresh."

    # Calculate time gap for context
    current_start = _parse_segment_time(current_segment)
    previous_end = _get_segment_end_time(segment)

    if current_start and previous_end:
        gap = current_start - previous_end
        gap_minutes = int(gap.total_seconds() / 60)
        time_note = f" ({gap_minutes} minutes ago)"
    else:
        time_note = ""

    lines = [f"## Previous State (from {segment}){time_note}", ""]

    active = state.get("active", [])
    ended = state.get("ended", [])

    if active:
        lines.append("**Active activities (may be continuing):**")
        for item in active:
            activity_id = item.get("activity", "")
            since = item.get("since", "")
            description = item.get("description", "")
            level = item.get("level", "")

            parts = [f"- {activity_id}"]
            if since:
                parts.append(f"(since {since})")
            if level:
                parts.append(f"[{level}]")
            if description:
                parts.append(f": {description}")
            lines.append(" ".join(parts))
        lines.append("")

    if ended:
        lines.append("**Recently ended:**")
        for item in ended:
            activity_id = item.get("activity", "")
            description = item.get("description", "")
            lines.append(
                f"- {activity_id}: {description}" if description else f"- {activity_id}"
            )
        lines.append("")

    if not active and not ended:
        lines.append("No activities were detected in the previous segment.")

    return "\n".join(lines)


def pre_process(context: dict) -> dict | None:
    """Build enriched context for activity state detection.

    Args:
        context: PreHookContext with day, segment, output_path, transcript, meta

    Returns:
        Dict with modified transcript containing activity context,
        or None if unable to process.
    """
    day = context.get("day")
    segment = context.get("segment")
    output_path = context.get("output_path", "")
    transcript = context.get("transcript", "")
    meta = context.get("meta", {})

    if not day or not segment:
        logger.warning("activity_state pre-hook requires day and segment")
        return None

    # Extract facet from output path
    facet = _extract_facet_from_output_path(output_path)
    if not facet:
        logger.warning("Could not extract facet from output_path: %s", output_path)
        return None

    # Get timeout from meta or use default
    timeout_seconds = meta.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS)

    # Build activity context
    activities_context = format_activities_context(facet)

    # Find and load previous segment state
    previous_segment = find_previous_segment(day, segment)
    previous_state = None
    timed_out = False

    if previous_segment:
        # Check timeout
        timed_out = check_timeout(segment, previous_segment, timeout_seconds)

        if not timed_out:
            previous_state, _ = load_previous_state(day, previous_segment, facet)

    # Format previous state context
    previous_context = format_previous_state(
        previous_state, previous_segment, segment, timed_out
    )

    # Build enriched transcript
    enriched_parts = [
        activities_context,
        "",
        previous_context,
        "",
        "## Current Segment Content",
        "",
        transcript,
    ]

    return {"transcript": "\n".join(enriched_parts)}
