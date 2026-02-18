# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Pre/post hooks for activity_state generator.

Pre-hook builds context for activity detection by:
1. Loading the facet's configured activities
2. Finding and loading previous segment's activity state
3. Formatting both as context for the prompt

Post-hook resolves timing metadata:
1. Stamps `since` (segment key) from tooling — never from LLM
2. Normalizes `state` from LLM values (continuing/new) to stored values (active)
3. Matches continuing/ended activities to previous state via activity type + fuzzy description
4. Drops redundant ended re-reports; promotes unmatched ended with novel descriptions to active
"""

import json
import logging
import os
from datetime import datetime, timedelta

from think.activities import make_activity_id
from think.callosum import callosum_send
from think.utils import day_path, iter_segments, segment_parse, segment_path

logger = logging.getLogger(__name__)

# Default timeout for activity continuity (1 hour in seconds)
DEFAULT_TIMEOUT_SECONDS = 3600


def _extract_facet_from_output_path(output_path: str) -> str | None:
    """Extract facet name from output path.

    Output paths for faceted generators follow the pattern:
    {day}/{segment}/agents/{facet}/activity_state.json

    Returns None if facet cannot be extracted.
    """
    if not output_path:
        return None

    filename = os.path.basename(output_path)
    if filename != "activity_state.json":
        return None

    parent = os.path.basename(os.path.dirname(output_path))
    if parent and parent != "agents":
        return parent
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


def _get_preceding_segments(
    day: str, current_segment: str, stream: str | None = None
) -> list[str]:
    """Return segment keys preceding *current_segment*, most-recent first.

    Uses iter_segments() to traverse the stream directory structure.
    When stream is provided, only segments in that stream are considered.
    """
    all_segments = iter_segments(day)
    if not all_segments:
        return []

    if stream:
        segment_keys = [
            s_key for s_stream, s_key, _s_path in all_segments if s_stream == stream
        ]
    else:
        segment_keys = [s_key for _s_stream, s_key, _s_path in all_segments]

    if not segment_keys:
        return []

    try:
        current_idx = segment_keys.index(current_segment)
    except ValueError:
        for i, seg in enumerate(segment_keys):
            if seg > current_segment:
                current_idx = i
                break
        else:
            current_idx = len(segment_keys)

    # Return preceding keys in reverse order (most recent first)
    return list(reversed(segment_keys[:current_idx]))


def find_previous_segment(
    day: str, current_segment: str, stream: str | None = None
) -> str | None:
    """Find the segment immediately before the current one.

    Returns None if current is the first segment or no segments found.
    """
    preceding = _get_preceding_segments(day, current_segment, stream=stream)
    return preceding[0] if preceding else None


def load_previous_state(
    day: str, segment: str, facet: str, stream: str | None = None
) -> tuple[list | None, str | None]:
    """Load activity state from a previous segment.

    Returns tuple of (state_list, segment_key) where state_list is the
    parsed JSON array or None if not found/invalid.
    """
    if stream:
        state_path = (
            segment_path(day, segment, stream)
            / "agents"
            / facet
            / "activity_state.json"
        )
    else:
        state_path = day_path(day) / segment / "agents" / facet / "activity_state.json"
    if not state_path.exists():
        return None, None

    try:
        content = state_path.read_text().strip()
        if not content:
            return None, segment

        data = json.loads(content)
        if isinstance(data, list):
            return data, segment
        # Unexpected format
        logger.warning("activity_state is not an array: %s", state_path)
        return None, segment
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to load previous state from %s: %s", state_path, e)
        return None, segment


# Maximum number of segments to walk back when looking for previous state.
_MAX_LOOKBACK = 10


def find_previous_state(
    day: str,
    current_segment: str,
    facet: str,
    stream: str | None = None,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> tuple[list | None, str | None]:
    """Walk backwards to find the most recent segment with valid state.

    Tries each preceding segment in reverse chronological order until it
    finds one with a valid activity_state.json for *facet*, or until it
    hits the timeout boundary or the look-back cap.

    Returns (state_list, segment_key) — same contract as load_previous_state.
    """
    preceding = _get_preceding_segments(day, current_segment, stream=stream)

    for seg in preceding[:_MAX_LOOKBACK]:
        if check_timeout(current_segment, seg, timeout_seconds):
            break
        state, seg_key = load_previous_state(day, seg, facet, stream=stream)
        if state is not None:
            return state, seg_key

    return None, None


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
        instructions = activity.get("instructions", "")
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

        if instructions:
            lines.append(f"  {instructions}")

    return "\n".join(lines)


def format_previous_state(
    state: list | None,
    segment: str | None,
    current_segment: str,
    timed_out: bool,
) -> str:
    """Format previous state as context for the prompt.

    Args:
        state: Previous segment's activity state list (flat array)
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

    active = [item for item in state if item.get("state") == "active"]
    ended = [item for item in state if item.get("state") == "ended"]

    if active:
        lines.append("**Active activities (may be continuing):**")
        for item in active:
            activity_id = item.get("activity", "")
            description = item.get("description", "")
            level = item.get("level", "")

            parts = [f"- {activity_id}"]
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


# ---------------------------------------------------------------------------
# Pre-hook
# ---------------------------------------------------------------------------


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
    stream = context.get("stream")
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
    previous_segment = find_previous_segment(day, segment, stream=stream)
    previous_state = None
    timed_out = False

    if previous_segment:
        # Check timeout
        timed_out = check_timeout(segment, previous_segment, timeout_seconds)

        if not timed_out:
            previous_state, _ = load_previous_state(
                day, previous_segment, facet, stream=stream
            )

            # Look further back if immediate predecessor has no state
            if previous_state is None:
                previous_state, found_seg = find_previous_state(
                    day,
                    segment,
                    facet,
                    stream=stream,
                    timeout_seconds=timeout_seconds,
                )
                if found_seg:
                    previous_segment = found_seg

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


# ---------------------------------------------------------------------------
# Post-hook
# ---------------------------------------------------------------------------


def _find_best_match(
    activity_id: str,
    description: str,
    candidates: list[tuple[int, dict]],
) -> tuple[int, dict] | None:
    """Find the best matching previous activity by type, then description.

    If only one candidate matches the activity type, returns it directly.
    If multiple match (rare — concurrent same-type activities), uses fuzzy
    description matching to pick the best one.

    Args:
        activity_id: Activity type to match (e.g., "meeting")
        description: Description from LLM output for fuzzy matching
        candidates: List of (index, item) tuples from previous active state

    Returns:
        (index, item) tuple of the best match, or None if no match found.
    """
    matches = [(i, c) for i, c in candidates if c.get("activity") == activity_id]

    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]

    # Multiple same-type activities — use fuzzy matching on description
    if not description:
        return matches[0]

    try:
        from rapidfuzz import fuzz, process

        # Build list of (description, index into matches) for fuzzy comparison.
        # Using a list avoids key collision when two activities share a description.
        desc_list: list[tuple[str, int]] = []
        for mi, (idx, m) in enumerate(matches):
            desc = m.get("description", "")
            if desc:
                desc_list.append((desc, mi))

        if not desc_list:
            return matches[0]

        result = process.extractOne(
            description,
            [d for d, _ in desc_list],
            scorer=fuzz.token_sort_ratio,
        )
        if result:
            _matched_str, _score, list_idx = result
            _, mi = desc_list[list_idx]
            return matches[mi]
    except ImportError:
        pass

    return matches[0]


def _is_redundant_ended(
    activity_id: str,
    description: str,
    prev_ended: list[dict],
) -> bool:
    """Check if an ended activity is a redundant re-report.

    Returns True if a matching activity already ended in the previous segment
    (same type with similar or empty description), meaning this is just the
    LLM re-reporting an ending that was already recorded.
    """
    ended_same_type = [e for e in prev_ended if e.get("activity") == activity_id]
    if not ended_same_type:
        return False

    # If only one match and description is close enough, it's redundant
    if not description:
        return True

    try:
        from rapidfuzz import fuzz

        for prev in ended_same_type:
            prev_desc = prev.get("description", "")
            if not prev_desc:
                return True
            if fuzz.token_sort_ratio(description, prev_desc) >= 70:
                return True
    except ImportError:
        # Without fuzzy matching, fall back to exact substring check
        for prev in ended_same_type:
            prev_desc = prev.get("description", "")
            if not prev_desc:
                return True
            if (
                description.lower() in prev_desc.lower()
                or prev_desc.lower() in description.lower()
            ):
                return True

    return False


def post_process(result: str, context: dict) -> str | None:
    """Resolve timing metadata on LLM activity state output.

    Stamps `since` field from tooling (never from LLM) and normalizes
    state values from LLM format (continuing/new/ended) to stored format
    (active/ended).

    Args:
        result: Raw LLM JSON output (flat array of activities)
        context: HookContext with day, segment, output_path, meta

    Returns:
        Transformed JSON string with since fields resolved,
        or None to keep original on error.
    """
    segment = context.get("segment")
    if not segment:
        logger.warning("activity_state post-hook requires segment")
        return None

    day = context.get("day")
    stream = context.get("stream")
    output_path = context.get("output_path", "")

    # Parse LLM output
    try:
        items = json.loads(result.strip())
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("Failed to parse activity_state LLM output: %s", e)
        return None

    if not isinstance(items, list):
        logger.warning("activity_state output is not an array")
        return None

    # Load previous state for since resolution
    prev_active: list[dict] = []
    prev_ended: list[dict] = []
    if day:
        facet = _extract_facet_from_output_path(output_path)
        if facet:
            previous_segment = find_previous_segment(day, segment, stream=stream)
            if previous_segment:
                prev_state, _ = load_previous_state(
                    day, previous_segment, facet, stream=stream
                )
                if prev_state:
                    prev_active = [
                        item for item in prev_state if item.get("state") == "active"
                    ]
                    prev_ended = [
                        item for item in prev_state if item.get("state") == "ended"
                    ]

    # Track which previous items have been claimed to avoid double-matching
    claimed: set[int] = set()

    resolved: list[dict] = []
    for item in items:
        activity_id = item.get("activity", "")
        state = item.get("state", "new")
        description = item.get("description", "")

        # Build unclaimed candidates with their original indices
        unclaimed = [(i, c) for i, c in enumerate(prev_active) if i not in claimed]

        active_entities = item.get("active_entities", [])

        if state == "continuing":
            result = _find_best_match(activity_id, description, unclaimed)
            if result:
                idx, matched = result
                claimed.add(idx)
                since = matched.get("since", segment)
            else:
                # No previous match — treat as new
                since = segment

            entry = {
                "id": make_activity_id(activity_id, since),
                "activity": activity_id,
                "state": "active",
                "since": since,
                "description": description,
                "level": item.get("level", "medium"),
            }
            if active_entities:
                entry["active_entities"] = active_entities
            resolved.append(entry)

        elif state == "ended":
            result = _find_best_match(activity_id, description, unclaimed)
            if result:
                idx, matched = result
                claimed.add(idx)
                since = matched.get("since", segment)
                resolved.append(
                    {
                        "id": make_activity_id(activity_id, since),
                        "activity": activity_id,
                        "state": "ended",
                        "since": since,
                        "description": description,
                    }
                )
            elif description and not _is_redundant_ended(
                activity_id, description, prev_ended
            ):
                # No active match but has a novel description — likely
                # a real activity the LLM mis-tagged as ended; treat as new
                entry = {
                    "id": make_activity_id(activity_id, segment),
                    "activity": activity_id,
                    "state": "active",
                    "since": segment,
                    "description": description,
                    "level": item.get("level", "medium"),
                }
                if active_entities:
                    entry["active_entities"] = active_entities
                resolved.append(entry)
            # else: redundant re-report of already ended activity — drop

        else:
            # "new" or any unrecognized state — stamp current segment
            entry = {
                "id": make_activity_id(activity_id, segment),
                "activity": activity_id,
                "state": "active",
                "since": segment,
                "description": description,
                "level": item.get("level", "medium"),
            }
            if active_entities:
                entry["active_entities"] = active_entities
            resolved.append(entry)

    # Emit activity.live events for active entries
    if day and facet:
        for entry in resolved:
            if entry.get("state") != "active":
                continue
            try:
                callosum_send(
                    "activity",
                    "live",
                    facet=facet,
                    day=day,
                    segment=segment,
                    id=entry["id"],
                    activity=entry["activity"],
                    since=entry["since"],
                    description=entry.get("description", ""),
                    level=entry.get("level", "medium"),
                    active_entities=entry.get("active_entities", []),
                )
            except Exception as e:
                logger.warning(
                    "Failed to emit activity.live for %s: %s", entry["id"], e
                )

    return json.dumps(resolved, ensure_ascii=False)
