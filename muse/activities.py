# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Pre/post hooks for the activities generator.

Pre-hook:
1. Finds previous segment and compares activity_state across all facets
2. Detects ended activities (explicit, implicit, timeout)
3. Walks segment chains to collect full activity data
4. Writes records to facets/{facet}/activities/{day}.jsonl (idempotent)
5. Builds LLM prompt with per-segment descriptions for synthesis

Post-hook:
1. Parses LLM JSON output with synthesized descriptions
2. Updates activity records with unified descriptions
"""

import json
import logging
import os

from muse.activity_state import (
    check_timeout,
    find_previous_segment,
)
from think.activities import (
    append_activity_record,
    level_avg,
    load_record_ids,
    update_record_description,
)
from think.utils import day_path, now_ms, segment_parse

logger = logging.getLogger(__name__)


def _make_activity_id(activity_type: str, since_segment: str) -> str:
    """Build activity record ID from type and start segment key."""
    return f"{activity_type}_{since_segment}"


def _list_facets_with_activity_state(day: str, segment: str) -> list[str]:
    """Find all facets that have activity_state.json in a segment."""
    agents_dir = day_path(day) / segment / "agents"
    if not agents_dir.is_dir():
        return []

    facets = []
    for entry in sorted(os.listdir(agents_dir)):
        entry_path = agents_dir / entry
        if entry_path.is_dir() and (entry_path / "activity_state.json").exists():
            facets.append(entry)
    return facets


def _load_activity_state(day: str, segment: str, facet: str) -> list[dict]:
    """Load activity_state.json for a facet in a segment. Returns [] on failure."""
    state_path = day_path(day) / segment / "agents" / facet / "activity_state.json"
    if not state_path.exists():
        return []
    try:
        data = json.loads(state_path.read_text().strip())
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def _detect_ended_activities(
    prev_state: list[dict],
    curr_state: list[dict],
    timed_out: bool,
) -> list[dict]:
    """Detect activities that ended between previous and current segment.

    An activity ended if:
    - It was active in prev and has state="ended" in current (explicit)
    - It was active in prev but absent from current (implicit)
    - Timeout occurred — all previous active items ended

    Returns list of previous-state dicts for activities that ended.
    """
    prev_active = [item for item in prev_state if item.get("state") == "active"]

    if timed_out:
        return prev_active

    # Build set of (activity, since) pairs that are still active in current
    curr_active_keys = set()
    for item in curr_state:
        if item.get("state") == "active":
            curr_active_keys.add((item.get("activity"), item.get("since")))

    ended = []
    for item in prev_active:
        key = (item.get("activity"), item.get("since"))
        if key not in curr_active_keys:
            ended.append(item)

    return ended


def _walk_activity_segments(
    day: str, facet: str, activity_type: str, since: str, end_segment: str
) -> dict:
    """Walk segment chain collecting all data for an activity span.

    Starts from the `since` segment and walks forward through all segments
    where this activity appears as active with the same `since` value.

    Returns dict with segments, descriptions, levels, and active_entities.
    """
    day_dir = day_path(day)
    if not day_dir.is_dir():
        return {"segments": [], "descriptions": [], "levels": [], "active_entities": []}

    # Collect all segments in order (validate with segment_parse to skip non-segment dirs)
    all_segments = sorted(
        entry
        for entry in os.listdir(day_dir)
        if os.path.isdir(day_dir / entry)
        and entry >= since
        and entry <= end_segment
        and segment_parse(entry)[0] is not None
    )

    segments = []
    descriptions = []
    levels = []
    all_entities: list[str] = []

    for seg in all_segments:
        state = _load_activity_state(day, seg, facet)
        for item in state:
            if (
                item.get("activity") == activity_type
                and item.get("since") == since
                and item.get("state") == "active"
            ):
                segments.append(seg)
                desc = item.get("description", "")
                level = item.get("level", "medium")
                descriptions.append((seg, level, desc))
                levels.append(level)
                for entity in item.get("active_entities", []):
                    if entity not in all_entities:
                        all_entities.append(entity)
                break

    return {
        "segments": segments,
        "descriptions": descriptions,
        "levels": levels,
        "active_entities": all_entities,
    }


def _estimate_duration_minutes(segments: list[str]) -> int:
    """Estimate total duration in minutes from a list of segment keys."""
    total_seconds = 0
    for seg in segments:
        start, end = segment_parse(seg)
        if start is not None and end is not None:
            from datetime import datetime

            dt_start = datetime(2000, 1, 1, start.hour, start.minute, start.second)
            dt_end = datetime(2000, 1, 1, end.hour, end.minute, end.second)
            total_seconds += (dt_end - dt_start).total_seconds()
    return max(1, int(total_seconds / 60))


def _format_prompt_section(facet: str, activities: list[dict]) -> str:
    """Format a facet's ended activities as a prompt section."""
    lines = [f"## #{facet}", ""]

    for act in activities:
        record_id = act["id"]
        seg_count = len(act["segments"])
        duration = _estimate_duration_minutes(act["segments"])
        lines.append(f"### {record_id} ({seg_count} segments, {duration} min)")
        lines.append("Segment descriptions:")

        for seg, level, desc in act["descriptions"]:
            time_part = seg[:6]
            lines.append(f"- [{time_part}] {level}: {desc}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pre-hook
# ---------------------------------------------------------------------------


def pre_process(context: dict) -> dict | None:
    """Detect ended activities across all facets and build synthesis prompt.

    Writes activity records to journal, then prepares LLM prompt for
    description synthesis. Returns skip_reason if no activities ended.
    """
    day = context.get("day")
    segment = context.get("segment")

    if not day or not segment:
        logger.warning("activities pre-hook requires day and segment")
        return {"skip_reason": "missing_day_or_segment"}

    # Find previous segment
    prev_segment = find_previous_segment(day, segment)
    if not prev_segment:
        return {"skip_reason": "no_previous_segment"}

    # Check timeout
    timed_out = check_timeout(segment, prev_segment)

    # Scan all facets that had activity_state in the previous segment
    prev_facets = _list_facets_with_activity_state(day, prev_segment)
    if not prev_facets:
        return {"skip_reason": "no_previous_activity_state"}

    # Collect ended activities across all facets
    all_ended: dict[str, list[dict]] = {}  # facet -> list of enriched activity dicts
    existing_ids_cache: dict[str, set[str]] = {}

    for facet in prev_facets:
        prev_state = _load_activity_state(day, prev_segment, facet)
        curr_state = _load_activity_state(day, segment, facet)

        ended_items = _detect_ended_activities(prev_state, curr_state, timed_out)
        if not ended_items:
            continue

        # Load existing IDs for idempotency
        if facet not in existing_ids_cache:
            existing_ids_cache[facet] = load_record_ids(facet, day)

        for item in ended_items:
            activity_type = item.get("activity", "")
            since = item.get("since", "")
            if not activity_type or not since:
                continue

            record_id = _make_activity_id(activity_type, since)

            # Skip if already recorded (idempotent)
            if record_id in existing_ids_cache[facet]:
                continue

            # Walk segment chain to collect full data
            walk = _walk_activity_segments(
                day, facet, activity_type, since, prev_segment
            )

            if not walk["segments"]:
                # No segments found — use minimal data from the ended item
                walk["segments"] = [since]
                walk["descriptions"] = [
                    (since, item.get("level", "medium"), item.get("description", ""))
                ]
                walk["levels"] = [item.get("level", "medium")]
                walk["active_entities"] = item.get("active_entities", [])

            # Build record
            record = {
                "id": record_id,
                "activity": activity_type,
                "segments": walk["segments"],
                "level_avg": level_avg(walk["levels"]),
                "description": (
                    walk["descriptions"][-1][2] if walk["descriptions"] else ""
                ),
                "active_entities": walk["active_entities"],
                "created_at": now_ms(),
            }

            # Write record (skip internal ID check — already verified above)
            if append_activity_record(facet, day, record, _checked=True):
                existing_ids_cache[facet].add(record_id)
                logger.info("Wrote activity record %s for #%s", record_id, facet)

            # Store for prompt building
            if facet not in all_ended:
                all_ended[facet] = []
            all_ended[facet].append(
                {
                    "id": record_id,
                    "segments": walk["segments"],
                    "descriptions": walk["descriptions"],
                }
            )

    if not all_ended:
        return {"skip_reason": "no_ended_activities"}

    # Build LLM prompt
    prompt_parts = []
    for facet in sorted(all_ended.keys()):
        prompt_parts.append(_format_prompt_section(facet, all_ended[facet]))

    transcript = "\n".join(prompt_parts)

    return {"transcript": transcript}


# ---------------------------------------------------------------------------
# Post-hook
# ---------------------------------------------------------------------------


def post_process(result: str, context: dict) -> str | None:
    """Update activity records with LLM-synthesized descriptions.

    Parses the LLM JSON output and updates descriptions in the JSONL files
    that were written by the pre-hook.
    """
    day = context.get("day")
    if not day:
        return None

    try:
        data = json.loads(result.strip())
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("Failed to parse activities LLM output: %s", e)
        return None

    if not isinstance(data, dict):
        logger.warning("activities output is not an object")
        return None

    updated_count = 0
    for facet, activities in data.items():
        if not isinstance(activities, list):
            continue
        for activity in activities:
            record_id = activity.get("id", "")
            description = activity.get("description", "")
            if record_id and description:
                if update_record_description(facet, day, record_id, description):
                    updated_count += 1

    if updated_count:
        logger.info("Updated %d activity record descriptions", updated_count)

    return None  # Don't modify the output file — it saves as-is
