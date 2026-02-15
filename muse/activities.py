# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Pre/post hooks for the activities generator.

Pre-hook:
1. Finds previous segment and compares activity_state across all facets
2. Detects ended activities (explicit, implicit, timeout)
3. Walks segment chains to collect full activity data
4. Writes records to facets/{facet}/activities/{day}.jsonl (idempotent)
5. Builds LLM prompt with per-segment descriptions for synthesis
6. Stashes record data in meta for post-hook event emission

Flush mode (context["flush"] is truthy):
- Triggered by supervisor when no new segments arrive after a timeout
- Treats all active activities in the target segment as ended
- Skips inter-segment comparison (supervisor owns the timeout decision)

Post-hook:
1. Parses LLM JSON output with synthesized descriptions
2. Updates activity records with unified descriptions
3. Emits activity.recorded callosum events for each completed activity
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
    estimate_duration_minutes,
    level_avg,
    load_record_ids,
    make_activity_id,
    update_record_description,
)
from think.callosum import callosum_send
from think.utils import day_path, iter_segments, now_ms, segment_path

logger = logging.getLogger(__name__)


def _list_facets_with_activity_state(
    day: str, segment: str, stream: str | None = None
) -> list[str]:
    """Find all facets that have activity_state.json in a segment."""
    if stream:
        agents_dir = segment_path(day, segment, stream) / "agents"
    else:
        agents_dir = day_path(day) / segment / "agents"
    if not agents_dir.is_dir():
        return []

    facets = []
    for entry in sorted(os.listdir(agents_dir)):
        entry_path = agents_dir / entry
        if entry_path.is_dir() and (entry_path / "activity_state.json").exists():
            facets.append(entry)
    return facets


def _load_activity_state(
    day: str, segment: str, facet: str, stream: str | None = None
) -> list[dict]:
    """Load activity_state.json for a facet in a segment. Returns [] on failure."""
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
    day: str,
    facet: str,
    activity_type: str,
    since: str,
    end_segment: str,
    stream: str | None = None,
) -> dict:
    """Walk segment chain collecting all data for an activity span.

    Starts from the `since` segment and walks forward through all segments
    where this activity appears as active with the same `since` value.

    When stream is provided, only segments belonging to that stream are
    included. This prevents incorrect chaining across interleaved streams.

    Returns dict with segments, descriptions, levels, and active_entities.
    """
    day_dir = day_path(day)
    if not day_dir.is_dir():
        return {"segments": [], "descriptions": [], "levels": [], "active_entities": []}

    # Collect all segments in order via iter_segments, filtering by range and stream
    all_segment_tuples = [
        (s_stream, s_key)
        for s_stream, s_key, _s_path in iter_segments(day)
        if s_key >= since
        and s_key <= end_segment
        and (not stream or s_stream == stream)
    ]

    segments = []
    descriptions = []
    levels = []
    all_entities: list[str] = []

    for seg_stream, seg in all_segment_tuples:
        state = _load_activity_state(day, seg, facet, stream=seg_stream)
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


def _format_prompt_section(facet: str, activities: list[dict]) -> str:
    """Format a facet's ended activities as a prompt section."""
    lines = [f"## #{facet}", ""]

    for act in activities:
        record_id = act["id"]
        seg_count = len(act["segments"])
        duration = estimate_duration_minutes(act["segments"])
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


def _collect_ended(
    day: str,
    facets: list[str],
    ended_by_facet: dict[str, list[dict]],
    end_segment: str,
    stream: str | None = None,
) -> dict[str, list[dict]]:
    """Walk segment chains, write records, and build prompt data for ended activities.

    Shared by both normal and flush pre-hook paths.

    Args:
        day: Day in YYYYMMDD format
        facets: Facets to process
        ended_by_facet: {facet: [ended activity_state items]}
        end_segment: Last segment to include in the walk
        stream: Stream name for segment path resolution

    Returns:
        {facet: [enriched activity dicts]} for prompt building and post-hook
    """
    all_ended: dict[str, list[dict]] = {}
    existing_ids_cache: dict[str, set[str]] = {}

    for facet in facets:
        ended_items = ended_by_facet.get(facet, [])
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

            record_id = make_activity_id(activity_type, since)

            # Skip if already recorded (idempotent)
            if record_id in existing_ids_cache[facet]:
                continue

            # Walk segment chain to collect full data
            walk = _walk_activity_segments(
                day, facet, activity_type, since, end_segment, stream=stream
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

            # Store for prompt building and post-hook event emission
            if facet not in all_ended:
                all_ended[facet] = []
            all_ended[facet].append(
                {
                    "id": record_id,
                    "activity": activity_type,
                    "segments": walk["segments"],
                    "descriptions": walk["descriptions"],
                    "description": record["description"],
                    "level_avg": record["level_avg"],
                    "active_entities": record["active_entities"],
                }
            )

    return all_ended


def _build_result(context: dict, all_ended: dict[str, list[dict]]) -> dict | None:
    """Build LLM prompt and stash meta from collected ended activities."""
    if not all_ended:
        return {"skip_reason": "no_ended_activities"}

    # Build LLM prompt
    prompt_parts = []
    for facet in sorted(all_ended.keys()):
        prompt_parts.append(_format_prompt_section(facet, all_ended[facet]))

    transcript = "\n".join(prompt_parts)

    # Stash record data in meta for post-hook event emission
    meta = context.get("meta", {})
    meta["activity_records"] = {
        facet: {a["id"]: a for a in activities}
        for facet, activities in all_ended.items()
    }

    return {"transcript": transcript, "meta": meta}


def _pre_process_flush(context: dict) -> dict | None:
    """Flush mode: end all active activities in the target segment.

    Called when supervisor determines no new segments have arrived after
    a timeout. All active activities in the target segment are treated
    as ended — no inter-segment comparison needed.
    """
    day = context.get("day")
    segment = context.get("segment")
    stream = context.get("stream")

    if not day or not segment:
        logger.warning("activities flush pre-hook requires day and segment")
        return {"skip_reason": "missing_day_or_segment"}

    # Find all facets with activity_state in the target segment
    facets = _list_facets_with_activity_state(day, segment, stream=stream)
    if not facets:
        return {"skip_reason": "no_activity_state"}

    # Treat all active entries as ended (timed_out=True, empty current state)
    ended_by_facet: dict[str, list[dict]] = {}
    for facet in facets:
        state = _load_activity_state(day, segment, facet, stream=stream)
        ended = _detect_ended_activities(state, [], timed_out=True)
        if ended:
            ended_by_facet[facet] = ended

    if not ended_by_facet:
        return {"skip_reason": "no_active_activities"}

    logger.info(
        "Flush: ending %d activities across %d facets",
        sum(len(v) for v in ended_by_facet.values()),
        len(ended_by_facet),
    )

    all_ended = _collect_ended(day, facets, ended_by_facet, segment, stream=stream)
    return _build_result(context, all_ended)


def _pre_process_normal(context: dict) -> dict | None:
    """Normal mode: detect ended activities by comparing adjacent segments."""
    day = context.get("day")
    segment = context.get("segment")
    stream = context.get("stream")

    if not day or not segment:
        logger.warning("activities pre-hook requires day and segment")
        return {"skip_reason": "missing_day_or_segment"}

    # Find previous segment
    prev_segment = find_previous_segment(day, segment, stream=stream)
    if not prev_segment:
        return {"skip_reason": "no_previous_segment"}

    # Check timeout
    timed_out = check_timeout(segment, prev_segment)

    # Scan all facets that had activity_state in the previous segment
    prev_facets = _list_facets_with_activity_state(day, prev_segment, stream=stream)
    if not prev_facets:
        return {"skip_reason": "no_previous_activity_state"}

    # Detect ended activities across all facets
    ended_by_facet: dict[str, list[dict]] = {}
    for facet in prev_facets:
        prev_state = _load_activity_state(day, prev_segment, facet, stream=stream)
        curr_state = _load_activity_state(day, segment, facet, stream=stream)
        ended = _detect_ended_activities(prev_state, curr_state, timed_out)
        if ended:
            ended_by_facet[facet] = ended

    all_ended = _collect_ended(
        day, prev_facets, ended_by_facet, prev_segment, stream=stream
    )
    return _build_result(context, all_ended)


def pre_process(context: dict) -> dict | None:
    """Detect ended activities across all facets and build synthesis prompt.

    Writes activity records to journal, then prepares LLM prompt for
    description synthesis. Returns skip_reason if no activities ended.

    In flush mode (context["flush"] is truthy), treats all active activities
    in the target segment as ended — triggered by supervisor after a timeout
    with no new segments.
    """
    if context.get("flush"):
        return _pre_process_flush(context)
    return _pre_process_normal(context)


# ---------------------------------------------------------------------------
# Post-hook
# ---------------------------------------------------------------------------


def post_process(result: str, context: dict) -> str | None:
    """Update activity records with LLM-synthesized descriptions.

    Parses the LLM JSON output, updates descriptions in the JSONL files
    that were written by the pre-hook, then emits activity.recorded events.
    """
    day = context.get("day")
    segment = context.get("segment")
    if not day:
        return None

    # Parse LLM output and update descriptions
    llm_descriptions: dict[str, dict[str, str]] = {}  # facet -> {id -> description}
    try:
        data = json.loads(result.strip())
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("Failed to parse activities LLM output: %s", e)
        data = None

    if isinstance(data, dict):
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
                        llm_descriptions.setdefault(facet, {})[record_id] = description

        if updated_count:
            logger.info("Updated %d activity record descriptions", updated_count)

    # Emit activity.recorded events using pre-hook record data
    meta = context.get("meta", {})
    activity_records = meta.get("activity_records", {})

    for facet, records in activity_records.items():
        for record_id, record in records.items():
            # Use LLM description if available, otherwise fall back to pre-hook
            description = llm_descriptions.get(facet, {}).get(record_id)
            if not description:
                description = record.get("description", "")
                logger.warning(
                    "No LLM description for %s in #%s, using pre-hook description",
                    record_id,
                    facet,
                )

            try:
                callosum_send(
                    "activity",
                    "recorded",
                    facet=facet,
                    day=day,
                    segment=segment,
                    id=record_id,
                    activity=record.get("activity", ""),
                    segments=record.get("segments", []),
                    level_avg=record.get("level_avg", 0.5),
                    description=description,
                    active_entities=record.get("active_entities", []),
                )
            except Exception as e:
                logger.warning(
                    "Failed to emit activity.recorded for %s: %s", record_id, e
                )

    return None  # Don't modify the output file — it saves as-is
