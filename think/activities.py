# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Activities system for tracking common activity types per facet.

Activities provide a consistent vocabulary for tagging time segments,
screen observations, and extracted events across the journal.

Also provides utilities for activity records — completed activity spans
stored as facets/{facet}/activities/{day}.jsonl.
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from think.utils import get_journal, segment_parse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default Activities
#
# These are predefined common activities that users can attach to facets.
# They serve as a starting vocabulary - facets must explicitly attach them.
# ---------------------------------------------------------------------------

DEFAULT_ACTIVITIES: list[dict[str, str]] = [
    {
        "id": "meeting",
        "name": "Meetings",
        "description": "Video calls, in-person meetings, and conferences",
        "icon": "📅",
        "instructions": (
            "Levels: high=actively speaking/presenting, medium=listening attentively,"
            " low=muted or multitasking during call."
            " Detect via: video call UI, multiple speakers, calendar event visible."
        ),
    },
    {
        "id": "coding",
        "name": "Coding",
        "description": "Programming, code review, and debugging",
        "icon": "💻",
        "instructions": (
            "Levels: high=writing or debugging code, medium=reading/reviewing code,"
            " low=IDE or editor open but not focused."
            " Detect via: editors, terminals with dev tools, AI coding assistants,"
            " git operations. Includes focused code reading and thinking."
        ),
    },
    {
        "id": "browsing",
        "name": "Browsing",
        "description": "Web browsing, research, and reading online",
        "icon": "🌐",
        "instructions": (
            "Levels: high=actively navigating/researching, medium=reading a page,"
            " low=browser open but idle."
            " Detect via: browser tabs, URL changes, search queries."
        ),
    },
    {
        "id": "email",
        "name": "Email",
        "description": "Email reading and composition",
        "icon": "📧",
        "instructions": (
            "Levels: high=composing or actively reading email,"
            " medium=scanning inbox, low=email client visible but idle."
            " Detect via: email client UI, inbox view, compose window."
        ),
    },
    {
        "id": "messaging",
        "name": "Messaging",
        "description": "Chat, Slack, Discord, and text messaging",
        "icon": "💬",
        "instructions": (
            "Levels: high=active conversation, medium=reading messages,"
            " low=chat app visible but idle."
            " Detect via: chat app UI, message notifications, typing indicators."
        ),
    },
    {
        "id": "writing",
        "name": "Writing",
        "description": "Documents, notes, and long-form writing",
        "icon": "✍️",
        "instructions": (
            "Levels: high=actively composing text, medium=editing/revising,"
            " low=document open but not being edited."
            " Detect via: document editors, note apps, text content changing."
        ),
    },
    {
        "id": "reading",
        "name": "Reading",
        "description": "PDFs, articles, and documentation",
        "icon": "📖",
        "instructions": (
            "Levels: high=focused reading, medium=skimming content,"
            " low=document open but attention elsewhere."
            " Detect via: PDF viewers, article pages, documentation sites."
            " Do not use for reading code — that is coding."
        ),
    },
    {
        "id": "video",
        "name": "Video",
        "description": "Watching videos and streaming content",
        "icon": "🎬",
        "instructions": (
            "Levels: high=actively watching, medium=video playing while"
            " doing something else, low=video paused or minimized."
            " Detect via: video player UI, streaming sites, playback controls."
        ),
    },
    {
        "id": "gaming",
        "name": "Gaming",
        "description": "Games and entertainment",
        "icon": "🎮",
        "instructions": (
            "Levels: high=actively playing, medium=in menus or waiting,"
            " low=game open but tabbed out."
            " Detect via: game window, controller input, game UI elements."
        ),
    },
    {
        "id": "social",
        "name": "Social Media",
        "description": "Social media browsing and interaction",
        "icon": "📱",
        "instructions": (
            "Levels: high=posting or actively engaging, medium=scrolling feed,"
            " low=social app open but idle."
            " Detect via: social media sites/apps, feed content, post composition."
        ),
    },
    {
        "id": "productivity",
        "name": "Productivity",
        "description": "Spreadsheets, slides, and task management",
        "icon": "📊",
        "instructions": (
            "Levels: high=actively editing or organizing, medium=reviewing data,"
            " low=app open but not focused."
            " Detect via: spreadsheet/slide editors, project management tools,"
            " calendar apps, task boards."
        ),
    },
    {
        "id": "terminal",
        "name": "Terminal",
        "description": "Command line and shell sessions",
        "icon": "⌨️",
        "instructions": (
            "Levels: high=running commands or scripts, medium=reading output,"
            " low=terminal open but idle."
            " Detect via: shell prompts, command output, tmux/screen sessions."
            " If terminal use is clearly coding-related, prefer coding instead."
        ),
    },
    {
        "id": "design",
        "name": "Design",
        "description": "Design tools and image editing",
        "icon": "🎨",
        "instructions": (
            "Levels: high=actively creating or editing, medium=reviewing designs,"
            " low=design tool open but idle."
            " Detect via: design apps (Figma, Photoshop, etc), canvas editing."
        ),
    },
    {
        "id": "music",
        "name": "Music",
        "description": "Music listening and audio",
        "icon": "🎵",
        "instructions": (
            "Levels: high=actively choosing or browsing music,"
            " medium=playlist running while working, low=ambient background audio."
            " Detect via: music player UI, audio playback indicators."
        ),
    },
]


def get_default_activities() -> list[dict[str, str]]:
    """Return the predefined activities list.

    These are common activities that users can attach to facets.
    Returns a copy to prevent mutation.
    """
    return [dict(a) for a in DEFAULT_ACTIVITIES]


def _get_activities_path(facet: str) -> Path:
    """Get the path to a facet's activities.jsonl file."""
    return Path(get_journal()) / "facets" / facet / "activities" / "activities.jsonl"


def _load_activities_jsonl(facet: str) -> list[dict[str, Any]]:
    """Load raw activities from a facet's JSONL file.

    Returns empty list if file doesn't exist.
    """
    path = _get_activities_path(facet)
    if not path.exists():
        return []

    activities = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    activities.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(
                        "Skipping malformed line %d in %s: %s", line_num, path, e
                    )
                    continue
    return activities


def _save_activities_jsonl(facet: str, activities: list[dict[str, Any]]) -> None:
    """Save activities to a facet's JSONL file."""
    path = _get_activities_path(facet)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for activity in activities:
            f.write(json.dumps(activity, ensure_ascii=False) + "\n")


def get_facet_activities(facet: str) -> list[dict[str, Any]]:
    """Load activities attached to a facet.

    Returns only activities that have been explicitly attached to the facet.
    Each returned activity includes all fields from the facet config merged
    with default metadata (name, icon) if the activity is predefined.

    Args:
        facet: Facet name

    Returns:
        List of activity dicts with keys:
        - id: Activity identifier
        - name: Display name
        - description: Activity description
        - icon: Emoji icon (if predefined)
        - priority: "high", "normal", or "low"
        - custom: True if user-created (not in defaults)
    """
    # Build lookup for defaults
    defaults_by_id = {a["id"]: a for a in DEFAULT_ACTIVITIES}

    # Load facet-specific activities
    facet_activities = _load_activities_jsonl(facet)

    result = []
    for fa in facet_activities:
        activity_id = fa.get("id")
        if not activity_id:
            continue

        # Start with default metadata if predefined
        if activity_id in defaults_by_id:
            activity = dict(defaults_by_id[activity_id])
            activity["custom"] = False
        else:
            activity = {"id": activity_id, "custom": True}

        # Apply facet overrides
        if "name" in fa:
            activity["name"] = fa["name"]
        if "description" in fa:
            activity["description"] = fa["description"]
        if "priority" in fa:
            activity["priority"] = fa["priority"]
        if "icon" in fa:
            activity["icon"] = fa["icon"]
        if "instructions" in fa:
            activity["instructions"] = fa["instructions"]

        # Ensure required fields have defaults
        activity.setdefault("name", activity_id.replace("_", " ").title())
        activity.setdefault("description", "")
        activity.setdefault("priority", "normal")

        result.append(activity)

    return result


def save_facet_activities(facet: str, activities: list[dict[str, Any]]) -> None:
    """Save activities configuration for a facet.

    Args:
        facet: Facet name
        activities: List of activity dicts to save. Each should have at minimum:
            - id: Activity identifier
            For custom activities, also include:
            - name: Display name
            - description: Activity description
            Optional for all:
            - priority: "high", "normal", or "low"
            - icon: Emoji icon
            - instructions: Detection/level instructions for the LLM
    """
    # Build lookup for defaults to determine what needs to be stored
    defaults_by_id = {a["id"]: a for a in DEFAULT_ACTIVITIES}

    entries = []
    for activity in activities:
        activity_id = activity.get("id")
        if not activity_id:
            continue

        entry: dict[str, Any] = {"id": activity_id}

        # For predefined activities, only store overrides
        if activity_id in defaults_by_id:
            default = defaults_by_id[activity_id]

            # Store description only if different from default
            if activity.get("description") and activity["description"] != default.get(
                "description"
            ):
                entry["description"] = activity["description"]

            # Store instructions only if different from default
            if activity.get("instructions") and activity["instructions"] != default.get(
                "instructions"
            ):
                entry["instructions"] = activity["instructions"]

            # Store priority if set
            if activity.get("priority") and activity["priority"] != "normal":
                entry["priority"] = activity["priority"]

        else:
            # Custom activity - store all fields
            entry["custom"] = True
            if activity.get("name"):
                entry["name"] = activity["name"]
            if activity.get("description"):
                entry["description"] = activity["description"]
            if activity.get("instructions"):
                entry["instructions"] = activity["instructions"]
            if activity.get("priority"):
                entry["priority"] = activity["priority"]
            if activity.get("icon"):
                entry["icon"] = activity["icon"]

        entries.append(entry)

    _save_activities_jsonl(facet, entries)


def get_activity_by_id(facet: str, activity_id: str) -> dict[str, Any] | None:
    """Look up a specific activity by ID.

    Args:
        facet: Facet name
        activity_id: Activity identifier

    Returns:
        Activity dict if found, None otherwise
    """
    activities = get_facet_activities(facet)
    for activity in activities:
        if activity.get("id") == activity_id:
            return activity
    return None


def generate_activity_id(name: str) -> str:
    """Generate a slug ID from an activity name.

    Args:
        name: Activity display name

    Returns:
        Slug identifier (lowercase, underscores)
    """
    # Lowercase and replace non-alphanumeric with underscores
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower())
    # Remove leading/trailing underscores
    slug = slug.strip("_")
    return slug or "activity"


def add_activity_to_facet(
    facet: str,
    activity_id: str,
    *,
    name: str | None = None,
    description: str | None = None,
    instructions: str | None = None,
    priority: str = "normal",
    icon: str | None = None,
) -> dict[str, Any]:
    """Add an activity to a facet.

    For predefined activities, only activity_id is required.
    For custom activities, name and description should be provided.

    Args:
        facet: Facet name
        activity_id: Activity identifier
        name: Display name (required for custom activities)
        description: Activity description
        instructions: Detection/level instructions for the LLM
        priority: "high", "normal", or "low"
        icon: Emoji icon

    Returns:
        The added activity dict
    """
    # Check if already attached
    existing = get_facet_activities(facet)
    for activity in existing:
        if activity.get("id") == activity_id:
            # Already attached - return existing
            return activity

    # Build new activity entry
    defaults_by_id = {a["id"]: a for a in DEFAULT_ACTIVITIES}

    if activity_id in defaults_by_id:
        # Predefined activity
        activity: dict[str, Any] = {"id": activity_id}
        if description:
            activity["description"] = description
        if instructions:
            activity["instructions"] = instructions
        if priority and priority != "normal":
            activity["priority"] = priority
    else:
        # Custom activity
        activity = {
            "id": activity_id,
            "custom": True,
            "name": name or activity_id.replace("_", " ").title(),
            "description": description or "",
        }
        if instructions:
            activity["instructions"] = instructions
        if priority and priority != "normal":
            activity["priority"] = priority
        if icon:
            activity["icon"] = icon

    # Add to existing activities and save
    existing_raw = _load_activities_jsonl(facet)
    existing_raw.append(activity)
    _save_activities_jsonl(facet, existing_raw)

    # Return the full activity with defaults merged
    return get_activity_by_id(facet, activity_id) or activity


def remove_activity_from_facet(facet: str, activity_id: str) -> bool:
    """Remove an activity from a facet.

    Args:
        facet: Facet name
        activity_id: Activity identifier to remove

    Returns:
        True if activity was removed, False if not found
    """
    existing = _load_activities_jsonl(facet)
    new_list = [a for a in existing if a.get("id") != activity_id]

    if len(new_list) == len(existing):
        # Nothing removed
        return False

    _save_activities_jsonl(facet, new_list)
    return True


def update_activity_in_facet(
    facet: str,
    activity_id: str,
    *,
    description: str | None = None,
    instructions: str | None = None,
    priority: str | None = None,
    name: str | None = None,
    icon: str | None = None,
) -> dict[str, Any] | None:
    """Update an activity's configuration in a facet.

    Args:
        facet: Facet name
        activity_id: Activity identifier
        description: New description (None to keep existing)
        instructions: New detection/level instructions (None to keep existing)
        priority: New priority (None to keep existing)
        name: New name - only applies to custom activities
        icon: New icon - only applies to custom activities

    Returns:
        Updated activity dict, or None if not found
    """
    existing = _load_activities_jsonl(facet)
    defaults_by_id = {a["id"]: a for a in DEFAULT_ACTIVITIES}

    found = False
    for activity in existing:
        if activity.get("id") == activity_id:
            found = True

            if description is not None:
                if description == "" and activity_id in defaults_by_id:
                    activity.pop("description", None)
                else:
                    activity["description"] = description
            if instructions is not None:
                if instructions == "" and activity_id in defaults_by_id:
                    activity.pop("instructions", None)
                else:
                    activity["instructions"] = instructions
            if priority is not None:
                if priority == "normal" and activity_id in defaults_by_id:
                    activity.pop("priority", None)
                else:
                    activity["priority"] = priority

            # Only allow name/icon changes for custom activities
            if activity.get("custom") or activity_id not in defaults_by_id:
                if name is not None:
                    activity["name"] = name
                if icon is not None:
                    activity["icon"] = icon

            break

    if not found:
        return None

    _save_activities_jsonl(facet, existing)
    return get_activity_by_id(facet, activity_id)


# ---------------------------------------------------------------------------
# Activity State — per-segment activity state loading
# ---------------------------------------------------------------------------


def load_segment_activity_state(
    day: str, segment: str, facet: str, activity_type: str
) -> dict[str, Any] | None:
    """Load activity state for a specific activity from a segment.

    Reads the activity_state.json written by the activity_state generator
    for a given segment and facet, and returns the entry matching the
    requested activity type.

    Args:
        day: Day in YYYYMMDD format
        segment: Segment key (HHMMSS_LEN)
        facet: Facet name
        activity_type: Activity type to find (e.g., "coding", "meeting")

    Returns:
        Activity state dict with keys like activity, state, description,
        level, active_entities — or None if not found.
    """
    from think.cluster import _find_segment_dir

    stream = os.environ.get("SOL_STREAM")
    seg_dir = _find_segment_dir(day, segment, stream)
    if not seg_dir:
        return None

    state_path = seg_dir / "agents" / facet / "activity_state.json"
    if not state_path.exists():
        return None

    try:
        with open(state_path, "r", encoding="utf-8") as f:
            states = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    if not isinstance(states, list):
        return None

    for entry in states:
        if entry.get("activity") == activity_type:
            return entry

    return None


# ---------------------------------------------------------------------------
# Activity Records — completed activity spans
# ---------------------------------------------------------------------------


def make_activity_id(activity_type: str, since_segment: str) -> str:
    """Build activity record ID from type and start segment key.

    Format: {activity_type}_{since_segment}, e.g. "coding_095809_303".
    Used by both activity_state (live tracking) and activities (records).
    """
    return f"{activity_type}_{since_segment}"


LEVEL_VALUES = {"high": 1.0, "medium": 0.5, "low": 0.25}


def _get_records_path(facet: str, day: str) -> Path:
    """Get path to a facet's activity records file for a day."""
    return Path(get_journal()) / "facets" / facet / "activities" / f"{day}.jsonl"


def get_activity_output_path(
    facet: str,
    day: str,
    activity_id: str,
    key: str,
    output_format: str | None = None,
) -> Path:
    """Return output path for an activity-scheduled agent.

    Output lives under the facet's activities directory, grouped by day
    and activity record ID:

        facets/{facet}/activities/{day}/{activity_id}/{agent}.{ext}

    Args:
        facet: Facet name
        day: Day in YYYYMMDD format
        activity_id: Activity record ID (e.g., "coding_095809_303")
        key: Agent key (e.g., "session_review", "chat:analysis")
        output_format: "json" for JSON, anything else for markdown

    Returns:
        Absolute path for the output file
    """
    from think.muse import get_output_name

    output_name = get_output_name(key)
    ext = "json" if output_format == "json" else "md"
    return (
        Path(get_journal())
        / "facets"
        / facet
        / "activities"
        / day
        / activity_id
        / f"{output_name}.{ext}"
    )


def load_activity_records(facet: str, day: str) -> list[dict[str, Any]]:
    """Load activity records for a facet and day.

    Returns list of record dicts, empty list if file doesn't exist.
    """
    path = _get_records_path(facet, day)
    if not path.exists():
        return []

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def load_record_ids(facet: str, day: str) -> set[str]:
    """Load just the IDs of existing activity records for idempotency checks."""
    return {r["id"] for r in load_activity_records(facet, day) if "id" in r}


def append_activity_record(
    facet: str, day: str, record: dict[str, Any], *, _checked: bool = False
) -> bool:
    """Append an activity record to the facet's day file.

    Checks for duplicate ID — returns False if record already exists.

    Args:
        facet: Facet name
        day: Day in YYYYMMDD format
        record: Activity record dict (must have 'id' field)
        _checked: If True, skip the duplicate ID check (caller already verified).

    Returns:
        True if record was written, False if duplicate ID found.
    """
    path = _get_records_path(facet, day)

    if not _checked:
        # Check for existing ID
        existing_ids = load_record_ids(facet, day)
        if record.get("id") in existing_ids:
            return False

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return True


def update_record_description(
    facet: str, day: str, record_id: str, description: str
) -> bool:
    """Update the description of an existing activity record.

    Rewrites the JSONL file atomically (write temp + rename) with the updated
    description for the matching record.

    Returns True if record was found and updated, False otherwise.
    """
    import tempfile

    path = _get_records_path(facet, day)
    if not path.exists():
        return False

    lines = path.read_text(encoding="utf-8").splitlines()
    updated = False
    new_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            new_lines.append(line)
            continue

        if record.get("id") == record_id:
            record["description"] = description
            updated = True

        new_lines.append(json.dumps(record, ensure_ascii=False))

    if updated:
        content = "\n".join(new_lines) + "\n"
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(content)
            os.replace(tmp, path)
        except BaseException:
            os.unlink(tmp)
            raise

    return updated


def estimate_duration_minutes(segments: list[str]) -> int:
    """Estimate total duration in minutes from a list of segment keys.

    Parses each HHMMSS_LEN segment key, sums the durations, returns minutes.
    Returns 1 as a minimum (for empty or unparseable inputs).
    """
    from datetime import datetime as dt

    total_seconds = 0
    for seg in segments:
        start, end = segment_parse(seg)
        if start is not None and end is not None:
            dt_start = dt(2000, 1, 1, start.hour, start.minute, start.second)
            dt_end = dt(2000, 1, 1, end.hour, end.minute, end.second)
            total_seconds += (dt_end - dt_start).total_seconds()
    return max(1, int(total_seconds / 60))


def level_avg(levels: list[str]) -> float:
    """Compute average engagement level from a list of level strings.

    Maps: high=1.0, medium=0.5, low=0.25. Unknown values use 0.5.
    Returns rounded to 2 decimal places.
    """
    if not levels:
        return 0.5
    values = [LEVEL_VALUES.get(level, 0.5) for level in levels]
    return round(sum(values) / len(values), 2)
