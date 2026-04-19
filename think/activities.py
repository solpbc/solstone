# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Activities system for tracking common activity types per facet.

Activities provide a consistent vocabulary for tagging time segments,
screen observations, and extracted events across the journal.

Also provides utilities for activity records — completed activity spans
stored as facets/{facet}/activities/{day}.jsonl.
"""

import difflib
import fcntl
import json
import logging
import os
import random
import re
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from think.utils import get_journal, segment_parse

logger = logging.getLogger(__name__)
ANTICIPATION_FUZZY_THRESHOLD = 0.85

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
        "always_on": True,
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
        "always_on": True,
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
        "always_on": True,
        "instructions": (
            "Levels: high=active conversation, medium=reading messages,"
            " low=chat app visible but idle."
            " Detect via: chat app UI, message notifications, typing indicators."
        ),
    },
    {
        "id": "ai_conversation",
        "name": "AI Conversation",
        "description": "Conversations with AI assistants like ChatGPT, Claude, and Gemini",
        "icon": "🤖",
        "instructions": (
            "Levels: high=actively prompting and reading responses,"
            " medium=reviewing AI output or refining prompts,"
            " low=AI chat open but idle."
            " Detect via: AI assistant interfaces (ChatGPT, Claude, Gemini),"
            " imported AI conversation transcripts, prompt-response patterns."
            " Do not confuse with messaging — AI conversation involves"
            " a human interacting with an AI model, not person-to-person chat."
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
        "description": "Books, PDFs, articles, highlights, and documentation",
        "icon": "📖",
        "instructions": (
            "Levels: high=focused reading, medium=skimming content,"
            " low=document open but attention elsewhere."
            " Detect via: PDF viewers, article pages, documentation sites,"
            " reading apps, imported book highlights and annotations."
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
        "id": "planning",
        "name": "Planning",
        "description": "Scheduling, calendar management, meeting preparation, and agenda setting",
        "icon": "📋",
        "instructions": (
            "Levels: high=actively scheduling or preparing agendas,"
            " medium=reviewing calendar or event details,"
            " low=calendar visible but not being interacted with."
            " Detect via: calendar apps, scheduling interfaces, event creation,"
            " imported calendar events, meeting invitations, agenda drafting."
            " Use for scheduling and preparation work."
            " Do not confuse with meeting — planning is the preparation,"
            " meeting is the actual synchronous interaction."
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
            " task boards."
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

    def modify_fn(_existing: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [dict(activity) for activity in activities]

    locked_modify(path, modify_fn, create_if_missing=True)


def get_facet_activities(facet: str) -> list[dict[str, Any]]:
    """Load activities attached to a facet.

    Returns activities explicitly attached to the facet plus any default
    activities marked ``always_on``. Always-on activities are auto-included
    even if the facet's ``activities.jsonl`` does not list them.

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
        - always_on: True if auto-included from defaults
    """
    # Build lookup for defaults
    defaults_by_id = {a["id"]: a for a in DEFAULT_ACTIVITIES}

    # Load facet-specific activities
    facet_activities = _load_activities_jsonl(facet)

    # If no explicit activities configured, use all defaults as the vocabulary
    if not facet_activities:
        result = []
        for default in DEFAULT_ACTIVITIES:
            activity = dict(default)
            activity["custom"] = False
            activity.setdefault("priority", "normal")
            result.append(activity)
        return result

    seen_ids: set[str] = set()
    result = []
    for fa in facet_activities:
        activity_id = fa.get("id")
        if not activity_id:
            continue

        seen_ids.add(activity_id)

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

    # Auto-include always-on defaults not already attached
    for default in DEFAULT_ACTIVITIES:
        if default.get("always_on") and default["id"] not in seen_ids:
            activity = dict(default)
            activity["custom"] = False
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
    # Check if already explicitly attached (in JSONL, not just defaults)
    existing_raw = _load_activities_jsonl(facet)
    for entry in existing_raw:
        if entry.get("id") == activity_id:
            # Already attached - return full activity with defaults merged
            return get_activity_by_id(facet, activity_id) or entry

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

    state_path = seg_dir / "talents" / facet / "activity_state.json"
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


def _read_jsonl_records(path: Path) -> list[dict[str, Any]]:
    """Load JSONL entries from *path*, skipping malformed lines."""
    if not path.exists():
        return []

    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_num, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning(
                    "Skipping malformed line %d in %s: %s", line_num, path, exc
                )
                continue
            if isinstance(data, dict):
                records.append(data)
    return records


def _write_jsonl_records(path: Path, records: list[dict[str, Any]]) -> None:
    """Atomically write JSONL entries to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        os.replace(tmp_name, path)
    except BaseException:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise


def _fallback_activity_title(record: dict[str, Any]) -> str:
    """Return the best available title for an activity record."""
    title = str(record.get("title") or "").strip()
    if title:
        return title

    description = str(record.get("description") or "").strip()
    if description:
        return description

    activity = str(record.get("activity") or record.get("id") or "").strip()
    if activity:
        return activity.replace("_", " ").title()

    return "Untitled activity"


def _normalize_activity_record(record: dict[str, Any]) -> dict[str, Any]:
    """Return a normalized activity record copy with schema defaults."""
    normalized = dict(record)
    normalized["title"] = _fallback_activity_title(record)
    normalized["details"] = str(record.get("details") or "")
    normalized["hidden"] = bool(record.get("hidden", False))

    edits = record.get("edits")
    normalized["edits"] = (
        [dict(edit) for edit in edits if isinstance(edit, dict)]
        if isinstance(edits, list)
        else []
    )
    return normalized


def locked_modify(
    path: Path,
    modify_fn: Any,
    *,
    create_if_missing: bool = False,
    max_retries: int = 3,
) -> None:
    """Perform a locked load-modify-save cycle on a JSONL file."""
    lock_path = path.parent / f"{path.name}.lock"

    last_error: OSError | None = None
    for attempt in range(max_retries):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(lock_path, "w", encoding="utf-8") as lock_file:
                fcntl.flock(lock_file, fcntl.LOCK_EX)
                try:
                    existed = path.exists()
                    if not existed and not create_if_missing:
                        raise FileNotFoundError(path)
                    current = _read_jsonl_records(path) if existed else []
                    updated = modify_fn([dict(item) for item in current])
                    if not isinstance(updated, list):
                        raise TypeError("modify_fn must return list[dict]")
                    if not existed and not updated:
                        return
                    if existed and updated == current:
                        return
                    _write_jsonl_records(path, updated)
                finally:
                    fcntl.flock(lock_file, fcntl.LOCK_UN)
            return
        except (FileNotFoundError, TypeError, ValueError):
            raise
        except OSError as exc:
            last_error = exc
            if attempt < max_retries - 1:
                time.sleep(random.uniform(0.05, 0.3) * (attempt + 1))

    if last_error is not None:
        raise last_error


def append_edit(
    record: dict[str, Any],
    *,
    actor: str,
    fields: list[str],
    note: str | None,
    payload: dict[str, Any]
    | None = None,  # Additive: Ledger close writes a `ledger_close` sub-dict alongside the audit edit; keep spread so edit readers see a flat entry.
) -> dict[str, Any]:
    """Append an edit entry to an activity record and return the record."""
    normalized = _normalize_activity_record(record)
    edits = [dict(edit) for edit in normalized.get("edits", [])]
    edit_entry: dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "actor": actor,
        "fields": list(fields),
        "note": note,
    }
    if payload is not None:
        if not isinstance(payload, dict):
            raise TypeError("payload must be dict[str, Any] when provided")
        collision_keys = sorted(set(payload) & set(edit_entry))
        if collision_keys:
            raise ValueError(
                "payload cannot overwrite canonical edit fields: "
                + ", ".join(collision_keys)
            )
        edit_entry = {**edit_entry, **payload}
    edits.append(edit_entry)
    normalized["edits"] = edits
    return normalized


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
    from think.talent import get_output_name

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


def load_activity_records(
    facet: str, day: str, *, include_hidden: bool = False
) -> list[dict[str, Any]]:
    """Load activity records for a facet and day.

    Returns list of record dicts, empty list if file doesn't exist.
    """
    path = _get_records_path(facet, day)
    records = [
        _normalize_activity_record(record) for record in _read_jsonl_records(path)
    ]
    if include_hidden:
        return records
    return [record for record in records if not record.get("hidden", False)]


def make_anticipation_id(
    activity_type: str,
    start: str | None,
    target_date: str,
) -> str:
    """Build the stable ID used for schedule-generated anticipated records."""
    activity_key = str(activity_type or "").strip()
    if not activity_key:
        raise ValueError("activity_type must be non-empty")

    try:
        parsed_target = datetime.strptime(target_date, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError("target_date must match YYYY-MM-DD") from exc

    if start is None:
        start_key = "000000"
    else:
        if not re.fullmatch(r"\d{2}:\d{2}:\d{2}", start):
            raise ValueError("start must match HH:MM:SS")
        start_key = start.replace(":", "")

    return f"anticipated_{activity_key}_{start_key}_{parsed_target.strftime('%m%d')}"


def dedup_anticipation(
    facet: str,
    target_day: str,
    new_record: dict[str, Any],
    *,
    threshold: float = ANTICIPATION_FUZZY_THRESHOLD,
) -> tuple[bool, list[str]]:
    """Check a new anticipated record for collisions and fuzzy supersedes."""

    new_id = str(new_record.get("id") or "").strip()
    if not new_id:
        raise ValueError("new_record.id is required")

    def _normalize_title(value: Any) -> str:
        return " ".join(str(value or "").lower().split())

    new_title = _normalize_title(new_record.get("title"))
    superseded_ids: list[str] = []

    for record in load_activity_records(facet, target_day, include_hidden=False):
        if record.get("source") != "anticipated":
            continue

        existing_id = str(record.get("id") or "").strip()
        if existing_id == new_id:
            return False, []

        existing_title = _normalize_title(record.get("title"))
        ratio = difflib.SequenceMatcher(None, new_title, existing_title).ratio()
        if ratio >= threshold:
            superseded_ids.append(existing_id)

    return True, superseded_ids


def load_record_ids(facet: str, day: str) -> set[str]:
    """Load just the IDs of existing activity records for idempotency checks."""
    return {
        r["id"]
        for r in load_activity_records(facet, day, include_hidden=True)
        if "id" in r
    }


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
    del _checked  # retained for compatibility; duplicate checks now happen under lock
    path = _get_records_path(facet, day)
    written = False

    def modify_fn(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        nonlocal written
        record_id = record.get("id")
        if record_id and any(item.get("id") == record_id for item in records):
            return records
        written = True
        return records + [_normalize_activity_record(record)]

    locked_modify(path, modify_fn, create_if_missing=True)
    return written


def update_record_fields(
    facet: str, day: str, record_id: str, fields: dict[str, Any]
) -> bool:
    """Update fields on an existing activity record.

    Rewrites the JSONL file atomically (write temp + rename) with the updated
    fields for the matching record.

    Returns True if record was found and updated, False otherwise.
    """
    path = _get_records_path(facet, day)
    updated = False

    try:

        def modify_fn(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
            nonlocal updated
            new_records: list[dict[str, Any]] = []
            for record in records:
                if record.get("id") == record_id:
                    merged = dict(record)
                    merged.update(fields)
                    new_records.append(_normalize_activity_record(merged))
                    updated = True
                else:
                    new_records.append(record)
            return new_records

        locked_modify(path, modify_fn)
    except FileNotFoundError:
        return False

    return updated


def update_record_description(
    facet: str,
    day: str,
    record_id: str,
    description: str,
    *,
    title: str | None = None,
    details: str | None = None,
) -> bool:
    """Update the description of an existing activity record."""
    patch: dict[str, Any] = {"description": description}
    current = get_activity_record(facet, day, record_id)
    if title is not None:
        patch["title"] = title
    elif current is not None:
        current_title = str(current.get("title") or "").strip()
        current_description = str(current.get("description") or "").strip()
        if not current_title or current_title == current_description:
            patch["title"] = description
    if details is not None:
        patch["details"] = details
    return update_record_fields(facet, day, record_id, patch)


def get_activity_record(facet: str, day: str, record_id: str) -> dict[str, Any] | None:
    """Return one activity record by ID, including hidden records."""
    for record in load_activity_records(facet, day, include_hidden=True):
        if record.get("id") == record_id:
            return record
    return None


def update_activity_record(
    facet: str,
    day: str,
    record_id: str,
    patch: dict[str, Any],
    *,
    actor: str,
    note: str,
) -> dict[str, Any] | None:
    """Apply a shallow patch to an activity record and append one edit."""
    allowed_fields = {"title", "description", "details"}
    if not patch:
        raise ValueError("patch cannot be empty")

    disallowed = sorted(set(patch) - allowed_fields)
    if disallowed:
        raise ValueError(f"patch contains disallowed fields: {', '.join(disallowed)}")

    updated_record: dict[str, Any] | None = None

    def modify_fn(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        nonlocal updated_record
        new_records: list[dict[str, Any]] = []
        for record in records:
            if record.get("id") == record_id:
                merged = _normalize_activity_record({**record, **patch})
                merged = append_edit(
                    merged,
                    actor=actor,
                    fields=list(patch.keys()),
                    note=note,
                )
                updated_record = merged
                new_records.append(merged)
            else:
                new_records.append(record)
        return new_records

    try:
        locked_modify(_get_records_path(facet, day), modify_fn)
    except FileNotFoundError:
        return None

    return updated_record


def merge_story_fields(
    facet: str,
    day: str,
    record_id: str,
    *,
    story: dict,
    commitments: list[dict],
    closures: list[dict],
    decisions: list[dict],
    actor: str,
    note: str | None = None,
) -> bool:
    """Replace story-derived fields on an activity record and append one edit."""
    updated = False
    path = _get_records_path(facet, day)

    def modify_fn(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        nonlocal updated
        new_records: list[dict[str, Any]] = []
        for record in records:
            if record.get("id") == record_id:
                merged = _normalize_activity_record(record)
                merged["story"] = dict(story)
                merged["commitments"] = [dict(entry) for entry in commitments]
                merged["closures"] = [dict(entry) for entry in closures]
                merged["decisions"] = [dict(entry) for entry in decisions]
                merged = append_edit(
                    merged,
                    actor=actor,
                    fields=["story", "commitments", "closures", "decisions"],
                    note=note,
                )
                new_records.append(merged)
                updated = True
            else:
                new_records.append(record)
        return new_records

    try:
        locked_modify(path, modify_fn, create_if_missing=False)
    except FileNotFoundError:
        logger.warning("story hook: activity record not found: %s", record_id)
        return False

    if not updated:
        logger.warning("story hook: activity record not found: %s", record_id)
    return updated


def _set_activity_hidden_state(
    facet: str,
    day: str,
    record_id: str,
    *,
    hidden: bool,
    actor: str,
    reason: str | None,
) -> dict[str, Any] | None:
    updated_record: dict[str, Any] | None = None

    def modify_fn(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        nonlocal updated_record
        new_records: list[dict[str, Any]] = []
        for record in records:
            if record.get("id") != record_id:
                new_records.append(record)
                continue

            normalized = _normalize_activity_record(record)
            if normalized.get("hidden", False) == hidden:
                updated_record = normalized
                new_records.append(normalized)
                continue

            normalized["hidden"] = hidden
            normalized = append_edit(
                normalized,
                actor=actor,
                fields=["hidden"],
                note=reason or ("muted" if hidden else "unmuted"),
            )
            updated_record = normalized
            new_records.append(normalized)
        return new_records

    try:
        locked_modify(_get_records_path(facet, day), modify_fn)
    except FileNotFoundError:
        return None

    return updated_record


def mute_activity_record(
    facet: str,
    day: str,
    record_id: str,
    *,
    actor: str,
    reason: str | None,
) -> dict[str, Any] | None:
    """Hide an activity record without deleting it."""
    return _set_activity_hidden_state(
        facet,
        day,
        record_id,
        hidden=True,
        actor=actor,
        reason=reason,
    )


def unmute_activity_record(
    facet: str,
    day: str,
    record_id: str,
    *,
    actor: str,
    reason: str | None,
) -> dict[str, Any] | None:
    """Restore a previously hidden activity record."""
    return _set_activity_hidden_state(
        facet,
        day,
        record_id,
        hidden=False,
        actor=actor,
        reason=reason,
    )


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


def _extract_activity_header(file_path: str | os.PathLike[str] | None) -> str:
    """Build a formatter header from an activities file path."""
    if not file_path:
        return "# Activities"

    path = Path(file_path)
    parts = path.parts
    try:
        facet_idx = parts.index("facets")
        facet_name = parts[facet_idx + 1]
    except (ValueError, IndexError):
        facet_name = "unknown"

    stem = path.stem
    if stem.isdigit() and len(stem) == 8:
        return f"# Activities: {facet_name} ({stem[:4]}-{stem[4:6]}-{stem[6:8]})"
    return f"# Activities: {facet_name}"


def _activity_time_range(segments: list[str]) -> str | None:
    """Return a compact HH:MM-HH:MM label for a list of segment keys."""
    if not segments:
        return None

    start_time, _ = segment_parse(segments[0])
    _, end_time = segment_parse(segments[-1])
    if start_time is None or end_time is None:
        return None

    return f"{start_time.strftime('%H:%M')}-{end_time.strftime('%H:%M')}"


def _format_participation(record: dict[str, Any]) -> str | None:
    """Format participation names for display."""
    participation = record.get("participation")
    if not isinstance(participation, list) or not participation:
        return None

    names = []
    for entry in participation:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name") or entry.get("entity_id") or "").strip()
        if name:
            names.append(name)

    if not names:
        return None
    return ", ".join(names)


def format_activities(
    entries: list[dict],
    context: dict | None = None,
) -> tuple[list[dict], dict]:
    """Format activity JSONL entries into markdown chunks."""
    ctx = context or {}
    meta: dict[str, Any] = {
        "header": _extract_activity_header(ctx.get("file_path")),
        "indexer": {"agent": "activity"},
    }
    chunks: list[dict[str, Any]] = []

    for entry in entries:
        if not isinstance(entry, dict):
            continue

        record = _normalize_activity_record(entry)
        lines = [f"### {_fallback_activity_title(record)}"]

        activity_type = str(record.get("activity") or record.get("id") or "").strip()
        if activity_type:
            lines.append(f"- Activity: {activity_type}")

        facet = str(record.get("facet") or "").strip()
        if facet:
            lines.append(f"- Facet: {facet}")

        day = str(record.get("day") or "").strip()
        if day:
            lines.append(f"- Day: {day}")

        time_range = _activity_time_range(record.get("segments", []))
        if time_range:
            lines.append(f"- Time: {time_range}")

        if "level_avg" in record:
            lines.append(f"- Level: {record['level_avg']}")

        description = str(record.get("description") or "").strip()
        if description:
            lines.append(f"- Description: {description}")

        details = str(record.get("details") or "").strip()
        if details:
            lines.append(f"- Details: {details}")

        participants = _format_participation(record)
        if participants:
            lines.append(f"- Participation: {participants}")

        story = record.get("story")
        if isinstance(story, dict):
            body = story.get("body")
            if isinstance(body, str) and body.strip():
                lines.append("")
                lines.append(body.strip())

            topics = story.get("topics")
            if isinstance(topics, list):
                topic_values = [
                    topic.strip()
                    for topic in topics
                    if isinstance(topic, str) and topic.strip()
                ]
                if topic_values:
                    lines.append(f"Topics: {', '.join(topic_values)}")

        if record.get("hidden", False):
            lines.append("- Hidden: yes")

        chunks.append(
            {
                "timestamp": int(record.get("created_at", 0) or 0),
                "markdown": "\n".join(lines),
                "source": record,
            }
        )

    return chunks, meta
