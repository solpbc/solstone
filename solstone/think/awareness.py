# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Awareness system — solstone's self-awareness about the user.

Tracks the system's evolving understanding: capture state, identity
persistence, imports, and awareness signals. Two-layer storage:

- ``awareness/current.json`` — materialized current state for fast reads
- ``awareness/YYYYMMDD.jsonl`` — append-only daily log of everything noticed

Designed to extend to cogitate (proactive agents),
learned preferences, and cross-session agent memory.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)
_LEGACY_AGENT_FIELD = "talent"


def _awareness_dir() -> Path:
    """Return path to the awareness directory, creating it if needed."""
    from solstone.think.utils import get_journal

    d = Path(get_journal()) / "awareness"
    d.mkdir(exist_ok=True)
    return d


def _now_ts() -> int:
    """Return current time in milliseconds."""
    return int(time.time() * 1000)


def _today() -> str:
    """Return today's date as YYYYMMDD."""
    return datetime.now().strftime("%Y%m%d")


def _now_iso() -> str:
    """Return current time as compact ISO string."""
    return datetime.now().strftime("%Y%m%dT%H:%M:%S")


def get_current() -> dict[str, Any]:
    """Read the current awareness state from ``awareness/current.json``.

    Returns an empty dict if no state exists yet.
    """
    path = _awareness_dir() / "current.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        logger.warning("Failed to read awareness/current.json, returning empty")
        return {}


def _write_current(state: dict[str, Any]) -> None:
    """Atomically write the current awareness state."""
    path = _awareness_dir() / "current.json"
    # Write to temp file then rename for atomicity
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(state, f, indent=2)
            f.write("\n")
        os.replace(tmp, path)
    except Exception:
        # Clean up temp file on failure
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def update_state(section: str, data: dict[str, Any]) -> dict[str, Any]:
    """Update a section of the current awareness state.

    Merges ``data`` into the named section (creates if missing).
    Returns the updated section.
    """
    state = get_current()
    existing = state.get(section, {})
    existing.update(data)
    state[section] = existing
    _write_current(state)
    return existing


def append_log(
    kind: str,
    *,
    key: str | None = None,
    message: str | None = None,
    data: dict[str, Any] | None = None,
    day: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Append an entry to the daily awareness log.

    Parameters
    ----------
    kind : str
        Entry type: "state", "observation", "nudge", "interaction", "preference"
    key : str, optional
        Dotted key for state entries (e.g., "onboarding.started")
    message : str, optional
        Human-readable message
    data : dict, optional
        Structured data payload
    day : str, optional
        Override day (defaults to today)
    **extra
        Additional fields merged into the entry

    Returns
    -------
    dict
        The entry that was written
    """
    entry: dict[str, Any] = {"ts": _now_ts(), "kind": kind}
    if key:
        entry["key"] = key
    if message:
        entry["message"] = message
    if data:
        entry["data"] = data
    entry.update(extra)

    log_day = day or _today()
    log_path = _awareness_dir() / f"{log_day}.jsonl"
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")

    return entry


def read_log(day: str | None = None) -> list[dict[str, Any]]:
    """Read all entries from a daily awareness log.

    Parameters
    ----------
    day : str, optional
        Day in YYYYMMDD format (defaults to today)

    Returns
    -------
    list[dict]
        Entries in chronological order, empty list if no log exists
    """
    log_day = day or _today()
    log_path = _awareness_dir() / f"{log_day}.jsonl"
    if not log_path.exists():
        return []
    entries = []
    for line in log_path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("Skipping malformed awareness log entry")
    return entries


# --- Import tracking convenience functions ---


def _ensure_imports_section() -> dict[str, Any]:
    """Ensure the imports section exists in current state, return it."""
    state = get_current()
    if "imports" not in state:
        state["imports"] = {
            "has_imported": False,
            "import_count": 0,
            "sources_used": [],
            "offer_declined": None,
            "last_nudge": None,
        }
        _write_current(state)
    return state["imports"]


def get_imports() -> dict[str, Any]:
    """Return the current import tracking state, or defaults if none."""
    state = get_current()
    return state.get(
        "imports",
        {
            "has_imported": False,
            "import_count": 0,
            "sources_used": [],
            "offer_declined": None,
            "last_nudge": None,
        },
    )


def _recent_chat_exchanges(limit: int = 10000) -> list[dict[str, Any]]:
    """Return owner-visible chat responses from chat stream history."""
    from solstone.think.utils import day_dirs

    try:
        days = day_dirs()
    except Exception:
        return []

    exchanges: list[dict[str, Any]] = []
    for day_name in sorted(days):
        day_path = Path(days[day_name])
        chat_root = day_path / "chat"
        if not chat_root.exists():
            continue
        for segment_dir in sorted(chat_root.iterdir()):
            if not segment_dir.is_dir():
                continue
            chat_path = segment_dir / "chat.jsonl"
            if not chat_path.exists():
                continue
            try:
                for line in chat_path.read_text().splitlines():
                    if not line.strip():
                        continue
                    event = json.loads(line)
                    if event.get("kind") != "sol_message":
                        continue
                    exchanges.append(
                        {
                            "talent": "chat",
                            "agent_response": event.get("text", ""),
                        }
                    )
            except (OSError, json.JSONDecodeError):
                logger.warning("Skipping malformed chat stream file: %s", chat_path)
    if limit <= 0:
        return []
    return exchanges[-limit:]


def compute_thickness() -> dict[str, Any]:
    """Compute journal thickness signals for naming ceremony readiness.

    Returns a dict with five signals and a composite ``ready`` boolean:

    - ``entity_depth``: count of entities with observation_depth >= 2
    - ``conversation_count``: conversation exchanges excluding legacy onboarding
    - ``recall_success``: exchanges where an entity name appears in agent_response
    - ``facet_count``: number of enabled (non-muted) facets
    - ``journal_days``: number of day directories with at least one segment
    - ``ready``: True when the naming ceremony should trigger
    """
    from solstone.think.facets import get_enabled_facets
    from solstone.think.indexer.journal import get_entity_strength
    from solstone.think.utils import day_dirs, iter_segments

    try:
        entities = get_entity_strength(limit=10000)
    except Exception:
        entities = []
    entity_depth = sum(1 for e in entities if e.get("observation_depth", 0) >= 2)

    try:
        exchanges = _recent_chat_exchanges(limit=10000)
    except Exception:
        exchanges = []
    non_onboarding = [
        ex
        for ex in exchanges
        if (ex.get("talent") or ex.get(_LEGACY_AGENT_FIELD, "")) != "onboarding"
    ]
    conversation_count = len(non_onboarding)

    entity_names = [e["entity_name"].lower() for e in entities if e.get("entity_name")]
    recall_success = 0
    for ex in non_onboarding:
        resp = (ex.get("agent_response") or "").lower()
        if resp and any(name in resp for name in entity_names):
            recall_success += 1

    try:
        facet_count = len(get_enabled_facets())
    except Exception:
        facet_count = 0

    try:
        days = day_dirs()
    except Exception:
        days = {}
    journal_days = 0
    for _day_name, day_path in days.items():
        try:
            if iter_segments(day_path):
                journal_days += 1
        except Exception:
            pass

    ready = (
        entity_depth >= 10 and conversation_count >= 5 and recall_success >= 1
    ) and (facet_count >= 2 or journal_days >= 3)

    return {
        "entity_depth": entity_depth,
        "conversation_count": conversation_count,
        "recall_success": recall_success,
        "facet_count": facet_count,
        "journal_days": journal_days,
        "ready": ready,
    }


def owner_detection_ready() -> dict[str, Any]:
    """Check if owner voice detection should be surfaced to the user.

    Same pattern as ``compute_thickness()`` for the naming ceremony.
    Returns a dict with a ``ready`` boolean and contextual fields.

    Checks in order:
    1. Owner centroid already exists → not ready
    2. Recent rejection within 14 days → not ready (cooldown)
    3. Calls ``detect_owner_candidate()`` → ready if positive recommendation
    """
    from solstone.apps.speakers.owner import detect_owner_candidate, load_owner_centroid

    if load_owner_centroid() is not None:
        return {"ready": False, "reason": "centroid_exists"}

    voiceprint = get_current().get("voiceprint", {})
    rejected_at = voiceprint.get("rejected_at")
    if rejected_at:
        try:
            rejection_time = datetime.fromisoformat(rejected_at)
            days_since = (datetime.now() - rejection_time).days
            if days_since < 14:
                return {
                    "ready": False,
                    "reason": "cooldown",
                    "days_remaining": 14 - days_since,
                }
        except (ValueError, TypeError):
            pass

    result = detect_owner_candidate()
    if result.get("recommendation") == "ready":
        return {
            "ready": True,
            "reason": "candidate_found",
            "cluster_size": result.get("cluster_size"),
            "streams_represented": result.get("streams_represented"),
            "samples": result.get("samples", []),
        }

    return {
        "ready": False,
        "reason": result.get("recommendation", result.get("status", "unknown")),
    }


def record_import(
    source_type: str,
    source_display: str | None = None,
    entries_written: int = 0,
) -> dict[str, Any]:
    """Record a completed import.

    Parameters
    ----------
    source_type : str
        Import source type (e.g., "chatgpt", "ics", "claude")
    source_display : str, optional
        Human-readable source display name
    entries_written : int
        Number of entries imported

    Returns
    -------
    dict
        The updated imports state
    """
    _ensure_imports_section()
    imports = get_imports()
    sources = imports.get("sources_used", [])
    if source_type not in sources:
        sources.append(source_type)
    update_data: dict[str, Any] = {
        "has_imported": True,
        "import_count": imports.get("import_count", 0) + 1,
        "sources_used": sources,
    }
    if source_display is not None:
        summary = (
            f"{entries_written} {source_display}" if entries_written else source_display
        )
        update_data["last_completed"] = _now_iso()
        update_data["last_result_summary"] = summary
    state = update_state("imports", update_data)
    append_log("state", key="imports.completed", data={"source_type": source_type})
    return state


def record_import_offer_declined() -> dict[str, Any]:
    """Record that the user declined an import offer.

    Returns
    -------
    dict
        The updated imports state
    """
    _ensure_imports_section()
    state = update_state(
        "imports",
        {"offer_declined": _now_iso()},
    )
    append_log("state", key="imports.offer_declined")
    return state


def record_import_nudge() -> dict[str, Any]:
    """Record that triage nudged the user about imports.

    Returns
    -------
    dict
        The updated imports state
    """
    _ensure_imports_section()
    state = update_state(
        "imports",
        {"last_nudge": _now_iso()},
    )
    append_log("state", key="imports.nudge_sent")
    return state
