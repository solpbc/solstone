# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Awareness system — solstone's self-awareness about the user.

Tracks the system's evolving understanding: onboarding state, observations,
nudges, and interactions. Two-layer storage:

- ``awareness/current.json`` — materialized current state for fast reads
- ``awareness/YYYYMMDD.jsonl`` — append-only daily log of everything noticed

Designed to extend beyond onboarding to cogitate (proactive agents),
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


def _awareness_dir() -> Path:
    """Return path to the awareness directory, creating it if needed."""
    from think.utils import get_journal

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


# --- Onboarding convenience functions ---


def get_onboarding() -> dict[str, Any]:
    """Return the current onboarding state, or empty dict if none."""
    return get_current().get("onboarding", {})


def start_onboarding(path: str) -> dict[str, Any]:
    """Record onboarding path selection.

    Parameters
    ----------
    path : str
        "a" for passive observation, "b" for conversational interview

    Returns
    -------
    dict
        The updated onboarding state
    """
    status = "observing" if path == "a" else "interviewing"
    state = update_state(
        "onboarding",
        {
            "path": path,
            "status": status,
            "started": _now_iso(),
            "observation_count": 0,
            "nudges_sent": 0,
        },
    )
    append_log(
        "state", key="onboarding.started", data={"path": path, "status": status}
    )
    return state


def skip_onboarding() -> dict[str, Any]:
    """Record onboarding skip."""
    state = update_state(
        "onboarding",
        {
            "status": "skipped",
            "started": _now_iso(),
        },
    )
    append_log("state", key="onboarding.skipped")
    return state


def complete_onboarding() -> dict[str, Any]:
    """Record onboarding completion."""
    state = update_state(
        "onboarding",
        {
            "status": "complete",
        },
    )
    append_log("state", key="onboarding.complete")
    return state
