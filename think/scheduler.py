# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Clock-aligned task scheduler for the supervisor.

Reads schedule definitions from config/schedules.json and submits tasks
via Callosum at hour and day boundaries. State (last-run times) persists
to health/scheduler.json across restarts.

Runtime functions (init, check) are used by the supervisor.
The main() function provides the ``sol schedule`` CLI.
"""

from __future__ import annotations

import argparse
import json
import logging
import tempfile
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from think.utils import get_journal, now_ms, setup_cli

logger = logging.getLogger(__name__)

# Valid schedule intervals
INTERVALS = {"hourly", "daily"}

# ---------------------------------------------------------------------------
# Module state (populated by init(), used by check())
# ---------------------------------------------------------------------------
_entries: dict[str, dict[str, Any]] = {}
_state: dict[str, dict[str, Any]] = {}
_callosum: Any = None  # CallosumConnection
_last_hour: datetime | None = None
_last_day: date | None = None


# ---------------------------------------------------------------------------
# Config + state I/O
# ---------------------------------------------------------------------------


def load_config() -> dict[str, dict[str, Any]]:
    """Read config/schedules.json and return validated entries."""
    config_path = Path(get_journal()) / "config" / "schedules.json"
    if not config_path.exists():
        return {}

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load schedules config: %s", exc)
        return {}

    if not isinstance(raw, dict):
        logger.warning(
            "schedules.json must be a JSON object, got %s", type(raw).__name__
        )
        return {}

    entries: dict[str, dict[str, Any]] = {}
    for name, entry in raw.items():
        if not isinstance(entry, dict):
            logger.warning("Schedule '%s': expected object, skipping", name)
            continue

        cmd = entry.get("cmd")
        if not cmd or not isinstance(cmd, list):
            logger.warning("Schedule '%s': missing or invalid 'cmd', skipping", name)
            continue

        every = entry.get("every")
        if every not in INTERVALS:
            logger.warning(
                "Schedule '%s': unknown interval '%s' (expected %s), skipping",
                name,
                every,
                "/".join(sorted(INTERVALS)),
            )
            continue

        if not entry.get("enabled", True):
            continue

        entries[name] = {"cmd": cmd, "every": every}

    return entries


def load_state() -> dict[str, dict[str, Any]]:
    """Read health/scheduler.json."""
    state_path = Path(get_journal()) / "health" / "scheduler.json"
    if not state_path.exists():
        return {}

    try:
        with open(state_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load scheduler state: %s", exc)
        return {}


def save_state() -> None:
    """Persist _state to health/scheduler.json atomically."""
    health_dir = Path(get_journal()) / "health"
    health_dir.mkdir(parents=True, exist_ok=True)
    state_path = health_dir / "scheduler.json"

    fd, tmp_path = tempfile.mkstemp(dir=health_dir, suffix=".tmp", prefix=".scheduler_")
    tmp_file = Path(tmp_path)
    try:
        with open(fd, "w", encoding="utf-8") as f:
            json.dump(_state, f, indent=2)
        tmp_file.replace(state_path)
    except BaseException:
        tmp_file.unlink(missing_ok=True)
        raise


# ---------------------------------------------------------------------------
# Boundary helpers
# ---------------------------------------------------------------------------


def _hour_mark(dt: datetime) -> datetime:
    """Truncate datetime to the start of its hour."""
    return dt.replace(minute=0, second=0, microsecond=0)


def _is_due(entry: dict, state_entry: dict | None, now: datetime) -> bool:
    """Check if an entry is due based on its interval and last_run."""
    last_run = (state_entry or {}).get("last_run")
    if last_run is None:
        return True

    try:
        last_dt = datetime.fromtimestamp(last_run)
    except (OSError, ValueError):
        return True

    every = entry["every"]
    if every == "hourly":
        return last_dt < _hour_mark(now)
    if every == "daily":
        return last_dt.date() < now.date()
    return False


# ---------------------------------------------------------------------------
# Runtime API (called by supervisor)
# ---------------------------------------------------------------------------


def init(callosum: Any) -> None:
    """Initialize scheduler with a Callosum connection. Load config and state."""
    global _entries, _state, _callosum, _last_hour, _last_day

    _callosum = callosum
    _entries = load_config()
    _state = load_state()

    now = datetime.now()
    _last_hour = _hour_mark(now)
    _last_day = now.date()

    if _entries:
        logger.info(
            "Scheduler initialized with %d schedule(s): %s",
            len(_entries),
            ", ".join(sorted(_entries)),
        )
    else:
        logger.info("Scheduler initialized (no schedules configured)")


def check() -> None:
    """Check for clock boundaries and submit due tasks.

    Called each supervisor tick (~1s). Does nothing unless an hour or day
    boundary has been crossed since the last check.
    """
    global _entries, _last_hour, _last_day

    if _last_hour is None:
        return

    now = datetime.now()
    current_hour = _hour_mark(now)
    current_day = now.date()

    hour_changed = current_hour != _last_hour
    day_changed = current_day != _last_day

    if not hour_changed and not day_changed:
        return

    # Boundary crossed — reload config for freshest definitions
    _entries = load_config()
    _last_hour = current_hour
    _last_day = current_day

    if not _entries:
        return

    submitted = False
    for name, entry in _entries.items():
        every = entry["every"]

        # Only check entries matching the boundary that changed
        if every == "hourly" and not hour_changed:
            continue
        if every == "daily" and not day_changed:
            continue

        if not _is_due(entry, _state.get(name), now):
            continue

        ref = f"sched:{name}:{now_ms()}"
        cmd = entry["cmd"]

        if _callosum:
            ok = _callosum.emit("supervisor", "request", cmd=cmd, ref=ref)
            if ok:
                logger.info(
                    "Scheduled task submitted: %s → %s (ref=%s)",
                    name,
                    " ".join(cmd),
                    ref,
                )
                _state.setdefault(name, {})["last_run"] = time.time()
                submitted = True
            else:
                logger.warning(
                    "Failed to emit scheduled task %s (callosum not connected)", name
                )
        else:
            logger.warning("No callosum connection for scheduled task: %s", name)

    if submitted:
        try:
            save_state()
        except Exception as exc:
            logger.warning("Failed to save scheduler state: %s", exc)


def collect_status() -> list[dict[str, Any]]:
    """Return schedule status for supervisor.status events."""
    now = datetime.now()
    result = []
    for name, entry in _entries.items():
        state_entry = _state.get(name)
        last_run = (state_entry or {}).get("last_run")
        result.append(
            {
                "name": name,
                "every": entry["every"],
                "last_run": last_run,
                "due": _is_due(entry, state_entry, now),
            }
        )
    return result


# ---------------------------------------------------------------------------
# CLI: sol schedule
# ---------------------------------------------------------------------------


def _format_timestamp(epoch: float | None) -> str:
    """Format an epoch timestamp for display."""
    if epoch is None:
        return "never"
    try:
        return datetime.fromtimestamp(epoch).strftime("%Y-%m-%d %H:%M")
    except (OSError, ValueError):
        return "invalid"


def _format_next_due(entry: dict, state_entry: dict | None, now: datetime) -> str:
    """Format the next due time for display."""
    if _is_due(entry, state_entry, now):
        return "now"

    every = entry["every"]
    if every == "hourly":
        nxt = _hour_mark(now) + timedelta(hours=1)
        return nxt.strftime("%H:%M")
    if every == "daily":
        return "midnight"
    return "?"


def main() -> None:
    """CLI entry point for sol schedule."""
    parser = argparse.ArgumentParser(description="Show scheduled tasks")
    setup_cli(parser)

    journal = Path(get_journal())
    config_path = journal / "config" / "schedules.json"
    state_path = journal / "health" / "scheduler.json"

    # Load config (all entries, including disabled for display)
    config: dict[str, Any] = {}
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"Error reading {config_path}: {exc}")
            return

    if not config:
        print("No schedules configured.")
        print(f"\nAdd schedules to: {config_path}")
        return

    # Load state
    state: dict[str, Any] = {}
    if state_path.exists():
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
        except (json.JSONDecodeError, OSError):
            pass

    now = datetime.now()

    # Compute column widths
    names = list(config.keys())
    name_width = max(max(len(n) for n in names), 4)
    every_width = 8
    last_run_width = 18
    next_due_width = 10

    # Header
    header = (
        f"  {'NAME':<{name_width}}  {'EVERY':<{every_width}}  "
        f"{'LAST RUN':<{last_run_width}}  {'NEXT DUE':<{next_due_width}}  CMD"
    )
    print(header)
    print()

    for name, raw_entry in sorted(config.items()):
        if not isinstance(raw_entry, dict):
            continue

        every = raw_entry.get("every", "?")
        cmd = raw_entry.get("cmd", [])
        enabled = raw_entry.get("enabled", True)
        state_entry = state.get(name)

        last_run_str = _format_timestamp((state_entry or {}).get("last_run"))

        # Build a validated entry for _is_due / _format_next_due
        if every in INTERVALS and enabled:
            entry = {"cmd": cmd, "every": every}
            next_due_str = _format_next_due(entry, state_entry, now)
        else:
            next_due_str = "disabled" if not enabled else "?"

        cmd_str = " ".join(cmd) if isinstance(cmd, list) else str(cmd)

        tags = ""
        if not enabled:
            tags = " [disabled]"

        line = (
            f"  {name:<{name_width}}  {every:<{every_width}}  "
            f"{last_run_str:<{last_run_width}}  {next_due_str:<{next_due_width}}  {cmd_str}{tags}"
        )
        print(line.rstrip())

    print()
    print(f"Config: {config_path}")
