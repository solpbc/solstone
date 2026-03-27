# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""User-defined routines engine for the supervisor.

Reads routine definitions from routines/config.json, evaluates cron
expressions each tick, and dispatches due routines as cogitate agents
via cortex. Output is written to routines/{routine-id}/{YYYYMMDD}.md.

Runtime functions (init, check) are used by the supervisor.
"""

from __future__ import annotations

import json
import logging
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from apps.calendar.event import EventDay
from think.callosum import callosum_send
from think.cortex_client import cortex_request, wait_for_agents
from think.facets import get_facets
from think.utils import get_journal

logger = logging.getLogger(__name__)

_config: dict[str, dict[str, Any]] = {}
_callosum: Any = None
_last_fired: dict[str, str] = {}  # routine_id -> "YYYY-MM-DD HH:MM" of last fire
_events_fired: dict[str, set[str]] = {}  # routine_id -> set of fired event keys


def _parse_cron_field(field: str, min_val: int, max_val: int) -> set[int]:
    """Parse a single cron field into a set of valid integers."""
    if "," in field:
        values: set[int] = set()
        for part in field.split(","):
            values.update(_parse_cron_field(part, min_val, max_val))
        return values

    if field == "*":
        return set(range(min_val, max_val + 1))

    if field.startswith("*/"):
        step = int(field[2:])
        if step <= 0:
            raise ValueError("Cron step must be > 0")
        return set(range(min_val, max_val + 1, step))

    if "/" in field:
        range_part, step_part = field.split("/", 1)
        step = int(step_part)
        if step <= 0:
            raise ValueError("Cron step must be > 0")
        if "-" not in range_part:
            raise ValueError(f"Invalid cron range step field: {field}")
        start_str, end_str = range_part.split("-", 1)
        start = int(start_str)
        end = int(end_str)
        if start > end:
            raise ValueError(f"Invalid cron range: {field}")
        if start < min_val or end > max_val:
            raise ValueError(f"Cron value out of range: {field}")
        return set(range(start, end + 1, step))

    if "-" in field:
        start_str, end_str = field.split("-", 1)
        start = int(start_str)
        end = int(end_str)
        if start > end:
            raise ValueError(f"Invalid cron range: {field}")
        if start < min_val or end > max_val:
            raise ValueError(f"Cron value out of range: {field}")
        return set(range(start, end + 1))

    value = int(field)
    if value < min_val or value > max_val:
        raise ValueError(f"Cron value out of range: {field}")
    return {value}


def cron_matches(expression: str, dt: datetime) -> bool:
    """Return whether a datetime matches a five-field cron expression."""
    fields = expression.split()
    if len(fields) != 5:
        raise ValueError("Cron expression must have exactly 5 fields")

    minute_set = _parse_cron_field(fields[0], 0, 59)
    hour_set = _parse_cron_field(fields[1], 0, 23)
    dom_set = _parse_cron_field(fields[2], 1, 31)
    month_set = _parse_cron_field(fields[3], 1, 12)
    dow_set = _parse_cron_field(fields[4], 0, 7)
    if 7 in dow_set:
        dow_set.remove(7)
        dow_set.add(0)

    dow = dt.isoweekday() % 7
    return (
        dt.minute in minute_set
        and dt.hour in hour_set
        and dt.day in dom_set
        and dt.month in month_set
        and dow in dow_set
    )


def get_config() -> dict[str, dict[str, Any]]:
    """Read routines/config.json."""
    config_path = Path(get_journal()) / "routines" / "config.json"
    if not config_path.exists():
        return {}

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load routines config: %s", exc)
        return {}

    if not isinstance(raw, dict):
        logger.warning(
            "routines/config.json must be a JSON object, got %s", type(raw).__name__
        )
        return {}
    return raw


def save_config(config: dict[str, dict[str, Any]]) -> None:
    """Persist routines/config.json atomically."""
    routines_dir = Path(get_journal()) / "routines"
    routines_dir.mkdir(parents=True, exist_ok=True)
    config_path = routines_dir / "config.json"

    fd, tmp_path = tempfile.mkstemp(dir=routines_dir, suffix=".tmp", prefix=".config_")
    tmp_file = Path(tmp_path)
    try:
        with open(fd, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        tmp_file.replace(config_path)
    except BaseException:
        tmp_file.unlink(missing_ok=True)
        raise


def _load_events_state() -> dict[str, set[str]]:
    """Load event trigger de-duplication state."""
    state_path = Path(get_journal()) / "routines" / "events_state.json"
    if not state_path.exists():
        return {}
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return {k: set(v) for k, v in raw.items()}
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load events state: %s", exc)
        return {}


def _save_events_state(state: dict[str, set[str]]) -> None:
    """Persist event trigger de-duplication state."""
    routines_dir = Path(get_journal()) / "routines"
    routines_dir.mkdir(parents=True, exist_ok=True)
    state_path = routines_dir / "events_state.json"
    serializable = {k: sorted(v) for k, v in state.items()}
    fd, tmp_path = tempfile.mkstemp(dir=routines_dir, suffix=".tmp", prefix=".events_")
    tmp_file = Path(tmp_path)
    try:
        with open(fd, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2)
        tmp_file.replace(state_path)
    except BaseException:
        tmp_file.unlink(missing_ok=True)
        raise


def init(callosum: Any) -> None:
    """Initialize routines runtime state."""
    global _callosum, _config, _events_fired
    _callosum = callosum
    _config = get_config()
    _events_fired = _load_events_state()
    logger.info("Routines initialized with %d routine(s)", len(_config))


def _log_health(routine_id: str, name: str, duration: int, outcome: str) -> None:
    """Append a line to health/routines.log."""
    health_dir = Path(get_journal()) / "health"
    health_dir.mkdir(parents=True, exist_ok=True)
    health_path = health_dir / "routines.log"
    ts = datetime.now(timezone.utc).isoformat()
    with open(health_path, "a", encoding="utf-8") as f:
        f.write(
            f"{ts} routine={routine_id} name={name} duration={duration}s outcome={outcome}\n"
        )


def _run_routine(routine: dict, event_context: dict | None = None) -> None:
    """Execute a single routine and persist its outcome."""
    routine_id = str(routine.get("id", "unknown"))
    name = str(routine.get("name", routine_id))
    start_time = time.monotonic()
    output_path: Path | None = None

    try:
        instruction = str(routine.get("instruction", ""))
        raw_cadence = routine.get("cadence", "")
        cadence = (
            "event-triggered" if isinstance(raw_cadence, dict) else str(raw_cadence)
        )
        facets = routine.get("facets") or []
        _template = routine.get("template")
        _notify = bool(routine.get("notify", False))

        journal = Path(get_journal())
        output_dir = journal / "routines" / routine_id
        output_dir.mkdir(parents=True, exist_ok=True)

        now_utc = datetime.now(timezone.utc)
        output_path = output_dir / f"{now_utc.strftime('%Y%m%d')}.md"
        if output_path.exists():
            output_path = output_dir / f"{now_utc.strftime('%Y%m%d-%H%M%S')}.md"

        previous_outputs = sorted(output_dir.glob("*.md"))
        prev_output_path = str(previous_outputs[-1]) if previous_outputs else None

        facets_line = f"**Facets:** {', '.join(facets)}" if facets else ""
        previous_line = (
            f"**Previous output:** {prev_output_path}" if prev_output_path else ""
        )
        event_section = ""
        if event_context:
            title = event_context.get("title", "")
            start = event_context.get("start", "")
            participants = event_context.get("participants") or []
            parts_line = ", ".join(participants) if participants else "none listed"
            event_section = (
                "\n**Upcoming Event:**\n"
                f"- Title: {title}\n"
                f"- Start: {start}\n"
                f"- Participants: {parts_line}\n"
            )
        prompt = (
            f"## Routine: {name}\n\n"
            f"**Instruction:** {instruction}\n\n"
            f"**Cadence:** {cadence}\n"
            f"{facets_line}\n"
            f"{previous_line}"
            f"{event_section}\n\n"
            "Execute this routine now. Write your output as concise, actionable markdown.\n"
        )

        callosum_send("routines", "started", routine_id=routine_id, name=name)
        agent_id = cortex_request(
            prompt=prompt,
            name="routine",
            config={"output_path": str(output_path), "output": "md"},
        )

        if agent_id is None:
            duration = int(time.monotonic() - start_time)
            logger.error("Failed to start routine %s", routine_id)
            _log_health(routine_id, name, duration, "error")
            callosum_send(
                "routines",
                "complete",
                routine_id=routine_id,
                name=name,
                outcome="error",
                output_path=str(output_path),
                duration_s=duration,
            )
            return

        completed, timed_out = wait_for_agents([agent_id], timeout=600)
        if agent_id in timed_out:
            outcome = "timeout"
        else:
            end_state = completed.get(agent_id, "error")
            outcome = "success" if end_state == "finish" else "error"

        duration = int(time.monotonic() - start_time)
        routine["last_run"] = datetime.now(timezone.utc).isoformat()
        _config[routine_id] = routine
        save_config(_config)

        callosum_send(
            "routines",
            "complete",
            routine_id=routine_id,
            name=name,
            outcome=outcome,
            output_path=str(output_path),
            duration_s=duration,
        )
        _log_health(routine_id, name, duration, outcome)
    except Exception as exc:
        duration = int(time.monotonic() - start_time)
        logger.exception("Routine %s failed: %s", routine_id, exc)
        try:
            _log_health(routine_id, name, duration, "error")
        except Exception:
            logger.exception("Failed to write routines health log for %s", routine_id)
        try:
            callosum_send(
                "routines",
                "complete",
                routine_id=routine_id,
                name=name,
                outcome="error",
                output_path=str(output_path) if output_path else "",
                duration_s=duration,
            )
        except Exception:
            logger.exception("Failed to emit routine completion for %s", routine_id)


def check() -> None:
    """Reload config and run any due routines."""
    global _config
    _config = get_config()

    now_utc = datetime.now(timezone.utc)
    for routine in _config.values():
        if not routine.get("enabled"):
            continue

        routine_id = routine.get("id")
        if not routine_id:
            continue

        tz = routine.get("timezone") or "UTC"
        try:
            local_now = now_utc.astimezone(ZoneInfo(tz))
        except ZoneInfoNotFoundError:
            logger.warning(
                "Routine %s has invalid timezone %r, skipping", routine_id, tz
            )
            continue
        cadence = routine.get("cadence")

        if isinstance(cadence, str):
            minute_key = local_now.strftime("%Y-%m-%d %H:%M")
            if _last_fired.get(routine_id) == minute_key:
                continue
            if cron_matches(cadence, local_now):
                _last_fired[routine_id] = minute_key
                _run_routine(routine)
        elif isinstance(cadence, dict) and cadence.get("type") == "event":
            _check_event_cadence(routine, str(routine_id), cadence, local_now)


def _check_event_cadence(
    routine: dict, routine_id: str, cadence: dict, local_now: datetime
) -> None:
    """Check calendar events and fire routine if within trigger window."""
    if cadence.get("trigger") != "calendar":
        logger.warning(
            "Routine %s has unsupported event trigger %r", routine_id, cadence
        )
        return

    offset_minutes = cadence.get("offset_minutes", -30)
    if not isinstance(offset_minutes, int):
        logger.warning(
            "Routine %s has invalid event offset %r", routine_id, offset_minutes
        )
        return

    facets_list = routine.get("facets") or []
    if not facets_list:
        try:
            facets_list = list(get_facets().keys())
        except Exception:
            logger.warning("Failed to discover facets for routine %s", routine_id)
            return

    today = local_now.strftime("%Y%m%d")
    now_minutes = local_now.hour * 60 + local_now.minute
    fired = _events_fired.setdefault(routine_id, set())

    for facet in facets_list:
        try:
            event_day = EventDay.load(today, facet)
        except Exception:
            logger.debug("Failed to load calendar for %s/%s", today, facet)
            continue

        for event in event_day.items:
            if event.cancelled:
                continue

            event_key = f"{today}:{facet}:{event.index}"
            if event_key in fired:
                continue

            try:
                parts = event.start.split(":")
                event_start_minutes = int(parts[0]) * 60 + int(parts[1])
            except (ValueError, IndexError):
                continue

            trigger_minutes = event_start_minutes + offset_minutes
            if trigger_minutes <= now_minutes < event_start_minutes:
                fired.add(event_key)
                event_context = {
                    "title": event.title,
                    "start": event.start,
                    "participants": event.participants,
                    "facet": facet,
                }
                _run_routine(routine, event_context=event_context)


def save_state() -> None:
    """Persist routines state."""
    save_config(_config)
    _save_events_state(_events_fired)
