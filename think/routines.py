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
from datetime import datetime, timedelta, timezone
from datetime import datetime as real_datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from think.callosum import callosum_send
from think.cortex_client import cortex_request, wait_for_uses
from think.utils import get_journal

logger = logging.getLogger(__name__)

_config: dict[str, dict[str, Any]] = {}
_callosum: Any = None
_last_fired: dict[str, str] = {}  # routine_id -> "YYYY-MM-DD HH:MM" of last fire
_fired_triggers: dict[str, dict[str, str]] = {}
# Crossover band for anticipation scan: load adjacent-day activity files when
# local_now is within this many minutes of midnight. Covers offset_minutes up to ~2h.
_ACTIVITY_ANTICIPATION_CROSSDAY_WINDOW_MINUTES = 120
_logged_unknown_cadence: set[str] = set()


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


def _format_cadence_human(cadence: object) -> str:
    """Format a cadence for human display in routine state."""
    return str(cadence)


def get_routine_state() -> list[dict[str, Any]]:
    """Return routine summaries for pre-hook injection.

    Reads config from disk and output files. Does not use module-level
    state or supervisor-only imports (cortex/callosum).
    """
    config = get_config()
    now_utc = datetime.now(timezone.utc)
    result = []
    for routine in config.values():
        routine_id = routine.get("id")
        if not routine_id:
            continue
        summary: dict[str, Any] = {
            "name": routine.get("name", ""),
            "cadence": _format_cadence_human(routine.get("cadence", "")),
            "last_run": routine.get("last_run"),
            "enabled": routine.get("enabled", False),
            "paused_until": routine.get("resume_date"),
        }
        output_summary = None
        last_run = routine.get("last_run")
        if last_run:
            try:
                last_dt = real_datetime.fromisoformat(last_run)
                if (now_utc - last_dt).total_seconds() < 43200:
                    output_dir = Path(get_journal()) / "routines" / routine_id
                    outputs = sorted(output_dir.glob("*.md"))
                    if outputs:
                        text = outputs[-1].read_text(encoding="utf-8").strip()
                        output_summary = text[:100]
            except (ValueError, OSError):
                pass
        summary["output_summary"] = output_summary
        result.append(summary)
    return result


def init(callosum: Any) -> None:
    """Initialize routines runtime state."""
    global _callosum, _config
    _callosum = callosum
    _config = get_config()
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


def _render_upcoming_activity_block(activity: dict | None, facet: str) -> str:
    """Render an upcoming activity context block for routine prompts."""
    if not activity:
        return ""

    participation = activity.get("participation") or []
    if not isinstance(participation, list):
        participation = []
    attendees = ", ".join(
        str(entry.get("name") or "").strip()
        for entry in participation
        if isinstance(entry, dict)
        and entry.get("role") == "attendee"
        and str(entry.get("name") or "").strip()
    )
    attendees = attendees or "(none listed)"

    title = str(activity.get("title") or "")
    activity_type = str(activity.get("activity") or "(unknown)")
    start = str(activity.get("start") or "(unknown)")
    end = str(activity.get("end") or "(unknown)")
    description = str(activity.get("description") or "(none)")
    details = str(activity.get("details") or "(none)")
    facet_display = str(facet or "(none)")

    return (
        "## Upcoming Activity\n\n"
        f"- **Title:** {title}\n"
        f"- **Type:** {activity_type}\n"
        f"- **Facet:** {facet_display}\n"
        f"- **Start:** {start}\n"
        f"- **End:** {end}\n"
        f"- **Description:** {description}\n"
        f"- **Details:** {details}\n"
        f"- **Attendees:** {attendees}\n\n"
    )


def _run_routine(routine: dict, trigger_context: dict | None = None) -> None:
    """Execute a single routine and persist its outcome."""
    routine_id = str(routine.get("id", "unknown"))
    name = str(routine.get("name", routine_id))
    start_time = time.monotonic()
    output_path: Path | None = None

    try:
        instruction = str(routine.get("instruction", ""))
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
        cadence_raw = routine.get("cadence")
        if isinstance(cadence_raw, dict):
            cadence_display = str(cadence_raw.get("type", ""))
        else:
            cadence_display = str(cadence_raw or "")

        upcoming_block = ""
        if trigger_context and "activity" in trigger_context:
            upcoming_block = _render_upcoming_activity_block(
                trigger_context["activity"], trigger_context.get("facet", "")
            )

        prompt = (
            f"## Routine: {name}\n\n"
            f"**Instruction:** {instruction}\n\n"
            f"**Cadence:** {cadence_display}\n"
            f"{facets_line}\n"
            f"{previous_line}\n\n"
            f"{upcoming_block}"
            "Execute this routine now. Write your output as concise, actionable markdown.\n"
        )

        callosum_send("routines", "started", routine_id=routine_id, name=name)
        use_id = cortex_request(
            prompt=prompt,
            name="routine",
            config={"output_path": str(output_path), "output": "md"},
        )

        if use_id is None:
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

        completed, timed_out = wait_for_uses([use_id], timeout=600)
        if use_id in timed_out:
            outcome = "timeout"
        else:
            end_state = completed.get(use_id, "error")
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


def _activity_anticipation_candidate_days(
    local_now: datetime, window_minutes: int
) -> list[str]:
    """Return chronological YYYYMMDD strings to scan for anticipation triggers.

    Always includes today. Adds yesterday when local_now falls within the first
    `window_minutes` after midnight, and tomorrow when local_now falls within the
    last `window_minutes` before midnight.
    """
    minutes_since_midnight = local_now.hour * 60 + local_now.minute
    days: list[str] = []
    if minutes_since_midnight < window_minutes:
        days.append((local_now - timedelta(days=1)).strftime("%Y%m%d"))
    days.append(local_now.strftime("%Y%m%d"))
    if minutes_since_midnight >= 24 * 60 - window_minutes:
        days.append((local_now + timedelta(days=1)).strftime("%Y%m%d"))
    return days


def _prune_fired_triggers(*, now_utc: datetime) -> None:
    """Drop in-memory activity trigger dedupe entries older than two days."""
    from datetime import timedelta

    cutoff = now_utc - timedelta(days=2)
    for routine_id, fired_for_routine in list(_fired_triggers.items()):
        for activity_id, fired_at in list(fired_for_routine.items()):
            try:
                fired_dt = real_datetime.fromisoformat(fired_at)
            except ValueError:
                del fired_for_routine[activity_id]
                continue
            if fired_dt < cutoff:
                del fired_for_routine[activity_id]
        if not fired_for_routine:
            del _fired_triggers[routine_id]


def check() -> None:
    """Reload config and run any due routines."""
    global _config
    _config = get_config()
    _prune_fired_triggers(now_utc=datetime.now(timezone.utc))

    config_changed = False
    for routine in _config.values():
        resume_date = routine.get("resume_date")
        if not resume_date or routine.get("enabled"):
            continue
        routine_id = routine.get("id")
        if not routine_id:
            continue
        tz = routine.get("timezone") or "UTC"
        try:
            local_today = (
                datetime.now(timezone.utc).astimezone(ZoneInfo(tz)).strftime("%Y-%m-%d")
            )
        except ZoneInfoNotFoundError:
            continue
        if resume_date <= local_today:
            routine["enabled"] = True
            routine.pop("resume_date", None)
            config_changed = True
            name = routine.get("name", routine_id)
            _log_health(routine_id, name, 0, "auto-resumed")
            logger.info("Auto-resumed routine %s (%s)", routine_id, name)
    if config_changed:
        save_config(_config)

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
        # Keep this cadence-object dispatch in sync with think.tools.routines._validate_routine_cadence().
        elif (
            isinstance(cadence, dict) and cadence.get("type") == "activity-anticipation"
        ):
            from datetime import timedelta

            from think.activities import load_activity_records
            from think.facets import get_facets

            offset_raw = cadence.get("offset_minutes", 0)
            try:
                offset_minutes = int(offset_raw)
            except (TypeError, ValueError):
                logger.warning(
                    "Routine %s has invalid offset_minutes %r, skipping",
                    routine_id,
                    offset_raw,
                )
                continue

            candidate_days = _activity_anticipation_candidate_days(
                local_now, _ACTIVITY_ANTICIPATION_CROSSDAY_WINDOW_MINUTES
            )
            fired_for_routine = _fired_triggers.setdefault(routine_id, {})

            for facet_name in get_facets().keys():
                for day_str in candidate_days:
                    try:
                        records = load_activity_records(facet_name, day_str)
                    except Exception:
                        logger.warning(
                            "Failed loading activities for facet %s on %s",
                            facet_name,
                            day_str,
                            exc_info=True,
                        )
                        continue

                    for record in records:
                        if record.get("source") != "anticipated":
                            continue
                        activity_id = record.get("id")
                        if not activity_id or activity_id in fired_for_routine:
                            continue
                        start_str = record.get("start")
                        if not start_str:
                            continue
                        try:
                            time_parts = [int(x) for x in str(start_str).split(":")]
                        except (ValueError, AttributeError):
                            continue
                        if len(time_parts) == 2:
                            h, m = time_parts
                            s = 0
                        elif len(time_parts) == 3:
                            h, m, s = time_parts
                        else:
                            continue
                        start_dt = datetime(
                            int(day_str[:4]),
                            int(day_str[4:6]),
                            int(day_str[6:8]),
                            h,
                            m,
                            s,
                            tzinfo=local_now.tzinfo,
                        )
                        trigger_dt = start_dt + timedelta(minutes=offset_minutes)
                        if abs((local_now - trigger_dt).total_seconds()) > 60:
                            continue
                        fired_for_routine[activity_id] = now_utc.isoformat()
                        _run_routine(
                            routine,
                            trigger_context={"activity": record, "facet": facet_name},
                        )
        elif isinstance(cadence, dict):
            cadence_type = str(cadence.get("type", "unknown"))
            if routine_id not in _logged_unknown_cadence:
                logger.info(
                    "Routine %s has unsupported cadence type %r, skipping",
                    routine_id,
                    cadence_type,
                )
                _logged_unknown_cadence.add(routine_id)


def save_state() -> None:
    """Persist routines state."""
    save_config(_config)
