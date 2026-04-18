# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Home app - Pulse landing page."""

from __future__ import annotations

import json
import logging
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

import frontmatter
from flask import Blueprint, jsonify, render_template

from convey.apps import _resolve_attention
from convey.bridge import get_cached_state
from think.awareness import get_current
from think.capture_health import get_capture_health
from think.facets import get_enabled_facets, get_facets
from think.indexer.journal import get_journal_index
from think.pipeline_health import pipeline_status_message, summarize_pipeline_day
from think.utils import get_journal

# Briefing phase thresholds
BRIEFING_MORNING_END_HOUR = 10
BRIEFING_EOD_HOUR = 20

# Section heading -> key mapping
_BRIEFING_SECTIONS = {
    "your day": "your_day",
    "yesterday": "yesterday",
    "needs attention": "needs_attention",
    "forward look": "forward_look",
    "reading": "reading",
}

home_bp = Blueprint(
    "app:home",
    __name__,
    url_prefix="/app/home",
)

_FIRST_WEEK_FRAMING = "Most of what I learn becomes useful in the third or fourth week, when I've seen enough patterns to surface them. For now, here's what's already happening:"

_ENTITY_TYPE_LABELS = {
    "company": ("company", "companies"),
    "decision": ("decision", "decisions"),
    "person": ("person", "people"),
    "place": ("place", "places"),
    "project": ("project", "projects"),
    "topic": ("topic", "topics"),
    "tool": ("tool", "tools"),
    "unknown": ("thing", "things"),
}


def _today() -> str:
    return datetime.now().strftime("%Y%m%d")


def _yesterday() -> str:
    return (datetime.now().astimezone() - timedelta(days=1)).strftime("%Y%m%d")


def _count_journal_age_days(today: str) -> int:
    chronicle_dir = Path(get_journal()) / "chronicle"
    if not chronicle_dir.is_dir():
        return 0

    earliest: datetime | None = None
    for child in chronicle_dir.iterdir():
        if not child.is_dir() or not child.name.isdigit() or len(child.name) != 8:
            continue
        try:
            day = datetime.strptime(child.name, "%Y%m%d")
        except ValueError:
            continue
        if earliest is None or day < earliest:
            earliest = day

    if earliest is None:
        return 0

    try:
        today_dt = datetime.strptime(today, "%Y%m%d")
    except ValueError:
        return 0
    return max(0, (today_dt - earliest).days)


def _load_flow_md(today: str) -> tuple[str | None, float | None]:
    """Load today's flow.md content and mtime. Returns (content, mtime) or (None, None)."""
    try:
        journal = Path(get_journal())
        flow_path = journal / today / "talents" / "flow.md"
        if flow_path.exists():
            return flow_path.read_text(), flow_path.stat().st_mtime
    except Exception:
        logger.warning("home: failed to load flow.md", exc_info=True)
    return None, None


def _load_pulse_md() -> tuple[str | None, dict | None, list[str]]:
    """Load sol/pulse.md if current for today.

    Returns (content, metadata, needs_you) or (None, None, []).
    """
    try:
        journal = Path(get_journal())
        pulse_path = journal / "sol" / "pulse.md"
        if not pulse_path.exists():
            return None, None, []
        post = frontmatter.load(str(pulse_path))
        updated = post.metadata.get("updated")
        if not updated:
            return None, None, []
        # Parse ISO datetime and check if from today
        if isinstance(updated, str):
            updated_dt = datetime.fromisoformat(updated)
        else:
            updated_dt = updated  # frontmatter may parse datetime objects
        if updated_dt.date() != datetime.now().date():
            return None, None, []
        # Extract ## needs you section
        needs = []
        in_needs = False
        for line in post.content.splitlines():
            if line.strip().lower() == "## needs you":
                in_needs = True
                continue
            if in_needs:
                if line.startswith("## "):
                    break
                stripped = line.strip()
                if stripped.startswith("- "):
                    needs.append(stripped[2:].strip())
        return post.content, post.metadata, needs
    except Exception:
        logger.warning("home: failed to load pulse.md", exc_info=True)
        return None, None, []


def _load_briefing_md(
    today: str | None = None,
) -> tuple[dict[str, str], dict | None, list[str]]:
    """Load today's briefing.md sections and needs_attention bullets."""
    try:
        today = today or _today()
        journal = Path(get_journal())
        briefing_path = journal / "sol" / "briefing.md"
        if not briefing_path.exists():
            return {}, None, []

        post = frontmatter.load(str(briefing_path))
        metadata = post.metadata
        if metadata.get("type") != "morning_briefing":
            return {}, None, []
        if str(metadata.get("date")) != today:
            return {}, None, []

        sections = {}
        current_key = None
        current_lines: list[str] = []

        def flush_section() -> None:
            nonlocal current_key, current_lines
            if not current_key:
                current_lines = []
                return
            body = "\n".join(current_lines).strip()
            if body:
                sections[current_key] = body
            current_lines = []

        for line in post.content.splitlines():
            if line.startswith("## "):
                flush_section()
                heading = line[3:].strip().lower()
                current_key = _BRIEFING_SECTIONS.get(heading)
                continue
            if current_key:
                current_lines.append(line)
        flush_section()

        needs_attention_items = []
        needs_body = sections.get("needs_attention", "")
        for line in needs_body.splitlines():
            stripped = line.strip()
            if stripped.startswith("- "):
                needs_attention_items.append(stripped[2:].strip())

        return sections, metadata, needs_attention_items
    except Exception:
        logger.warning("home: failed to load briefing.md", exc_info=True)
        return {}, None, []


def _compute_briefing_phase(
    segment_count: int, hour: int, briefing_exists: bool
) -> str:
    """Compute briefing display phase from current time and activity."""
    if hour >= BRIEFING_EOD_HOUR:
        return "eod"
    if not briefing_exists and hour < BRIEFING_MORNING_END_HOUR:
        return "pending"
    if briefing_exists and (segment_count == 0 or hour < BRIEFING_MORNING_END_HOUR):
        return "morning"
    if briefing_exists and segment_count > 0:
        return "active"
    return "eod"


def _normalize_item(text: str) -> str:
    return " ".join(text.lower().split())


def _briefing_summary(sections: dict[str, str], needs_count: int) -> str:
    """Generate a short collapsed summary for the briefing card."""
    meeting_count = 0
    your_day = sections.get("your_day", "")
    for line in your_day.splitlines():
        stripped = line.strip()
        if stripped.startswith("- ") and "**" in stripped:
            after_bullet = stripped[2:]
            if after_bullet.startswith("**") and after_bullet.count("**") >= 2:
                time_part = after_bullet.split("**", 2)[1]
                if len(time_part) == 5 and time_part[2] == ":":
                    meeting_count += 1

    if meeting_count or needs_count:
        meeting_label = "meeting" if meeting_count == 1 else "meetings"
        needs_label = "item needs" if needs_count == 1 else "items need"
        return (
            f"Morning briefing — {meeting_count} {meeting_label}, "
            f"{needs_count} {needs_label} attention"
        )

    for content in sections.values():
        for line in content.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            stripped = stripped.removeprefix("- ").strip()
            if len(stripped) > 58:
                stripped = stripped[:55].rstrip() + "..."
            return f"Morning briefing — {stripped}"
    return "Morning briefing"


def _load_stats(today: str) -> dict[str, Any]:
    """Load per-day stats.json. Returns empty dict if missing."""
    try:
        journal = Path(get_journal())
        stats_path = journal / today / "stats.json"
        if stats_path.exists():
            return json.loads(stats_path.read_text())
    except Exception:
        logger.warning("home: failed to load stats", exc_info=True)
    return {}


def _load_yesterday_stats(yesterday: str) -> dict[str, Any] | None:
    try:
        stats_path = Path(get_journal()) / "chronicle" / yesterday / "stats.json"
        if not stats_path.exists():
            return None
        return json.loads(stats_path.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("home: failed to load yesterday stats", exc_info=True)
        return None


def _load_yesterday_pipeline_summary(yesterday: str) -> dict[str, Any]:
    return summarize_pipeline_day(yesterday)


def _collect_todos(today: str) -> list[dict[str, Any]]:
    """Collect pending todos across all facets."""
    from apps.todos.todo import get_todos

    todos = []
    try:
        facets = get_facets()
    except Exception:
        logger.warning("home: failed to get facets for todos", exc_info=True)
        return []

    for facet_name in facets:
        facet_todos = get_todos(today, facet_name)
        if facet_todos is None:
            continue
        for todo in facet_todos:
            if not todo.get("completed") and not todo.get("cancelled"):
                todo["facet"] = facet_name
                todos.append(todo)
    return todos


def _collect_anticipated_activities(today: str) -> list[dict[str, Any]]:
    """Collect anticipated activities across all facets."""
    from think.activities import load_activity_records

    try:
        anticipated_activities = []
        for facet_name in get_facets():
            for record in load_activity_records(facet_name, today):
                if record.get("source") != "anticipated":
                    continue

                participants = []
                for entry in record.get("participation", []):
                    if not isinstance(entry, dict) or entry.get("role") != "attendee":
                        continue
                    name = str(entry.get("name") or "").strip()
                    if name:
                        participants.append(name)

                anticipated_activities.append(
                    {
                        "title": record.get("title", ""),
                        "start": record.get("start") or "",
                        "end": record.get("end") or "",
                        "facet": facet_name,
                        "occurred": False,
                        "participants": participants,
                    }
                )
        return anticipated_activities
    except Exception:
        logger.warning("home: failed to collect anticipated activities", exc_info=True)
        return []


def _collect_activities(today: str) -> list[dict[str, Any]]:
    """Collect recent activities across all facets, last ~4 hours."""
    from think.activities import load_activity_records

    activities = []
    try:
        facets = get_facets()
    except Exception:
        logger.warning("home: failed to get facets for activities", exc_info=True)
        return []

    now = datetime.now()
    cutoff_ts = (now - timedelta(hours=4)).timestamp() * 1000  # ms

    for facet_name in facets:
        records = load_activity_records(facet_name, today)
        for record in records:
            if record.get("source") == "anticipated":
                continue
            created_at = record.get("created_at", 0)
            if created_at < cutoff_ts:
                continue
            # Convert ms timestamp to HH:MM for display
            try:
                dt = datetime.fromtimestamp(created_at / 1000)
                record["display_time"] = dt.strftime("%H:%M")
            except (OSError, ValueError):
                record["display_time"] = ""
            record["facet"] = facet_name
            activities.append(record)

    activities.sort(key=lambda a: a.get("created_at", 0), reverse=True)
    return activities


def _collect_entities_today(today: str) -> list[dict[str, Any]]:
    """Get today's entities from entity_signals table."""
    try:
        conn, _ = get_journal_index()
        try:
            rows = conn.execute(
                """SELECT entity_name, COUNT(*) as signal_count,
                          GROUP_CONCAT(DISTINCT signal_type) as types
                   FROM entity_signals
                   WHERE day = ?
                   GROUP BY entity_name
                   ORDER BY signal_count DESC
                   LIMIT 8""",
                (today,),
            ).fetchall()

            entity_meta = {}
            meta_rows = conn.execute(
                "SELECT entity_id, name, type FROM entities WHERE source='identity'"
            ).fetchall()
            for row in meta_rows:
                entity_meta[row[0]] = {"name": row[1], "type": row[2] or "unknown"}
                entity_meta[row[1].lower()] = {
                    "name": row[1],
                    "type": row[2] or "unknown",
                }

            entities = []
            for row in rows:
                name = row[0]
                meta = entity_meta.get(name, entity_meta.get(name.lower(), {}))
                entities.append(
                    {
                        "name": meta.get("name", name),
                        "signal_count": row[1],
                        "types": row[2] or "",
                        "entity_type": meta.get("type", "unknown"),
                    }
                )
            return entities
        finally:
            conn.close()
    except Exception:
        logger.warning("home: failed to collect entities", exc_info=True)
        return []


def _collect_entities_yesterday(yesterday: str) -> list[dict[str, Any]]:
    """Get yesterday's entities from entity_signals table."""
    try:
        conn, _ = get_journal_index()
        try:
            rows = conn.execute(
                """SELECT entity_name, COUNT(*) as signal_count,
                          GROUP_CONCAT(DISTINCT signal_type) as types
                   FROM entity_signals
                   WHERE day = ?
                   GROUP BY entity_name
                   ORDER BY signal_count DESC""",
                (yesterday,),
            ).fetchall()

            entity_meta = {}
            meta_rows = conn.execute(
                "SELECT entity_id, name, type FROM entities WHERE source='identity'"
            ).fetchall()
            for row in meta_rows:
                entity_meta[row[0]] = {"name": row[1], "type": row[2] or "unknown"}
                entity_meta[row[1].lower()] = {
                    "name": row[1],
                    "type": row[2] or "unknown",
                }

            entities = []
            for row in rows:
                name = row[0]
                meta = entity_meta.get(name, entity_meta.get(name.lower(), {}))
                entities.append(
                    {
                        "name": meta.get("name", name),
                        "signal_count": row[1],
                        "types": row[2] or "",
                        "entity_type": meta.get("type", "unknown"),
                    }
                )
            return entities
        finally:
            conn.close()
    except Exception:
        logger.warning("home: failed to collect yesterday entities", exc_info=True)
        return []


def _normalize_activity_title(record: dict[str, Any]) -> str:
    title = str(record.get("description") or "").strip()
    if title:
        return title
    activity = str(record.get("activity") or "").strip().replace("_", " ")
    if activity:
        return activity.title()
    return "Untitled activity"


def _collect_top_activities_yesterday(yesterday: str) -> list[dict[str, Any]]:
    from think.activities import estimate_duration_minutes, load_activity_records

    activities = []
    try:
        facets = get_enabled_facets()
    except Exception:
        logger.warning(
            "home: failed to get enabled facets for yesterday activities",
            exc_info=True,
        )
        return []

    for facet_name in facets:
        for record in load_activity_records(facet_name, yesterday):
            segments = record.get("segments", [])
            activities.append(
                {
                    **record,
                    "facet": facet_name,
                    "title": _normalize_activity_title(record),
                    "duration_minutes": estimate_duration_minutes(segments),
                }
            )

    activities.sort(
        key=lambda record: (
            -int(record.get("duration_minutes", 0)),
            record.get("title", "").lower(),
            record.get("facet", ""),
        )
    )
    return activities


def _top_heatmap_hours(stats_data: dict[str, Any]) -> list[int]:
    hours = stats_data.get("heatmap_data", {}).get("hours", {})
    ranked = []
    for hour, minutes in hours.items():
        try:
            hour_int = int(hour)
            minutes_value = float(minutes)
        except (TypeError, ValueError):
            continue
        if minutes_value <= 0:
            continue
        ranked.append((hour_int, minutes_value))

    ranked.sort(key=lambda item: (-item[1], item[0]))
    return [hour for hour, _minutes in ranked[:3]]


def _knowledge_graph_freshness(yesterday: str) -> dict[str, Any]:
    path = (
        Path(get_journal()) / "chronicle" / yesterday / "talents" / "knowledge_graph.md"
    )
    if not path.exists():
        return {"exists": False, "fresh": False, "updated_label": None}

    try:
        start_of_yesterday_local = datetime.strptime(yesterday, "%Y%m%d").astimezone()
        updated_at = datetime.fromtimestamp(path.stat().st_mtime).astimezone()
    except Exception:
        logger.warning(
            "home: failed to inspect knowledge graph freshness", exc_info=True
        )
        return {"exists": True, "fresh": False, "updated_label": None}

    return {
        "exists": True,
        "fresh": updated_at >= start_of_yesterday_local,
        "updated_label": updated_at.strftime("%-I:%M%p").lower(),
    }


def _briefing_freshness(today: str) -> dict[str, Any]:
    briefing_path = Path(get_journal()) / "sol" / "briefing.md"
    if not briefing_path.exists():
        return {"exists": False, "valid": False, "generated_label": None}

    try:
        post = frontmatter.load(str(briefing_path))
    except Exception:
        logger.warning("home: failed to load briefing freshness", exc_info=True)
        return {"exists": True, "valid": False, "generated_label": None}

    if post.metadata.get("type") != "morning_briefing":
        return {"exists": True, "valid": False, "generated_label": None}

    generated = post.metadata.get("generated")
    if generated is None:
        return {"exists": True, "valid": False, "generated_label": None}

    try:
        if isinstance(generated, str):
            generated_dt = datetime.fromisoformat(generated)
        else:
            generated_dt = generated
        if generated_dt.tzinfo is None:
            generated_dt = generated_dt.astimezone()
        else:
            generated_dt = generated_dt.astimezone()
    except Exception:
        return {"exists": True, "valid": False, "generated_label": None}

    return {
        "exists": True,
        "valid": generated_dt.strftime("%Y%m%d") == today,
        "generated_label": generated_dt.strftime("%-I:%M%p").lower(),
    }


def _newsletter_attempts_from_think_logs(yesterday: str) -> tuple[int, int]:
    journal = Path(get_journal())
    successful = len(list(journal.glob(f"facets/*/news/{yesterday}.md")))

    failed = 0
    health_dir = journal / "chronicle" / yesterday / "health"
    if health_dir.is_dir():
        for path in sorted(health_dir.glob("*_daily.jsonl")):
            try:
                with path.open(encoding="utf-8") as handle:
                    for raw_line in handle:
                        line = raw_line.strip()
                        if not line:
                            continue
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if (
                            record.get("event") == "talent.fail"
                            and record.get("facet")
                            and record.get("name") == "facet_newsletter"
                        ):
                            failed += 1
            except OSError:
                logger.warning(
                    "home: failed to read newsletter think log %s",
                    path,
                    exc_info=True,
                )

    return successful, successful + failed


def _format_duration(total_minutes: float) -> str:
    rounded_minutes = int(round(total_minutes))
    if rounded_minutes < 60:
        return f"{rounded_minutes} min"

    rounded_hours = round(total_minutes / 60, 1)
    if float(rounded_hours).is_integer():
        hours_int = int(rounded_hours)
        return f"{hours_int} hour{'s' if hours_int != 1 else ''}"
    return f"{rounded_hours:.1f} hours"


def _format_hour_label(start_hour: int, end_hour: int) -> str:
    def render(hour: int, *, include_meridiem: bool) -> str:
        normalized = hour % 24
        display_hour = normalized % 12 or 12
        meridiem = "am" if normalized < 12 else "pm"
        return f"{display_hour}{meridiem}" if include_meridiem else str(display_hour)

    start_meridiem = "am" if start_hour % 24 < 12 else "pm"
    end_meridiem = "am" if end_hour % 24 < 12 else "pm"
    return (
        f"{render(start_hour, include_meridiem=start_meridiem != end_meridiem)}-"
        f"{render(end_hour, include_meridiem=True)}"
    )


def _join_phrases(parts: list[str]) -> str:
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    return ", ".join(parts[:-1]) + f", and {parts[-1]}"


def _format_entity_count(type_key: str, count: int) -> str:
    singular, plural = _ENTITY_TYPE_LABELS.get(type_key, (type_key, f"{type_key}s"))
    label = singular if count == 1 else plural
    return f"{count} {label}"


def _format_entity_summary(entities: list[dict[str, Any]]) -> str | None:
    counts: dict[str, int] = {}
    for entity in entities:
        type_key = str(entity.get("entity_type") or "unknown").strip().lower()
        counts[type_key] = counts.get(type_key, 0) + 1

    if not counts:
        return None

    ordered_keys = []
    if counts.get("person"):
        ordered_keys.append("person")
    ordered_keys.extend(
        key
        for key, _count in sorted(
            (
                (key, count)
                for key, count in counts.items()
                if key != "person" and count > 0
            ),
            key=lambda item: (-item[1], item[0]),
        )
    )

    labels = [
        _format_entity_count(key, counts[key]) for key in ordered_keys if counts[key]
    ]
    if not labels:
        return None
    return f"I recognized {_join_phrases(labels)}."


def _format_activity_label(activity: dict[str, Any]) -> str:
    return (
        f"I took notes on {activity.get('title', 'Untitled activity')} "
        f"for {_format_duration(activity.get('duration_minutes', 0))} in "
        f"{activity.get('facet', 'unknown')}."
    )


def _format_newsletter_summary(successful: int, attempted: int) -> str:
    if attempted == 0:
        return "I didn't produce any facet newsletters."
    if attempted > successful:
        return (
            f"I wrote {successful} of {attempted} newsletter"
            f"{'s' if attempted != 1 else ''}."
        )
    return f"I wrote {successful} newsletter{'s' if successful != 1 else ''}."


def _format_processing_summary(
    mode: str,
    successful_newsletters: int,
    attempted_newsletters: int,
    knowledge_graph: dict[str, Any],
    briefing: dict[str, Any],
) -> str:
    if mode == "degraded":
        if attempted_newsletters == 0 and successful_newsletters == 0:
            return (
                "I didn't produce any facet newsletters, and some overnight "
                "processing didn't finish."
            )
        if attempted_newsletters > successful_newsletters:
            return (
                f"I wrote {successful_newsletters} of {attempted_newsletters} "
                "newsletters, but some overnight processing didn't finish."
            )
        return (
            f"I wrote {successful_newsletters} newsletter"
            f"{'s' if successful_newsletters != 1 else ''}, but some overnight "
            "processing didn't finish."
        )

    actions = []
    if successful_newsletters > 0:
        actions.append(
            f"wrote {successful_newsletters} newsletter"
            f"{'s' if successful_newsletters != 1 else ''}"
        )
    if knowledge_graph.get("fresh"):
        actions.append("refreshed your knowledge graph")
    if briefing.get("valid"):
        actions.append("prepared your morning briefing")
    if not actions:
        return _format_newsletter_summary(successful_newsletters, attempted_newsletters)
    return f"I {_join_phrases(actions)}."


def _format_heatmap_summary(stats_data: dict[str, Any]) -> str | None:
    hours = sorted(_top_heatmap_hours(stats_data))
    if not hours:
        return None

    ranges = []
    range_start = hours[0]
    range_end = hours[0] + 1
    for hour in hours[1:]:
        if hour == range_end:
            range_end += 1
            continue
        ranges.append(_format_hour_label(range_start, range_end))
        range_start = hour
        range_end = hour + 1
    ranges.append(_format_hour_label(range_start, range_end))
    return "I watched most closely during " + " · ".join(ranges) + "."


def _format_gap_bullets(
    pipeline_summary: dict[str, Any],
    knowledge_graph: dict[str, Any],
    briefing: dict[str, Any],
) -> list[str]:
    bullets = []
    anomalies = pipeline_summary.get("anomalies", [])
    has_daily = any(
        anomaly.get("kind") == "daily_agents_missing" for anomaly in anomalies
    )
    has_activity = any(
        anomaly.get("kind") == "activity_agents_missing" for anomaly in anomalies
    )
    has_failure = any(anomaly.get("kind") == "talent_failure" for anomaly in anomalies)

    if has_daily:
        bullets.append("I didn't finish the full overnight review.")
    if has_activity:
        bullets.append("I didn't finish writing all of yesterday's notes.")
    if has_failure and not has_daily and not has_activity:
        bullets.append("Some of my overnight work didn't finish.")

    if not knowledge_graph.get("fresh"):
        bullets.append("I didn't refresh your knowledge graph overnight.")
    if not briefing.get("valid"):
        bullets.append("I didn't prepare your morning briefing overnight.")
    return bullets


def _summarize_yesterday_processing(
    yesterday: str, journal_age_days: int
) -> dict[str, Any] | None:
    stats_data = _load_yesterday_stats(yesterday)
    if stats_data is None or journal_age_days == 0:
        return None

    stats = stats_data.get("stats", {})
    transcript_seconds = float(stats.get("transcript_duration", 0) or 0)
    transcript_segments = int(stats.get("transcript_segments", 0) or 0)
    facet_data = stats_data.get("facet_data", {})
    has_facet_activity = any(
        float(facet.get("minutes", 0) or 0) > 0 or int(facet.get("count", 0) or 0) > 0
        for facet in facet_data.values()
    )

    entities = _collect_entities_yesterday(yesterday)
    activities = _collect_top_activities_yesterday(yesterday)
    if (
        transcript_seconds <= 0
        and transcript_segments <= 0
        and not has_facet_activity
        and not activities
        and not entities
    ):
        return None

    pipeline_summary = _load_yesterday_pipeline_summary(yesterday)
    knowledge_graph = _knowledge_graph_freshness(yesterday)
    briefing = _briefing_freshness(_today())
    successful_newsletters, attempted_newsletters = (
        _newsletter_attempts_from_think_logs(yesterday)
    )

    is_sparse = (
        (transcript_seconds > 0 or transcript_segments > 0)
        and not has_facet_activity
        and not activities
    )

    status_reasons = []
    if attempted_newsletters > successful_newsletters:
        status_reasons.append("newsletter_partial")
    if pipeline_summary.get("status") != "healthy":
        status_reasons.append("pipeline_warning")
    if not knowledge_graph.get("fresh"):
        status_reasons.append("knowledge_graph_stale")
    if not briefing.get("valid"):
        status_reasons.append("briefing_missing")

    if is_sparse:
        mode = "sparse"
    elif status_reasons:
        mode = "degraded"
    else:
        mode = "healthy"

    first_week_framing = (
        _FIRST_WEEK_FRAMING if journal_age_days <= 7 and mode != "sparse" else None
    )

    if mode == "sparse":
        summary_line = (
            f"I watched {_format_duration(transcript_seconds / 60)} yesterday."
        )
        return {
            "title": "Yesterday's processing",
            "mode": mode,
            "default_collapsed": False,
            "first_week_framing": None,
            "summary_line": summary_line,
            "details": None,
            "sparse_lines": [
                "I didn't produce any facet newsletters.",
                "There wasn't much else to process.",
            ],
            "status_reasons": status_reasons,
        }

    details = []
    if mode == "degraded":
        details.extend(_format_gap_bullets(pipeline_summary, knowledge_graph, briefing))

    details.append(
        _format_newsletter_summary(successful_newsletters, attempted_newsletters)
    )
    if knowledge_graph.get("fresh"):
        details.append(
            "I refreshed your knowledge graph"
            + (
                f" at {knowledge_graph['updated_label']}."
                if knowledge_graph.get("updated_label")
                else "."
            )
        )
    if briefing.get("valid"):
        details.append(
            "I prepared your morning briefing"
            + (
                f" at {briefing['generated_label']}."
                if briefing.get("generated_label")
                else "."
            )
        )

    heatmap_summary = _format_heatmap_summary(stats_data)
    if heatmap_summary:
        details.append(heatmap_summary)

    for activity in activities[:2]:
        details.append(_format_activity_label(activity))

    entity_summary = _format_entity_summary(entities)
    if entity_summary:
        details.append(entity_summary)

    default_collapsed = mode == "healthy" and journal_age_days >= 8
    return {
        "title": (
            "⚠ Yesterday's processing"
            if mode == "degraded"
            else "Yesterday's processing"
        ),
        "mode": mode,
        "default_collapsed": default_collapsed,
        "first_week_framing": first_week_framing,
        "summary_line": _format_processing_summary(
            mode,
            successful_newsletters,
            attempted_newsletters,
            knowledge_graph,
            briefing,
        ),
        "details": details,
        "sparse_lines": None,
        "status_reasons": status_reasons,
    }


def _freshness_hours(cadence) -> int:
    """Return freshness window in hours based on routine cadence type."""
    if isinstance(cadence, dict):
        return 24
    if isinstance(cadence, str):
        fields = cadence.split()
        if len(fields) == 5:
            dom, dow = fields[2], fields[4]
            if dom == "*" and dow == "*":
                return 24
            return 168
    return 24


def _extract_summary(output_path: Path) -> str:
    """Extract a concise routine summary from a markdown output file."""
    try:
        lines = output_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return ""

    if lines and lines[0].strip() == "---":
        for i in range(1, len(lines)):
            if lines[i].strip() == "---":
                lines = lines[i + 1 :]
                break

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if len(stripped) > 80:
            return stripped[:79] + "…"
        return stripped
    return ""


def _load_routines_state() -> dict[str, Any]:
    """Load routines seen state from routines/state.json."""
    state_path = Path(get_journal()) / "routines" / "state.json"
    if not state_path.exists():
        return {}
    try:
        with open(state_path, encoding="utf-8") as f:
            raw = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}
    return raw if isinstance(raw, dict) else {}


def _save_routines_state(state: dict[str, Any]) -> None:
    """Persist routines seen state to routines/state.json."""
    routines_dir = Path(get_journal()) / "routines"
    routines_dir.mkdir(parents=True, exist_ok=True)
    state_path = routines_dir / "state.json"

    fd, tmp_path = tempfile.mkstemp(dir=routines_dir, suffix=".tmp", prefix=".state_")
    tmp_file = Path(tmp_path)
    try:
        with open(fd, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        tmp_file.replace(state_path)
    except BaseException:
        tmp_file.unlink(missing_ok=True)
        raise


def _load_skills_state() -> dict[str, Any]:
    """Load skills seen state from skills/state.json."""
    state_path = Path(get_journal()) / "skills" / "state.json"
    if not state_path.exists():
        return {}
    try:
        with open(state_path, encoding="utf-8") as f:
            raw = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}
    return raw if isinstance(raw, dict) else {}


def _save_skills_state(state: dict[str, Any]) -> None:
    """Persist skills seen state to skills/state.json."""
    skills_dir = Path(get_journal()) / "skills"
    skills_dir.mkdir(parents=True, exist_ok=True)
    state_path = skills_dir / "state.json"

    fd, tmp_path = tempfile.mkstemp(dir=skills_dir, suffix=".tmp", prefix=".state_")
    tmp_file = Path(tmp_path)
    try:
        with open(fd, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        tmp_file.replace(state_path)
    except BaseException:
        tmp_file.unlink(missing_ok=True)
        raise


def _collect_routines() -> list[dict[str, Any]]:
    """Collect recent routine outputs for display."""
    from think.routines import get_config as get_routines_config

    try:
        config = get_routines_config()
        state = _load_routines_state()
        last_seen = state.get("routines_last_seen")
        last_seen_dt = datetime.fromisoformat(last_seen) if last_seen else None

        now = datetime.now()
        journal = Path(get_journal())
        routines = []

        for value in config.values():
            if not isinstance(value, dict):
                continue
            if not value.get("enabled"):
                continue
            last_run = value.get("last_run")
            if not last_run:
                continue

            try:
                last_run_dt = datetime.fromisoformat(
                    last_run.replace("Z", "+00:00")
                ).replace(tzinfo=None)
            except (ValueError, AttributeError):
                continue

            freshness = _freshness_hours(value.get("cadence"))
            if (now - last_run_dt).total_seconds() > freshness * 3600:
                continue

            delta = now - last_run_dt
            if delta.total_seconds() < 60:
                run_time_display = "just now"
            elif delta.total_seconds() < 3600:
                run_time_display = f"{int(delta.total_seconds() / 60)}m ago"
            else:
                run_time_display = f"{int(delta.total_seconds() / 3600)}h ago"

            routine_id = value.get("id", "")
            output_dir = journal / "routines" / routine_id
            summary = ""
            if output_dir.exists():
                outputs = sorted(
                    output_dir.glob("*.md"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                if outputs:
                    summary = _extract_summary(outputs[0])

            seen = last_seen_dt is not None and last_run_dt <= last_seen_dt

            routines.append(
                {
                    "id": routine_id,
                    "name": value.get("name", routine_id),
                    "last_run": last_run,
                    "run_time_display": run_time_display,
                    "summary": summary,
                    "seen": seen,
                }
            )

        routines.sort(key=lambda r: r["last_run"], reverse=True)
        return routines
    except Exception:
        logger.warning("home: failed to collect routines", exc_info=True)
        return []


def _collect_skills() -> list[dict[str, Any]]:
    """Collect mature skills from all enabled facets."""
    try:
        journal = Path(get_journal())
        state = _load_skills_state()
        last_seen = state.get("skills_last_seen")
        last_seen_dt = datetime.fromisoformat(last_seen) if last_seen else None

        skills = []
        for facet_name in get_enabled_facets():
            skills_dir = journal / "facets" / facet_name / "skills"
            if not skills_dir.exists():
                continue

            # Read patterns.jsonl for mature skills (skill_generated: true)
            patterns_path = skills_dir / "patterns.jsonl"
            if not patterns_path.exists():
                continue

            mature_ids = set()
            try:
                with open(patterns_path, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            pattern = json.loads(line)
                            if pattern.get("skill_generated"):
                                mature_ids.add(pattern.get("id", ""))
                        except json.JSONDecodeError:
                            continue
            except OSError:
                continue

            for skill_id in sorted(mature_ids):
                skill_path = skills_dir / f"{skill_id}.md"
                if not skill_path.exists():
                    continue
                try:
                    post = frontmatter.load(str(skill_path))
                    meta = post.metadata

                    summary = meta.get("activity_type", "")
                    typical_time = meta.get("typical_time", "")
                    if typical_time:
                        summary = (
                            f"{summary} · {typical_time}" if summary else typical_time
                        )

                    observations = meta.get("observations", 0)
                    last_seen_str = meta.get("last_seen", "")

                    try:
                        mtime_dt = datetime.fromtimestamp(skill_path.stat().st_mtime)
                    except OSError:
                        mtime_dt = None

                    seen = (
                        last_seen_dt is not None
                        and mtime_dt is not None
                        and mtime_dt <= last_seen_dt
                    )

                    skills.append(
                        {
                            "id": skill_id,
                            "name": meta.get("name", skill_id),
                            "facet": facet_name,
                            "summary": summary,
                            "observations": observations,
                            "last_seen": last_seen_str,
                            "content": post.content,
                            "seen": seen,
                        }
                    )
                except Exception:
                    logger.warning(
                        "home: failed to load skill %s/%s",
                        facet_name,
                        skill_id,
                        exc_info=True,
                    )
                    continue

        skills.sort(key=lambda s: s.get("last_seen", ""), reverse=True)
        return skills
    except Exception:
        logger.warning("home: failed to collect skills", exc_info=True)
        return []


def _build_pulse_context() -> dict[str, Any]:
    """Build the full Pulse page context."""
    today = _today()
    yesterday = _yesterday()
    now = datetime.now()
    journal_age_days = _count_journal_age_days(today)

    capture_status = get_capture_health()["status"]
    cached = get_cached_state()
    last_observe_ts = cached.get("last_observe_ts")
    attention = _resolve_attention(get_current())

    stats_data = _load_stats(today)
    stats = stats_data.get("stats", {})
    segment_count = stats.get("transcript_segments", 0)
    duration_seconds = stats.get("transcript_duration", 0)
    duration_minutes = round(duration_seconds / 60) if duration_seconds else 0
    facet_data = stats_data.get("facet_data", {})

    flow_content, flow_mtime = _load_flow_md(today)
    flow_updated_at = None
    if flow_mtime:
        flow_updated_at = datetime.fromtimestamp(flow_mtime).strftime("%H:%M")

    # Try pulse.md as primary narrative, fall back to flow.md
    pulse_content, pulse_meta, pulse_needs = _load_pulse_md()
    if pulse_content:
        narrative_content = pulse_content
        narrative_source = "pulse"
        narrative_header = "pulse"
        updated = pulse_meta.get("updated", "")
        if isinstance(updated, str):
            try:
                narrative_updated_at = datetime.fromisoformat(updated).strftime("%H:%M")
            except ValueError:
                narrative_updated_at = flow_updated_at
        elif hasattr(updated, "strftime"):
            narrative_updated_at = updated.strftime("%H:%M")
        else:
            narrative_updated_at = flow_updated_at
    else:
        narrative_content = flow_content
        narrative_source = "flow"
        narrative_header = "today's flow"
        narrative_updated_at = flow_updated_at
        pulse_needs = []

    anticipated_activities = _collect_anticipated_activities(today)
    activities = _collect_activities(today)
    todos = _collect_todos(today)
    entities = _collect_entities_today(today)
    routines = _collect_routines()
    skills = _collect_skills()

    last_observe_relative = None
    if last_observe_ts:
        try:
            delta = now - datetime.fromtimestamp(last_observe_ts)
            if delta.total_seconds() < 60:
                last_observe_relative = "just now"
            elif delta.total_seconds() < 3600:
                mins = int(delta.total_seconds() / 60)
                last_observe_relative = f"{mins}m ago"
            else:
                hours = int(delta.total_seconds() / 3600)
                last_observe_relative = f"{hours}h ago"
        except Exception:
            logger.warning(
                "home: failed to compute last_observe_relative", exc_info=True
            )

    # Briefing card
    briefing_sections, briefing_meta, briefing_needs = _load_briefing_md(today)
    briefing_exists = bool(briefing_sections)
    briefing_phase = _compute_briefing_phase(segment_count, now.hour, briefing_exists)
    unseen_routines = [r for r in routines if not r["seen"]]
    unseen_skills = [s for s in skills if not s["seen"]]
    show_welcome = (
        narrative_content is None
        and not anticipated_activities
        and not activities
        and not todos
        and not entities
        and not unseen_routines
        and not skills
        and not briefing_exists
        and not attention
        and not pulse_needs
    )

    # Section summaries for collapsed state
    narrative_summary = ""
    if narrative_content:
        narrative_summary = narrative_header
        if narrative_updated_at:
            narrative_summary += f" — updated {narrative_updated_at}"

    routines_summary = ""
    if unseen_routines:
        n = len(unseen_routines)
        routines_summary = f"{n} new routine{'s' if n != 1 else ''}"

    skills_summary = ""
    if skills:
        new_count = len(unseen_skills)
        total = len(skills)
        if new_count:
            skills_summary = f"{new_count} new, {total} total"
        else:
            skills_summary = f"{total} skill{'s' if total != 1 else ''}"

    skills_content = {s["id"]: s["content"] for s in skills}

    today_summary_parts = []
    if anticipated_activities:
        n = len(anticipated_activities)
        today_summary_parts.append(f"{n} anticipated activit{'ies' if n != 1 else 'y'}")
    if activities:
        n = len(activities)
        today_summary_parts.append(f"{n} {'activities' if n != 1 else 'activity'}")
    today_summary = ", ".join(today_summary_parts)

    needs_count = len(pulse_needs) + len(todos) + (1 if attention else 0)
    needs_summary = ""
    if needs_count:
        needs_summary = (
            f"{needs_count} item{'s' if needs_count != 1 else ''} "
            f"need{'s' if needs_count == 1 else ''} attention"
        )

    network_summary = ""
    if entities:
        people = sum(1 for e in entities if e.get("entity_type") == "person")
        others = len(entities) - people
        if people and others:
            network_summary = (
                f"{people} {'people' if people != 1 else 'person'}, {others} more"
            )
        elif people:
            network_summary = f"{people} {'people' if people != 1 else 'person'}"
        else:
            network_summary = (
                f"{len(entities)} {'entities' if len(entities) != 1 else 'entity'}"
            )

    pulse_needs_normalized = {_normalize_item(item) for item in pulse_needs}
    briefing_needs_deduped = []
    briefing_needs_shared_count = 0
    for item in briefing_needs:
        if _normalize_item(item) in pulse_needs_normalized:
            briefing_needs_shared_count += 1
        else:
            briefing_needs_deduped.append(item)

    briefing_needs_badge = None
    if briefing_needs_shared_count > 0:
        s = "" if briefing_needs_shared_count == 1 else "s"
        briefing_needs_badge = (
            f"{briefing_needs_shared_count} item{s} also in Pulse needs"
        )

    briefing_summary = None
    if briefing_phase == "active":
        briefing_summary = _briefing_summary(
            briefing_sections, len(briefing_needs_deduped)
        )

    try:
        _summary = summarize_pipeline_day(_today())
        pipeline_status = pipeline_status_message(_summary)
    except Exception:
        logger.warning("pipeline_status unavailable", exc_info=True)
        pipeline_status = None

    yesterday_processing = _summarize_yesterday_processing(yesterday, journal_age_days)

    return {
        "today": today,
        "now": now,
        "capture_status": capture_status,
        "last_observe_relative": last_observe_relative,
        "attention": attention,
        "pipeline_status": pipeline_status,
        "segment_count": segment_count,
        "duration_minutes": duration_minutes,
        "facet_data": facet_data,
        "narrative_content": narrative_content,
        "narrative_updated_at": narrative_updated_at,
        "narrative_source": narrative_source,
        "narrative_header": narrative_header,
        "pulse_needs": pulse_needs,
        "flow_content": flow_content,
        "flow_updated_at": flow_updated_at,
        "anticipated_activities": anticipated_activities,
        "activities": activities,
        "todos": todos,
        "entities": entities,
        "routines": routines,
        "skills": skills,
        "skills_summary": skills_summary,
        "skills_content": skills_content,
        "briefing_sections": briefing_sections,
        "briefing_meta": briefing_meta,
        "briefing_phase": briefing_phase,
        "briefing_exists": briefing_exists,
        "briefing_summary": briefing_summary,
        "briefing_needs_deduped": briefing_needs_deduped,
        "briefing_needs_shared_count": briefing_needs_shared_count,
        "briefing_needs_badge": briefing_needs_badge,
        "yesterday_processing": yesterday_processing,
        "show_welcome": show_welcome,
        "narrative_summary": narrative_summary,
        "routines_summary": routines_summary,
        "today_summary": today_summary,
        "needs_summary": needs_summary,
        "network_summary": network_summary,
    }


@home_bp.route("/")
def index():
    ctx = _build_pulse_context()
    return render_template("app.html", **ctx)


@home_bp.route("/api/pulse")
def api_pulse():
    """Aggregated JSON for client-side refresh after WebSocket events."""
    ctx = _build_pulse_context()
    attention = ctx.get("attention")
    if attention:
        ctx["attention"] = {
            "placeholder_text": attention.placeholder_text,
            "context_lines": attention.context_lines,
        }
    ctx.pop("show_welcome", None)
    ctx["now"] = ctx["now"].isoformat()
    return jsonify(ctx)


@home_bp.route("/api/routines/seen", methods=["POST"])
def api_routines_seen():
    """Mark routines as seen."""
    state = _load_routines_state()
    state["routines_last_seen"] = datetime.utcnow().isoformat()
    _save_routines_state(state)
    return jsonify({"ok": True})


@home_bp.route("/api/skills/seen", methods=["POST"])
def api_skills_seen():
    """Mark skills as seen."""
    state = _load_skills_state()
    state["skills_last_seen"] = datetime.now(timezone.utc).isoformat()
    _save_skills_state(state)
    return jsonify({"ok": True})


@home_bp.route("/api/briefing")
def api_briefing():
    """Briefing-specific JSON for WebSocket-triggered refresh."""
    ctx = _build_pulse_context()
    meta = ctx.get("briefing_meta")
    if meta:
        generated = meta.get("generated")
        if hasattr(generated, "isoformat"):
            meta = dict(meta)
            meta["generated"] = generated.isoformat()
        if "date" in meta:
            meta["date"] = str(meta["date"])
    return jsonify(
        {
            "exists": ctx["briefing_exists"],
            "phase": ctx["briefing_phase"],
            "summary": ctx["briefing_summary"],
            "meta": meta,
            "sections": ctx["briefing_sections"],
            "needs_deduped": ctx["briefing_needs_deduped"],
            "needs_shared_count": ctx["briefing_needs_shared_count"],
            "needs_badge": ctx["briefing_needs_badge"],
        }
    )
