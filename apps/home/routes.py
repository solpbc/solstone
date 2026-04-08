# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Home app - Pulse landing page."""

from __future__ import annotations

import json
import logging
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

import frontmatter
from flask import Blueprint, jsonify, render_template

from convey.apps import _resolve_attention
from convey.bridge import get_cached_state
from think.awareness import get_current
from think.facets import get_facets
from think.indexer.journal import get_journal_index
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


def _today() -> str:
    return datetime.now().strftime("%Y%m%d")


def _load_flow_md(today: str) -> tuple[str | None, float | None]:
    """Load today's flow.md content and mtime. Returns (content, mtime) or (None, None)."""
    try:
        journal = Path(get_journal())
        flow_path = journal / today / "agents" / "flow.md"
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


def _collect_events(today: str) -> list[dict[str, Any]]:
    """Collect calendar events across all facets."""
    from think.indexer.journal import get_events

    try:
        events = get_events(today)
        for event in events:
            if event.get("start") is None:
                event["start"] = ""
            if event.get("end") is None:
                event["end"] = ""
        return events
    except Exception:
        logger.warning("home: failed to collect events", exc_info=True)
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


def _build_pulse_context() -> dict[str, Any]:
    """Build the full Pulse page context."""
    today = _today()
    now = datetime.now()

    awareness = get_current()
    capture_status = awareness.get("capture", {}).get("status", "unknown")
    cached = get_cached_state()
    last_observe_ts = cached.get("last_observe_ts")
    attention = _resolve_attention(awareness)

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

    events = _collect_events(today)
    activities = _collect_activities(today)
    todos = _collect_todos(today)
    entities = _collect_entities_today(today)
    routines = _collect_routines()

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
    show_welcome = (
        narrative_content is None
        and not events
        and not activities
        and not todos
        and not entities
        and not unseen_routines
        and not briefing_exists
        and not attention
        and not pulse_needs
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

    return {
        "today": today,
        "now": now,
        "capture_status": capture_status,
        "last_observe_relative": last_observe_relative,
        "attention": attention,
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
        "events": events,
        "activities": activities,
        "todos": todos,
        "entities": entities,
        "routines": routines,
        "briefing_sections": briefing_sections,
        "briefing_meta": briefing_meta,
        "briefing_phase": briefing_phase,
        "briefing_exists": briefing_exists,
        "briefing_summary": briefing_summary,
        "briefing_needs_deduped": briefing_needs_deduped,
        "briefing_needs_shared_count": briefing_needs_shared_count,
        "briefing_needs_badge": briefing_needs_badge,
        "show_welcome": show_welcome,
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
