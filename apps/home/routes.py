# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Home app - Pulse landing page."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from flask import Blueprint, jsonify, render_template

from think.awareness import get_current
from think.facets import get_facets
from think.indexer.journal import get_journal_index
from think.utils import get_journal

from convey.apps import _resolve_attention
from convey.bridge import get_cached_state

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
        pass
    return None, None


def _load_stats(today: str) -> dict[str, Any]:
    """Load per-day stats.json. Returns empty dict if missing."""
    try:
        journal = Path(get_journal())
        stats_path = journal / today / "stats.json"
        if stats_path.exists():
            return json.loads(stats_path.read_text())
    except Exception:
        pass
    return {}


def _collect_todos(today: str) -> list[dict[str, Any]]:
    """Collect pending todos across all facets."""
    from apps.todos.todo import get_todos

    todos = []
    try:
        facets = get_facets()
    except Exception:
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
        return get_events(today)
    except Exception:
        return []


def _collect_activities(today: str) -> list[dict[str, Any]]:
    """Collect recent activities across all facets, last ~4 hours."""
    from think.activities import load_activity_records

    activities = []
    try:
        facets = get_facets()
    except Exception:
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

    events = _collect_events(today)
    activities = _collect_activities(today)
    todos = _collect_todos(today)
    entities = _collect_entities_today(today)

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
            pass

    return {
        "today": today,
        "now": now,
        "capture_status": capture_status,
        "last_observe_relative": last_observe_relative,
        "attention": attention,
        "segment_count": segment_count,
        "duration_minutes": duration_minutes,
        "facet_data": facet_data,
        "flow_content": flow_content,
        "flow_updated_at": flow_updated_at,
        "events": events,
        "activities": activities,
        "todos": todos,
        "entities": entities,
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
    ctx["now"] = ctx["now"].isoformat()
    return jsonify(ctx)
