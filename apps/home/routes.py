# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Home app - daily dashboard with facet-aware summary."""

from __future__ import annotations

import re
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from flask import Blueprint, jsonify, redirect, render_template, url_for

from apps.todos.todo import get_todos
from convey import state
from convey.utils import DATE_RE
from think.entities import load_entities
from think.facets import get_facet_news, get_facets
from think.indexer.journal import get_events

# Lookback window for recent entities (days)
RECENT_ENTITIES_LOOKBACK = 1

home_bp = Blueprint(
    "app:home",
    __name__,
    url_prefix="/app/home",
)


def _get_recent_entities(day: str, facet_names: list[str]) -> dict[str, list[str]]:
    """Load attached entities with last_seen within lookback window.

    Args:
        day: Reference day in YYYYMMDD format
        facet_names: List of facet names to check

    Returns:
        Dict mapping facet name to list of entity names seen recently
    """
    # Calculate threshold date
    try:
        ref_date = datetime.strptime(day, "%Y%m%d")
        threshold = (ref_date - timedelta(days=RECENT_ENTITIES_LOOKBACK)).strftime(
            "%Y%m%d"
        )
    except ValueError:
        return {}

    recent: dict[str, list[str]] = {}

    for facet_name in facet_names:
        try:
            # Load attached entities (no day param = attached, not detected)
            entities = load_entities(facet_name)
            # Filter by last_seen >= threshold
            facet_recent = [
                e.get("name", "")
                for e in entities
                if e.get("name") and e.get("last_seen", "") >= threshold
            ]
            if facet_recent:
                recent[facet_name] = facet_recent[:10]  # Limit to 10
        except Exception:
            pass

    return recent


def _get_day_summary(day: str) -> dict[str, Any]:
    """Aggregate dashboard data for a day, organized by facet.

    Returns dict with facet-keyed data for todos, events, entities, and news.
    Also includes facet_meta with title, color, emoji, and has_activity flag.
    """
    try:
        facet_map = get_facets()
    except Exception:
        facet_map = {}

    facet_names = list(facet_map.keys())

    # Aggregate data per facet
    result: dict[str, Any] = {
        "day": day,
        "facets": {},
        "facet_meta": {},
        "totals": {
            "todos_pending": 0,
            "todos_completed": 0,
            "events": 0,
            "entities": 0,
        },
    }

    # Events by agent (aggregated across facets for totals)
    events_by_agent: dict[str, int] = defaultdict(int)

    for facet_name in facet_names:
        facet_config = facet_map.get(facet_name, {})
        facet_data: dict[str, Any] = {
            "todos": [],
            "events_by_agent": {},
            "entities": [],
            "news_content": None,
        }

        # Todos - return actual items
        todos = get_todos(day, facet_name)
        if todos:
            for todo in todos:
                facet_data["todos"].append(
                    {
                        "text": todo.get("text", ""),
                        "completed": todo.get("completed", False),
                    }
                )
                if todo.get("completed"):
                    result["totals"]["todos_completed"] += 1
                else:
                    result["totals"]["todos_pending"] += 1

        # Events (load directly from source files)
        try:
            events = get_events(day, facet=facet_name)
            facet_events: dict[str, int] = defaultdict(int)
            for event in events:
                agent = event.get("agent", "other")
                facet_events[agent] += 1
                events_by_agent[agent] += 1
                result["totals"]["events"] += 1
            facet_data["events_by_agent"] = dict(facet_events)
        except Exception:
            pass

        # Detected entities for this day
        try:
            entities = load_entities(facet_name, day)
            entity_names = [e.get("name", "") for e in entities if e.get("name")]
            facet_data["entities"] = entity_names[:10]  # Limit to 10
            result["totals"]["entities"] += len(entities)
        except Exception:
            pass

        # News content
        try:
            news = get_facet_news(facet_name, day=day)
            days = news.get("days", [])
            if days:
                content = days[0].get("raw_content", "")
                if content:
                    facet_data["news_content"] = content
        except Exception:
            pass

        result["facets"][facet_name] = facet_data

        # Determine if facet has any activity for this day
        has_activity = bool(
            facet_data["todos"]
            or facet_data["events_by_agent"]
            or facet_data["entities"]
        )

        # Extract daily goal (first pending todo)
        facet_data["goal"] = next(
            (t["text"] for t in facet_data["todos"] if not t["completed"]), None
        )

        # Add facet metadata
        result["facet_meta"][facet_name] = {
            "title": facet_config.get("title", facet_name.title()),
            "color": facet_config.get("color", "#6b7280"),
            "emoji": facet_config.get("emoji", "📁"),
            "has_activity": has_activity,
        }

    # Add aggregated events by agent to totals
    result["totals"]["events_by_agent"] = dict(events_by_agent)

    # Load upcoming items (next 3 days)
    upcoming_data = []
    try:
        ref_date = datetime.strptime(day, "%Y%m%d")
        for i in range(1, 4):
            next_day = (ref_date + timedelta(days=i)).strftime("%Y%m%d")
            day_events = get_events(next_day)
            for e in day_events:
                upcoming_data.append(
                    {
                        "type": "event",
                        "day": next_day,
                        "title": e.get("title") or e.get("summary", "Untitled Event"),
                        "facet": e.get("facet"),
                    }
                )
            for f_name in facet_names:
                f_todos = get_todos(next_day, f_name)
                if f_todos:
                    for t in f_todos:
                        if not t.get("completed") and not t.get("cancelled"):
                            upcoming_data.append(
                                {
                                    "type": "todo",
                                    "day": next_day,
                                    "title": t.get("text"),
                                    "facet": f_name,
                                }
                            )
    except Exception:
        pass
    result["upcoming"] = upcoming_data[:5]  # Limit to 5 items

    # Load recent entities (attached entities with last_seen in lookback window)
    recent_entities = _get_recent_entities(day, facet_names)
    result["recent_entities"] = recent_entities
    result["totals"]["recent_entities"] = sum(
        len(names) for names in recent_entities.values()
    )

    return result


@home_bp.route("/")
def index():
    """Redirect to today's dashboard."""
    today = date.today().strftime("%Y%m%d")
    return redirect(url_for("app:home.home_day", day=today))


@home_bp.route("/<day>")
def home_day(day: str) -> str:
    """Dashboard view for a specific day."""
    if not DATE_RE.fullmatch(day):
        return "", 404

    return render_template("app.html")


@home_bp.route("/api/summary/<day>")
def api_summary(day: str):
    """Return aggregated summary data for a day."""
    if not DATE_RE.fullmatch(day):
        return jsonify({"error": "Invalid day format"}), 400

    data = _get_day_summary(day)
    return jsonify(data)


@home_bp.route("/api/stats/<month>")
def api_stats(month: str):
    """Return activity indicator for each day in a month.

    Returns simple count (todos + events) for month picker heatmap.
    """
    if not re.fullmatch(r"\d{6}", month):
        return jsonify({"error": "Invalid month format, expected YYYYMM"}), 400

    from think.utils import day_dirs

    stats: dict[str, int] = {}

    # Get all facets
    try:
        facet_map = get_facets()
    except Exception:
        facet_map = {}

    journal_root = Path(state.journal_root)

    for day_name in day_dirs().keys():
        if not day_name.startswith(month):
            continue

        count = 0

        # Count todos across facets
        for facet_name in facet_map.keys():
            todos = get_todos(day_name, facet_name)
            if todos:
                count += len(todos)

        # Count if day directory exists (has journal data)
        day_dir = journal_root / day_name
        if day_dir.is_dir():
            # Check for agent outputs
            agents_dir = day_dir / "agents"
            if agents_dir.is_dir():
                count += len(list(agents_dir.glob("*.md")))
                count += len(list(agents_dir.glob("*/*.md")))

        if count > 0:
            stats[day_name] = count

    return jsonify(stats)
