"""Home app - daily dashboard with facet-aware summary."""

from __future__ import annotations

import re
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any

from flask import Blueprint, jsonify, redirect, render_template, url_for

from apps.todos.todo import get_todos
from convey import state
from convey.utils import DATE_RE
from think.entities import load_entities
from think.facets import get_facet_news, get_facets
from think.indexer import search_events

home_bp = Blueprint(
    "app:home",
    __name__,
    url_prefix="/app/home",
)


def _get_day_summary(day: str) -> dict[str, Any]:
    """Aggregate dashboard data for a day, organized by facet.

    Returns dict with facet-keyed data for todos, events, entities, and news.
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
        "totals": {
            "todos_pending": 0,
            "todos_completed": 0,
            "events": 0,
            "entities": 0,
        },
    }

    # Events by topic (aggregated across facets for totals)
    events_by_topic: dict[str, int] = defaultdict(int)

    for facet_name in facet_names:
        facet_data: dict[str, Any] = {
            "todos": [],
            "events_by_topic": {},
            "entities": [],
            "news_preview": None,
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

        # Events (search for this facet on this day)
        try:
            _, events = search_events(query="", day=day, facet=facet_name, limit=100)
            facet_events: dict[str, int] = defaultdict(int)
            for event in events:
                topic = event.get("metadata", {}).get("topic", "other")
                facet_events[topic] += 1
                events_by_topic[topic] += 1
                result["totals"]["events"] += 1
            facet_data["events_by_topic"] = dict(facet_events)
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

        # News preview
        try:
            news = get_facet_news(facet_name, day=day)
            days = news.get("days", [])
            if days:
                content = days[0].get("raw_content", "")
                if content:
                    facet_data["news_preview"] = content[:150] + (
                        "..." if len(content) > 150 else ""
                    )
        except Exception:
            pass

        result["facets"][facet_name] = facet_data

    # Add aggregated events by topic to totals
    result["totals"]["events_by_topic"] = dict(events_by_topic)

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
            # Check for insights
            insights_dir = day_dir / "insights"
            if insights_dir.is_dir():
                count += len(list(insights_dir.glob("*.md")))

        if count > 0:
            stats[day_name] = count

    return jsonify(stats)
