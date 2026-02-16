# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

import re
from typing import Any

from flask import Blueprint, jsonify, request

from convey.utils import format_date
from think.facets import get_facets
from think.indexer.journal import search_counts, search_journal

search_bp = Blueprint(
    "app:search",
    __name__,
    url_prefix="/app/search",
)

# Agent icons for display
AGENT_ICONS = {
    "flow": "📝",
    "knowledge_graph": "🗺️",
    "meetings": "📅",
    "event": "📅",
    "audio": "🎤",
    "screen": "🖥️",
    "todo": "✅",
    "entity": "👤",
    "entity:attached": "👤",
    "entity:detected": "👤",
    "news": "📰",
    "import": "📥",
}

# Agent display names
AGENT_LABELS = {
    "flow": "Flow",
    "knowledge_graph": "Knowledge Graph",
    "meetings": "Meetings",
    "event": "Event",
    "audio": "Transcript",
    "screen": "Screen",
    "todo": "Todo",
    "entity": "Entity",
    "entity:attached": "Entity",
    "entity:detected": "Entity",
    "news": "News",
    "import": "Import",
}


def _parse_facet_filter() -> str | None:
    """Parse facet filter from request args."""
    return request.args.get("facet", "").strip() or None


def _parse_agent_filter() -> str | None:
    """Parse single agent filter from request args."""
    return request.args.get("agent", "").strip() or None


def _parse_stream_filter() -> str | None:
    """Parse stream filter from request args."""
    return request.args.get("stream", "").strip() or None


def _highlight_query_terms(text: str, query: str) -> str:
    """Add bold highlighting around query terms in text."""
    if not query:
        return text
    # Extract individual words from query (ignore FTS operators)
    terms = [t for t in query.split() if t.upper() not in ("AND", "OR", "NOT")]
    for term in terms:
        # Remove quotes and wildcards for matching
        clean_term = term.strip('"*')
        if len(clean_term) >= 2:
            # Case-insensitive replacement with bold
            pattern = re.compile(re.escape(clean_term), re.IGNORECASE)
            text = pattern.sub(lambda m: f"<strong>{m.group()}</strong>", text)
    return text


def _format_result(result: dict, query: str, facets_map: dict) -> dict:
    """Format a search result for API response."""
    meta = result.get("metadata", {})
    agent = meta.get("agent", "")
    text = result.get("text", "")
    facet_name = meta.get("facet", "")

    # Get facet metadata
    facet_info = facets_map.get(facet_name, {})

    # Clean and truncate text
    words = text.split()
    if len(words) > 50:
        display_text = " ".join(words[:50]) + "..."
    else:
        display_text = text

    # Apply highlighting
    display_text = _highlight_query_terms(display_text, query)

    return {
        "id": result.get("id", ""),
        "day": meta.get("day", ""),
        "agent": agent,
        "agent_icon": AGENT_ICONS.get(agent, "📄"),
        "agent_label": AGENT_LABELS.get(agent, agent.title()),
        "facet": facet_name,
        "facet_title": facet_info.get("title", facet_name),
        "facet_color": facet_info.get("color", ""),
        "facet_emoji": facet_info.get("emoji", ""),
        "text": display_text,
        "stream": meta.get("stream", ""),
        "path": meta.get("path", ""),
        "idx": meta.get("idx", 0),
        "score": result.get("score", 0.0),
    }


@search_bp.route("/api/search")
def search_journal_api() -> Any:
    """Unified journal search endpoint with day grouping.

    Query parameters:
        q: Search query (required)
        limit: Max results per day (default 5)
        offset: Day offset for pagination (default 0)
        facet: Filter by facet name (optional, empty string for no-facet items)
        agent: Filter by single agent (optional)

    Returns:
        JSON with:
        - total: Total match count
        - days: List of day groups, each with date info and results
        - facets: List of facets with counts for filter sidebar
        - agents: List of agents with counts for filter sidebar
    """
    query = request.args.get("q", "").strip()

    # Parse parameters
    results_per_day = int(request.args.get("limit", 5))
    day_offset = int(request.args.get("offset", 0))
    facet_filter = _parse_facet_filter()
    agent_filter = _parse_agent_filter()
    stream_filter = _parse_stream_filter()

    # Load facet metadata for enriching results
    facets_map = get_facets()

    # Get aggregation counts efficiently (lightweight query, no content)
    # First get unfiltered counts for sidebar display
    base_counts = search_counts(query, stream=stream_filter)
    facet_counts = dict(base_counts["facets"])
    agent_counts = dict(base_counts["agents"])

    # Get filtered counts for results
    filtered_counts = search_counts(
        query, facet=facet_filter, agent=agent_filter, stream=stream_filter
    )
    day_counts = dict(filtered_counts["days"])

    # Determine which days to show (sorted descending)
    sorted_days = sorted(day_counts.keys(), reverse=True)

    # Apply day pagination
    paginated_days = sorted_days[day_offset : day_offset + 20]

    # Fetch results for each paginated day
    days_response = []
    for day in paginated_days:
        _, day_results = search_journal(
            query,
            limit=results_per_day,
            offset=0,
            day=day,
            facet=facet_filter,
            agent=agent_filter,
            stream=stream_filter,
        )
        total_in_day = day_counts.get(day, 0)

        formatted_results = [_format_result(r, query, facets_map) for r in day_results]

        days_response.append(
            {
                "day": day,
                "date": format_date(day),
                "total": total_in_day,
                "showing": len(formatted_results),
                "has_more": total_in_day > results_per_day,
                "results": formatted_results,
            }
        )

    # Build facet list for sidebar with counts (unfiltered counts for discovery)
    facets_list = []
    for name, data in facets_map.items():
        if data.get("muted"):
            continue
        facets_list.append(
            {
                "name": name,
                "title": data.get("title", name),
                "color": data.get("color", ""),
                "emoji": data.get("emoji", ""),
                "count": facet_counts.get(name, 0),
            }
        )
    # Sort by count descending
    facets_list.sort(key=lambda x: x["count"], reverse=True)

    # Build agent list for sidebar (unfiltered counts for discovery)
    agents_list = []
    for agent, count in sorted(agent_counts.items(), key=lambda x: -x[1]):
        agents_list.append(
            {
                "name": agent,
                "label": AGENT_LABELS.get(agent, agent.title()),
                "icon": AGENT_ICONS.get(agent, "📄"),
                "count": count,
            }
        )

    return jsonify(
        {
            "total": filtered_counts["total"],
            "total_days": len(sorted_days),
            "showing_days": len(days_response),
            "days": days_response,
            "facets": facets_list,
            "agents": agents_list,
        }
    )


@search_bp.route("/api/day_results")
def day_results_api() -> Any:
    """Get more results for a specific day.

    Query parameters:
        q: Search query
        day: Day to get results for (YYYYMMDD)
        offset: Result offset within the day (default 0)
        limit: Max results (default 20)
        facet: Facet filter (optional)
        agent: Single agent filter (optional)
    """
    query = request.args.get("q", "").strip()
    day = request.args.get("day", "").strip()
    if not day:
        return jsonify({"results": [], "total": 0})

    offset = int(request.args.get("offset", 0))
    limit = int(request.args.get("limit", 20))
    facet_filter = _parse_facet_filter()
    agent_filter = _parse_agent_filter()
    stream_filter = _parse_stream_filter()

    facets_map = get_facets()

    # Get total count for this day with filters
    counts = search_counts(
        query, day=day, facet=facet_filter, agent=agent_filter, stream=stream_filter
    )
    total_in_day = counts["total"]

    # Fetch paginated results
    _, rows = search_journal(
        query,
        limit=limit,
        offset=offset,
        day=day,
        facet=facet_filter,
        agent=agent_filter,
        stream=stream_filter,
    )

    formatted = [_format_result(r, query, facets_map) for r in rows]

    return jsonify(
        {
            "day": day,
            "total": total_in_day,
            "offset": offset,
            "results": formatted,
        }
    )
