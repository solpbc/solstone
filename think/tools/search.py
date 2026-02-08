# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Journal search and event query functions.

These functions query the journal SQLite index and can be imported directly.
"""

from datetime import datetime, timedelta
from typing import Any

from think.indexer.journal import get_events as get_events_impl
from think.indexer.journal import search_counts as search_counts_impl
from think.indexer.journal import search_journal as search_journal_impl


def _bucket_day_counts(day_counts: dict[str, int]) -> dict[str, Any]:
    """Bucket day counts into recent days, top days, and bucketed days.

    Returns:
        Dict with:
        - recent_days: Last 7 days individually (includes 0 counts)
        - top_days: Top 20 days by count
        - bucketed_days: Older days grouped by week (day_from-day_to format)
    """
    today = datetime.now()

    # Generate last 7 days (including today)
    recent_dates = []
    for i in range(7):
        d = today - timedelta(days=i)
        recent_dates.append(d.strftime("%Y%m%d"))

    recent_days = {d: day_counts.get(d, 0) for d in recent_dates}

    # Top 20 days by count
    sorted_days = sorted(day_counts.items(), key=lambda x: (-x[1], x[0]))
    top_days = dict(sorted_days[:20])

    # Weekly buckets for days older than 7 days
    cutoff = (today - timedelta(days=7)).strftime("%Y%m%d")
    older_days = {d: c for d, c in day_counts.items() if d < cutoff}

    weekly_buckets: dict[str, int] = {}
    for day_str, count in older_days.items():
        try:
            day_date = datetime.strptime(day_str, "%Y%m%d")
            # Find the Monday of that week
            week_start = day_date - timedelta(days=day_date.weekday())
            week_end = week_start + timedelta(days=6)
            bucket_key = (
                f"{week_start.strftime('%Y%m%d')}-{week_end.strftime('%Y%m%d')}"
            )
            weekly_buckets[bucket_key] = weekly_buckets.get(bucket_key, 0) + count
        except ValueError:
            continue

    # Sort bucketed days by start date descending, omit empty weeks
    bucketed_days = dict(
        sorted(
            ((k, v) for k, v in weekly_buckets.items() if v > 0),
            key=lambda x: x[0],
            reverse=True,
        )
    )

    return {
        "recent_days": recent_days,
        "top_days": top_days,
        "bucketed_days": bucketed_days,
    }


def search_journal(
    query: str = "",
    limit: int = 10,
    offset: int = 0,
    *,
    day: str | None = None,
    day_from: str | None = None,
    day_to: str | None = None,
    facet: str | None = None,
    topic: str | None = None,
) -> dict[str, Any]:
    """Search across all journal content using semantic full-text search.

    This tool searches through all indexed journal content including insights,
    transcripts, events, entities, and todos. Use filters to narrow results
    to specific content types or contexts.

    Args:
        query: Search query. Words are AND'd by default; use OR to match any
            (e.g., "apple OR orange"), quotes for exact phrases, * for prefix match.
            Empty string returns all content matching the filters.
        limit: Maximum number of results to return (default: 10)
        offset: Number of results to skip for pagination (default: 0)
        day: Filter by exact day in ``YYYYMMDD`` format (mutually exclusive with day_from/day_to)
        day_from: Filter by date range start (``YYYYMMDD``, inclusive)
        day_to: Filter by date range end (``YYYYMMDD``, inclusive)
        facet: Filter by facet name (e.g., "work", "personal")
        topic: Filter by topic (e.g., "flow", "audio", "event", "entity:detected")

    Returns:
        Dictionary containing:
        - total: Total number of matching results
        - limit: Current limit value
        - offset: Current offset value
        - query: Echo of query text and applied filters
        - counts: Aggregation metadata with facets, topics, and bucketed days
        - results: List of matches with day, facet, topic, text, path, and idx

    Examples:
        - search_journal("machine learning")
        - search_journal("meeting notes", day="20240101")
        - search_journal("project planning", facet="work")
        - search_journal("standup", topic="audio")
        - search_journal("weekly sync", day_from="20241201", day_to="20241207")
        - search_journal(topic="audio", day="20240101")  # Browse all audio for a day
    """
    try:
        kwargs: dict[str, Any] = {}
        filters: dict[str, Any] = {}
        if day is not None:
            kwargs["day"] = day
            filters["day"] = day
        if day_from is not None:
            kwargs["day_from"] = day_from
            filters["day_from"] = day_from
        if day_to is not None:
            kwargs["day_to"] = day_to
            filters["day_to"] = day_to
        if facet is not None:
            kwargs["facet"] = facet
            filters["facet"] = facet
        if topic is not None:
            kwargs["topic"] = topic
            filters["topic"] = topic

        # Get search results
        total, results = search_journal_impl(query, limit, offset, **kwargs)

        # Get aggregation counts
        counts_data = search_counts_impl(query, **kwargs)

        # Build result items with full metadata
        items = []
        for r in results:
            meta = r.get("metadata", {})
            items.append(
                {
                    "day": meta.get("day", ""),
                    "facet": meta.get("facet", ""),
                    "topic": meta.get("topic", ""),
                    "text": r.get("text", ""),
                    "path": meta.get("path", ""),
                    "idx": meta.get("idx", 0),
                }
            )

        # Build counts structure
        day_buckets = _bucket_day_counts(dict(counts_data["days"]))
        counts = {
            "facets": dict(counts_data["facets"]),
            "topics": dict(counts_data["topics"]),
            **day_buckets,
        }

        return {
            "total": total,
            "limit": limit,
            "offset": offset,
            "query": {"text": query, "filters": filters},
            "counts": counts,
            "results": items,
        }
    except Exception as exc:
        return {
            "error": f"Failed to search journal: {exc}",
            "suggestion": "try adjusting the query or ensure the index exists (run sol indexer --rescan)",
        }


def get_events(
    day: str,
    facet: str | None = None,
) -> dict[str, Any]:
    """Get structured events for a specific day.

    This tool retrieves full event data including titles, summaries,
    start/end times, and participants. Use this when you need complete
    event information rather than text search results.

    Args:
        day: Day in ``YYYYMMDD`` format
        facet: Optional facet name to filter by

    Returns:
        Dictionary containing:
        - day: The requested day
        - facet: The facet filter (if any)
        - events: List of event objects with full structured data

    Examples:
        - get_events("20240101")
        - get_events("20240101", facet="work")
    """
    try:
        events = get_events_impl(day, facet)
        return {
            "day": day,
            "facet": facet or "",
            "events": events,
        }
    except Exception as exc:
        return {
            "error": f"Failed to get events: {exc}",
            "suggestion": "verify the day parameter is valid",
        }
