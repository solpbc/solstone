"""MCP tools for search operations.

Note: These functions are registered as MCP tools by muse/mcp.py
They can also be imported and called directly for testing or internal use.
"""

from typing import Any

from think.indexer import search_events as search_events_impl
from think.indexer import search_insights as search_insights_impl
from think.indexer import search_transcripts as search_transcripts_impl


def search_insights(
    query: str,
    limit: int = 5,
    offset: int = 0,
    *,
    topic: str | None = None,
    day: str | None = None,
) -> dict[str, Any]:
    """Search across journal insight summaries using semantic full-text search.

    This tool searches through pre-processed insight summaries that represent
    key themes and subjects from your journal entries. Use this when looking
    for high-level concepts, themes, or when you need an overview of topics
    discussed over time.

    Args:
        query: Natural language search query (e.g., "meetings product launch")
        limit: Optional maximum number of results to return (default: 5, max: 20)
        offset: Optional number of results to skip for pagination (default: 0)
        topic: Optional topic name to filter results by
        day: Optional day to filter results by in ``YYYYMMDD`` format

    Returns:
        Dictionary containing:
        - total: Total number of matching insights
        - limit: Current limit value used for this query
        - offset: Current offset value used for this query
        - results: List of matching insights with day, topic, and text excerpt, ordered by text relevance

    Examples:
        - search_insights("machine learning projects")
        - search_insights("team retrospectives", limit=10)
        - search_insights("planning", topic="standup")
        - search_insights("meetings", day="20240101")
    """
    try:
        kwargs = {}
        if topic is not None:
            kwargs["topic"] = topic
        if day is not None:
            kwargs["day"] = day
        total, results = search_insights_impl(query, limit, offset, **kwargs)

        items = []
        for r in results:
            meta = r.get("metadata", {})
            topic = meta.get("topic", "")
            items.append(
                {
                    "day": meta.get("day", ""),
                    "topic": topic,
                    "text": r.get("text", ""),
                }
            )

        return {"total": total, "limit": limit, "offset": offset, "results": items}
    except Exception as exc:
        return {
            "error": f"Failed to search insights: {exc}",
            "suggestion": "try adjusting the query or ensure indexes exist",
        }


def search_transcripts(
    query: str,
    day: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    limit: int = 5,
    offset: int = 0,
) -> dict[str, Any]:
    """Search raw transcripts and screen diffs for a specific day or date range.

    This tool scans raw audio transcripts (``*_audio.json``) and screenshot
    diffs (``*_diff.json``) produced throughout the day(s). Use it when you need
    to recall exact wording, short snippets, or visual context from given dates.

    Args:
        query: Natural language search query (e.g., "error message")
        day: Optional specific day to search in ``YYYYMMDD`` format
        start_date: Optional start date for range search in ``YYYYMMDD`` format
        end_date: Optional end date for range search in ``YYYYMMDD`` format
        limit: Optional maximum number of results to return (default: 5, max: 20)
        offset: Optional number of results to skip for pagination (default: 0)

    Returns:
        Dictionary containing:
        - total: Total number of matching raw entries
        - limit: Current limit value used for this query
        - offset: Current offset value used for this query
        - results: List of entries with day, time, type, and text snippet

    Examples:
        - search_transcripts("error message", day="20240101")
        - search_transcripts("feature flag", start_date="20240101", end_date="20240107")
        - search_transcripts("bug fix")  # searches all days
    """
    try:
        total, results = search_transcripts_impl(
            query,
            limit=limit,
            offset=offset,
            day=day,
            start_date=start_date,
            end_date=end_date,
        )

        items = []
        for r in results:
            meta = r.get("metadata", {})
            items.append(
                {
                    "day": meta.get("day", ""),
                    "time": meta.get("time", ""),
                    "type": meta.get("type", ""),
                    "text": r.get("text", ""),
                }
            )

        return {"total": total, "limit": limit, "offset": offset, "results": items}
    except Exception as exc:
        return {
            "error": f"Failed to search raw data: {exc}",
            "suggestion": "verify the day parameter or adjust the query",
        }


def search_events(
    query: str,
    limit: int = 5,
    offset: int = 0,
    *,
    day: str | None = None,
    facet: str | None = None,
    topic: str | None = None,
    start: str | None = None,
    end: str | None = None,
) -> dict[str, Any]:
    """Search structured events extracted from journal summaries.

    This tool searches JSON event data generated from your daily summaries.
    Use it to find meetings, tasks, or other notable activities. Results may
    be filtered by day, facet, topic, or a time range.

    Args:
        query: Natural language search query (e.g., "team standup")
        limit: Optional maximum number of events to return (default: 5)
        offset: Optional number of results to skip for pagination (default: 0)
        day: Optional ``YYYYMMDD`` day to filter results
        facet: Optional facet name to filter results by (e.g., "work", "personal")
        topic: Optional topic name to filter by
        start: Optional start time to filter events starting on or after this ``HH:MM:SS`` time
        end: Optional end time to filter events ending on or before this ``HH:MM:SS`` time

    Returns:
        Dictionary with ``limit``, ``offset`` and ``results`` list containing day, facet, topic,
        start/end times and short event summaries.
        Ordered by day and start time (most recent first).

    Examples:
        - search_events("sprint review")
        - search_events("planning", facet="work")
        - search_events("planning", day="20240101", limit=10)
        - search_events("standup", limit=5, offset=10)
    """

    try:
        total, rows = search_events_impl(
            query,
            limit=limit,
            offset=offset,
            day=day,
            facet=facet,
            start=start,
            end=end,
            topic=topic,
        )

        items = []
        for r in rows:
            meta = r.get("metadata", {})
            occ = r.get("event", {})
            items.append(
                {
                    "day": meta.get("day", ""),
                    "facet": meta.get("facet", ""),
                    "topic": meta.get("topic", ""),
                    "start": meta.get("start", ""),
                    "end": meta.get("end", ""),
                    "title": occ.get("title") or r.get("text", ""),
                    "summary": occ.get("summary", ""),
                }
            )

        return {"total": total, "limit": limit, "offset": offset, "results": items}
    except Exception as exc:
        return {
            "error": f"Failed to search events: {exc}",
            "suggestion": "try adjusting the query or filters",
        }
