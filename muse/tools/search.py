"""MCP tools for search operations.

Note: These functions are registered as MCP tools by muse/mcp.py
They can also be imported and called directly for testing or internal use.
"""

from typing import Any

from think.indexer.journal import get_events as get_events_impl
from think.indexer.journal import search_journal as search_journal_impl


def search_journal(
    query: str,
    limit: int = 10,
    offset: int = 0,
    *,
    day: str | None = None,
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
        limit: Maximum number of results to return (default: 10)
        offset: Number of results to skip for pagination (default: 0)
        day: Filter by day in ``YYYYMMDD`` format
        facet: Filter by facet name (e.g., "work", "personal")
        topic: Filter by topic (e.g., "flow", "audio", "event", "entity:detected")

    Returns:
        Dictionary containing:
        - total: Total number of matching results
        - limit: Current limit value
        - offset: Current offset value
        - results: List of matches with day, facet, topic, and text

    Examples:
        - search_journal("machine learning")
        - search_journal("meeting notes", day="20240101")
        - search_journal("project planning", facet="work")
        - search_journal("standup", topic="audio")
    """
    try:
        kwargs: dict[str, Any] = {}
        if day is not None:
            kwargs["day"] = day
        if facet is not None:
            kwargs["facet"] = facet
        if topic is not None:
            kwargs["topic"] = topic

        total, results = search_journal_impl(query, limit, offset, **kwargs)

        items = []
        for r in results:
            meta = r.get("metadata", {})
            items.append(
                {
                    "day": meta.get("day", ""),
                    "facet": meta.get("facet", ""),
                    "topic": meta.get("topic", ""),
                    "text": r.get("text", ""),
                }
            )

        return {"total": total, "limit": limit, "offset": offset, "results": items}
    except Exception as exc:
        return {
            "error": f"Failed to search journal: {exc}",
            "suggestion": "try adjusting the query or ensure the index exists (run think-indexer --rescan)",
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
