#!/usr/bin/env python3
"""MCP server for Sunstone journal assistant."""

import json
import os
from pathlib import Path
from typing import Any

from fastmcp import FastMCP
from fastmcp.resources import TextResource

from think.cluster import cluster_range
from think.indexer import search_occurrences as search_occurrences_impl
from think.indexer import search_raws as search_raws_impl
from think.indexer import search_topics as search_topics_impl

# Create the MCP server
mcp = FastMCP("sunstone")


@mcp.tool
def search_topic(
    query: str, limit: int = 5, offset: int = 0, *, topic: str | None = None
) -> dict[str, Any]:
    """Search across journal topic summaries using semantic full-text search.

    This tool searches through pre-processed topic summaries that represent
    key themes and subjects from your journal entries. Use this when looking
    for high-level concepts, themes, or when you need an overview of topics
    discussed over time.

    Args:
        query: Natural language search query (e.g., "meetings about product launch")
        limit: Optional maximum number of results to return (default: 5, max: 20)
        offset: Optional number of results to skip for pagination (default: 0)
        topic: Optional topic name to filter results by

    Returns:
        Dictionary containing:
        - total: Total number of matching topics
        - limit: Current limit value used for this query
        - offset: Current offset value used for this query
        - results: List of matching topics with day, topic, and text excerpt

    Examples:
        - search_topic("machine learning projects")
        - search_topic("team retrospectives", limit=10)
        - search_topic("planning", topic="standup")
    """
    try:
        kwargs = {}
        if topic is not None:
            kwargs["topic"] = topic
        total, results = search_topics_impl(query, limit, offset, **kwargs)

        items = []
        for r in results:
            meta = r.get("metadata", {})
            topic_label = meta.get("topic", "")
            if topic_label.endswith(".md"):
                topic_label = topic_label[:-3]
            topic_label = os.path.basename(topic_label)
            items.append(
                {
                    "day": meta.get("day", ""),
                    "topic": topic_label,
                    "text": r.get("text", ""),
                }
            )

        return {"total": total, "limit": limit, "offset": offset, "results": items}
    except Exception as exc:
        return {
            "error": f"Failed to search topics: {exc}",
            "suggestion": "try adjusting the query or ensure indexes exist",
        }


@mcp.tool
def search_raw(query: str, day: str, limit: int = 5, offset: int = 0) -> dict[str, Any]:
    """Search raw transcripts and screen diffs for a specific day.

    This tool scans raw audio transcripts (``*_audio.json``) and screenshot
    diffs (``*_diff.json``) produced throughout the day. Use it when you need
    to recall exact wording, short snippets, or visual context from a given
    date.

    Args:
        query: Natural language search query (e.g., "error message")
        day: Day folder to search in ``YYYYMMDD`` format
        limit: Optional maximum number of results to return (default: 5, max: 20)
        offset: Optional number of results to skip for pagination (default: 0)

    Returns:
        Dictionary containing:
        - total: Total number of matching raw entries
        - limit: Current limit value used for this query
        - offset: Current offset value used for this query
        - results: List of entries with day, time, type, and text snippet

    Examples:
        - search_raw("error message", day="20240101")
        - search_raw("feature flag", day="20240102", limit=10)
    """
    try:
        total, results = search_raws_impl(query, limit=limit, offset=offset, day=day)

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


@mcp.tool
def search_events(
    query: str,
    limit: int = 5,
    offset: int = 0,
    *,
    day: str | None = None,
    topic: str | None = None,
    start: str | None = None,
    end: str | None = None,
) -> dict[str, Any]:
    """Search structured events extracted from journal summaries.

    This tool searches JSON event data generated from your daily summaries.
    Use it to find meetings, tasks, or other notable activities. Results may
    be filtered by day, topic, or a time range.

    Args:
        query: Natural language search query (e.g., "team standup")
        limit: Optional maximum number of events to return (default: 5)
        offset: Optional number of results to skip for pagination (default: 0)
        day: Optional ``YYYYMMDD`` day to filter results
        topic: Optional topic name to filter by
        start: Optional start time to filter events starting on or after this ``HH:MM:SS`` time
        end: Optional end time to filter events ending on or before this ``HH:MM:SS`` time

    Returns:
        Dictionary with ``limit``, ``offset`` and ``results`` list containing day, topic,
        start/end times and short event summaries.

    Examples:
        - search_events("sprint review")
        - search_events("planning", day="20240101", limit=10)
        - search_events("standup", limit=5, offset=10)
    """

    try:
        total, rows = search_occurrences_impl(
            query,
            limit=limit,
            offset=offset,
            day=day,
            start=start,
            end=end,
            topic=topic,
        )

        items = []
        for r in rows:
            meta = r.get("metadata", {})
            occ = r.get("occurrence", {})
            items.append(
                {
                    "day": meta.get("day", ""),
                    "topic": meta.get("topic", ""),
                    "start": meta.get("start", ""),
                    "end": meta.get("end", ""),
                    "title": occ.get("title") or r.get("text", ""),
                    "summary": occ.get("summary", ""),
                }
            )

        return {"total": total, "limit": limit, "results": items}
    except Exception as exc:
        return {
            "error": f"Failed to search events: {exc}",
            "suggestion": "try adjusting the query or filters",
        }


@mcp.resource("journal://summary/{day}/{topic}")
def get_topic_summary(day: str, topic: str) -> TextResource:
    """Return JSON summary for a topic markdown file."""
    journal = os.getenv("JOURNAL_PATH", "journal")
    md_path = Path(journal) / day / "topics" / f"{topic}.md"

    if not md_path.is_file():
        content = {"error": f"Topic '{topic}' not found for day {day}"}
    else:
        text = md_path.read_text(encoding="utf-8")
        content = {
            "day": day,
            "topic": topic,
            "summary": text,
            "word_count": len(text.split()),
        }

    return TextResource(
        uri=f"journal://summary/{day}/{topic}",
        name=f"Summary: {topic} ({day})",
        description=f"Summary of {topic} topic from {day}",
        mime_type="application/json",
        text=json.dumps(content, indent=2),
    )


@mcp.resource("journal://raw/{day}/{time}/{length}")
def get_raw_cluster(day: str, time: str, length: str) -> TextResource:
    """Return raw audio and screen transcripts for a specific time range.

    This resource provides raw audio and screen transcripts for a given
    time range. The data is organized into 5-minute intervals and formatted
    as markdown. Each 5 minute segment could potentially be very large if there was a lot of activity, so it is recommended to use this with a specific minimum time range.

    Args:
        day: Day in YYYYMMDD format
        time: Start time in HHMMSS format
        length: Length in minutes for the time range
    """
    try:
        # Parse the length as minutes and convert to end time
        length_minutes = int(length)
        from datetime import datetime, timedelta

        # Parse start time
        start_dt = datetime.strptime(f"{day}{time}", "%Y%m%d%H%M%S")
        # Calculate end time
        end_dt = start_dt + timedelta(minutes=length_minutes)
        end_time = end_dt.strftime("%H%M%S")

        # Use cluster_range with raw screen data
        markdown_content = cluster_range(
            day=day, start=time, end=end_time, screen="raw"
        )

        return TextResource(
            uri=f"journal://raw/{day}/{time}/{length}",
            name=f"Raw Cluster: {day} {time} ({length}min)",
            description=f"Raw screen activity cluster from {day} starting at {time} for {length} minutes",
            mime_type="text/markdown",
            text=markdown_content,
        )

    except Exception as e:
        error_content = f"# Error\n\nFailed to generate raw cluster for {day} {time} ({length}min): {str(e)}"
        return TextResource(
            uri=f"journal://raw/{day}/{time}/{length}",
            name=f"Raw Cluster Error: {day} {time} ({length}min)",
            description="Error generating raw screen cluster",
            mime_type="text/markdown",
            text=error_content,
        )


if __name__ == "__main__":
    # When run directly, use stdio transport (default)
    mcp.run()
