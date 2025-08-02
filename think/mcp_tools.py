#!/usr/bin/env python3
"""MCP tools for the Sunstone journal assistant."""

import base64
import os
import sys
from pathlib import Path
from typing import Any

from fastmcp import FastMCP
from fastmcp.resources import FileResource, TextResource

from think.cluster import cluster_range
from think.indexer import search_events as search_events_impl
from think.indexer import search_summaries as search_summaries_impl
from think.indexer import search_transcripts as search_transcripts_impl
from think.utils import get_raw_file

# Create the MCP server
mcp = FastMCP("sunstone")


@mcp.tool
def search_summaries(
    query: str,
    limit: int = 5,
    offset: int = 0,
    *,
    topic: str | None = None,
    day: str | None = None,
) -> dict[str, Any]:
    """Search across journal topic summaries using semantic full-text search.

    This tool searches through pre-processed topic summaries that represent
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
        - total: Total number of matching topics
        - limit: Current limit value used for this query
        - offset: Current offset value used for this query
        - results: List of matching topics with day, topic, and text excerpt, ordered by text relevance

    Examples:
        - search_summaries("machine learning projects")
        - search_summaries("team retrospectives", limit=10)
        - search_summaries("planning", topic="standup")
        - search_summaries("meetings", day="20240101")
    """
    try:
        kwargs = {}
        if topic is not None:
            kwargs["topic"] = topic
        if day is not None:
            kwargs["day"] = day
        total, results = search_summaries_impl(query, limit, offset, **kwargs)

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
            "error": f"Failed to search topics: {exc}",
            "suggestion": "try adjusting the query or ensure indexes exist",
        }


@mcp.tool
def search_transcripts(
    query: str, day: str, limit: int = 5, offset: int = 0
) -> dict[str, Any]:
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
        - results: List of entries with day, time, type, and text snippet, ordered by time series in the given range

    Examples:
        - search_transcripts("error message", day="20240101")
        - search_transcripts("feature flag", day="20240102", limit=10)
    """
    try:
        total, results = search_transcripts_impl(
            query, limit=limit, offset=offset, day=day
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
        Ordered by day and start time (most recent first).

    Examples:
        - search_events("sprint review")
        - search_events("planning", day="20240101", limit=10)
        - search_events("standup", limit=5, offset=10)
    """

    try:
        total, rows = search_events_impl(
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

        return {"total": total, "limit": limit, "offset": offset, "results": items}
    except Exception as exc:
        return {
            "error": f"Failed to search events: {exc}",
            "suggestion": "try adjusting the query or filters",
        }


@mcp.tool
async def get_resource(uri: str) -> object:
    """Return the contents of a journal resource.

    Many MCP clients cannot read ``journal://`` resources directly. This tool
    acts as a wrapper around the server resources so they can be fetched via a
    normal tool call.

    The following resource types are supported:

    - ``journal://summary/{day}/{topic}`` — markdown topic summaries
    - ``journal://raw/{day}/{time}/{length}`` — raw transcripts for a time range
    - ``journal://media/{day}/{name}`` — raw FLAC or PNG media files

    Args:
        uri: Resource URI to fetch.

    Returns:
        ``Image`` or ``Audio`` objects for binary media, or a plain string for
        text resources.
    """

    try:
        resource = await mcp._resource_manager.get_resource(uri)
        data = await resource.read()

        if isinstance(data, bytes):
            # Return base64 encoded data for binary content
            return base64.b64encode(data).decode("utf-8")

        # text content
        return str(data)
    except Exception as exc:  # pragma: no cover - unexpected failure
        return {"error": f"Failed to fetch resource: {exc}"}


@mcp.resource("journal://summary/{day}/{topic}")
def get_summary(day: str, topic: str) -> TextResource:
    """Return the markdown summary for a topic."""
    journal = os.getenv("JOURNAL_PATH", "journal")
    md_path = Path(journal) / day / "topics" / f"{topic}.md"

    if not md_path.is_file():
        text = f"Topic '{topic}' not found for day {day}"
    else:
        text = md_path.read_text(encoding="utf-8")

    return TextResource(
        uri=f"journal://summary/{day}/{topic}",
        name=f"Summary: {topic} ({day})",
        description=f"Summary of {topic} topic from {day}",
        mime_type="text/markdown",
        text=text,
    )


@mcp.resource("journal://raw/{day}/{time}/{length}")
def get_transcripts(day: str, time: str, length: str) -> TextResource:
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
            name=f"Transcripts: {day} {time} ({length}min)",
            description=f"Raw screen activity from {day} starting at {time} for {length} minutes",
            mime_type="text/markdown",
            text=markdown_content,
        )

    except Exception as e:
        error_content = f"# Error\n\nFailed to generate transcripts for {day} {time} ({length}min): {str(e)}"
        return TextResource(
            uri=f"journal://raw/{day}/{time}/{length}",
            name=f"Transcripts Error: {day} {time} ({length}min)",
            description="Error generating raw screen transcripts",
            mime_type="text/markdown",
            text=error_content,
        )


@mcp.resource("journal://media/{day}/{name}")
def get_media(day: str, name: str) -> FileResource:
    """Return a raw FLAC or PNG file referenced by a transcript.

    Parameters
    ----------
    day:
        Day folder in ``YYYYMMDD`` format.
    name:
        Transcript JSON filename such as ``HHMMSS_audio.json`` or
        ``HHMMSS_monitor_1_diff.json``.

    Returns
    -------
    FileResource
        Resource pointing to the raw media file referenced by ``name``.
    """

    rel_path, mime, _ = get_raw_file(day, name)
    journal = os.getenv("JOURNAL_PATH", "journal")
    abs_path = Path(journal) / day / rel_path
    return FileResource(
        uri=f"journal://media/{day}/{name}",
        name=f"Media: {name}",
        description=f"Raw media file from {day}",
        mime_type=mime,
        path=abs_path,
    )


def main() -> None:
    """Run the MCP server using the requested transport."""

    transport = "stdio"
    if len(sys.argv) > 1:
        transport = sys.argv[1]

    if transport == "http":
        host = os.getenv("SUNSTONE_MCP_HOST", "127.0.0.1")
        port = int(os.getenv("SUNSTONE_MCP_PORT", "8000"))
        path = os.getenv("SUNSTONE_MCP_PATH", "/mcp/")
        mcp.run(
            transport="http",
            host=host,
            port=port,
            path=path,
            show_banner=False,
        )
    else:
        # default stdio transport
        mcp.run(show_banner=False)


if __name__ == "__main__":
    main()
