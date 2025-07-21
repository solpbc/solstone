#!/usr/bin/env python3
"""MCP server for Sunstone journal assistant."""

import os
from pathlib import Path
from typing import Any

from fastmcp import FastMCP

from think.indexer import search_raws as search_raws_impl
from think.indexer import search_topics as search_topics_impl

# Create the MCP server
mcp = FastMCP("sunstone")


@mcp.tool
def search_topic(query: str, limit: int = 5, offset: int = 0) -> dict[str, Any]:
    """Search across journal topic summaries using semantic full-text search.

    This tool searches through pre-processed topic summaries that represent
    key themes and subjects from your journal entries. Use this when looking
    for high-level concepts, themes, or when you need an overview of topics
    discussed over time.

    Args:
        query: Natural language search query (e.g., "meetings about product launch")
        limit: Maximum number of results to return (default: 5, max: 20)
        offset: Number of results to skip for pagination (default: 0)

    Returns:
        Dictionary containing:
        - total: Total number of matching topics
        - results: List of matching topics with day, filename, and text excerpt

    Examples:
        - search_topic("machine learning projects")
        - search_topic("team retrospectives", limit=10)
    """
    journal = os.getenv("JOURNAL_PATH", "journal")
    total, results = search_topics_impl(journal, query, limit, offset)

    items = []
    for r in results:
        meta = r.get("metadata", {})
        topic = meta.get("topic", "")
        if topic.endswith(".md"):
            topic = topic[:-3]
        items.append(
            {"day": meta.get("day", ""), "filename": topic, "text": r.get("text", "")}
        )

    return {"total": total, "results": items}


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
        limit: Maximum number of results to return (default: 5, max: 20)
        offset: Number of results to skip for pagination (default: 0)

    Returns:
        Dictionary containing:
        - total: Total number of matching raw entries
        - results: List of entries with day, time, type, and text snippet

    Examples:
        - search_raw("error message", day="20240101")
        - search_raw("feature flag", day="20240102", limit=10)
    """
    journal = os.getenv("JOURNAL_PATH", "journal")
    total, results = search_raws_impl(
        journal, query, limit=limit, offset=offset, day=day
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

    return {"total": total, "results": items}


@mcp.tool
def read_markdown(date: str, filename: str) -> str:
    """Return journal markdown contents."""
    journal = os.getenv("JOURNAL_PATH", "journal")
    md_path = Path(journal) / date / f"{filename}.md"

    if not md_path.is_file():
        raise FileNotFoundError(f"Markdown not found: {md_path}")

    return md_path.read_text(encoding="utf-8")


if __name__ == "__main__":
    # When run directly, use stdio transport (default)
    mcp.run()
