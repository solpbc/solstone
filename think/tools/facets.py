# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""MCP tools for facet management.

Note: These functions are registered as MCP tools by think/mcp.py
They can also be imported and called directly for testing or internal use.
"""

from pathlib import Path
from typing import Any

from think.facets import facet_summary
from think.utils import get_journal


def get_facet(facet: str) -> dict[str, Any]:
    """Get a comprehensive summary of a facet including its metadata and entities.

    This tool generates a formatted markdown summary for a specified facet in the journal.
    The summary includes the facet's title, description, and tracked entities.
    Use this when you need an overview of a facet's current state and its associated entities.

    Args:
        facet: The facet name to retrieve the summary for

    Returns:
        Dictionary containing:
        - facet: The facet name that was queried
        - summary: Formatted markdown text with the complete facet summary including:
            - Facet title
            - Facet description
            - List of tracked entities

    Examples:
        - get_facet("personal")
        - get_facet("work_projects")
        - get_facet("research")

    Returns:
        If the facet doesn't exist, returns an error dictionary with an error message
        and suggestion for resolution.
    """
    try:
        # Get the facet summary markdown
        summary_text = facet_summary(facet)
        return {"facet": facet, "summary": summary_text}
    except FileNotFoundError:
        return {
            "error": f"Facet '{facet}' not found",
            "suggestion": "verify the facet name exists in the journal",
        }
    except Exception as exc:
        return {
            "error": f"Failed to get facet summary: {exc}",
            "suggestion": "check that the facet exists and has valid metadata",
        }


def facet_news(facet: str, day: str, markdown: str | None = None) -> dict[str, Any]:
    """Read or write news for a specific facet and day.

    This tool manages facet-specific news stored in markdown files organized by date.
    When markdown content is provided, it writes/updates the news file for that day.
    When markdown is not provided, it reads and returns the existing news for that day.
    News files are stored as `facets/<facet>/news/YYYYMMDD.md`.

    Args:
        facet: The facet name to manage news for
        day: The day in YYYYMMDD format
        markdown: Optional markdown content to write. If not provided, reads existing news.
                 Should follow the format with dated header and news entries with source/time.

    Returns:
        Dictionary containing either:
        - facet, day, and news content when reading
        - facet, day, and success message when writing
        - error and suggestion if operation fails

    Examples:
        - facet_news("ml_research", "20250118")  # Read news for the day
        - facet_news("work", "20250118", "# 2025-01-18 News...")  # Write news
    """
    try:
        journal_path = Path(get_journal())
        facet_path = journal_path / "facets" / facet

        # Check if facet exists
        if not facet_path.exists():
            return {
                "error": f"Facet '{facet}' not found",
                "suggestion": "Create the facet first or check the facet name",
            }

        # Ensure news directory exists
        news_dir = facet_path / "news"
        news_dir.mkdir(exist_ok=True)

        # Path to the specific day's news file
        news_file = news_dir / f"{day}.md"

        if markdown is not None:
            # Write mode - save the markdown content
            news_file.write_text(markdown, encoding="utf-8")
            return {
                "facet": facet,
                "day": day,
                "message": f"News for {day} saved successfully in facet '{facet}'",
            }
        else:
            # Read mode - return existing news or empty message
            if news_file.exists():
                news_content = news_file.read_text(encoding="utf-8")
                return {"facet": facet, "day": day, "news": news_content}
            else:
                return {
                    "facet": facet,
                    "day": day,
                    "news": None,
                    "message": f"No news recorded for {day} in facet '{facet}'",
                }

    except Exception as exc:
        return {
            "error": f"Failed to process facet news: {exc}",
            "suggestion": "check facet exists and has proper permissions",
        }
