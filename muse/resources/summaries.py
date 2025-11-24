"""MCP resource handlers for summaries."""

import os
from pathlib import Path

from fastmcp.resources import TextResource

from muse.mcp import mcp


@mcp.resource("journal://summary/{day}/{topic}")
def get_summary(day: str, topic: str) -> TextResource:
    """Return the markdown summary for a topic."""
    journal = os.getenv("JOURNAL_PATH", "journal")
    md_path = Path(journal) / day / "insights" / f"{topic}.md"

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
