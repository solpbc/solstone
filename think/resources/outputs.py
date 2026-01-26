# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""MCP resource handlers for insights."""

from pathlib import Path

from fastmcp.resources import TextResource

from think.mcp import mcp
from think.utils import get_journal


@mcp.resource("journal://insight/{day}/{topic}")
def get_insight(day: str, topic: str) -> TextResource:
    """Return the markdown insight for a topic."""
    md_path = Path(get_journal()) / day / "insights" / f"{topic}.md"

    if not md_path.is_file():
        text = f"Topic '{topic}' not found for day {day}"
    else:
        text = md_path.read_text(encoding="utf-8")

    return TextResource(
        uri=f"journal://insight/{day}/{topic}",
        name=f"Insight: {topic} ({day})",
        description=f"Insight on {topic} topic from {day}",
        mime_type="text/markdown",
        text=text,
    )
