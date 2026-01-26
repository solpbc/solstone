# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""MCP resource handlers for agent outputs."""

from pathlib import Path

from fastmcp.resources import TextResource

from think.mcp import mcp
from think.utils import get_journal


@mcp.resource("journal://agents/{day}/{topic}")
def get_agent_output(day: str, topic: str) -> TextResource:
    """Return the markdown output for a topic."""
    md_path = Path(get_journal()) / day / "agents" / f"{topic}.md"

    if not md_path.is_file():
        text = f"Topic '{topic}' not found for day {day}"
    else:
        text = md_path.read_text(encoding="utf-8")

    return TextResource(
        uri=f"journal://agents/{day}/{topic}",
        name=f"Output: {topic} ({day})",
        description=f"Agent output on {topic} from {day}",
        mime_type="text/markdown",
        text=text,
    )
