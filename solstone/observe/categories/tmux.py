# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Formatter for tmux category content.

Renders tmux terminal capture to markdown with pane contents.
"""

import re
from typing import Any

# ANSI escape code pattern
ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape codes from text."""
    return ANSI_ESCAPE.sub("", text)


def format(content: Any, context: dict) -> str:
    """Format tmux capture to markdown.

    Args:
        content: Tmux capture dict with session, window, panes
        context: Dict with frame, file_path, timestamp_str

    Returns:
        Formatted markdown string
    """
    if not isinstance(content, dict):
        return ""

    lines = []

    # Session and window header
    session = content.get("session", "unknown")
    window = content.get("window", {})
    window_name = (
        window.get("name", "unknown") if isinstance(window, dict) else "unknown"
    )

    lines.append(f"**Tmux** ({session}:{window_name})")
    lines.append("")

    # Pane contents
    panes = content.get("panes", [])
    for pane in panes:
        if not isinstance(pane, dict):
            continue

        pane_content = pane.get("content", "")
        if not pane_content:
            continue

        # Strip ANSI codes for clean markdown
        clean_content = _strip_ansi(pane_content).rstrip()
        if not clean_content:
            continue

        # Label pane if multiple panes
        if len(panes) > 1:
            pane_idx = pane.get("index", 0)
            active = " (active)" if pane.get("active") else ""
            lines.append(f"**Pane {pane_idx}{active}:**")
            lines.append("")

        lines.append("```")
        lines.append(clean_content)
        lines.append("```")
        lines.append("")

    return "\n".join(lines)
