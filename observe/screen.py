#!/usr/bin/env python3
"""
Screen analysis formatter for indexing and clustering.

Provides format_screen() and format_screen_text() functions for converting
screen.jsonl frame analyses to markdown format.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from observe.utils import load_analysis_frames, parse_screen_filename

logger = logging.getLogger(__name__)


def format_screen(
    entries: list[dict],
    context: dict | None = None,
) -> tuple[list[dict], dict]:
    """Format screen.jsonl entries to markdown chunks.

    This is the formatter function used by the formatters registry.

    Args:
        entries: Raw JSONL entries (first line is metadata, rest are frames)
        context: Optional context with:
            - file_path: Path to JSONL file (for extracting base timestamp)
            - entity_names: Comma-separated entity names for context
            - include_entity_context: Whether to include entity header

    Returns:
        Tuple of (chunks, meta) where:
            - chunks: List of dicts with keys:
                - timestamp: int (unix ms)
                - markdown: str
                - source: dict (original frame entry)
            - meta: Dict with optional "header" and "error" keys
    """
    ctx = context or {}
    file_path = ctx.get("file_path")
    entity_names = ctx.get("entity_names", "")
    include_entity_context = ctx.get("include_entity_context", True)

    # Separate metadata from frame entries
    # Only first entry can be metadata (has "raw" key but no "timestamp" key)
    frame_entries = []
    skipped_count = 0
    for i, entry in enumerate(entries):
        if i == 0 and "timestamp" not in entry and "raw" in entry:
            pass  # Skip metadata entry
        elif "timestamp" in entry:
            frame_entries.append(entry)
        else:
            skipped_count += 1

    # Build meta dict with optional error
    meta: dict[str, Any] = {}
    if skipped_count > 0:
        error_msg = f"Skipped {skipped_count} entries missing 'timestamp' field"
        if file_path:
            error_msg += f" in {file_path}"
        meta["error"] = error_msg
        logger.info(error_msg)

    chunks: list[dict[str, Any]] = []

    # Extract position/connector from filename for header
    # e.g., "center_DP-3_screen.jsonl" -> position="center", connector="DP-3"
    position, connector = "unknown", "unknown"
    if file_path:
        file_path = Path(file_path)
        position, connector = parse_screen_filename(file_path.stem)

    # Build header with entity context if requested
    header_lines = []
    if include_entity_context and entity_names:
        header_lines = [
            "# Entity Context",
            "",
            f"Frequently used names that may appear: {entity_names}",
            "",
            "---",
            "",
        ]

    # Add frame analyses header with monitor info if available
    if position != "unknown" and connector != "unknown":
        header_lines.append(f"# Frame Analyses ({position} - {connector})")
    else:
        header_lines.append("# Frame Analyses")

    meta["header"] = "\n".join(header_lines)

    # Extract base timestamp from segment directory (HHMMSS_LEN)
    # Expected structure: YYYYMMDD/HHMMSS_LEN/screen.jsonl
    base_hour = base_minute = base_second = 0
    base_timestamp_ms = 0  # Unix timestamp in milliseconds for segment start
    if file_path:
        try:
            from think.utils import segment_parse

            # Get segment start time from parent directory
            file_path = Path(file_path)
            start_time, _ = segment_parse(file_path.parent.name)
            if start_time:
                base_hour = start_time.hour
                base_minute = start_time.minute
                base_second = start_time.second

                # Try to get day from grandparent directory for unix timestamp
                day_dir = file_path.parent.parent.name
                if len(day_dir) == 8 and day_dir.isdigit():
                    day_date = datetime.strptime(day_dir, "%Y%m%d").date()
                    dt = datetime.combine(day_date, start_time)
                    base_timestamp_ms = int(dt.timestamp() * 1000)
        except (ValueError, AttributeError):
            pass

    # Sort all frames chronologically
    sorted_frames = sorted(frame_entries, key=lambda f: f.get("timestamp", 0))

    for frame in sorted_frames:
        lines = []

        # Calculate absolute time
        frame_offset = frame.get("timestamp", 0)
        total_seconds = (
            base_hour * 3600 + base_minute * 60 + base_second + int(frame_offset)
        )
        abs_hour = (total_seconds // 3600) % 24
        abs_minute = (total_seconds // 60) % 60
        abs_second = total_seconds % 60

        # Build frame header with timestamp
        frame_header = f"### {abs_hour:02d}:{abs_minute:02d}:{abs_second:02d}"

        lines.append(frame_header)
        lines.append("")

        # Add analysis if present
        analysis = frame.get("analysis", {})
        if analysis:
            # Extract category from primary region, fall back to legacy visible field
            primary = analysis.get("primary", {})
            if primary:
                category = primary.get("category", "unknown")
            else:
                category = analysis.get("visible", "unknown")
            description = analysis.get("visual_description", "")

            lines.append(f"**Category:** {category}")
            lines.append("")
            if description:
                lines.append(description)
                lines.append("")

        # Add extracted text if present
        extracted_text = frame.get("extracted_text")
        if extracted_text:
            lines.append("**Extracted Text:**")
            lines.append("")
            lines.append("```")
            lines.append(extracted_text.strip())
            lines.append("```")
            lines.append("")

        # Add meeting analysis if present
        meeting = frame.get("meeting_analysis")
        if meeting:
            lines.append("**Meeting Analysis:**")
            lines.append("")
            lines.append("```json")
            lines.append(json.dumps(meeting, indent=2))
            lines.append("```")
            lines.append("")

        # Calculate absolute unix timestamp in milliseconds
        frame_timestamp_ms = base_timestamp_ms + int(frame_offset * 1000)

        chunks.append(
            {
                "timestamp": frame_timestamp_ms,
                "markdown": "\n".join(lines),
                "source": frame,
            }
        )

    # Indexer metadata - topic is always "screen" for screen analysis
    meta["indexer"] = {"topic": "screen"}

    return chunks, meta


def format_screen_text(jsonl_path: Path) -> str:
    """Load and format screen.jsonl to markdown text.

    Convenience function for cluster.py that loads frames and formats to text.

    Args:
        jsonl_path: Path to screen.jsonl file

    Returns:
        Formatted markdown string
    """
    frames = load_analysis_frames(jsonl_path)
    if not frames:
        return ""

    context = {"file_path": jsonl_path, "include_entity_context": False}
    chunks, meta = format_screen(frames, context)

    parts = []
    if meta.get("header"):
        parts.append(meta["header"])
    parts.extend(chunk["markdown"] for chunk in chunks)
    return "\n".join(parts)
