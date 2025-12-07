"""Event formatting for the journal index.

This module provides the formatter function for converting event JSONL
entries to markdown chunks for the unified journal index.
"""

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any


def format_events(
    entries: list[dict],
    context: dict | None = None,
) -> tuple[list[dict], dict]:
    """Format event JSONL entries to markdown chunks.

    This is the formatter function used by the formatters registry.

    Args:
        entries: Raw JSONL entries (one event per line)
        context: Optional context with:
            - file_path: Path to JSONL file (for extracting facet name and day)

    Returns:
        Tuple of (chunks, meta) where:
            - chunks: List of {"timestamp": int, "markdown": str} dicts, one per event
            - meta: Dict with optional "header" and "error" keys
    """
    ctx = context or {}
    file_path = ctx.get("file_path")
    meta: dict[str, Any] = {}
    chunks: list[dict[str, Any]] = []
    skipped_count = 0

    # Extract facet name and day from path
    facet_name = "unknown"
    day_str: str | None = None

    if file_path:
        file_path = Path(file_path)

        # Extract facet name from path: facets/{facet}/events/YYYYMMDD.jsonl
        path_str = str(file_path)
        facet_match = re.search(r"facets/([^/]+)/events", path_str)
        if facet_match:
            facet_name = facet_match.group(1)

        # Extract day from filename
        if file_path.stem.isdigit() and len(file_path.stem) == 8:
            day_str = file_path.stem

    # Calculate base timestamp (midnight of the event day) in milliseconds
    base_ts = 0
    if day_str:
        try:
            dt = datetime.strptime(day_str, "%Y%m%d")
            base_ts = int(dt.timestamp() * 1000)
        except ValueError:
            pass

    # Build header
    if day_str:
        formatted_day = f"{day_str[:4]}-{day_str[4:6]}-{day_str[6:8]}"
        meta["header"] = f"# Events for '{facet_name}' facet on {formatted_day}"
    else:
        meta["header"] = f"# Events for '{facet_name}' facet"

    # Format each event as a chunk
    for event in entries:
        # Skip entries without title
        title = event.get("title")
        if not title:
            skipped_count += 1
            continue

        event_type = event.get("type", "event").capitalize()
        occurred = event.get("occurred", True)

        # Calculate timestamp from day + start time
        ts = base_ts
        start_time = event.get("start", "")
        if start_time and base_ts:
            try:
                # Parse HH:MM:SS or HH:MM
                time_parts = start_time.split(":")
                hours = int(time_parts[0])
                minutes = int(time_parts[1]) if len(time_parts) > 1 else 0
                seconds = int(time_parts[2]) if len(time_parts) > 2 else 0
                ts = base_ts + (hours * 3600 + minutes * 60 + seconds) * 1000
            except (ValueError, IndexError):
                pass

        # Build markdown
        type_prefix = "Planned " if not occurred else ""
        lines = [f"### {type_prefix}{event_type}: {title}\n", ""]

        # Time range (24h format, strip seconds for display)
        end_time = event.get("end", "")
        time_label = "Occurred" if occurred else "Scheduled"
        if start_time:
            start_display = start_time[:5] if len(start_time) >= 5 else start_time
            if end_time:
                end_display = end_time[:5] if len(end_time) >= 5 else end_time
                lines.append(f"**Time {time_label}:** {start_display} - {end_display}")
            else:
                lines.append(f"**Time {time_label}:** {start_display}")

        # Participants
        participants = event.get("participants", [])
        if participants and isinstance(participants, list):
            participants_label = (
                "Expected Participants" if not occurred else "Participants"
            )
            lines.append(f"**{participants_label}:** {', '.join(participants)}")

        # For anticipations, show when it was created (from source path)
        if not occurred:
            source = event.get("source", "")
            # Extract YYYYMMDD from source path like "20240101/insights/schedule.md"
            source_match = re.match(r"(\d{8})/", source)
            if source_match:
                created_day = source_match.group(1)
                created_formatted = (
                    f"{created_day[:4]}-{created_day[4:6]}-{created_day[6:8]}"
                )
                lines.append(f"**Created on:** {created_formatted}")

        lines.append("")

        # Summary
        summary = event.get("summary", "")
        if summary:
            lines.append(summary)
            lines.append("")

        # Details
        details = event.get("details", "")
        if details:
            lines.append(details)
            lines.append("")

        chunks.append({"timestamp": ts, "markdown": "\n".join(lines)})

    # Report skipped entries
    if skipped_count > 0:
        error_msg = f"Skipped {skipped_count} entries missing 'title' field"
        if file_path:
            error_msg += f" in {file_path}"
        meta["error"] = error_msg
        logging.info(error_msg)

    return chunks, meta
