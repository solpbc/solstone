# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Formatter for structured import JSONL files.

Converts entries from file importers (ICS, Obsidian, Kindle) into markdown
chunks for the search index. Each entry type gets a source-appropriate
markdown representation.

Import JSONL layout: YYYYMMDD/import.{source}/imported.jsonl
  Line 1: header {"import": {"id": "...", "source": "ics"}, "entry_count": N}
  Line 2+: entries with type, ts, content fields (source-specific extras)
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any

_SOURCE_LABELS = {"chatgpt": "ChatGPT", "claude": "Claude", "gemini": "Gemini"}


def format_imported(
    entries: list[dict],
    context: dict | None = None,
) -> tuple[list[dict], dict]:
    """Format structured import JSONL entries to markdown chunks.

    Args:
        entries: Raw JSONL entries (header + content entries)
        context: Optional context with file_path

    Returns:
        Tuple of (chunks, meta) per formatter contract.
    """
    meta: dict[str, Any] = {}
    chunks: list[dict[str, Any]] = []

    if not entries:
        return chunks, meta

    # First line is the header
    header_entry = entries[0]
    import_info = header_entry.get("import", {})
    source = import_info.get("source", "unknown")
    entry_count = header_entry.get("entry_count", 0)

    meta["indexer"] = {"agent": f"import.{source}"}
    meta["header"] = f"# Imported from {source} ({entry_count} entries)"

    # Format each content entry (skip header)
    for entry in entries[1:]:
        entry_type = entry.get("type", "")
        ts_str = entry.get("ts", "")

        # Parse timestamp for ordering
        ts_ms = 0
        if ts_str:
            try:
                dt = datetime.fromisoformat(ts_str)
                ts_ms = int(dt.timestamp() * 1000)
            except (ValueError, OSError):
                pass

        md = _format_entry(entry_type, entry)
        if md:
            chunks.append({"markdown": md, "timestamp": ts_ms, "source": entry})

    return chunks, meta


def format_ai_chat(
    entries: list[dict],
    context: dict | None = None,
) -> tuple[list[dict], dict]:
    """Format AI chat import JSONL entries to markdown chunks."""
    meta: dict[str, Any] = {}
    chunks: list[dict[str, Any]] = []

    if not entries:
        return chunks, meta

    metadata = entries[0] if entries and "start" not in entries[0] else {}
    transcript_entries = [entry for entry in entries if "start" in entry]

    source_key = "ai_chat"
    source = "AI chat"
    imported = metadata.get("imported", {})

    ctx = context or {}
    file_path = ctx.get("file_path")
    if file_path:
        parts = Path(str(file_path)).parts
        for part in parts:
            if part.startswith("import."):
                source_key = part.replace("import.", "").lower()
                source = _SOURCE_LABELS.get(source_key, source_key.capitalize())
                break

    meta["indexer"] = {"agent": f"import.{source_key}"}

    model = metadata.get("model")
    header_parts = [f"# {source} conversation"]
    if model:
        header_parts.append(f"Model: {model}")
    if isinstance(imported, dict) and imported.get("facet"):
        header_parts.append(f"Facet: {imported['facet']}")
    meta["header"] = (
        "\n".join(header_parts) if len(header_parts) > 1 else header_parts[0]
    )

    base_timestamp = 0
    if file_path:
        parts = Path(str(file_path)).parts
        rev_parts = list(reversed(parts))
        for i, part in enumerate(rev_parts):
            if re.match(r"^\d{8}$", part):
                day_str = part
                from solstone.think.utils import segment_parse

                for j in range(1, i):
                    parsed_time, _ = segment_parse(rev_parts[j])
                    if parsed_time is not None:
                        day_date = datetime.strptime(day_str, "%Y%m%d").date()
                        dt_obj = datetime.combine(day_date, parsed_time)
                        base_timestamp = int(dt_obj.timestamp() * 1000)
                        break
                break

    for entry in transcript_entries:
        speaker = entry.get("speaker", "")
        text = entry.get("text", "")
        if not text:
            continue

        start = entry.get("start", "")
        entry_timestamp = base_timestamp
        if start:
            try:
                h, m, s = map(int, start.split(":"))
                entry_timestamp = base_timestamp + (h * 3600 + m * 60 + s) * 1000
            except (ValueError, AttributeError):
                pass

        chunks.append(
            {
                "timestamp": entry_timestamp,
                "markdown": f"**{speaker}:** {text}",
                "source": entry,
            }
        )

    return chunks, meta


def _format_entry(entry_type: str, entry: dict) -> str:
    """Dispatch to type-specific formatter."""
    if entry_type == "calendar_event":
        return _format_calendar_event(entry)
    elif entry_type == "note":
        return _format_note(entry)
    elif entry_type == "highlight":
        return _format_highlight(entry)
    else:
        return _format_generic(entry)


def _format_calendar_event(entry: dict) -> str:
    """Format a calendar event entry."""
    title = entry.get("title", "Untitled event")
    ts = entry.get("ts", "")
    lines = [f"## {title}"]

    # Time info
    time_parts = []
    if ts:
        try:
            dt = datetime.fromisoformat(ts)
            time_parts.append(dt.strftime("%I:%M %p"))
        except (ValueError, OSError):
            pass
    duration = entry.get("duration_minutes")
    if duration:
        time_parts.append(f"{duration} min")
    if time_parts:
        lines.append(" | ".join(time_parts))

    location = entry.get("location")
    if location:
        lines.append(f"Location: {location}")

    attendees = entry.get("attendees", [])
    if attendees:
        names = [
            a.get("name") or a.get("email", "")
            for a in attendees
            if isinstance(a, dict)
        ]
        if not names:
            names = [str(a) for a in attendees]
        if names:
            lines.append(f"Attendees: {', '.join(names)}")

    content = entry.get("content", "")
    if content:
        lines.append("")
        lines.append(content)

    return "\n".join(lines)


def _format_note(entry: dict) -> str:
    """Format a note entry (Obsidian/Logseq)."""
    title = entry.get("title", "Untitled note")
    lines = [f"## {title}"]

    tags = entry.get("tags", [])
    if tags:
        lines.append(f"Tags: {', '.join(tags)}")

    wikilinks = entry.get("wikilinks", [])
    if wikilinks:
        lines.append(f"Links: {', '.join(wikilinks)}")

    content = entry.get("content", "")
    if content:
        lines.append("")
        lines.append(content)

    return "\n".join(lines)


def _format_highlight(entry: dict) -> str:
    """Format a Kindle highlight entry."""
    book = entry.get("book_title", "Unknown book")
    author = entry.get("author", "")
    content = entry.get("content", "")
    clip_type = entry.get("clip_type", "highlight")

    header = f"## {book}"
    if author:
        header += f" by {author}"
    lines = [header]

    loc_parts = []
    page = entry.get("page")
    location = entry.get("location")
    if page:
        loc_parts.append(f"Page {page}")
    if location:
        loc_parts.append(f"Location {location}")
    if loc_parts:
        lines.append(" | ".join(loc_parts))

    if content:
        if clip_type == "note":
            lines.append("")
            lines.append(f"Note: {content}")
        else:
            lines.append("")
            lines.append(f"> {content}")

    return "\n".join(lines)


def _format_generic(entry: dict) -> str:
    """Format an unknown entry type."""
    title = entry.get("title", "")
    content = entry.get("content", "")

    lines = []
    if title:
        lines.append(f"## {title}")
    if content:
        if lines:
            lines.append("")
        lines.append(content)

    return "\n".join(lines) if lines else ""
