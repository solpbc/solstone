# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Audio transcript utilities for observe package."""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any


def load_transcript(
    file_path: str | os.PathLike,
) -> tuple[dict, list[dict] | None, str]:
    """Load a transcript JSONL file with metadata, entries, and formatted text.

    The JSONL format has metadata as the first line (may be empty {})
    and transcript entries as subsequent lines. Handles both native
    transcripts (segment/audio.jsonl) and imported transcripts (segment/imported_audio.jsonl).

    Args:
        file_path: Path to the JSONL transcript file

    Returns:
        Tuple of (metadata, entries, formatted_text) where:
        - metadata: Dict from first line. Native transcripts may have empty {}
                   or contain "topics"/"setting". Imported transcripts contain
                   {"imported": {"id": "...", "facet": "...", ...}}.
                   On error, returns {"error": "message"}.
        - entries: List of entry dicts from subsequent lines, each with fields
                  like "start", "text", "source", etc. Returns None on error.
        - formatted_text: Human-readable formatted text with header and entries.
                         Format: "Start: 2024-06-15 10:05a Setting: work\n[00:00:15] (mic) Speaker 1: Hello"

    Examples:
        # Load a native transcript
        metadata, entries, formatted_text = load_transcript("20250101/120000/audio.jsonl")
        if entries is None:
            print(f"Error: {metadata.get('error')}")
            return
        print(formatted_text)  # Human-readable output
        for entry in entries:
            print(f"{entry['start']}: {entry['text']}")

        # Load an imported transcript
        metadata, entries, formatted_text = load_transcript("20250101/120000/imported_audio.jsonl")
        if entries is not None:
            import_id = metadata.get("imported", {}).get("id")
            facet = metadata.get("imported", {}).get("facet")
            print(f"Imported from {import_id} (facet: {facet})")

        # Check for topics/setting in native transcript
        metadata, entries, formatted_text = load_transcript(path)
        if entries is not None:
            topics = metadata.get("topics")
            setting = metadata.get("setting")
    """
    try:
        path = Path(file_path)
        if not path.exists():
            return (
                {"error": f"File not found: {file_path}"},
                None,
                f"Error loading transcript: File not found: {file_path}",
            )

        content = path.read_text(encoding="utf-8").strip()
        if not content:
            return (
                {"error": "File is empty"},
                None,
                "Error loading transcript: File is empty",
            )

        lines = content.split("\n")

        # Parse metadata from first line
        try:
            metadata = json.loads(lines[0])
            if not isinstance(metadata, dict):
                return (
                    {"error": "First line must be a JSON object"},
                    None,
                    "Error loading transcript: First line must be a JSON object",
                )
        except json.JSONDecodeError as e:
            return (
                {"error": f"Invalid JSON in metadata line: {e}"},
                None,
                f"Error loading transcript: Invalid JSON in metadata line: {e}",
            )

        # Parse entries from remaining lines
        entries = []
        for i, line in enumerate(lines[1:], start=2):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if not isinstance(entry, dict):
                    return (
                        {"error": f"Line {i} is not a JSON object"},
                        None,
                        f"Error loading transcript: Line {i} is not a JSON object",
                    )
                entries.append(entry)
            except json.JSONDecodeError as e:
                return (
                    {"error": f"Invalid JSON at line {i}: {e}"},
                    None,
                    f"Error loading transcript: Invalid JSON at line {i}: {e}",
                )

        # Format the transcript as human-readable text
        formatted_text = _format_transcript_entries(path, metadata, entries)

        return metadata, entries, formatted_text

    except Exception as e:
        return (
            {"error": f"Failed to load transcript: {e}"},
            None,
            f"Error loading transcript: {e}",
        )


def format_audio(
    entries: list[dict],
    context: dict | None = None,
) -> tuple[list[dict], dict]:
    """Format audio transcript entries to markdown chunks.

    This is the formatter function used by the formatters registry.

    Args:
        entries: Raw JSONL entries (first line is metadata, rest are transcript entries)
        context: Optional context with:
            - file_path: Path to JSONL file (for extracting base timestamp)

    Returns:
        Tuple of (chunks, meta) where:
            - chunks: List of dicts with keys:
                - timestamp: int (unix ms)
                - markdown: str
                - source: dict (original transcript entry)
            - meta: Dict with optional "header" and "error" keys
    """
    ctx = context or {}
    file_path = ctx.get("file_path")

    # Separate metadata from transcript entries
    # Only first entry can be metadata (has no "start" key)
    metadata = {}
    transcript_entries = []
    skipped_count = 0
    for i, entry in enumerate(entries):
        if i == 0 and "start" not in entry:
            metadata = entry
        elif "start" in entry:
            transcript_entries.append(entry)
        else:
            skipped_count += 1

    # Build meta dict with optional error
    meta: dict[str, Any] = {}
    if skipped_count > 0:
        logger = logging.getLogger(__name__)
        error_msg = f"Skipped {skipped_count} entries missing 'start' field"
        if file_path:
            error_msg += f" in {file_path}"
        meta["error"] = error_msg
        logger.info(error_msg)

    chunks: list[dict[str, Any]] = []

    # Parse day and time from path structure
    # Supports both layouts:
    #   YYYYMMDD/HHMMSS_LEN/audio.jsonl           (legacy)
    #   YYYYMMDD/stream/HHMMSS_LEN/audio.jsonl    (stream-based)
    day_str = None
    start_time = None
    base_timestamp = 0

    if file_path:
        file_path = Path(file_path)
        parts = file_path.parts
        rev_parts = list(reversed(parts))

        # Try to find YYYYMMDD in path
        for i, part in enumerate(rev_parts):
            if re.match(r"^\d{8}$", part):
                day_str = part
                # Scan parts between file and day for a valid segment key
                from think.utils import segment_parse

                for j in range(1, i):
                    parsed_time, _ = segment_parse(rev_parts[j])
                    if parsed_time is not None:
                        start_time = parsed_time
                        break
                break

    # Build header line
    header_parts = []

    # Add start time if we could parse it
    if day_str and start_time:
        try:
            day_date = datetime.strptime(day_str, "%Y%m%d").date()
            dt = datetime.combine(day_date, start_time)
            # Format as "2024-06-15 10:05a"
            time_formatted = dt.strftime("%Y-%m-%d %I:%M%p").lower()
            header_parts.append(f"Start: {time_formatted}")
            # Calculate base timestamp for entries (milliseconds)
            base_timestamp = int(dt.timestamp() * 1000)
        except ValueError:
            pass

    # Add metadata fields (excluding special fields)
    skip_fields = {"error", "raw", "imported"}

    for key, value in metadata.items():
        if key in skip_fields:
            continue

        # Format the value
        if isinstance(value, list):
            value_str = ", ".join(str(v) for v in value)
        else:
            value_str = str(value)

        if value_str:
            header_parts.append(f"{key.capitalize()}: {value_str}")

    # Handle imported metadata specially
    if "imported" in metadata and isinstance(metadata["imported"], dict):
        imported = metadata["imported"]
        if "facet" in imported:
            header_parts.append(f"Facet: {imported['facet']}")
        if "id" in imported:
            header_parts.append(f"Import ID: {imported['id']}")

    # Build header from metadata parts
    if header_parts:
        meta["header"] = " ".join(header_parts)

    # Format transcript entries
    for entry in transcript_entries:
        entry_parts = []

        # Timestamp
        start = entry.get("start", "")
        entry_timestamp = base_timestamp
        if start:
            entry_parts.append(f"[{start}]")
            # Parse timestamp for chunk ordering (HH:MM:SS format, offset in ms)
            try:
                h, m, s = map(int, start.split(":"))
                entry_timestamp = base_timestamp + (h * 3600 + m * 60 + s) * 1000
            except (ValueError, AttributeError):
                pass

        # Source (mic/sys)
        source = entry.get("source", "")
        if source:
            entry_parts.append(f"({source})")

        # Speaker - handle both int and string formats (optional, for legacy transcripts)
        speaker = entry.get("speaker")
        if speaker is not None:
            if isinstance(speaker, int):
                entry_parts.append(f"Speaker {speaker}:")
            else:
                entry_parts.append(f"{speaker}:")
        else:
            entry_parts.append("")

        # Text - prefer corrected text if available
        text = entry.get("corrected") or entry.get("text", "")

        # Emotion (tone, delivery) - skip if "neutral"
        emotion = entry.get("emotion", "")
        if emotion and emotion.lower() == "neutral":
            emotion = ""

        # Combine into markdown
        prefix = " ".join(entry_parts).strip()
        if prefix:
            markdown = f"{prefix} {text}" if text else prefix
        elif text:
            markdown = text
        else:
            continue  # Skip empty entries

        # Append emotion in italics if present and not neutral
        if emotion:
            markdown = f"{markdown} *({emotion})*"

        chunks.append(
            {
                "timestamp": entry_timestamp,
                "markdown": markdown,
                "source": entry,
            }
        )

    # Indexer metadata - agent is always "audio" for audio transcripts
    meta["indexer"] = {"agent": "audio"}

    return chunks, meta


def _format_transcript_entries(path: Path, metadata: dict, entries: list[dict]) -> str:
    """Format transcript metadata and entries as human-readable text.

    This is a convenience wrapper around format_audio() that returns
    a single concatenated string.
    """
    # Reconstruct full entries list with metadata as first entry
    # (format_audio expects raw JSONL entries with metadata on first line)
    full_entries = [metadata] + entries
    context = {"file_path": path}
    chunks, meta = format_audio(full_entries, context)
    parts = []
    if meta.get("header"):
        parts.append(meta["header"])
    parts.extend(chunk["markdown"] for chunk in chunks)
    return "\n".join(parts)
