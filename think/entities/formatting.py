# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Entity formatting for indexer.

This module provides format_entities() which is registered in the formatters
registry to convert entity JSONL files into markdown chunks for indexing.
"""

import re
from datetime import datetime
from pathlib import Path
from typing import Any

from think.entities.core import EntityDict, entity_last_active_ts


def format_entities(
    entries: list[EntityDict],
    context: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Format entity JSONL entries to markdown chunks.

    This is the formatter function used by the formatters registry.
    Works for both attached entities (facets/*/entities/<id>/entity.json) and
    detected entities (facets/*/entities/*.jsonl).

    Args:
        entries: Raw JSONL entries (one entity per line)
        context: Optional context with:
            - file_path: Path to JSONL file (for extracting facet name and type)

    Returns:
        Tuple of (chunks, meta) where:
            - chunks: List of dicts with keys:
                - timestamp: int (unix ms)
                - markdown: str
                - source: dict (original entity entry)
            - meta: Dict with optional "header" and "error" keys
    """
    ctx = context or {}
    file_path = ctx.get("file_path")
    meta: dict[str, Any] = {}
    chunks: list[dict[str, Any]] = []

    # Determine if attached or detected, extract facet name and day
    facet_name = "unknown"
    is_detected = False
    day_str: str | None = None

    if file_path:
        file_path = Path(file_path)

        # Extract facet name from path
        # Pattern: facets/{facet}/entities/{day}.jsonl (detected entities)
        path_str = str(file_path)
        facet_match = re.search(r"facets/([^/]+)/entities", path_str)
        if facet_match:
            facet_name = facet_match.group(1)

        # Check if detected (has day in filename)
        if file_path.parent.name == "entities" and file_path.stem.isdigit():
            is_detected = True
            day_str = file_path.stem

    # Build header
    if is_detected and day_str:
        # Format day as YYYY-MM-DD for readability
        formatted_day = f"{day_str[:4]}-{day_str[4:6]}-{day_str[6:8]}"
        header_title = f"# Detected Entities: {facet_name} ({formatted_day})\n"
    else:
        header_title = f"# Attached Entities: {facet_name}\n"

    entity_count = len(entries)
    meta["header"] = f"{header_title}\n{entity_count} entities"

    # Calculate base timestamp for detected entities (midnight of that day)
    detected_base_ts = 0
    if is_detected and day_str:
        try:
            dt = datetime.strptime(day_str, "%Y%m%d")
            detected_base_ts = int(dt.timestamp() * 1000)
        except ValueError:
            pass

    # Format each entity as a chunk
    for entity in entries:
        etype = entity.get("type", "Unknown")
        name = entity.get("name", "Unnamed")
        description = entity.get("description", "")

        # Determine timestamp
        if is_detected:
            ts = detected_base_ts
        else:
            # Attached: use activity timestamp (full fallback chain)
            ts = entity_last_active_ts(entity)

        # Build markdown for this entity
        lines = [
            f"### {etype}: {name}\n",
            "",
        ]

        # Description or placeholder
        if description:
            lines.append(description)
        else:
            lines.append("*(No description available)*")
        lines.append("")

        # Additional fields (skip core fields, timestamp fields, id, and detached flag)
        skip_fields = {
            "id",
            "type",
            "name",
            "description",
            "updated_at",
            "attached_at",
            "last_seen",
            "detached",
        }

        # Handle tags specially
        tags = entity.get("tags")
        if tags and isinstance(tags, list):
            lines.append(f"**Tags:** {', '.join(tags)}")

        # Handle aka specially
        aka = entity.get("aka")
        if aka and isinstance(aka, list):
            lines.append(f"**Also known as:** {', '.join(aka)}")

        # Other custom fields
        for key, value in entity.items():
            if key in skip_fields or key in ("tags", "aka"):
                continue
            # Format value appropriately
            if isinstance(value, list):
                value_str = ", ".join(str(v) for v in value)
            else:
                value_str = str(value)
            # Capitalize first letter of key for display
            display_key = key.replace("_", " ").title()
            lines.append(f"**{display_key}:** {value_str}")

        lines.append("")

        chunks.append(
            {
                "timestamp": ts,
                "markdown": "\n".join(lines),
                "source": entity,
            }
        )

    # Indexer metadata - agent depends on attached vs detected
    agent = "entity:detected" if is_detected else "entity:attached"
    meta["indexer"] = {"agent": agent}

    return chunks, meta
