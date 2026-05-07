# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Entity formatting for indexer.

Formatters for entity-related files (JSONL and JSON), registered in the formatters
registry to convert structured data into markdown chunks for indexing.
"""

import re
from datetime import datetime
from pathlib import Path
from typing import Any

from solstone.think.entities.core import EntityDict, entity_last_active_ts


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


def format_entity_identity(
    entries: list[EntityDict],
    context: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Format a standalone entity identity JSON file to markdown chunks."""
    _ = context
    meta: dict[str, Any] = {"indexer": {"agent": "entity"}}
    chunks: list[dict[str, Any]] = []

    if not entries:
        return chunks, meta

    entity = entries[0]
    day = entity.get("last_seen")
    if isinstance(day, str) and re.fullmatch(r"\d{8}", day):
        meta["indexer"]["day"] = day

    etype = entity.get("type", "Unknown")
    name = entity.get("name", "Unnamed")
    description = entity.get("description", "")

    lines = [
        f"### {etype}: {name}\n",
        "",
    ]

    if description:
        lines.append(description)
    else:
        lines.append("*(No description available)*")
    lines.append("")

    skip_fields = {
        "id",
        "type",
        "name",
        "description",
        "is_principal",
        "blocked",
        "updated_at",
        "attached_at",
        "last_seen",
        "created_at",
        "detached",
    }

    tags = entity.get("tags")
    if tags and isinstance(tags, list):
        lines.append(f"**Tags:** {', '.join(tags)}")

    aka = entity.get("aka")
    if aka and isinstance(aka, list):
        lines.append(f"**Also known as:** {', '.join(aka)}")

    for key, value in entity.items():
        if key in skip_fields or key in ("tags", "aka"):
            continue
        if isinstance(value, list):
            value_str = ", ".join(str(v) for v in value)
        else:
            value_str = str(value)
        display_key = key.replace("_", " ").title()
        lines.append(f"**{display_key}:** {value_str}")

    lines.append("")

    chunks.append(
        {
            "timestamp": entity_last_active_ts(entity),
            "markdown": "\n".join(lines),
            "source": entity,
        }
    )

    return chunks, meta


def format_observations(
    entries: list[dict[str, Any]],
    context: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Format entity observation JSONL entries to markdown chunks.

    This formatter handles observations stored under:
    facets/{facet}/entities/{slug}/observations.jsonl.

    Args:
        entries: Raw JSONL entries (one observation per line)
        context: Optional context with:
            - file_path: Path to observations file (for extracting slug)

    Returns:
        Tuple of (chunks, meta) where:
            - chunks: List of dicts with keys:
                - timestamp: int (observed_at in milliseconds, default 0)
                - markdown: str
                - source: dict (original observation entry)
            - meta: Dict with "header" and "indexer" keys
    """
    ctx = context or {}
    file_path = ctx.get("file_path")
    slug = Path(file_path).parent.name if file_path else "unknown"
    entity_name = slug.replace("_", " ").title()

    meta: dict[str, Any] = {
        "header": f"# Observations: {entity_name}\n\n{len(entries)} observations",
        "indexer": {"agent": "observation"},
    }

    chunks: list[dict[str, Any]] = []
    for entry in entries:
        content = entry.get("content", "")
        markdown = f"- {content}"
        source_day = entry.get("source_day")
        if source_day:
            markdown += f" (observed: {source_day})"

        chunks.append(
            {
                "timestamp": entry.get("observed_at", 0),
                "markdown": markdown,
                "source": entry,
            }
        )

    return chunks, meta
