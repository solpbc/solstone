# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Unified journal index for all content types.

This module provides a single FTS5 index over journal content:
- Agent outputs (markdown files)
- Events (facet event JSONL)
- Entities (facet entity JSONL)
- Todos (facet todo JSONL)
- Action logs (facet/journal-level JSONL)

All content is converted to markdown chunks via the formatters framework,
then indexed with metadata fields for filtering (day, facet, agent).
Raw audio/screen transcripts are formattable but not indexed by default.
"""

import json
import logging
import math
import os
import re
import sqlite3
import time
from datetime import date
from pathlib import Path
from typing import Any

from think.entities.core import atomic_write, entity_slug
from think.formatters import (
    extract_path_metadata,
    find_formattable_files,
    format_file,
    get_formatter,
    load_jsonl,
)
from think.utils import DATE_RE, get_journal, now_ms

logger = logging.getLogger(__name__)

# Noise entity patterns — transcript artifacts that should not be indexed.
# Matches "Speaker N", "Unknown/Unidentified <word>" (single-word role).
# Multi-word patterns ("Speaker Diarization") and bare "Unknown" are kept.
_NOISE_ENTITY_RE = re.compile(
    r"^Speaker \d+(?:\s*\(.*\))?$"
    r"|^(?:Unknown|Unidentified) \w+(?:\s*\d+)?(?:\s*\(.*\))?$",
    re.IGNORECASE,
)


def is_noise_entity(name: str) -> bool:
    """Return True if the entity name is a transcript artifact (noise)."""
    return bool(_NOISE_ENTITY_RE.match(name))


def _strip_md_formatting(text: str) -> str:
    """Strip bold markers and backticks from a markdown table cell value."""
    text = text.strip()
    # Remove bold markers
    if text.startswith("**") and text.endswith("**"):
        text = text[2:-2]
    # Remove backticks
    if text.startswith("`") and text.endswith("`"):
        text = text[1:-1]
    return text.strip()


# Database constants
INDEX_DIR = "indexer"
DB_NAME = "journal.sqlite"

# Schema for the unified journal index
SCHEMA = [
    "CREATE TABLE IF NOT EXISTS files(path TEXT PRIMARY KEY, mtime INTEGER)",
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING fts5(
        content,
        path UNINDEXED,
        day UNINDEXED,
        facet UNINDEXED,
        agent UNINDEXED,
        stream UNINDEXED,
        idx UNINDEXED
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS entities(
        entity_id TEXT NOT NULL,
        source TEXT NOT NULL,
        facet TEXT,
        day TEXT,
        name TEXT,
        type TEXT,
        description TEXT,
        tags TEXT,
        contact TEXT,
        aka TEXT,
        is_principal INTEGER,
        blocked INTEGER,
        observation_count INTEGER,
        last_observed TEXT,
        attached_at TEXT,
        updated_at TEXT,
        last_seen TEXT,
        created_at TEXT,
        detached INTEGER,
        path TEXT NOT NULL,
        PRIMARY KEY (path, entity_id, source)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_entities_id ON entities(entity_id)",
    "CREATE INDEX IF NOT EXISTS idx_entities_facet ON entities(facet)",
    "CREATE INDEX IF NOT EXISTS idx_entities_source ON entities(source)",
    """
    CREATE TABLE IF NOT EXISTS entity_signals(
        signal_type TEXT NOT NULL,
        entity_name TEXT NOT NULL,
        entity_type TEXT,
        target_name TEXT,
        relationship_type TEXT,
        day TEXT,
        facet TEXT,
        event_title TEXT,
        event_type TEXT,
        path TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_signals_type ON entity_signals(signal_type)",
    "CREATE INDEX IF NOT EXISTS idx_signals_entity ON entity_signals(entity_name)",
    "CREATE INDEX IF NOT EXISTS idx_signals_day ON entity_signals(day)",
    "CREATE INDEX IF NOT EXISTS idx_signals_path ON entity_signals(path)",
    "CREATE INDEX IF NOT EXISTS idx_signals_facet ON entity_signals(facet)",
]

ENTITY_COLUMNS = [
    "entity_id",
    "source",
    "facet",
    "day",
    "name",
    "type",
    "description",
    "tags",
    "contact",
    "aka",
    "is_principal",
    "blocked",
    "observation_count",
    "last_observed",
    "attached_at",
    "updated_at",
    "last_seen",
    "created_at",
    "detached",
    "path",
]

SIGNAL_COLUMNS = [
    "signal_type",
    "entity_name",
    "entity_type",
    "target_name",
    "relationship_type",
    "day",
    "facet",
    "event_title",
    "event_type",
    "path",
]


def _insert_entity_row(conn: sqlite3.Connection, row: dict[str, Any]) -> None:
    """Insert one entity row into the entities table."""
    columns = ",".join(ENTITY_COLUMNS)
    placeholders = ",".join("?" for _ in ENTITY_COLUMNS)
    values = [row.get(col) for col in ENTITY_COLUMNS]
    conn.execute(f"INSERT INTO entities({columns}) VALUES ({placeholders})", values)


def _insert_signal_row(conn: sqlite3.Connection, row: dict[str, Any]) -> None:
    """Insert one signal row into the entity_signals table."""
    columns = ",".join(SIGNAL_COLUMNS)
    placeholders = ",".join("?" for _ in SIGNAL_COLUMNS)
    values = [row.get(col) for col in SIGNAL_COLUMNS]
    conn.execute(
        f"INSERT INTO entity_signals({columns}) VALUES ({placeholders})", values
    )


def _entity_source_from_path(rel_path: str) -> str | None:
    """Infer entity source type from a relative path."""
    rel = rel_path.replace("\\", "/")
    if re.fullmatch(r"entities/[^/]+/entity\.json", rel):
        return "identity"
    if re.fullmatch(r"facets/[^/]+/entities/[^/]+/entity\.json", rel):
        return "relationship"
    if re.match(r"\d{8}\.jsonl$", Path(rel).name) and re.fullmatch(
        r"facets/[^/]+/entities/\d{8}\.jsonl", rel
    ):
        return "detected"
    if re.fullmatch(r"facets/[^/]+/entities/[^/]+/observations\.jsonl", rel):
        return "observation"
    return None


def _find_entity_files(journal: str) -> dict[str, tuple[str, str]]:
    """Find all supported entity source files.

    Returns:
        Mapping from relative path to tuple of (absolute_path, source_type).
    """
    journal_path = Path(journal)
    files: dict[str, tuple[str, str]] = {}

    for path in journal_path.glob("entities/*/entity.json"):
        if path.is_file():
            rel = path.relative_to(journal_path).as_posix()
            files[rel] = (str(path), "identity")

    for path in journal_path.glob("facets/*/entities/*/entity.json"):
        if path.is_file():
            rel = path.relative_to(journal_path).as_posix()
            files[rel] = (str(path), "relationship")

    for path in journal_path.glob("facets/*/entities/*.jsonl"):
        filename = path.name
        if re.match(r"\d{8}\.jsonl$", filename):
            rel = path.relative_to(journal_path).as_posix()
            files[rel] = (str(path), "detected")

    for path in journal_path.glob("facets/*/entities/*/observations.jsonl"):
        if path.is_file():
            rel = path.relative_to(journal_path).as_posix()
            files[rel] = (str(path), "observation")

    return files


def _find_signal_files(journal: str) -> dict[str, tuple[str, str]]:
    """Find all signal source files (KG markdown + event JSONL)."""
    journal_path = Path(journal)
    files: dict[str, tuple[str, str]] = {}

    for path in journal_path.glob("*/agents/knowledge_graph.md"):
        if path.is_file():
            rel = path.relative_to(journal_path).as_posix()
            files[rel] = (str(path), "kg")

    for path in journal_path.glob("facets/*/events/*.jsonl"):
        if path.is_file() and re.match(r"\d{8}\.jsonl$", path.name):
            rel = path.relative_to(journal_path).as_posix()
            files[rel] = (str(path), "event")

    return files


def _extract_entity_identity(journal: str, rel_path: str) -> list[dict[str, Any]]:
    """Read a journal entity file and return row data."""
    abs_path = os.path.join(journal, rel_path)
    rel_parts = rel_path.replace("\\", "/").split("/")
    if len(rel_parts) < 2:
        return []
    entity_id = rel_parts[1]

    try:
        with open(abs_path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Skipping %s: %s", rel_path, e)
        return []

    return [
        {
            "entity_id": entity_id,
            "source": "identity",
            "facet": None,
            "day": None,
            "name": data.get("name"),
            "type": data.get("type"),
            "description": None,
            "tags": None,
            "contact": None,
            "aka": json.dumps(data["aka"]) if data.get("aka") else None,
            "is_principal": 1 if data.get("is_principal") else 0,
            "blocked": 1 if data.get("blocked") else 0,
            "observation_count": None,
            "last_observed": None,
            "attached_at": None,
            "updated_at": data.get("updated_at"),
            "last_seen": None,
            "created_at": data.get("created_at"),
            "detached": None,
            "path": rel_path,
        }
    ]


def _extract_entity_relationship(journal: str, rel_path: str) -> list[dict[str, Any]]:
    """Read a relationship entity file and return row data."""
    abs_path = os.path.join(journal, rel_path)
    rel_parts = rel_path.replace("\\", "/").split("/")
    if len(rel_parts) < 4:
        return []
    facet = rel_parts[1]
    entity_id = rel_parts[3]

    try:
        with open(abs_path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Skipping %s: %s", rel_path, e)
        return []

    return [
        {
            "entity_id": entity_id,
            "source": "relationship",
            "facet": facet,
            "day": None,
            "name": None,
            "type": None,
            "description": data.get("description"),
            "tags": json.dumps(data["tags"]) if data.get("tags") else None,
            "contact": data.get("contact"),
            "aka": None,
            "is_principal": None,
            "blocked": None,
            "observation_count": None,
            "last_observed": None,
            "attached_at": data.get("attached_at"),
            "updated_at": data.get("updated_at"),
            "last_seen": data.get("last_seen"),
            "created_at": None,
            "detached": 1 if data.get("detached") else 0,
            "path": rel_path,
        }
    ]


def _extract_entity_detected(journal: str, rel_path: str) -> list[dict[str, Any]]:
    """Read an entities JSONL file and return one row per valid entity."""
    abs_path = os.path.join(journal, rel_path)
    rel_parts = rel_path.replace("\\", "/").split("/")
    if len(rel_parts) < 3:
        return []
    facet = rel_parts[1]
    day = Path(rel_path).name.removesuffix(".jsonl")

    rows: list[dict[str, Any]] = []
    try:
        with open(abs_path, encoding="utf-8") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError as e:
                    logger.warning("Skipping malformed JSONL in %s: %s", rel_path, e)
                    continue

                name = data.get("name")
                if not name:
                    continue

                entity_id = data.get("id") or entity_slug(name)
                rows.append(
                    {
                        "entity_id": entity_id,
                        "source": "detected",
                        "facet": facet,
                        "day": day,
                        "name": name,
                        "type": data.get("type"),
                        "description": data.get("description"),
                        "tags": None,
                        "contact": None,
                        "aka": None,
                        "is_principal": None,
                        "blocked": None,
                        "observation_count": None,
                        "last_observed": None,
                        "attached_at": None,
                        "updated_at": None,
                        "last_seen": None,
                        "created_at": None,
                        "detached": None,
                        "path": rel_path,
                    }
                )
    except OSError as e:
        logger.warning("Skipping %s: %s", rel_path, e)
        return []

    return rows


def _extract_entity_observations(journal: str, rel_path: str) -> list[dict[str, Any]]:
    """Summarize observations for an entity into a single row."""
    abs_path = os.path.join(journal, rel_path)
    rel_parts = rel_path.replace("\\", "/").split("/")
    if len(rel_parts) < 4:
        return []
    facet = rel_parts[1]
    entity_id = rel_parts[3]

    count = 0
    last_observed = None

    try:
        with open(abs_path, encoding="utf-8") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError as e:
                    logger.warning("Skipping malformed JSONL in %s: %s", rel_path, e)
                    continue

                count += 1
                observed_at = data.get("observed_at")
                if observed_at and (
                    last_observed is None or observed_at > last_observed
                ):
                    last_observed = observed_at
    except OSError as e:
        logger.warning("Skipping %s: %s", rel_path, e)
        return []

    if count == 0:
        return []

    return [
        {
            "entity_id": entity_id,
            "source": "observation",
            "facet": facet,
            "day": None,
            "name": None,
            "type": None,
            "description": None,
            "tags": None,
            "contact": None,
            "aka": None,
            "is_principal": None,
            "blocked": None,
            "observation_count": count,
            "last_observed": last_observed,
            "attached_at": None,
            "updated_at": None,
            "last_seen": None,
            "created_at": None,
            "detached": None,
            "path": rel_path,
        }
    ]


def _is_historical_signal_file(rel_path: str, source_type: str) -> bool:
    """Check if a signal source file is historical and should be skipped in light mode.

    KG files are never historical — they are accumulated (indexed once, kept forever)
    like events. Only today's KG file changes; historical KG files are stable and should
    be indexed incrementally then left alone.
    """
    return False


def _load_facet_entities_for_day(journal: str, day: str) -> dict[str, set[str]]:
    """Load entity names detected in each facet for a given day.

    Returns a mapping of lowercase entity name → set of facets where detected.
    """
    journal_path = Path(journal)
    result: dict[str, set[str]] = {}

    for detection_file in journal_path.glob(f"facets/*/entities/{day}.jsonl"):
        facet = detection_file.relative_to(journal_path).parts[1]
        try:
            with open(detection_file, encoding="utf-8") as f:
                for raw in f:
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        data = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    name = data.get("name", "").strip()
                    if name:
                        result.setdefault(name.lower(), set()).add(facet)
        except OSError:
            continue

    return result


def _extract_signal_kg(journal: str, rel_path: str) -> list[dict[str, Any]]:
    """Extract KG appearance and edge signals from a knowledge graph markdown file."""
    abs_path = os.path.join(journal, rel_path)
    day = rel_path.split("/")[0]

    try:
        with open(abs_path, encoding="utf-8") as f:
            content = f.read()
    except OSError as e:
        logger.warning("Skipping %s: %s", rel_path, e)
        return []

    # Split entity vs relationship sections by finding the relationship table
    # header row. Requires both "Source" and either "Target" or "Relationship"
    # in the same pipe-delimited row to avoid false positives on entity rows
    # that happen to contain "Source" as part of an entity name.
    source_hdr = re.search(
        r"^\|[^|]*Source\s*\w*[^|]*\|[^|]*(?:Target|Relationship)\s*\w*[^|]*\|",
        content,
        re.MULTILINE | re.IGNORECASE,
    )
    if source_hdr:
        entity_section = content[: source_hdr.start()]
        relationship_section = content[source_hdr.start() :]
    else:
        entity_section = content
        relationship_section = ""

    rows: list[dict[str, Any]] = []

    # Match entity rows in pipe-delimited tables. Handles bold (**Name**),
    # backtick (`Name`), or plain text entity names. The second column must
    # look like an entity type (not a separator row or header underline).
    appearance_re = re.compile(
        r"^\|\s*(?:\*\*(.+?)\*\*|`([^`]+)`|([^|*`\n][^|\n]*?))\s*\|\s*([^|]+?)\s*\|",
        re.MULTILINE,
    )
    # Column header values that should not be treated as entity types
    _HEADER_VALUES = frozenset(
        {
            "entity type",
            "type",
            "category",
            "classification",
            "role",
            "entity name",
            "name",
            "first appearance",
            "total engagement",
            "context",
            "description",
            "notes",
        }
    )
    for m in appearance_re.finditer(entity_section):
        entity_name = (m.group(1) or m.group(2) or m.group(3) or "").strip()
        entity_type = m.group(4).strip()
        if not entity_name:
            continue
        if entity_type.startswith(":") or entity_type.startswith("-"):
            continue
        if entity_type.lower() in _HEADER_VALUES:
            continue
        if is_noise_entity(entity_name):
            continue
        rows.append(
            {
                "signal_type": "kg_appearance",
                "entity_name": entity_name,
                "entity_type": entity_type,
                "target_name": None,
                "relationship_type": None,
                "day": day,
                "facet": None,
                "event_title": None,
                "event_type": None,
                "path": rel_path,
            }
        )

    # --- Edge extraction (table format) ---
    # KG files use varied table formats (LLM-generated). Parse the relationship
    # table header to detect column positions, then extract by index. Handles
    # bold/non-bold entity names and any column order.
    if relationship_section:
        rel_lines = relationship_section.split("\n")
        source_col = target_col = rel_col = -1
        data_start = 0
        for li, line in enumerate(rel_lines):
            cells = [c.strip().lower() for c in line.split("|")]
            for ci, cell in enumerate(cells):
                if "source" in cell:
                    source_col = ci
                elif "target" in cell:
                    target_col = ci
                elif "relationship" in cell:
                    rel_col = ci
            if source_col >= 0 and (target_col >= 0 or rel_col >= 0):
                data_start = li + 1
                break

        # Skip separator row (| :--- | :--- | ... |)
        if data_start < len(rel_lines) and re.match(
            r"^\s*\|[\s:\-|]+\|\s*$", rel_lines[data_start]
        ):
            data_start += 1

        if source_col >= 0 and target_col >= 0 and rel_col >= 0:
            for line in rel_lines[data_start:]:
                if not line.strip().startswith("|"):
                    if line.strip().startswith("#") or (line.strip() == "---"):
                        break  # next section
                    continue
                cells = [c.strip() for c in line.split("|")]
                if max(source_col, target_col, rel_col) >= len(cells):
                    continue
                source = _strip_md_formatting(cells[source_col])
                target = _strip_md_formatting(cells[target_col])
                rel_type = _strip_md_formatting(cells[rel_col])
                if not source or not target or not rel_type:
                    continue
                if rel_type.startswith(":") or rel_type.startswith("-"):
                    continue
                if is_noise_entity(source) or is_noise_entity(target):
                    continue
                rows.append(
                    {
                        "signal_type": "kg_edge",
                        "entity_name": source,
                        "entity_type": None,
                        "target_name": target,
                        "relationship_type": rel_type,
                        "day": day,
                        "facet": None,
                        "event_title": None,
                        "event_type": None,
                        "path": rel_path,
                    }
                )

    # --- Edge extraction (bullet-list formats, early KG files) ---
    # Format C: * **Source** `rel-type` **Target**
    edge_re_bullet = re.compile(
        r"^\s*[\*\-]\s+\*\*(.+?)\*\*\s+`([^`]+)`\s+\*\*(.+?)\*\*",
        re.MULTILINE,
    )
    # Format D: `Source` -> `rel-type` -> `Target`
    edge_re_arrow = re.compile(
        r"^\s*[\*\-]\s+`([^`]+)`\s*->\s*`([^`]+)`\s*->\s*`([^`]+)`",
        re.MULTILINE,
    )
    # Format E: (Source) -> `rel-type` -> (Target)
    edge_re_paren = re.compile(
        r"^\s*[\*\-]\s+\(([^)]+)\)\s*->\s*`([^`]+)`\s*->\s*\(([^)]+)\)",
        re.MULTILINE,
    )
    # Format F: `Source` **rel-type** `Target`
    edge_re_bold_rel = re.compile(
        r"^\s*[\*\-]\s+`([^`]+)`\s+\*\*([^*]+)\*\*\s+`([^`]+)`",
        re.MULTILINE,
    )
    # Format G: Source `rel-type` Target (no bold, no parens — plain text with backtick rel)
    edge_re_plain = re.compile(
        r"^\s*[\*\-]\s+([A-Z][^`\n]+?)\s+`([^`]+)`\s+([A-Z][^`\n]+?)$",
        re.MULTILINE,
    )
    # Format H: **Source** ---[rel]---> **Target** or **Source** --- `rel` ---> **Target**
    edge_re_graph = re.compile(
        r"^\s*[\*\-]\s+\*\*(.+?)\*\*\s*(?:\([^)]*\)\s*)?-+\s*[\[`]([^\]`]+)[\]`]\s*-+>?\s*\*\*(.+?)\*\*",
        re.MULTILINE,
    )
    # Format I: (Source) `rel-type` (Target) (parens without arrows)
    edge_re_paren_backtick = re.compile(
        r"^\s*[\*\-]\s+\(([^)]+)\)\s+`([^`]+)`\s+\(([^)]+)\)",
        re.MULTILINE,
    )
    # Format J: **Source:** X -> **Relationship:** `rel` -> **Target:** Y (labeled arrow)
    edge_re_labeled = re.compile(
        r"^\s*[\*\-]\s+\*\*Source:\*\*\s*(.+?)\s*->\s*\*\*Relationship:\*\*\s*`([^`]+)`\s*->\s*\*\*Target:\*\*\s*(.+?)$",
        re.MULTILINE,
    )
    # Format K: `(Source) -> [rel] -> (Target)` (backtick-wrapped arrow with bracket rel)
    edge_re_backtick_arrow = re.compile(
        r"^\s*[\*\-]\s+`\(([^)]+)\)\s*->\s*\[([^\]]+)\]\s*->\s*\(([^)]+)\)`",
        re.MULTILINE,
    )

    # Scope bullet-list edge matching to the relationship section only (or full
    # content when no section split was found) to avoid false matches from
    # bullet-formatted engagement notes in entity appearance tables.
    bullet_search_text = relationship_section or content
    for regex in (
        edge_re_bullet,
        edge_re_arrow,
        edge_re_paren,
        edge_re_bold_rel,
        edge_re_plain,
        edge_re_graph,
        edge_re_paren_backtick,
        edge_re_labeled,
        edge_re_backtick_arrow,
    ):
        for m in regex.finditer(bullet_search_text):
            source = m.group(1).strip()
            rel_type = m.group(2).strip()
            target = m.group(3).strip()
            if is_noise_entity(source) or is_noise_entity(target):
                continue
            rows.append(
                {
                    "signal_type": "kg_edge",
                    "entity_name": source,
                    "entity_type": None,
                    "target_name": target,
                    "relationship_type": rel_type,
                    "day": day,
                    "facet": None,
                    "event_title": None,
                    "event_type": None,
                    "path": rel_path,
                }
            )

    # Assign facets by cross-referencing entity names against facet detection data
    facet_entities = _load_facet_entities_for_day(journal, day)
    if not facet_entities:
        return rows  # no detection data for this day — all stay facet=None

    expanded: list[dict[str, Any]] = []
    for row in rows:
        if row["signal_type"] == "kg_appearance":
            facets = facet_entities.get(row["entity_name"].lower(), set())
            if facets:
                for facet in sorted(facets):
                    expanded.append({**row, "facet": facet})
            else:
                expanded.append(row)
        elif row["signal_type"] == "kg_edge":
            source_facets = facet_entities.get(row["entity_name"].lower(), set())
            target_facets = facet_entities.get(
                (row["target_name"] or "").lower(), set()
            )
            shared = source_facets & target_facets
            if shared:
                for facet in sorted(shared):
                    expanded.append({**row, "facet": facet})
            else:
                expanded.append(row)
        else:
            expanded.append(row)

    return expanded


def _extract_signal_event_participants(
    journal: str, rel_path: str
) -> list[dict[str, Any]]:
    """Extract participant signals from an event JSONL file."""
    abs_path = os.path.join(journal, rel_path)
    parts = rel_path.replace("\\", "/").split("/")
    if len(parts) < 4:
        return []
    facet = parts[1]
    day = Path(rel_path).stem

    rows: list[dict[str, Any]] = []
    try:
        with open(abs_path, encoding="utf-8") as f:
            for event_index, raw in enumerate(f):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError as e:
                    logger.warning("Skipping malformed JSONL in %s: %s", rel_path, e)
                    continue

                participants = event.get("participants", [])
                if not participants:
                    continue

                event_path = f"{rel_path}#{event_index}"
                title = event.get("title", "")
                event_type = event.get("type", "")
                for name in participants:
                    if not name or is_noise_entity(name):
                        continue
                    rows.append(
                        {
                            "signal_type": "event_participant",
                            "entity_name": name,
                            "entity_type": None,
                            "target_name": None,
                            "relationship_type": None,
                            "day": day,
                            "facet": facet,
                            "event_title": title,
                            "event_type": event_type,
                            "path": event_path,
                        }
                    )
    except OSError as e:
        logger.warning("Skipping %s: %s", rel_path, e)
        return []

    return rows


def _ensure_schema(conn: sqlite3.Connection) -> None:
    """Create required tables if they don't exist."""
    for statement in SCHEMA:
        conn.execute(statement)


def get_journal_index(journal: str | None = None) -> tuple[sqlite3.Connection, str]:
    """Return SQLite connection for the journal index.

    Args:
        journal: Path to journal root. Uses JOURNAL_PATH env var if not provided.

    Returns:
        Tuple of (connection, db_path)
    """
    journal = journal or get_journal()

    db_dir = os.path.join(journal, INDEX_DIR)
    os.makedirs(db_dir, exist_ok=True)
    db_path = os.path.join(db_dir, DB_NAME)

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    _ensure_schema(conn)

    return conn, db_path


def reset_journal_index(journal: str) -> None:
    """Remove the journal index database file."""
    db_path = os.path.join(journal, INDEX_DIR, DB_NAME)
    try:
        os.unlink(db_path)
    except FileNotFoundError:
        pass


def index_file(journal: str, file_path: str, verbose: bool = False) -> bool:
    """Index a single file into the journal index.

    Validates that the file exists, is under the journal directory, and has
    a registered formatter. Then indexes it (replacing any existing chunks).

    Args:
        journal: Path to journal root directory
        file_path: Absolute or journal-relative path to file
        verbose: If True, log detailed progress

    Returns:
        True if file was indexed successfully

    Raises:
        ValueError: If file is outside journal or has no formatter
        FileNotFoundError: If file doesn't exist
    """
    journal_path = Path(journal).resolve()

    # Resolve file path (handle both absolute and relative)
    if os.path.isabs(file_path):
        abs_path = Path(file_path).resolve()
    else:
        abs_path = (journal_path / file_path).resolve()

    # Validate file exists
    if not abs_path.is_file():
        raise FileNotFoundError(f"File not found: {abs_path}")

    # Validate file is under journal
    try:
        rel_path = str(abs_path.relative_to(journal_path))
    except ValueError:
        raise ValueError(f"File is outside journal directory: {abs_path}") from None

    # Validate formatter exists
    if get_formatter(rel_path) is None:
        raise ValueError(f"No formatter found for: {rel_path}")

    # Get file mtime
    mtime = int(os.path.getmtime(abs_path))

    # Index the file
    conn, _ = get_journal_index(journal)

    # Delete existing chunks for this file
    conn.execute("DELETE FROM chunks WHERE path=?", (rel_path,))

    if verbose:
        logger.info("Indexing %s", rel_path)

    stream = _extract_stream(journal, rel_path)
    _index_file(conn, rel_path, str(abs_path), verbose, stream=stream)

    # Update file mtime
    conn.execute("REPLACE INTO files(path, mtime) VALUES (?, ?)", (rel_path, mtime))

    conn.commit()
    conn.close()

    return True


def _extract_stream(journal: str, rel: str) -> str | None:
    """Extract stream name from a journal-relative path's segment directory.

    Reads stream.json from the segment dir if the path is inside a segment
    (e.g., "20240101/142500_300/agents/facet/flow.md").

    Returns stream name string or None for non-segment paths or pre-stream segments.
    """
    from think.streams import read_segment_stream
    from think.utils import segment_key

    parts = rel.replace("\\", "/").split("/")
    # Segment paths: parts[0]=day, parts[1]=stream, parts[2]=segment, parts[3+]=file
    if len(parts) >= 3 and segment_key(parts[2]):
        seg_dir = os.path.join(journal, parts[0], parts[1], parts[2])
        marker = read_segment_stream(seg_dir)
        if marker:
            return marker.get("stream")
    return None


def _index_file(
    conn: sqlite3.Connection,
    rel: str,
    path: str,
    verbose: bool,
    stream: str | None = None,
) -> None:
    """Index a single file into the chunks table.

    Uses format_file() to convert content to markdown chunks,
    then inserts each chunk with metadata.

    Metadata is sourced from two places:
    - Path-derived: day and facet from extract_path_metadata()
    - Formatter-provided: agent from meta["indexer"]["agent"]
    For markdown files, agent is also path-derived.
    """
    try:
        chunks, meta = format_file(path)
    except (ValueError, FileNotFoundError) as e:
        logger.warning("Skipping %s: %s", rel, e)
        return

    # Get path-derived metadata (day, facet, agent for .md files)
    path_meta = extract_path_metadata(rel)

    # Get formatter-provided metadata (agent for JSONL files)
    formatter_indexer = meta.get("indexer", {})

    # Merge: formatter values override path values, normalize to lowercase
    day = formatter_indexer.get("day") or path_meta["day"]
    facet = (formatter_indexer.get("facet") or path_meta["facet"]).lower()
    agent = (formatter_indexer.get("agent") or path_meta["agent"]).lower()

    if verbose:
        logger.info(
            "  %s chunks, day=%s, facet=%s, agent=%s, stream=%s",
            len(chunks),
            day,
            facet,
            agent,
            stream,
        )

    for idx, chunk in enumerate(chunks):
        content = chunk.get("markdown", "")
        if not content:
            continue

        conn.execute(
            "INSERT INTO chunks(content, path, day, facet, agent, stream, idx) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (content, rel, day, facet, agent, stream, idx),
        )


def _is_historical_day(rel_path: str) -> bool:
    """Check if path is in a historical YYYYMMDD directory (before today).

    Returns True for paths like "20240101/..." where the date is before today.
    Returns False for non-day paths (facets/, imports/, apps/) or today/future.
    """
    from datetime import datetime

    if not rel_path or "/" not in rel_path:
        return False

    first_part = rel_path.split("/")[0]
    if not DATE_RE.fullmatch(first_part):
        return False  # Not a day directory

    today = datetime.now().strftime("%Y%m%d")
    return first_part < today


def _is_historical_entity_file(rel_path: str, source_type: str | None) -> bool:
    """Check if a detected entity file is historical based on filename day."""
    if source_type != "detected":
        return False

    from datetime import datetime

    filename = Path(rel_path).name
    match = re.match(r"\d{8}\.jsonl$", filename)
    if not match:
        return False

    today = datetime.now().strftime("%Y%m%d")
    day = filename[:-6]
    return day < today


def scan_entities(
    journal: str,
    conn: sqlite3.Connection,
    verbose: bool = False,
    full: bool = False,
) -> bool:
    """Scan and index entity source files."""
    entity_files = _find_entity_files(journal)

    in_scope: dict[str, tuple[str, str]] = entity_files
    if not full:
        in_scope = {
            rel: (path, source_type)
            for rel, (path, source_type) in entity_files.items()
            if not _is_historical_entity_file(rel, source_type)
        }

    logger.info("Scanning %s entity files...", len(in_scope))

    # Get current file mtimes from database.
    # Entity paths are stored with an "entity:" prefix to avoid collisions
    # with the chunk indexer, which also tracks some of the same files.
    ENTITY_PREFIX = "entity:"
    db_mtimes = {
        path: mtime
        for path, mtime in conn.execute(
            "SELECT path, mtime FROM files WHERE path LIKE 'entity:%'"
        )
    }
    to_index: list[tuple[str, str, int, str]] = []
    for rel, (path, source_type) in in_scope.items():
        try:
            mtime = int(os.path.getmtime(path))
        except OSError as e:
            logger.warning("Unable to stat %s: %s", rel, e)
            continue
        if db_mtimes.get(ENTITY_PREFIX + rel) != mtime:
            to_index.append((rel, path, mtime, source_type))

    cached = len(in_scope) - len(to_index)
    logger.info(
        "%s total entity files, %s cached, %s to index",
        len(in_scope),
        cached,
        len(to_index),
    )

    start = time.time()

    for i, (rel, path, mtime, source_type) in enumerate(to_index, 1):
        if verbose:
            logger.info("[%s/%s] %s", i, len(to_index), rel)

        conn.execute("DELETE FROM entities WHERE path=?", (rel,))

        if source_type == "identity":
            rows = _extract_entity_identity(journal, rel)
        elif source_type == "relationship":
            rows = _extract_entity_relationship(journal, rel)
        elif source_type == "detected":
            rows = _extract_entity_detected(journal, rel)
        elif source_type == "observation":
            rows = _extract_entity_observations(journal, rel)
        else:
            rows = []

        for row in rows:
            _insert_entity_row(conn, row)

        conn.execute(
            "REPLACE INTO files(path, mtime) VALUES (?, ?)",
            (ENTITY_PREFIX + rel, mtime),
        )

    # Entity files removed from scope (full vs light)
    removed: set[str] = set()
    db_entity_paths = {
        row[0] for row in conn.execute("SELECT DISTINCT path FROM entities").fetchall()
    }

    if full:
        removed = db_entity_paths - set(entity_files)
    else:
        in_scope_db = {
            path
            for path in db_entity_paths
            if not _is_historical_entity_file(path, _entity_source_from_path(path))
        }
        removed = in_scope_db - set(in_scope)

    for rel in removed:
        conn.execute("DELETE FROM entities WHERE path=?", (rel,))
        conn.execute("DELETE FROM files WHERE path=?", (ENTITY_PREFIX + rel,))

    if to_index or removed:
        conn.commit()

    elapsed = time.time() - start
    logger.info(
        "%s entity files indexed, %s entity rows removed in %.2f seconds",
        len(to_index),
        len(removed),
        elapsed,
    )

    return bool(to_index or removed)


def scan_signals(
    journal: str,
    conn: sqlite3.Connection,
    verbose: bool = False,
    full: bool = False,
) -> bool:
    """Scan and index signal source files."""
    signal_files = _find_signal_files(journal)

    in_scope: dict[str, tuple[str, str]] = signal_files
    if not full:
        in_scope = {
            rel: (path, source_type)
            for rel, (path, source_type) in signal_files.items()
            if not _is_historical_signal_file(rel, source_type)
        }

    logger.info("Scanning %s signal files...", len(in_scope))

    SIGNAL_PREFIX = "signal:"
    db_mtimes = {
        path: mtime
        for path, mtime in conn.execute(
            "SELECT path, mtime FROM files WHERE path LIKE 'signal:%'"
        )
    }
    to_index: list[tuple[str, str, int, str]] = []
    for rel, (path, source_type) in in_scope.items():
        try:
            mtime = int(os.path.getmtime(path))
        except OSError as e:
            logger.warning("Unable to stat %s: %s", rel, e)
            continue
        if db_mtimes.get(SIGNAL_PREFIX + rel) != mtime:
            to_index.append((rel, path, mtime, source_type))

    cached = len(in_scope) - len(to_index)
    logger.info(
        "%s total signal files, %s cached, %s to index",
        len(in_scope),
        cached,
        len(to_index),
    )

    start = time.time()

    for i, (rel, path, mtime, source_type) in enumerate(to_index, 1):
        if verbose:
            logger.info("[%s/%s] %s", i, len(to_index), rel)

        conn.execute(
            "DELETE FROM entity_signals WHERE path=? OR path LIKE ?",
            (rel, rel + "#%"),
        )

        if source_type == "kg":
            rows = _extract_signal_kg(journal, rel)
        elif source_type == "event":
            rows = _extract_signal_event_participants(journal, rel)
        else:
            rows = []

        for row in rows:
            _insert_signal_row(conn, row)

        conn.execute(
            "REPLACE INTO files(path, mtime) VALUES (?, ?)",
            (SIGNAL_PREFIX + rel, mtime),
        )

    removed: set[str] = set()
    db_signal_paths = {
        row[0]
        for row in conn.execute("SELECT DISTINCT path FROM entity_signals").fetchall()
    }

    if full:
        removed = {p for p in db_signal_paths if p.split("#")[0] not in signal_files}
    else:
        in_scope_db = {
            path
            for path in db_signal_paths
            if not _is_historical_signal_file(
                path.split("#")[0],
                "kg" if "/agents/knowledge_graph.md" in path else "event",
            )
        }
        removed = {p for p in in_scope_db if p.split("#")[0] not in in_scope}

    for rel in removed:
        conn.execute("DELETE FROM entity_signals WHERE path=?", (rel,))
        conn.execute("DELETE FROM files WHERE path=?", (SIGNAL_PREFIX + rel,))

    if to_index or removed:
        conn.commit()

    elapsed = time.time() - start
    logger.info(
        "%s signal files indexed, %s removed in %.2f seconds",
        len(to_index),
        len(removed),
        elapsed,
    )

    return bool(to_index or removed)


def consolidate_segment_entities(journal: str, full: bool = False) -> int:
    """Consolidate per-segment entity detections into the journal entity store.

    Reads agents/entities.jsonl files from all day/stream/segment directories,
    deduplicates by (name, type), and writes to entities/<slug>/entity.json for
    new entities. Skips any entity whose entity.json already exists (preserves
    user-managed and previously-consolidated records).

    Args:
        journal: Path to journal root directory
        full: If True, scan all day directories. If False, scan today only.

    Returns:
        Number of new journal entities written.
    """
    from datetime import datetime

    journal_path = Path(journal)
    today = datetime.now().strftime("%Y%m%d")

    # Collect all matching segment entity files across day/stream/segment dirs
    segment_files = []
    for path in journal_path.glob("**/agents/entities.jsonl"):
        if not path.is_file():
            continue
        try:
            day = path.relative_to(journal_path).parts[0]
        except (ValueError, IndexError):
            continue
        if not DATE_RE.fullmatch(day):
            continue  # Not a journal day directory
        if full or day == today:
            segment_files.append(path)

    if not segment_files:
        logger.info(
            "consolidated 0 entities from 0 segment files → 0 new journal entities written"
        )
        return 0

    # Collect and deduplicate entities from all segment files.
    # Key: (name.lower().strip(), type.lower().strip())
    # Value: {"name": ..., "type": ..., "description": ...}
    seen: dict[tuple[str, str], dict] = {}

    for seg_file in segment_files:
        try:
            with open(seg_file, encoding="utf-8") as f:
                for raw in f:
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        data = json.loads(raw)
                    except json.JSONDecodeError as e:
                        logger.warning(
                            "Skipping malformed JSONL in %s: %s", seg_file, e
                        )
                        continue

                    name = (data.get("name") or "").strip()
                    etype = (data.get("type") or "").strip()
                    description = (data.get("description") or "").strip()

                    if not name or not etype:
                        continue
                    if is_noise_entity(name):
                        continue

                    key = (name.lower(), etype.lower())
                    if key not in seen:
                        seen[key] = {
                            "name": name,
                            "type": etype,
                            "description": description,
                        }
                    elif len(description) > len(seen[key]["description"]):
                        # Longest description wins
                        seen[key]["description"] = description
        except OSError as e:
            logger.warning("Skipping %s: %s", seg_file, e)
            continue

    total_entities = len(seen)

    if not seen:
        logger.info(
            "consolidated 0 entities from %d segment files → 0 new journal entities written",
            len(segment_files),
        )
        return 0

    # Write new entities, skipping any entity.json that already exists.
    ts = now_ms()
    written = 0

    for (name_lower, _type_lower), data in seen.items():
        name = data["name"]
        etype = data["type"]
        description = data["description"]

        base_slug = entity_slug(name)
        if not base_slug:
            continue

        # Resolve slug: skip if entity already exists for this name, handle
        # collisions (different entity at same slug) by appending _2, _3, etc.
        final_slug = None
        for attempt in range(1, 102):
            candidate = base_slug if attempt == 1 else f"{base_slug}_{attempt}"
            candidate_path = journal_path / "entities" / candidate / "entity.json"

            if not candidate_path.exists():
                final_slug = candidate
                break

            # Path exists — check if it's the same entity
            try:
                with open(candidate_path, encoding="utf-8") as f:
                    existing = json.load(f)
                if (existing.get("name") or "").lower().strip() == name_lower:
                    # Entity already exists (prior run or user-created), skip
                    break
                # Different entity occupies this slug — try next suffix
            except (json.JSONDecodeError, OSError):
                # Unreadable file — treat as occupied, try next suffix
                continue
        else:
            logger.warning("Too many slug collisions for '%s', skipping", name)
            continue

        if final_slug is None:
            continue  # Entity already exists

        entity: dict[str, Any] = {
            "id": final_slug,
            "name": name,
            "type": etype,
            "source": "detected",
            "created_at": ts,
            "updated_at": ts,
        }
        if description:
            entity["description"] = description

        entity_path = journal_path / "entities" / final_slug / "entity.json"
        try:
            content = json.dumps(entity, ensure_ascii=False, indent=2) + "\n"
            atomic_write(entity_path, content, prefix=".entity_")
            written += 1
        except OSError as e:
            logger.warning("Failed to write entity %s: %s", final_slug, e)

    logger.info(
        "consolidated %d entities from %d segment files → %d new journal entities written",
        total_entities,
        len(segment_files),
        written,
    )
    return written


def scan_journal(journal: str, verbose: bool = False, full: bool = False) -> bool:
    """Scan and index journal content.

    Args:
        journal: Path to journal root directory
        verbose: If True, log detailed progress
        full: If True, scan all files. If False (default), exclude historical
            YYYYMMDD directories (before today) for lighter incremental scans.

    Returns:
        True if any files were indexed or removed
    """
    conn, db_path = get_journal_index(journal)
    files = find_formattable_files(journal)

    # Light mode: exclude historical day directories
    if not full:
        files = {
            rel: path for rel, path in files.items() if not _is_historical_day(rel)
        }

    logger.info("Scanning %s files...", len(files))

    # Get current file mtimes from database
    db_mtimes = {
        path: mtime
        for path, mtime in conn.execute("SELECT path, mtime FROM files")
        if not (path.startswith("entity:") or path.startswith("signal:"))
    }

    to_index = []
    for rel, path in files.items():
        try:
            mtime = int(os.path.getmtime(path))
        except OSError:
            continue
        if db_mtimes.get(rel) != mtime:
            to_index.append((rel, path, mtime))

    cached = len(files) - len(to_index)
    logger.info(
        "%s total files, %s cached, %s to index", len(files), cached, len(to_index)
    )

    start = time.time()

    for i, (rel, path, mtime) in enumerate(to_index, 1):
        if verbose:
            logger.info("[%s/%s] %s", i, len(to_index), rel)

        # Delete existing chunks for this file
        conn.execute("DELETE FROM chunks WHERE path=?", (rel,))

        # Index the file
        stream = _extract_stream(journal, rel)
        _index_file(conn, rel, path, verbose, stream=stream)

        # Update file mtime
        conn.execute("REPLACE INTO files(path, mtime) VALUES (?, ?)", (rel, mtime))

    # Remove files that no longer exist
    # In full mode: remove all missing entries
    # In light mode: only remove entries that would have been scanned (non-historical)
    removed: set[str] = set()
    if full:
        removed = set(db_mtimes) - set(files)
    else:
        # Filter db entries to those in light mode's scan scope, then find missing
        in_scope_db = {rel for rel in db_mtimes if not _is_historical_day(rel)}
        removed = in_scope_db - set(files)

    for rel in removed:
        conn.execute("DELETE FROM chunks WHERE path=?", (rel,))
        conn.execute("DELETE FROM files WHERE path=?", (rel,))

    if to_index or removed:
        conn.commit()

    elapsed = time.time() - start
    logger.info(
        "%s indexed, %s removed in %.2f seconds", len(to_index), len(removed), elapsed
    )

    consolidate_segment_entities(journal, full=full)
    entity_changed = scan_entities(journal, conn, verbose=verbose, full=full)
    signal_changed = scan_signals(journal, conn, verbose=verbose, full=full)

    conn.close()
    return bool(to_index or removed or entity_changed or signal_changed)


def sanitize_fts_query(query: str) -> str:
    """Sanitize query for FTS5: keep alphanumeric, spaces, quotes, apostrophes, and *.

    This allows FTS5 operators (OR, AND, NOT), quoted phrases, and prefix
    matching while preventing syntax errors from special characters.
    """
    result = re.sub(r"[^a-zA-Z0-9\s\"'*]", " ", query)
    # Remove all quotes if unbalanced
    if result.count('"') % 2:
        result = result.replace('"', "")
    return result


def _build_where_clause(
    query: str,
    day: str | None = None,
    day_from: str | None = None,
    day_to: str | None = None,
    facet: str | None = None,
    agent: str | None = None,
    stream: str | None = None,
) -> tuple[str, list[Any]]:
    """Build WHERE clause and params for FTS5 search.

    Args:
        query: FTS5 search query
        day: Filter by exact day (YYYYMMDD) - mutually exclusive with day_from/day_to
        day_from: Filter by date range start (YYYYMMDD, inclusive)
        day_to: Filter by date range end (YYYYMMDD, inclusive)
        facet: Filter by facet name
        agent: Filter by agent
        stream: Filter by stream name

    Returns:
        Tuple of (where_clause, params)
    """
    params: list[Any] = []

    if query:
        sanitized = sanitize_fts_query(query)
        where_clause = f"chunks MATCH '{sanitized}'"
    else:
        where_clause = "1=1"

    if day:
        where_clause += " AND day=?"
        params.append(day)
    elif day_from or day_to:
        if day_from:
            where_clause += " AND day>=?"
            params.append(day_from)
        if day_to:
            where_clause += " AND day<=?"
            params.append(day_to)
    if facet:
        where_clause += " AND facet=?"
        params.append(facet.lower())
    if agent:
        where_clause += " AND agent=?"
        params.append(agent.lower())
    if stream:
        where_clause += " AND stream=?"
        params.append(stream)

    return where_clause, params


def search_journal(
    query: str,
    limit: int = 10,
    offset: int = 0,
    *,
    day: str | None = None,
    day_from: str | None = None,
    day_to: str | None = None,
    facet: str | None = None,
    agent: str | None = None,
    stream: str | None = None,
) -> tuple[int, list[dict[str, Any]]]:
    """Search the journal index.

    Args:
        query: FTS5 search query. Words are AND'd by default; use OR to match any,
            quotes for exact phrases, * for prefix match. Empty string returns all.
        limit: Maximum results to return
        offset: Number of results to skip for pagination
        day: Filter by exact day (YYYYMMDD) - mutually exclusive with day_from/day_to
        day_from: Filter by date range start (YYYYMMDD, inclusive)
        day_to: Filter by date range end (YYYYMMDD, inclusive)
        facet: Filter by facet name
        agent: Filter by agent (e.g., "flow", "event", "news")
        stream: Filter by stream name

    Returns:
        Tuple of (total_count, results) where each result has:
            - id: "{path}:{idx}"
            - text: The matched markdown chunk
            - metadata: {day, facet, agent, stream, path, idx}
            - score: BM25 relevance score
    """
    conn, _ = get_journal_index()
    where_clause, params = _build_where_clause(
        query, day, day_from, day_to, facet, agent, stream
    )

    # Get total count
    total = conn.execute(
        f"SELECT count(*) FROM chunks WHERE {where_clause}", params
    ).fetchone()[0]

    # Get results
    cursor = conn.execute(
        f"""
        SELECT content, path, day, facet, agent, stream, idx, bm25(chunks) as rank
        FROM chunks WHERE {where_clause}
        ORDER BY rank LIMIT ? OFFSET ?
        """,
        params + [limit, offset],
    )

    results = []
    for (
        content,
        path,
        day_val,
        facet_val,
        agent_val,
        stream_val,
        idx,
        rank,
    ) in cursor.fetchall():
        results.append(
            {
                "id": f"{path}:{idx}",
                "text": content,
                "metadata": {
                    "day": day_val,
                    "facet": facet_val,
                    "agent": agent_val,
                    "stream": stream_val,
                    "path": path,
                    "idx": idx,
                },
                "score": rank,
            }
        )

    conn.close()
    return total, results


def search_counts(
    query: str,
    *,
    day: str | None = None,
    day_from: str | None = None,
    day_to: str | None = None,
    facet: str | None = None,
    agent: str | None = None,
    stream: str | None = None,
) -> dict[str, Any]:
    """Get aggregated counts for a search query.

    Uses single query + Python aggregation for efficiency.

    Args:
        query: FTS5 search query (empty string for all)
        day: Filter by exact day (YYYYMMDD) - mutually exclusive with day_from/day_to
        day_from: Filter by date range start (YYYYMMDD, inclusive)
        day_to: Filter by date range end (YYYYMMDD, inclusive)
        facet: Filter by facet name
        agent: Filter by agent
        stream: Filter by stream name

    Returns:
        Dict with:
            - total: Total matching chunks
            - facets: Counter of facet_name -> count
            - agents: Counter of agent_name -> count
            - days: Counter of day -> count
            - streams: Counter of stream_name -> count
    """
    from collections import Counter

    conn, _ = get_journal_index()
    where_clause, params = _build_where_clause(
        query, day, day_from, day_to, facet, agent, stream
    )

    rows = conn.execute(
        f"SELECT facet, agent, day, stream FROM chunks WHERE {where_clause}", params
    ).fetchall()

    conn.close()

    return {
        "total": len(rows),
        "facets": Counter(r[0] for r in rows if r[0]),
        "agents": Counter(r[1] for r in rows if r[1]),
        "days": Counter(r[2] for r in rows if r[2]),
        "streams": Counter(r[3] for r in rows if r[3]),
    }


def get_events(
    day: str,
    facet: str | None = None,
) -> list[dict[str, Any]]:
    """Get structured events for a day, re-hydrated from source files.

    This function reads source JSONL files directly from both
    facets/*/events/{day}.jsonl and facets/*/calendar/{day}.jsonl to return
    full event objects with all fields (title, summary, start, end,
    participants, etc.). Cancelled calendar entries are excluded.

    Args:
        day: Day in YYYYMMDD format
        facet: Optional facet name to filter by

    Returns:
        List of event dicts with full structured data
    """
    events = []
    facets_dir = Path(get_journal()) / "facets"

    if not facets_dir.is_dir():
        return events

    for facet_dir in facets_dir.iterdir():
        if not facet_dir.is_dir():
            continue

        facet_name = facet_dir.name
        if facet and facet_name.lower() != facet.lower():
            continue

        events_file = facet_dir / "events" / f"{day}.jsonl"
        if events_file.is_file():
            entries = load_jsonl(str(events_file))
            for entry in entries:
                # Add facet to event if not present
                entry.setdefault("facet", facet_name)
                events.append(entry)

        # Also check calendar/ subdir for user-created events
        calendar_file = facet_dir / "calendar" / f"{day}.jsonl"
        if calendar_file.is_file():
            cal_entries = load_jsonl(str(calendar_file))
            for entry in cal_entries:
                if entry.get("cancelled"):
                    continue
                entry.setdefault("facet", facet_name)
                entry.setdefault("agent", "user")
                entry.setdefault("occurred", False)
                events.append(entry)

    return events


def _load_index_entity_dicts(
    conn: sqlite3.Connection,
    principal_only: bool = False,
) -> list[dict[str, Any]]:
    """Load identity entities from the journal index as entity dicts.

    Returns dicts with "id", "name", and "aka" suitable for
    build_name_resolution_map().
    """
    where = "source='identity'"
    if principal_only:
        where += " AND is_principal=1"
    rows = conn.execute(
        f"SELECT entity_id, name, aka FROM entities WHERE {where}"
    ).fetchall()
    entity_dicts: list[dict[str, Any]] = []
    for entity_id, name, aka_str in rows:
        d: dict[str, Any] = {"id": entity_id, "name": name, "aka": []}
        if aka_str:
            try:
                aka_list = json.loads(aka_str)
                if isinstance(aka_list, list):
                    d["aka"] = aka_list
            except (ValueError, TypeError):
                pass
        entity_dicts.append(d)
    return entity_dicts


def get_principal_entity_names(conn: sqlite3.Connection) -> set[str]:
    """Return signal entity_names that correspond to principal entities.

    Uses shared name resolution (build_name_resolution_map) to ensure
    consistent matching — the same name resolves the same way everywhere.
    """
    from think.entities.matching import build_name_resolution_map

    entity_dicts = _load_index_entity_dicts(conn, principal_only=True)
    if not entity_dicts:
        return set()

    signal_names = [
        r[0]
        for r in conn.execute(
            "SELECT DISTINCT entity_name FROM entity_signals"
        ).fetchall()
    ]

    name_map = build_name_resolution_map(signal_names, entity_dicts)
    return set(name_map.keys())


def _build_entity_name_map(conn: sqlite3.Connection) -> dict[str, str]:
    """Map signal entity_names to entity_ids via shared name resolution.

    Returns dict mapping entity_name -> entity_id. Uses the same tiered
    matching as all other name resolution call sites.
    """
    from think.entities.matching import build_name_resolution_map

    entity_dicts = _load_index_entity_dicts(conn)
    signal_names = [
        r[0]
        for r in conn.execute(
            "SELECT DISTINCT entity_name FROM entity_signals"
        ).fetchall()
    ]

    return build_name_resolution_map(signal_names, entity_dicts)


# Half-life of 30 days: decay reaches ~0.5 at 30 days, ~0.1 at ~100 days.
_RECENCY_DECAY_LAMBDA = math.log(2) / 30


def _recency_decay(days_since: int) -> float:
    """Exponential recency decay.

    Returns ~1.0 at day 0, ~0.5 at 30 days, ~0.1 at ~100 days.
    """
    return round(math.exp(-_RECENCY_DECAY_LAMBDA * max(0, days_since)), 4)


def _strength_score(
    kg_edge_count: int,
    co_occurrence: int,
    recency: float,
    observation_depth: int,
    facet_breadth: int,
) -> float:
    """Compute composite strength score from components.

    Weights: kg_edge=5, co_occurrence=4, recency=3, observation_depth=2, facet_breadth=1
    """
    return round(
        5 * kg_edge_count
        + 4 * co_occurrence
        + 3 * recency
        + 2 * observation_depth
        + 1 * facet_breadth,
        4,
    )


def _compute_strength_scores(
    conn: sqlite3.Connection,
    facet: str | None = None,
    since: str | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    """Compute strength score components for all entities in signals table."""
    where_parts: list[str] = []
    params: list[Any] = []
    if facet:
        where_parts.append("facet=?")
        params.append(facet.lower())
    if since:
        where_parts.append("day>=?")
        params.append(since)
    where = " AND ".join(where_parts) if where_parts else "1=1"

    rows = conn.execute(
        f"""
        SELECT entity_name, COUNT(*) as appearance, MAX(day) as last_day,
               COUNT(DISTINCT facet) as facet_breadth
        FROM entity_signals
        WHERE {where}
        GROUP BY entity_name
        """,
        params,
    ).fetchall()

    scores: dict[str, dict[str, Any]] = {}
    for entity_name, appearance, last_day, facet_breadth in rows:
        scores[entity_name] = {
            "appearance": appearance,
            "last_day": last_day or "",
            "facet_breadth": facet_breadth,
            "co_occurrence": 0,
            "kg_edge_count": 0,
        }

    # Count distinct KG relationship edges per entity (both source and target roles).
    kg_where_parts: list[str] = ["signal_type='kg_edge'"]
    kg_params: list[Any] = []
    if facet:
        kg_where_parts.append("facet=?")
        kg_params.append(facet.lower())
    if since:
        kg_where_parts.append("day>=?")
        kg_params.append(since)
    kg_where = " AND ".join(kg_where_parts)

    kg_rows = conn.execute(
        f"""
        SELECT entity, COUNT(*) as edge_count FROM (
            SELECT DISTINCT entity_name as entity, target_name, relationship_type
            FROM entity_signals WHERE {kg_where}
            UNION
            SELECT DISTINCT target_name as entity, entity_name, relationship_type
            FROM entity_signals WHERE {kg_where}
        ) GROUP BY entity
        """,
        kg_params + kg_params,
    ).fetchall()
    for entity_name, edge_count in kg_rows:
        if entity_name in scores:
            scores[entity_name]["kg_edge_count"] = edge_count

    # Exclude principal entities from co-occurrence calculations — they
    # co-occur with essentially everyone and inflate scores meaninglessly.
    principal_names = get_principal_entity_names(conn)

    co_where_parts: list[str] = []
    co_params: list[Any] = []
    if facet:
        co_where_parts.append("s1.facet=?")
        co_params.append(facet.lower())
    if since:
        co_where_parts.append("s1.day>=?")
        co_params.append(since)
    if principal_names:
        ph = ",".join("?" for _ in principal_names)
        co_where_parts.append(f"s1.entity_name NOT IN ({ph})")
        co_params.extend(principal_names)
        co_where_parts.append(f"s2.entity_name NOT IN ({ph})")
        co_params.extend(principal_names)
    co_where = " AND ".join(co_where_parts) if co_where_parts else "1=1"

    co_rows = conn.execute(
        f"""
        SELECT s1.entity_name, COUNT(DISTINCT s2.entity_name) as co_count
        FROM entity_signals s1
        JOIN entity_signals s2
          ON s1.path = s2.path
         AND s1.entity_name != s2.entity_name
        WHERE {co_where}
        GROUP BY s1.entity_name
        """,
        co_params,
    ).fetchall()
    for entity_name, co_count in co_rows:
        if entity_name in scores:
            scores[entity_name]["co_occurrence"] = co_count

    obs_rows = conn.execute(
        "SELECT entity_id, observation_count FROM entities WHERE source='observation'"
    ).fetchall()
    obs_map = {
        entity_id: int(observation_count or 0)
        for entity_id, observation_count in obs_rows
    }

    return scores, obs_map


def get_entity_strength(
    facet: str | None = None,
    since: str | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Rank entities by composite relationship strength score."""
    conn, _ = get_journal_index()
    try:
        name_map = _build_entity_name_map(conn)
        scores, obs_map = _compute_strength_scores(conn, facet, since)

        today = date.today().strftime("%Y%m%d")
        results: dict[str, dict[str, Any]] = {}

        for entity_name, components in scores.items():
            entity_id = name_map.get(entity_name)
            key = entity_id or entity_name

            if key not in results:
                results[key] = {
                    "entity_id": entity_id,
                    "entity_name": entity_name,
                    "kg_edge_count": 0,
                    "co_occurrence": 0,
                    "appearance": 0,
                    "recency": 0.0,
                    "facet_breadth": 0,
                    "observation_depth": 0,
                }

            r = results[key]
            r["appearance"] += components["appearance"]
            r["kg_edge_count"] = max(
                r["kg_edge_count"], components.get("kg_edge_count", 0)
            )
            r["co_occurrence"] = max(
                r["co_occurrence"], components.get("co_occurrence", 0)
            )
            r["facet_breadth"] = max(
                r["facet_breadth"], components.get("facet_breadth", 0)
            )
            last_day = components.get("last_day", "")
            if last_day and last_day > r.get("_last_day", ""):
                r["_last_day"] = last_day

        for r in results.values():
            last_day = r.pop("_last_day", "")
            if last_day:
                try:
                    last = date(
                        int(last_day[:4]), int(last_day[4:6]), int(last_day[6:8])
                    )
                    ref = date(int(today[:4]), int(today[4:6]), int(today[6:8]))
                    days_since = max(0, (ref - last).days)
                    r["recency"] = _recency_decay(days_since)
                except (ValueError, IndexError):
                    r["recency"] = 0.0
            if r["entity_id"] and r["entity_id"] in obs_map:
                r["observation_depth"] = obs_map[r["entity_id"]]

            r["score"] = _strength_score(
                r["kg_edge_count"],
                r["co_occurrence"],
                r["recency"],
                r["observation_depth"],
                r["facet_breadth"],
            )

        ranked = sorted(results.values(), key=lambda x: x["score"], reverse=True)
        return ranked[:limit]
    finally:
        conn.close()


def _extract_match_candidates(fts_results: list[dict[str, Any]]) -> set[str]:
    """Extract candidate entity names from FTS result text."""
    names: set[str] = set()
    for result in fts_results:
        text = result.get("text", "")
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("### "):
                name = stripped[4:].strip()
                if name.startswith("Project: "):
                    names.add(name[len("Project: ") :].strip())
                elif name.startswith("Person: "):
                    names.add(name[len("Person: ") :].strip())
                elif name:
                    names.add(name)
    return names


def search_entities(
    query: str | None = None,
    entity_type: str | None = None,
    facet: str | None = None,
    since: str | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Search entities by text query, type, facet, and/or signal activity."""
    conn, _ = get_journal_index()
    try:
        id_where_parts = ["source='identity'"]
        id_params: list[Any] = []
        if entity_type:
            id_where_parts.append("LOWER(type)=LOWER(?)")
            id_params.append(entity_type)

        id_rows = conn.execute(
            f"SELECT entity_id, name, type, description FROM entities WHERE {' AND '.join(id_where_parts)}",
            id_params,
        ).fetchall()

        entities: dict[str, dict[str, Any]] = {
            r[0]: {
                "entity_id": r[0],
                "name": r[1],
                "type": r[2],
                "description": r[3] or "",
            }
            for r in id_rows
        }

        if facet or since:
            sig_where = []
            sig_params: list[Any] = []
            if facet:
                sig_where.append("facet=?")
                sig_params.append(facet.lower())
            if since:
                sig_where.append("day>=?")
                sig_params.append(since)

            sig_rows = conn.execute(
                f"SELECT DISTINCT entity_name FROM entity_signals WHERE {' AND '.join(sig_where)}",
                sig_params,
            ).fetchall()
            name_map = _build_entity_name_map(conn)
            active_ids = set()
            for (sname,) in sig_rows:
                eid = name_map.get(sname)
                if eid:
                    active_ids.add(eid)

            if facet:
                rel_rows = conn.execute(
                    "SELECT DISTINCT entity_id FROM entities WHERE source='relationship' AND facet=?",
                    [facet.lower()],
                ).fetchall()
                for (eid,) in rel_rows:
                    active_ids.add(eid)

            entities = {eid: e for eid, e in entities.items() if eid in active_ids}

        if query:
            _, fts_results = search_journal(query, limit=100, agent="entity:detected")
            fts_ids = set()
            for r in fts_results:
                path = r.get("metadata", {}).get("path", "")
                parts = path.split("/")
                if "entities" in parts:
                    idx = parts.index("entities")
                    if idx + 1 < len(parts):
                        candidate = parts[idx + 1]
                        if candidate and "." not in candidate:
                            fts_ids.add(candidate)

            if not fts_ids:
                name_map = _build_entity_name_map(conn)
                match_names = _extract_match_candidates(fts_results)
                for sname, eid in name_map.items():
                    if sname in match_names:
                        fts_ids.add(eid)

            if not fts_ids:
                like_term = f"%{query.lower()}%"
                identity_rows = conn.execute(
                    """
                    SELECT DISTINCT entity_id
                    FROM entities
                    WHERE source='identity'
                      AND (
                        LOWER(name) LIKE ?
                        OR LOWER(type) LIKE ?
                        OR LOWER(description) LIKE ?
                      )
                    """,
                    [like_term, like_term, like_term],
                ).fetchall()
                for (eid,) in identity_rows:
                    fts_ids.add(eid)

            entities = {eid: e for eid, e in entities.items() if eid in fts_ids}

        name_map = _build_entity_name_map(conn)
        reverse_map: dict[str, list[str]] = {}
        for sname, eid in name_map.items():
            reverse_map.setdefault(eid, []).append(sname)

        result_list = []
        for eid, e in entities.items():
            signal_names = reverse_map.get(eid, [])
            if signal_names:
                placeholders = ",".join("?" for _ in signal_names)
                row = conn.execute(
                    f"SELECT COUNT(*), MAX(day), COUNT(DISTINCT facet) FROM entity_signals WHERE entity_name IN ({placeholders})",
                    signal_names,
                ).fetchone()
                e["signal_count"] = row[0]
                e["last_active"] = row[1] or ""
                facet_rows = conn.execute(
                    f"SELECT DISTINCT facet FROM entity_signals WHERE entity_name IN ({placeholders}) AND facet IS NOT NULL AND facet != ''",
                    signal_names,
                ).fetchall()
                e["facets"] = [r[0] for r in facet_rows if r[0]]
            else:
                e["signal_count"] = 0
                e["last_active"] = ""
                e["facets"] = []

            rel_facets = conn.execute(
                "SELECT DISTINCT facet FROM entities WHERE entity_id=? AND source='relationship' AND facet IS NOT NULL AND facet != ''",
                [eid],
            ).fetchall()
            for (rf,) in rel_facets:
                if rf and rf not in e["facets"]:
                    e["facets"].append(rf)

            result_list.append(e)

        result_list.sort(key=lambda x: (-x["signal_count"], x["name"]))
        return result_list[:limit]
    finally:
        conn.close()


def get_entity_intelligence(
    entity: str,
    facet: str | None = None,
) -> dict[str, Any] | None:
    """Get a full intelligence briefing for an entity."""
    conn, _ = get_journal_index()
    try:
        slug = entity_slug(entity)

        identity_row = conn.execute(
            """
            SELECT entity_id, name, type, description, tags, contact, aka, is_principal
            FROM entities WHERE source='identity' AND entity_id=?
            """,
            [slug],
        ).fetchone()

        if not identity_row:
            candidates = conn.execute(
                """
                SELECT entity_id, name, type, description, tags, contact, aka, is_principal
                FROM entities WHERE source='identity' AND entity_id LIKE ?
                ORDER BY entity_id
                """,
                [slug + "_%"],
            ).fetchall()
            if candidates:
                identity_row = candidates[0]

        if not identity_row:
            return None

        identity = {
            "entity_id": identity_row[0],
            "name": identity_row[1],
            "type": identity_row[2],
            "description": identity_row[3] or "",
            "tags": identity_row[4] or "",
            "contact": identity_row[5] or "",
            "aka": identity_row[6] or "",
            "is_principal": bool(identity_row[7]),
        }
        entity_id = identity["entity_id"]

        rel_where = "entity_id=? AND source='relationship'"
        rel_params: list[Any] = [entity_id]
        if facet:
            rel_where += " AND facet=?"
            rel_params.append(facet.lower())

        rel_rows = conn.execute(
            f"SELECT facet, description, attached_at, updated_at, detached FROM entities WHERE {rel_where}",
            rel_params,
        ).fetchall()
        relationships = [
            {
                "facet": r[0],
                "description": r[1] or "",
                "attached_at": r[2] or "",
                "updated_at": r[3] or "",
                "detached": bool(r[4]),
            }
            for r in rel_rows
        ]

        obs_rows = conn.execute(
            "SELECT facet, observation_count, last_observed FROM entities WHERE entity_id=? AND source='observation'",
            [entity_id],
        ).fetchall()
        observations = [
            {
                "facet": r[0] or "",
                "observation_count": int(r[1] or 0),
                "last_observed": r[2] or "",
            }
            for r in obs_rows
        ]

        name_map = _build_entity_name_map(conn)
        signal_names = [sname for sname, eid in name_map.items() if eid == entity_id]
        if identity["name"] not in signal_names:
            signal_names.append(identity["name"])

        activity: list[dict[str, Any]] = []
        network: dict[str, int] = {}
        signal_facets: list[str] = []

        if signal_names:
            placeholders = ",".join("?" for _ in signal_names)
            sig_rows = conn.execute(
                f"SELECT signal_type, entity_name, target_name, relationship_type, day, facet, event_title, event_type, path FROM entity_signals WHERE entity_name IN ({placeholders}) ORDER BY day DESC",
                signal_names,
            ).fetchall()

            for row in sig_rows:
                activity.append(
                    {
                        "signal_type": row[0],
                        "entity_name": row[1],
                        "target_name": row[2] or "",
                        "relationship_type": row[3] or "",
                        "day": row[4] or "",
                        "facet": row[5] or "",
                        "event_title": row[6] or "",
                        "event_type": row[7] or "",
                        "path": row[8],
                    }
                )

            # Exclude principal entities from co-occurrence — they co-occur
            # with everyone and would dominate the network section.
            principal_names = get_principal_entity_names(conn)
            co_extra_where = ""
            co_params: list[Any] = signal_names + signal_names
            if principal_names:
                ph = ",".join("?" for _ in principal_names)
                co_extra_where = f" AND s2.entity_name NOT IN ({ph})"
                co_params.extend(list(principal_names))

            co_rows = conn.execute(
                f"""
                SELECT s2.entity_name, COUNT(DISTINCT s2.path) as co_count
                FROM entity_signals s1
                JOIN entity_signals s2 ON s1.path = s2.path
                WHERE s1.entity_name IN ({placeholders}) AND s2.entity_name NOT IN ({placeholders}){co_extra_where}
                GROUP BY s2.entity_name
                ORDER BY COUNT(DISTINCT s2.path) DESC
                """,
                co_params,
            ).fetchall()
            network = {r[0]: r[1] for r in co_rows}

            facet_rows = conn.execute(
                f"SELECT DISTINCT facet FROM entity_signals WHERE entity_name IN ({placeholders}) AND facet IS NOT NULL AND facet != ''",
                signal_names,
            ).fetchall()
            signal_facets = [r[0] for r in facet_rows if r[0]]

        all_facets = list(
            set(signal_facets + [r["facet"] for r in relationships if r["facet"]])
        )

        # Compute strength components for this entity only (single-entity fast
        # path — avoids the full co-occurrence self-join across all entities).
        total_appearance = 0
        max_kg_edges = 0
        max_co = 0
        max_breadth = 0
        best_last_day = ""

        if signal_names:
            placeholders_s = ",".join("?" for _ in signal_names)
            facet_filter = " AND facet=?" if facet else ""
            facet_params = [facet.lower()] if facet else []

            # Basic signal stats for this entity's signal names
            stat_row = conn.execute(
                f"""
                SELECT COUNT(*) as appearance, MAX(day) as last_day,
                       COUNT(DISTINCT facet) as facet_breadth
                FROM entity_signals
                WHERE entity_name IN ({placeholders_s}){facet_filter}
                """,
                signal_names + facet_params,
            ).fetchone()
            if stat_row:
                total_appearance = stat_row[0]
                best_last_day = stat_row[1] or ""
                max_breadth = stat_row[2]

            # KG edge count for this entity
            kg_facet_filter = " AND facet=?" if facet else ""
            kg_facet_params = [facet.lower()] if facet else []
            kg_row = conn.execute(
                f"""
                SELECT COUNT(*) FROM (
                    SELECT DISTINCT entity_name as entity, target_name, relationship_type
                    FROM entity_signals WHERE signal_type='kg_edge' AND entity_name IN ({placeholders_s}){kg_facet_filter}
                    UNION
                    SELECT DISTINCT target_name as entity, entity_name, relationship_type
                    FROM entity_signals WHERE signal_type='kg_edge' AND target_name IN ({placeholders_s}){kg_facet_filter}
                )
                """,
                signal_names + kg_facet_params + signal_names + kg_facet_params,
            ).fetchone()
            if kg_row:
                max_kg_edges = kg_row[0]

            # Co-occurrence count for this entity (excluding principals)
            max_co = len(network)

        obs_row = conn.execute(
            "SELECT observation_count FROM entities WHERE entity_id=? AND source='observation'",
            [entity_id],
        ).fetchone()
        obs_depth = int(obs_row[0] or 0) if obs_row else 0

        recency = 0.0
        if best_last_day:
            try:
                last = date(
                    int(best_last_day[:4]),
                    int(best_last_day[4:6]),
                    int(best_last_day[6:8]),
                )
                ref = date.today()
                days_since = max(0, (ref - last).days)
                recency = _recency_decay(days_since)
            except (ValueError, IndexError):
                pass

        strength = {
            "score": _strength_score(
                max_kg_edges, max_co, recency, obs_depth, max_breadth
            ),
            "kg_edge_count": max_kg_edges,
            "co_occurrence": max_co,
            "appearance": total_appearance,
            "recency": recency,
            "facet_breadth": max_breadth,
            "observation_depth": obs_depth,
        }

        return {
            "identity": identity,
            "relationships": relationships,
            "observations": observations,
            "activity": activity,
            "strength": strength,
            "network": network,
            "facets": all_facets,
        }
    finally:
        conn.close()
