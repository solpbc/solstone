# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Hook for extracting entities from insight results and writing to JSONL.

This hook is invoked via "hook": "entities" in insight frontmatter.
It parses the markdown entity list and writes deduplicated entities
to a JSONL file in the segment directory.
"""

import json
import logging
import re
from pathlib import Path

# Pattern to match: * Type: Name - Description
# Requires whitespace around " - " to allow dashes in names (e.g., DO-260C)
ENTITY_PATTERN = re.compile(r"^\*\s*([^:]+):\s*(.+?)\s+-\s+(.+)$")

# Alternate pattern: * type: Tool - name: make - description: ...
LABELED_PATTERN = re.compile(
    r"^\*\s*type:\s*(.+?)\s+-\s*name:\s*(.+?)\s+-\s*description:\s*(.+)$",
    re.IGNORECASE,
)


def _strip_bold(text: str) -> str:
    """Remove markdown bold formatting from text."""
    return text.replace("**", "").strip()


def _parse_entity_line(line: str) -> dict | None:
    """Parse a markdown entity line into structured data.

    Expected formats:
    - * Type: Name - Description sentence
    - * type: Tool - name: make - description: A build tool.

    Returns:
        Dict with type, name, description keys or None if line doesn't match.
    """
    line = line.strip()
    if not line:
        return None

    # Try standard format first: * Type: Name - Description
    match = ENTITY_PATTERN.match(line)
    if match:
        return {
            "type": _strip_bold(match.group(1)),
            "name": _strip_bold(match.group(2)),
            "description": match.group(3).strip(),
        }

    # Try labeled format: * type: X - name: Y - description: Z
    match = LABELED_PATTERN.match(line)
    if match:
        return {
            "type": _strip_bold(match.group(1)),
            "name": _strip_bold(match.group(2)),
            "description": match.group(3).strip(),
        }

    return None


def process(result: str, context: dict) -> str | None:
    """Parse entity list and write to segment JSONL file.

    Args:
        result: The generated output content (markdown entity list).
        context: Hook context with keys:
            - day: YYYYMMDD string
            - segment: segment key (HHMMSS_LEN)
            - name: e.g., "entities"
            - output_path: absolute path to output file
            - meta: dict with frontmatter
            - transcript: the clustered transcript markdown

    Returns:
        None - this hook does not modify the output result.
    """
    segment = context.get("segment")
    if not segment:
        logging.warning("entities hook requires segment mode")
        return None

    # Parse entities from result
    entities = []
    unparsed = []
    for line in result.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        entity = _parse_entity_line(line)
        if entity:
            entities.append(entity)
        elif line.startswith("*"):
            unparsed.append(line)

    if unparsed:
        print(f"Warning: {len(unparsed)} unparsed entity lines:")
        for line in unparsed:
            print(f"  {line}")

    if not entities:
        logging.info("entities hook: no entities extracted")
        return None

    # Deduplicate by (type, name) - keep first occurrence
    seen = set()
    unique_entities = []
    for entity in entities:
        key = (entity["type"].lower(), entity["name"].lower())
        if key not in seen:
            seen.add(key)
            unique_entities.append(entity)

    if len(unique_entities) < len(entities):
        logging.info(
            "entities hook: deduplicated %d -> %d entities",
            len(entities),
            len(unique_entities),
        )

    # Determine output path: segment_dir/entities.jsonl
    output_path = Path(context.get("output_path", ""))
    segment_dir = output_path.parent
    jsonl_path = segment_dir / "entities.jsonl"

    # Write JSONL file
    try:
        with open(jsonl_path, "w") as f:
            for entity in unique_entities:
                f.write(json.dumps(entity) + "\n")
        print(f"Entities written to: {jsonl_path} ({len(unique_entities)} entities)")
    except Exception as e:
        logging.error("entities hook: failed to write JSONL: %s", e)

    return None  # Don't modify insight result
