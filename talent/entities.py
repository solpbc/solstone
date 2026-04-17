# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Hook for extracting entities from insight results and writing to JSONL.

This hook is invoked via "hook": {"post": "entities"} in generator frontmatter.
It parses the markdown entity list and writes deduplicated entities
to a JSONL file next to the agent output.
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


def post_process(result: str, context: dict) -> str | None:
    """Parse entity list and write to an adjacent JSONL file.

    Args:
        result: The generated output content (markdown entity list).
        context: HookContext with keys including day, segment, name,
            output_path, meta, transcript.

    Returns:
        None - this hook does not modify the output result.
    """
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
        logging.warning("entities hook: %d unparsed entity lines", len(unparsed))
        for line in unparsed:
            logging.warning("entities hook: unparsed line: %s", line)

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

    # Write entities.jsonl alongside the agent output in the talents/ directory
    output_path_value = context.get("output_path")
    if not output_path_value:
        logging.error("entities hook: missing output_path in context")
        return None

    output_path = Path(output_path_value)
    agents_dir = output_path.parent
    jsonl_path = agents_dir / "entities.jsonl"

    # Write JSONL file
    try:
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for entity in unique_entities:
                f.write(json.dumps(entity) + "\n")
        logging.info(
            "entities hook: wrote %d entities to %s",
            len(unique_entities),
            jsonl_path,
        )
    except Exception as e:
        logging.error("entities hook: failed to write JSONL: %s", e)

    return None  # Don't modify insight result
