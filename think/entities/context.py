# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Context assembly for entity_observer generate agent.

Pre-computes the observation context that the cogitate version built
through sequential tool calls. Used by the entity_observer pre-hook
to inject $observer_context into the prompt.
"""

from __future__ import annotations

from pathlib import Path

from think.entities.activity import parse_knowledge_graph_entities
from think.entities.loading import load_entities
from think.entities.matching import find_matching_entity
from think.entities.observations import load_observations
from think.utils import get_journal


def _active_entity_ids(facet: str, day: str, attached: list[dict]) -> set[str]:
    active_ids: set[str] = set()

    for name in parse_knowledge_graph_entities(day):
        match = find_matching_entity(name, attached)
        if match:
            entity_id = match.get("id")
            if entity_id:
                active_ids.add(entity_id)

    for detected in load_entities(facet, day):
        match = find_matching_entity(detected.get("name", ""), attached)
        if match:
            entity_id = match.get("id")
            if entity_id:
                active_ids.add(entity_id)

    return active_ids


def _load_knowledge_graph(day: str) -> str:
    kg_path = Path(get_journal()) / day / "agents" / "knowledge_graph.md"
    if not kg_path.exists():
        return "No knowledge graph available for this day."

    try:
        content = kg_path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        return "No knowledge graph available for this day."

    return content or "No knowledge graph available for this day."


def _format_observations(facet: str, entity_id: str) -> list[str]:
    observations = load_observations(facet, entity_id)[-3:]
    if not observations:
        return ["No prior observations."]

    lines = []
    for observation in observations:
        content = str(observation.get("content", "")).strip()
        if not content:
            continue
        source_day = observation.get("source_day") or "unknown"
        lines.append(f"- {content} (source: {source_day})")

    return lines or ["No prior observations."]


def _format_entity_section(facet: str, entity: dict) -> str:
    entity_id = entity.get("id", "")
    entity_name = entity.get("name", entity_id)
    description = entity.get("description", "") or ""

    lines = [
        f"#### {entity_name} ({entity_id})",
        f"- Type: {entity.get('type', '')}",
        f"- Description: {description}",
    ]

    aka_list = entity.get("aka")
    if isinstance(aka_list, list) and aka_list:
        lines.append(f"- AKA: {', '.join(str(item) for item in aka_list if item)}")

    lines.append("")
    lines.append("Recent observations:")
    lines.extend(_format_observations(facet, entity_id))

    return "\n".join(lines)


def assemble_observer_context(facet: str, day: str) -> str:
    """Assemble structured observation context for a facet/day run."""
    attached = load_entities(facet)
    active_ids = _active_entity_ids(facet, day, attached)

    if not active_ids:
        return "No active entities found for this day."

    active_entities = [entity for entity in attached if entity.get("id") in active_ids]
    kg_content = _load_knowledge_graph(day)

    lines = [
        "# Entity Observer Context",
        "",
        f"## Facet: {facet}",
        f"## Day: {day}",
        f"## Active Entities: {len(active_entities)} of {len(attached)} attached",
        "",
        "### Entities",
        "",
    ]

    for index, entity in enumerate(active_entities):
        if index:
            lines.extend(["", "---", ""])
        lines.append(_format_entity_section(facet, entity))

    lines.extend(
        [
            "",
            "### Knowledge Graph",
            "",
            kg_content,
        ]
    )

    return "\n".join(lines)
