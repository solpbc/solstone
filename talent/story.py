# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Hook for merging storyteller outputs onto activity records."""

from __future__ import annotations

import json
import logging
import math
from typing import Any

from think.activities import merge_story_fields
from think.entities.loading import load_entities
from think.entities.matching import find_matching_entity

logger = logging.getLogger(__name__)

ALLOWED_RESOLUTIONS = frozenset({"sent", "done", "signed", "dropped", "deferred"})


def _normalize_topics(value: Any) -> list[str] | None:
    if not isinstance(value, list):
        logger.warning("story hook: missing topics list")
        return None

    topics: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str):
            logger.warning("story hook: invalid topics list")
            return None
        topic = item.strip().lower()
        if not topic or topic in seen:
            continue
        seen.add(topic)
        topics.append(topic)
        if len(topics) >= 10:
            break

    if not topics:
        logger.warning("story hook: empty topics after normalization")
        return None

    return topics


def _normalize_confidence(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        logger.warning("story hook: invalid confidence value")
        return None

    confidence = float(value)
    if math.isnan(confidence):
        logger.warning("story hook: invalid confidence value")
        return None

    clamped = min(1.0, max(0.0, confidence))
    if clamped != confidence:
        logger.warning("story hook: clamped confidence %s to %s", confidence, clamped)
    return clamped


def _resolve_entity_id(name: str, entities: list[dict[str, Any]]) -> str | None:
    match = find_matching_entity(name, entities, fuzzy_threshold=90)
    return match.get("id") if match else None


def _validate_fields(
    entry: dict[str, Any], required_fields: tuple[str, ...]
) -> dict[str, str] | None:
    normalized: dict[str, str] = {}
    for field in required_fields:
        value = entry.get(field)
        if not isinstance(value, str):
            return None
        normalized[field] = value
    return normalized


def post_process(result: str, context: dict) -> str:
    """Validate storyteller JSON and merge it onto an activity record."""
    try:
        data = json.loads(result.strip())
    except (json.JSONDecodeError, ValueError) as exc:
        logger.error("story hook: failed to parse JSON: %s", exc)
        return ""

    if not isinstance(data, dict):
        logger.warning("story hook: expected top-level object")
        return ""

    body = data.get("body")
    topics = data.get("topics")
    confidence = data.get("confidence")
    commitments = data.get("commitments")
    closures = data.get("closures")
    decisions = data.get("decisions")

    if not isinstance(body, str) or not body.strip():
        logger.warning("story hook: missing body")
        return ""
    topics = _normalize_topics(topics)
    if topics is None:
        return ""
    confidence = _normalize_confidence(confidence)
    if confidence is None:
        return ""
    if not isinstance(commitments, list):
        logger.warning("story hook: missing commitments list")
        return ""
    if not isinstance(closures, list):
        logger.warning("story hook: missing closures list")
        return ""
    if not isinstance(decisions, list):
        logger.warning("story hook: missing decisions list")
        return ""

    activity = context.get("activity")
    if not isinstance(activity, dict):
        logger.warning("story hook: missing activity context")
        return ""

    record_id = activity.get("id")
    if not isinstance(record_id, str) or not record_id:
        logger.warning("story hook: missing activity record id")
        return ""

    facet = context.get("facet")
    day = context.get("day")
    if not isinstance(facet, str) or not facet or not isinstance(day, str) or not day:
        logger.warning("story hook: missing facet/day context")
        return ""

    entities = load_entities(facet=facet, day=day)

    resolved_commitments: list[dict[str, Any]] = []
    for index, entry in enumerate(commitments):
        if not isinstance(entry, dict):
            logger.warning(
                "story hook: skipping commitment[%d]: expected object", index
            )
            continue
        normalized = _validate_fields(
            entry, ("owner", "action", "counterparty", "when", "context")
        )
        if normalized is None:
            logger.warning(
                "story hook: skipping commitment[%d]: missing required string field",
                index,
            )
            continue
        resolved_commitment = dict(normalized)
        resolved_commitment["owner_entity_id"] = _resolve_entity_id(
            normalized["owner"], entities
        )
        resolved_commitment["counterparty_entity_id"] = _resolve_entity_id(
            normalized["counterparty"], entities
        )
        resolved_commitments.append(resolved_commitment)

    resolved_closures: list[dict[str, Any]] = []
    for index, entry in enumerate(closures):
        if not isinstance(entry, dict):
            logger.warning("story hook: skipping closure[%d]: expected object", index)
            continue
        normalized = _validate_fields(
            entry, ("owner", "action", "counterparty", "resolution", "context")
        )
        if normalized is None:
            logger.warning(
                "story hook: skipping closure[%d]: missing required string field",
                index,
            )
            continue
        if normalized["resolution"] not in ALLOWED_RESOLUTIONS:
            logger.warning(
                "story hook: skipping closure[%d]: invalid resolution '%s'",
                index,
                normalized["resolution"],
            )
            continue
        resolved_closure = dict(normalized)
        resolved_closure["owner_entity_id"] = _resolve_entity_id(
            normalized["owner"], entities
        )
        resolved_closure["counterparty_entity_id"] = _resolve_entity_id(
            normalized["counterparty"], entities
        )
        resolved_closures.append(resolved_closure)

    resolved_decisions: list[dict[str, Any]] = []
    for index, entry in enumerate(decisions):
        if not isinstance(entry, dict):
            logger.warning("story hook: skipping decision[%d]: expected object", index)
            continue
        normalized = _validate_fields(entry, ("owner", "action", "context"))
        if normalized is None:
            logger.warning(
                "story hook: skipping decision[%d]: missing required string field",
                index,
            )
            continue
        resolved_decision = dict(normalized)
        resolved_decision["owner_entity_id"] = _resolve_entity_id(
            normalized["owner"], entities
        )
        resolved_decisions.append(resolved_decision)

    talent_name = context.get("name") or ""
    if not talent_name:
        logger.warning("story hook: missing talent name in context")

    story = {
        "talent": talent_name,
        "body": body.strip(),
        "topics": topics,
        "confidence": confidence,
    }

    merge_story_fields(
        facet,
        day,
        record_id,
        story=story,
        commitments=resolved_commitments,
        closures=resolved_closures,
        decisions=resolved_decisions,
        actor="story",
        note=None,
    )

    return ""
