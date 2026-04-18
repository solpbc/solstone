# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Hook for merging participation data onto activity records."""

import json
import logging

from think.activities import update_record_fields
from think.entities.loading import load_entities
from think.entities.matching import find_matching_entity

logger = logging.getLogger(__name__)


def post_process(result: str, context: dict) -> str | None:
    """Resolve participation entries and merge them onto an activity record."""
    try:
        data = json.loads(result.strip())
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("participation hook: failed to parse JSON: %s", exc)
        return None

    if not isinstance(data, dict):
        logger.warning("participation hook: expected top-level object")
        return None

    activity = context.get("activity")
    if not isinstance(activity, dict):
        logger.warning("participation hook: missing activity context")
        return None

    record_id = activity.get("id")
    if not record_id:
        logger.warning("participation hook: missing activity record id")
        return None

    facet = context.get("facet")
    day = context.get("day")
    if not facet or not day:
        logger.warning("participation hook: missing facet/day context")
        return None

    participation = data.get("participation")
    if not isinstance(participation, list):
        logger.warning("participation hook: missing participation list")
        return None

    entities_list = load_entities(facet=facet, day=day)

    resolved_entries = []
    for entry in participation:
        if not isinstance(entry, dict):
            logger.warning("participation hook: skipping non-object entry")
            continue

        resolved_entry = dict(entry)
        match = find_matching_entity(resolved_entry.get("name", ""), entities_list)
        resolved_entry["entity_id"] = match.get("id") if match else None
        resolved_entries.append(resolved_entry)

    payload = {"participation": resolved_entries}
    participation_confidence = data.get("participation_confidence")
    if participation_confidence is not None:
        payload["participation_confidence"] = participation_confidence

    if not update_record_fields(facet, day, record_id, payload):
        logger.warning("participation hook: activity record not found: %s", record_id)

    return None
