# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Entity observer talent hook — pre-computes context and persists observations.

pre_process:  Assembles entity context (attached entities, recent observations,
              KG excerpts) and injects it as $observer_context template variable.

post_process: Parses JSON output, validates entity_ids against attached entities,
              and persists valid observations via add_observation().
"""

from __future__ import annotations

import json
import logging

from solstone.think.entities.context import assemble_observer_context
from solstone.think.entities.loading import load_entities
from solstone.think.entities.observations import add_observation, load_observations

logger = logging.getLogger(__name__)


def pre_process(context: dict) -> dict | None:
    facet = context.get("facet")
    day = context.get("day")
    if not facet or not day:
        return None

    observer_context = assemble_observer_context(facet, day)
    return {"template_vars": {"observer_context": observer_context}}


def post_process(result: str, context: dict) -> str | None:
    facet = context.get("facet")
    day = context.get("day")
    if not facet or not day:
        return None

    try:
        data = json.loads(result)
    except json.JSONDecodeError:
        logger.warning("entity_observer: could not parse result as JSON")
        return None

    if not isinstance(data, dict):
        return None

    observations = data.get("observations")
    if not isinstance(observations, list):
        logger.warning("entity_observer: observations is not a list")
        return None
    if not observations:
        return None

    valid_entity_ids = {
        entity.get("id") for entity in load_entities(facet) if entity.get("id")
    }

    for entry in observations:
        if not isinstance(entry, dict):
            logger.debug("Skipping non-dict observation entry: %r", entry)
            continue
        entity_id = entry.get("entity_id")
        items = entry.get("items")
        if not isinstance(entity_id, str) or not isinstance(items, list):
            logger.debug("Skipping malformed observation entry: %r", entry)
            continue
        if entity_id not in valid_entity_ids:
            logger.debug("Skipping unrecognized entity_id: %s", entity_id)
            continue

        existing = {
            obs.get("content", "").strip().lower()
            for obs in load_observations(facet, entity_id)
        }

        for item in items:
            if not isinstance(item, dict):
                continue
            content = str(item.get("content", "")).strip()
            if not content:
                continue
            if content.lower() in existing:
                logger.debug(
                    "Skipping duplicate observation for %s: %s", entity_id, content[:60]
                )
                continue
            add_observation(facet, entity_id, content, day)
            existing.add(content.lower())

    return None
