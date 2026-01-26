# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Entity activity tracking via Callosum event handlers.

Updates last_seen on attached entities when they appear in daily outputs.
Triggered after dream processing completes for each day.
"""

import logging

from apps.events import EventContext, on_event
from think.entities import parse_knowledge_graph_entities, touch_entities_from_activity
from think.facets import get_facets

logger = logging.getLogger(__name__)


@on_event("dream", "generators_completed")
def update_entity_activity(ctx: EventContext) -> None:
    """Update last_seen for entities mentioned in today's knowledge graph.

    Triggered after generator processing completes. Parses the knowledge graph
    for entity names and updates last_seen on matching attached entities
    across all facets.
    """
    # Only process daily mode (knowledge graph is a daily insight)
    if ctx.msg.get("mode") != "daily":
        return

    day = ctx.msg.get("day")
    if not day:
        logger.warning("generators_completed event missing day field")
        return

    # Parse entity names from knowledge graph
    kg_names = parse_knowledge_graph_entities(day)
    if not kg_names:
        logger.debug(f"No entities found in knowledge graph for {day}")
        return

    logger.info(f"Found {len(kg_names)} entities in knowledge graph for {day}")

    # Update each facet's attached entities
    facets = get_facets()
    total_updated = 0
    total_matched = 0

    for facet_name in facets:
        result = touch_entities_from_activity(facet_name, kg_names, day)
        matched_count = len(result["matched"])
        updated_count = len(result["updated"])

        if matched_count > 0:
            logger.info(
                f"Facet '{facet_name}': matched {matched_count}, "
                f"updated {updated_count} entities for {day}"
            )
            total_matched += matched_count
            total_updated += updated_count

    if total_matched > 0:
        logger.info(
            f"Entity activity update complete for {day}: "
            f"{total_matched} matches, {total_updated} updates across {len(facets)} facets"
        )
