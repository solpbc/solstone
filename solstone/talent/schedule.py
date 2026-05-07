# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Hook for writing schedule-derived planned items as activity records."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from typing import Any

from solstone.think.activities import (
    append_activity_record,
    append_edit,
    dedup_anticipation,
    make_anticipation_id,
    mute_activity_record,
)
from solstone.think.entities.loading import load_entities
from solstone.think.entities.matching import find_matching_entity
from solstone.think.facets import get_facets

logger = logging.getLogger(__name__)

_TIME_RE = re.compile(r"^\d{2}:\d{2}:\d{2}$")


def _require_text(item: dict[str, Any], key: str) -> str:
    value = item.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"missing required field '{key}'")
    return value.strip()


def _optional_time(item: dict[str, Any], key: str) -> str | None:
    value = item.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not _TIME_RE.fullmatch(value):
        raise ValueError(f"invalid {key!r}: expected HH:MM:SS or null")
    return value


def post_process(result: str, context: dict) -> None:
    """Persist schedule-derived planned items as activity records."""
    try:
        events = json.loads(result.strip())
    except json.JSONDecodeError as exc:
        snippet = result.strip()[:200]
        logger.error("schedule hook: failed to parse JSON: %s snippet=%r", exc, snippet)
        return None

    if not isinstance(events, list):
        logger.error("schedule hook: expected top-level array")
        return None

    day = str(context.get("day") or "")
    try:
        current_day = datetime.strptime(day, "%Y%m%d").date()
    except ValueError:
        logger.error("schedule hook: invalid context day %r", day)
        return None

    known_facets = set(get_facets().keys())
    entity_cache: dict[tuple[str, str], list[dict[str, Any]]] = {}

    for raw_event in events:
        try:
            if not isinstance(raw_event, dict):
                raise ValueError("expected object")

            activity = _require_text(raw_event, "activity")
            target_date = _require_text(raw_event, "target_date")
            title = _require_text(raw_event, "title")
            description = _require_text(raw_event, "description")
            facet = _require_text(raw_event, "facet")
            if facet not in known_facets:
                raise ValueError(f"unknown facet {facet!r}")

            target_day = datetime.strptime(target_date, "%Y-%m-%d").date()
            if target_day <= current_day:
                raise ValueError(
                    f"target_date must be after context day ({target_date} <= {day})"
                )

            start = _optional_time(raw_event, "start")
            end = _optional_time(raw_event, "end")
            cancelled = bool(raw_event.get("cancelled", False))
            details = str(raw_event.get("details") or "")
            participation_confidence = raw_event.get("participation_confidence")
            participation = raw_event.get("participation", [])
            if not isinstance(participation, list):
                raise ValueError("participation must be a list")

            cache_key = (facet, target_day.strftime("%Y%m%d"))
            entities_list = entity_cache.get(cache_key)
            if entities_list is None:
                entities_list = load_entities(facet=facet, day=cache_key[1])
                entity_cache[cache_key] = entities_list

            resolved_participation: list[dict[str, Any]] = []
            active_entities: list[str] = []
            seen_active_entities: set[str] = set()
            for entry in participation:
                if not isinstance(entry, dict):
                    continue

                resolved_entry = dict(entry)
                match = find_matching_entity(
                    resolved_entry.get("name", ""), entities_list
                )
                entity_id = match.get("id") if match else None
                resolved_entry["entity_id"] = entity_id
                resolved_participation.append(resolved_entry)

                if resolved_entry.get("role") != "attendee" or not entity_id:
                    continue
                if entity_id in seen_active_entities:
                    continue
                seen_active_entities.add(entity_id)
                active_entities.append(entity_id)

            new_id = make_anticipation_id(activity, start, target_date)
            record = {
                "id": new_id,
                "activity": activity,
                "target_date": target_date,
                "start": start,
                "end": end,
                "title": title,
                "description": description,
                "details": details,
                "facet": facet,
                "source": "anticipated",
                "active_entities": active_entities,
                "participation": resolved_participation,
                "participation_confidence": participation_confidence,
                "cancelled": cancelled,
                "hidden": cancelled,
            }
            record = append_edit(
                record,
                actor="schedule",
                fields=[
                    "activity",
                    "target_date",
                    "start",
                    "end",
                    "title",
                    "description",
                    "details",
                    "source",
                    "active_entities",
                    "participation",
                    "participation_confidence",
                    "cancelled",
                    "hidden",
                ],
                note=(
                    "created by schedule (cancelled on calendar)"
                    if cancelled
                    else "created by schedule"
                ),
            )

            should_write, superseded_ids = dedup_anticipation(
                facet,
                target_day.strftime("%Y%m%d"),
                record,
            )
            if not should_write:
                logger.info(
                    "schedule hook: duplicate anticipated activity id=%s", new_id
                )
                continue

            written = append_activity_record(
                facet,
                target_day.strftime("%Y%m%d"),
                record,
            )
            if not written:
                logger.info("schedule hook: append lost race for id=%s", new_id)
                continue

            for superseded_id in superseded_ids:
                mute_activity_record(
                    facet,
                    target_day.strftime("%Y%m%d"),
                    superseded_id,
                    actor="schedule",
                    reason=f"superseded by {new_id}",
                )
        except Exception as exc:
            logger.warning(
                "schedule hook: skipping invalid item %r: %s",
                raw_event,
                exc,
            )

    return None
