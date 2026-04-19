# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Post-hook for structured storytelling span rows."""

from __future__ import annotations

import json
import logging
import math
import re
from pathlib import Path
from typing import Any

from think.activities import locked_modify
from think.utils import get_journal, segment_parse

logger = logging.getLogger(__name__)


def _strip_code_fences(result: str) -> str:
    stripped = result.strip()
    stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
    return re.sub(r"\s*```$", "", stripped)


def _normalize_topics(value: Any) -> list[str] | None:
    if not isinstance(value, list):
        logger.warning("spans hook: missing topics list")
        return None

    topics: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str):
            logger.warning("spans hook: invalid topics list")
            return None
        topic = item.strip().lower()
        if not topic or topic in seen:
            continue
        seen.add(topic)
        topics.append(topic)
        if len(topics) >= 10:
            break

    if not topics:
        logger.warning("spans hook: empty topics after normalization")
        return None

    return topics


def _normalize_confidence(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        logger.warning("spans hook: invalid confidence value")
        return None

    confidence = float(value)
    if math.isnan(confidence):
        logger.warning("spans hook: invalid confidence value")
        return None

    clamped = min(1.0, max(0.0, confidence))
    if clamped != confidence:
        logger.warning(
            "spans hook: clamped confidence %.3f to %.3f", confidence, clamped
        )
    return clamped


def _activity_time_bounds(segments: Any) -> tuple[str, str] | None:
    if not isinstance(segments, list) or not segments:
        logger.warning("spans hook: missing activity segments")
        return None

    start_time, _ = segment_parse(str(segments[0]))
    _, end_time = segment_parse(str(segments[-1]))
    if start_time is None or end_time is None:
        logger.warning("spans hook: invalid activity segments")
        return None

    return start_time.strftime("%H:%M:%S"), end_time.strftime("%H:%M:%S")


def _spans_path(facet: str, day: str) -> Path:
    return Path(get_journal()) / "facets" / facet / "spans" / f"{day}.jsonl"


def post_process(result: str, context: dict) -> str:
    """Parse model JSON and persist a single storytelling span row."""
    try:
        try:
            data = json.loads(_strip_code_fences(result))
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("spans hook: failed to parse JSON: %s", exc)
            return ""

        if not isinstance(data, dict):
            logger.warning("spans hook: expected top-level object")
            return ""

        body = data.get("body")
        if not isinstance(body, str) or not body.strip():
            logger.warning("spans hook: missing body")
            return ""
        normalized_body = body.strip()

        topics = _normalize_topics(data.get("topics"))
        if topics is None:
            return ""

        confidence = _normalize_confidence(data.get("confidence"))
        if confidence is None:
            return ""

        activity = context.get("activity")
        if not isinstance(activity, dict):
            logger.warning("spans hook: missing activity context")
            return ""

        facet = str(context.get("facet") or "").strip()
        day = str(context.get("day") or "").strip()
        if not facet or not day:
            logger.warning("spans hook: missing facet/day context")
            return ""

        span_id = str(activity.get("id") or "").strip()
        activity_type = str(activity.get("activity") or "").strip()
        talent = str(context.get("name") or "").strip()
        if not span_id or not activity_type or not talent:
            logger.warning("spans hook: missing span metadata")
            return ""

        bounds = _activity_time_bounds(activity.get("segments"))
        if bounds is None:
            return ""
        start, end = bounds

        row = {
            "span_id": span_id,
            "talent": talent,
            "facet": facet,
            "day": day,
            "activity_type": activity_type,
            "start": start,
            "end": end,
            "body": normalized_body,
            "topics": topics,
            "confidence": confidence,
        }

        def modify_fn(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
            updated: list[dict[str, Any]] = []
            replaced = False
            for record in records:
                if record.get("span_id") == span_id and record.get("talent") == talent:
                    if not replaced:
                        updated.append(dict(row))
                        replaced = True
                    continue
                updated.append(record)
            if not replaced:
                updated.append(dict(row))
            return updated

        locked_modify(_spans_path(facet, day), modify_fn, create_if_missing=True)
    except Exception as exc:
        logger.warning("spans hook: failed to persist row: %s", exc)

    return ""
