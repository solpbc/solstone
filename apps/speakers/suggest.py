# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Speaker curation suggestions - computed on the fly from existing data."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import time
from pathlib import Path
from typing import Any

from think.utils import day_dirs, get_journal, segment_parse

logger = logging.getLogger(__name__)


def suggest_speakers(limit: int = 5) -> list[dict[str, Any]]:
    """Return prioritized speaker curation suggestions.

    Priority order: unknown_recurring > import_linkable > name_variant >
    low_confidence_review. Returns at most ``limit`` suggestions total.
    """
    suggestions: list[dict[str, Any]] = []
    suggestions.extend(_suggest_unknown_recurring())
    suggestions.extend(_suggest_import_linkable())
    suggestions.extend(_suggest_name_variants())
    suggestions.extend(_suggest_low_confidence_review())
    return suggestions[:limit]


def _suggest_unknown_recurring() -> list[dict[str, Any]]:
    """Transform the discovery cache into actionable cluster suggestions."""
    cache_path = Path(get_journal()) / "awareness" / "discovery_clusters.json"
    if not cache_path.exists():
        return []

    try:
        cache_data = json.loads(cache_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        logger.warning("Failed to read discovery cache", exc_info=True)
        return []

    clusters = cache_data.get("clusters")
    if not isinstance(clusters, dict):
        return []

    suggestions: list[dict[str, Any]] = []
    for cluster_id_str, records in clusters.items():
        if not isinstance(records, list):
            continue
        try:
            cluster_id = int(cluster_id_str)
        except (TypeError, ValueError):
            continue

        segment_keys = {
            (record.get("day"), record.get("segment_key"))
            for record in records
            if isinstance(record, dict)
        }
        segments = sorted(
            {
                f"{record['day']}/{record['stream']}/{record['segment_key']}"
                for record in records
                if isinstance(record, dict)
                and record.get("day")
                and record.get("stream")
                and record.get("segment_key")
            }
        )
        suggestions.append(
            {
                "type": "unknown_recurring",
                "cluster_id": cluster_id,
                "size": len(records),
                "segment_count": len(
                    {
                        (day, segment_key)
                        for day, segment_key in segment_keys
                        if day and segment_key
                    }
                ),
                "segments": segments,
                "samples": records[:3],
                "import_hints": {
                    "calendar_overlap": _calendar_overlap_for_segments(segments)
                },
            }
        )

    suggestions.sort(key=lambda item: item["size"], reverse=True)
    return suggestions


def _suggest_import_linkable() -> list[dict[str, Any]]:
    """Suggest meeting participants who appear in events but lack voiceprints."""
    participant_data: dict[str, dict[str, Any]] = {}
    for events_path in Path(get_journal()).glob("facets/*/events/*.jsonl"):
        day = events_path.stem
        try:
            with open(events_path, encoding="utf-8") as handle:
                for line in handle:
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if (
                        event.get("type") != "meeting"
                        or event.get("occurred") is not True
                    ):
                        continue
                    participants = event.get("participants")
                    if not isinstance(participants, list) or not participants:
                        continue
                    for name in participants:
                        if not isinstance(name, str) or not name.strip():
                            continue
                        entry = participant_data.setdefault(
                            name,
                            {"count": 0, "day_events": []},
                        )
                        entry["count"] += 1
                        entry["day_events"].append((day, event))
        except OSError:
            logger.warning("Failed reading event file %s", events_path, exc_info=True)
            continue

    if not participant_data:
        return []

    from think.entities.journal import load_journal_entity, scan_journal_entities

    name_to_entity: dict[str, dict[str, Any]] = {}
    for entity_id in scan_journal_entities():
        entity = load_journal_entity(entity_id)
        if not entity:
            continue
        entity_names = [entity.get("name", "")]
        entity_names.extend(entity.get("aka", []))
        for name in entity_names:
            if isinstance(name, str) and name.strip():
                name_to_entity[name.lower()] = entity

    suggestions: list[dict[str, Any]] = []
    journal = Path(get_journal())
    for name, entry in participant_data.items():
        matched_entity = name_to_entity.get(name.lower())
        has_voiceprint = False
        if matched_entity is not None:
            voiceprint_path = (
                journal / "entities" / matched_entity["id"] / "voiceprints.npz"
            )
            has_voiceprint = voiceprint_path.exists()
        if has_voiceprint:
            continue

        overlapping_segments: set[str] = set()
        segments_by_day: dict[str, list[tuple[str, str, time, time]]] = {}
        for day, event in entry["day_events"]:
            event_start = _parse_event_time(event.get("start"))
            event_end = _parse_event_time(event.get("end"))
            if event_start is None or event_end is None:
                continue
            day_segments = segments_by_day.setdefault(day, _iter_day_segments(day))
            for stream, segment_key, seg_start, seg_end in day_segments:
                if _time_overlaps(seg_start, seg_end, event_start, event_end):
                    overlapping_segments.add(f"{day}/{stream}/{segment_key}")

        suggestions.append(
            {
                "type": "import_linkable",
                "name": name,
                "source": "meetings",
                "meetings_count": entry["count"],
                "has_voiceprint": False,
                "overlapping_segments": sorted(overlapping_segments),
            }
        )

    suggestions.sort(key=lambda item: item["meetings_count"], reverse=True)
    return suggestions


def _suggest_name_variants() -> list[dict[str, Any]]:
    """Return high-similarity speaker name pairs with resolved entity IDs."""
    from apps.speakers.bootstrap import resolve_name_variants
    from think.entities.journal import load_journal_entity, scan_journal_entities

    stats = resolve_name_variants(dry_run=True)
    matches = stats.get("matches_found", [])
    if not isinstance(matches, list):
        return []

    name_to_entity_id: dict[str, str] = {}
    for entity_id in scan_journal_entities():
        entity = load_journal_entity(entity_id)
        if not entity:
            continue
        name = entity.get("name")
        if isinstance(name, str) and name.strip():
            name_to_entity_id[name.lower()] = entity_id

    suggestions = []
    for match in matches:
        if not isinstance(match, dict):
            continue
        name_a = match.get("name_a")
        name_b = match.get("name_b")
        if not isinstance(name_a, str) or not isinstance(name_b, str):
            continue
        suggestions.append(
            {
                "type": "name_variant",
                "names": [name_a, name_b],
                "entity_ids": [
                    name_to_entity_id.get(name_a.lower()),
                    name_to_entity_id.get(name_b.lower()),
                ],
                "similarity": match.get("similarity"),
            }
        )

    return suggestions


def _suggest_low_confidence_review() -> list[dict[str, Any]]:
    """Suggest days with a large number of medium or null speaker labels."""
    journal = Path(get_journal())
    by_day: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"medium_count": 0, "null_count": 0, "segments": set()}
    )

    for day in day_dirs().keys():
        day_dir = journal / day
        if not day_dir.is_dir():
            continue
        for stream_dir in sorted(day_dir.iterdir()):
            if not stream_dir.is_dir():
                continue
            for seg_dir in sorted(stream_dir.iterdir()):
                if not seg_dir.is_dir():
                    continue
                labels_path = seg_dir / "agents" / "speaker_labels.json"
                if not labels_path.exists():
                    continue
                try:
                    data = json.loads(labels_path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    continue
                needs_review = False
                for label in data.get("labels", []):
                    if label.get("confidence") == "medium":
                        by_day[day]["medium_count"] += 1
                        needs_review = True
                    if label.get("speaker") is None:
                        by_day[day]["null_count"] += 1
                        needs_review = True
                if needs_review:
                    by_day[day]["segments"].add(
                        f"{day}/{stream_dir.name}/{seg_dir.name}"
                    )

    suggestions = []
    for day, info in by_day.items():
        total = info["medium_count"] + info["null_count"]
        if total <= 10:
            continue
        suggestions.append(
            {
                "type": "low_confidence_review",
                "day": day,
                "medium_count": info["medium_count"],
                "null_count": info["null_count"],
                "segments_needing_review": sorted(info["segments"]),
            }
        )

    suggestions.sort(
        key=lambda item: item["medium_count"] + item["null_count"],
        reverse=True,
    )
    return suggestions


def _calendar_overlap_for_segments(segments: list[str]) -> list[dict[str, Any]]:
    """Find meeting events that overlap with the given segment strings.

    Args:
        segments: list of "day/stream/segment_key" strings
    """
    segments_by_day: dict[str, list[tuple[str, str, time, time]]] = defaultdict(list)
    for segment in segments:
        parts = segment.split("/")
        if len(parts) != 3:
            continue
        day, stream, segment_key = parts
        seg_start, seg_end = segment_parse(segment_key)
        if seg_start is None or seg_end is None:
            continue
        segments_by_day[day].append((stream, segment_key, seg_start, seg_end))

    overlaps: list[dict[str, Any]] = []
    for day, segment_entries in segments_by_day.items():
        for event in _load_day_events(day):
            if event.get("type") != "meeting":
                continue
            participants = event.get("participants")
            if not isinstance(participants, list) or not participants:
                continue
            event_start = _parse_event_time(event.get("start"))
            event_end = _parse_event_time(event.get("end"))
            if event_start is None or event_end is None:
                continue

            overlapping_segments = []
            for stream, segment_key, seg_start, seg_end in segment_entries:
                segment_label = f"{day}/{stream}/{segment_key}"
                if _time_overlaps(seg_start, seg_end, event_start, event_end):
                    overlapping_segments.append(segment_label)

            if overlapping_segments:
                overlaps.append(
                    {
                        "day": day,
                        "title": event.get("title", ""),
                        "facet": event.get("facet", ""),
                        "participants": participants,
                        "start": event.get("start"),
                        "end": event.get("end"),
                        "segments": sorted(set(overlapping_segments)),
                    }
                )

    return overlaps


def _time_overlaps(
    seg_start: time, seg_end: time, event_start: time, event_end: time
) -> bool:
    """Return True if two time ranges overlap."""
    return seg_start < event_end and event_start < seg_end


def _load_day_events(day: str) -> list[dict[str, Any]]:
    """Load all events for a day from facets/*/events/{day}.jsonl."""
    events: list[dict[str, Any]] = []
    for events_path in Path(get_journal()).glob(f"facets/*/events/{day}.jsonl"):
        try:
            with open(events_path, encoding="utf-8") as handle:
                for line in handle:
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(event, dict):
                        continue
                    event.setdefault("facet", events_path.parent.parent.name)
                    events.append(event)
        except OSError:
            logger.warning(
                "Failed reading day events from %s", events_path, exc_info=True
            )
            continue
    return events


def _iter_day_segments(day: str) -> list[tuple[str, str, time, time]]:
    """Return (stream, segment_key, start_time, end_time) for all segments on a day."""
    day_dir = Path(get_journal()) / day
    if not day_dir.is_dir():
        return []

    segments = []
    for stream_dir in sorted(day_dir.iterdir()):
        if not stream_dir.is_dir():
            continue
        for segment_dir in sorted(stream_dir.iterdir()):
            if not segment_dir.is_dir():
                continue
            start_time, end_time = segment_parse(segment_dir.name)
            if start_time is None or end_time is None:
                continue
            segments.append((stream_dir.name, segment_dir.name, start_time, end_time))
    return segments


def _parse_event_time(value: Any) -> time | None:
    """Parse an event time string if valid."""
    if not isinstance(value, str) or not value:
        return None
    try:
        return time.fromisoformat(value)
    except ValueError:
        return None
