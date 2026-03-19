# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Speaker curation suggestion helpers."""

from __future__ import annotations

import json
import logging
import re
from datetime import time
from pathlib import Path
from typing import Any

from think.utils import day_dirs, get_journal, iter_segments, segment_parse

logger = logging.getLogger(__name__)

_MEETING_LINE_RE = re.compile(r"^-\s+(\d{2}:\d{2})\s+(.*)")
_PARTICIPANTS_RE = re.compile(
    r"\*\*Participants?\*\*\s*[:\u2013\u2014\-]\s*(.*)",
    re.IGNORECASE,
)
_PAREN_RE = re.compile(r"\(([^)]+)\)")
_WITH_RE = re.compile(r"\bwith\s+(.+?)(?:\s*\(|$)", re.IGNORECASE)
_SKIP_PARTICIPANT_TERMS = ("presenting", "private", "unscheduled")


def _bootstrap_helpers():
    from apps.speakers.bootstrap import resolve_name_variants

    return resolve_name_variants


def _discovery_helpers():
    from apps.speakers.discovery import discover_unknown_speakers

    return discover_unknown_speakers


def _split_participants(text: str) -> list[str]:
    parts = re.split(r",|\band\b", text, flags=re.IGNORECASE)
    return [part.strip().strip("*").strip() for part in parts if part.strip()]


def _name_matches_entity(participant: str, names: set[str]) -> bool:
    participant_lower = participant.strip().lower()
    if not participant_lower:
        return False
    if participant_lower in names:
        return True
    first_word = participant_lower.split()[0]
    return any(name.split()[0] == first_word for name in names if name)


def _parse_meetings(day_path: str) -> list[dict[str, Any]]:
    meetings_path = Path(day_path) / "agents" / "meetings.md"
    if not meetings_path.exists():
        return []

    try:
        content = meetings_path.read_text(encoding="utf-8")
    except OSError:
        return []

    meetings: list[dict[str, Any]] = []
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        participants_match = _PARTICIPANTS_RE.search(line)
        if participants_match:
            participants = [
                name
                for name in _split_participants(participants_match.group(1))
                if len(name) >= 2
            ]
            meetings.append(
                {
                    "time": None,
                    "line": raw_line,
                    "participants": participants,
                }
            )
            continue

        match = _MEETING_LINE_RE.match(line)
        if not match:
            continue

        meeting_time, description = match.groups()
        participants: list[str] = []

        for paren_match in _PAREN_RE.findall(description):
            for name in _split_participants(paren_match):
                if len(name) < 2:
                    continue
                if any(term in name.lower() for term in _SKIP_PARTICIPANT_TERMS):
                    continue
                participants.append(name)

        with_match = _WITH_RE.search(description)
        if with_match:
            participants.extend(
                name
                for name in _split_participants(with_match.group(1))
                if len(name) >= 2
            )

        meetings.append(
            {
                "time": meeting_time,
                "line": raw_line,
                "participants": list(dict.fromkeys(participants)),
            }
        )

    return meetings


def _meetings_overlapping_segment(
    meetings: list[dict[str, Any]], segment_key: str
) -> list[str]:
    start_time, end_time = segment_parse(segment_key)
    if start_time is None or end_time is None:
        return []

    overlaps: list[str] = []
    for meeting in meetings:
        meeting_time = meeting.get("time")
        if not meeting_time:
            continue
        try:
            hour, minute = meeting_time.split(":", 1)
            meeting_clock = time(hour=int(hour), minute=int(minute))
        except (TypeError, ValueError):
            continue
        if start_time <= meeting_clock <= end_time:
            overlaps.append(meeting.get("line", ""))
    return overlaps


def _unknown_recurring() -> list[dict[str, Any]]:
    discover_unknown_speakers = _discovery_helpers()

    result = discover_unknown_speakers()
    clusters = result.get("clusters", [])

    suggestions: list[dict[str, Any]] = []
    meetings_cache: dict[str, list[dict[str, Any]]] = {}
    all_days = day_dirs()

    for cluster in clusters:
        samples = cluster.get("samples", [])
        segments: list[str] = []
        seen_segments: set[str] = set()
        overlap_lines: list[str] = []
        seen_overlap_keys: set[tuple[str, str]] = set()

        for sample in samples:
            segment_ref = f"{sample['day']}/{sample['stream']}/{sample['segment_key']}"
            if segment_ref not in seen_segments:
                seen_segments.add(segment_ref)
                segments.append(segment_ref)

            day = sample["day"]
            segment_key = sample["segment_key"]
            overlap_key = (day, segment_key)
            if overlap_key in seen_overlap_keys:
                continue
            seen_overlap_keys.add(overlap_key)

            if day not in meetings_cache:
                dp = all_days.get(day)
                meetings_cache[day] = _parse_meetings(dp) if dp else []
            for line in _meetings_overlapping_segment(meetings_cache[day], segment_key):
                if line not in overlap_lines:
                    overlap_lines.append(line)

        suggestions.append(
            {
                "type": "unknown_recurring",
                "cluster_id": cluster["cluster_id"],
                "size": cluster["size"],
                "segment_count": cluster["segment_count"],
                "segments": segments,
                "samples": samples,
                "import_hints": {"calendar_overlap": overlap_lines},
            }
        )

    return suggestions


def _import_linkable() -> list[dict[str, Any]]:
    from think.entities.journal import load_all_journal_entities

    entities = load_all_journal_entities()
    # Track both the count of meeting lines and which days, per participant name
    participant_info: dict[str, dict[str, Any]] = {}

    for day, dp in day_dirs().items():
        for meeting in _parse_meetings(dp):
            for participant in meeting.get("participants", []):
                key = participant.lower()
                info = participant_info.setdefault(key, {"count": 0, "days": set()})
                info["count"] += 1
                info["days"].add(day)

    suggestions: list[dict[str, Any]] = []
    journal_path = Path(get_journal())

    for entity_id, entity in entities.items():
        if entity.get("is_principal") or entity.get("blocked"):
            continue

        if (journal_path / "entities" / entity_id / "voiceprints.npz").exists():
            continue

        names = {
            str(name).strip().lower()
            for name in [entity.get("name"), *(entity.get("aka", []))]
            if str(name).strip()
        }
        if not names:
            continue

        mention_count = 0
        matched_days: set[str] = set()
        for participant, info in participant_info.items():
            if _name_matches_entity(participant, names):
                mention_count += info["count"]
                matched_days.update(info["days"])

        if not matched_days:
            continue

        suggestions.append(
            {
                "type": "import_linkable",
                "entity_id": entity_id,
                "name": entity["name"],
                "meetings_mentioned": mention_count,
                "meeting_days": sorted(matched_days),
            }
        )

    suggestions.sort(
        key=lambda item: item["meetings_mentioned"],
        reverse=True,
    )
    return suggestions


def _name_variant() -> list[dict[str, Any]]:
    resolve_name_variants = _bootstrap_helpers()
    from think.entities.journal import load_all_journal_entities

    result = resolve_name_variants(dry_run=True)
    entities = load_all_journal_entities()
    name_to_id = {
        entity.get("name", "").strip().lower(): entity_id
        for entity_id, entity in entities.items()
        if entity.get("name")
    }

    suggestions: list[dict[str, Any]] = []
    for pair in result.get("matches_found", []):
        name_a = pair["name_a"]
        name_b = pair["name_b"]
        lower_a = name_a.lower()
        lower_b = name_b.lower()
        first_word_match = (
            lower_a.split()[0] == lower_b or lower_b.split()[0] == lower_a
        )
        substring_match = lower_a in lower_b or lower_b in lower_a
        if not (first_word_match or substring_match):
            continue

        suggestions.append(
            {
                "type": "name_variant",
                "entity_a": {"id": name_to_id.get(lower_a), "name": name_a},
                "entity_b": {"id": name_to_id.get(lower_b), "name": name_b},
                "similarity": pair["similarity"],
            }
        )

    suggestions.sort(key=lambda item: item["similarity"], reverse=True)
    return suggestions


def _low_confidence_review() -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    for day in sorted(day_dirs().keys()):
        for stream, segment_key, seg_path in iter_segments(day):
            labels_path = seg_path / "agents" / "speaker_labels.json"
            if not labels_path.exists():
                continue
            try:
                data = json.loads(labels_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue

            labels = data.get("labels", [])
            if not isinstance(labels, list):
                continue

            medium_or_null = 0
            null_count = 0
            total = 0
            for label in labels:
                if not isinstance(label, dict):
                    continue
                total += 1
                if label.get("confidence") != "high":
                    medium_or_null += 1
                if not label.get("speaker"):
                    null_count += 1

            if medium_or_null <= 10:
                continue

            speakers_path = seg_path / "agents" / "speakers.json"
            has_speakers = speakers_path.is_file()
            null_proportion = null_count / total if total else 0.0
            results.append(
                {
                    "type": "low_confidence_review",
                    "day": day,
                    "segment_key": segment_key,
                    "stream": stream,
                    "medium_or_null_count": medium_or_null,
                    "total_labels": total,
                    "has_speakers": has_speakers,
                    "null_proportion": null_proportion,
                }
            )

    results.sort(key=lambda item: (not item["has_speakers"], -item["null_proportion"]))
    return results


def suggest_opportunities(limit: int = 5) -> list[dict[str, Any]]:
    suggestions: list[dict[str, Any]] = []
    for generator in [
        _unknown_recurring,
        _import_linkable,
        _name_variant,
        _low_confidence_review,
    ]:
        if len(suggestions) >= limit:
            break
        try:
            suggestions.extend(generator())
        except Exception:
            logger.exception("Suggestion generator %s failed", generator.__name__)
    return suggestions[:limit]


def format_suggestions(suggestions: list[dict[str, Any]]) -> str:
    if not suggestions:
        return "No speaker curation suggestions found."

    lines: list[str] = []
    for suggestion in suggestions:
        suggestion_type = suggestion.get("type")
        if suggestion_type == "unknown_recurring":
            lines.append(
                "Unknown recurring speaker "
                f"(cluster {suggestion['cluster_id']}): "
                f"{suggestion['size']} samples across "
                f"{suggestion['segment_count']} segments"
            )
            segments = suggestion.get("segments", [])
            if segments:
                lines.append(f"  Segments: {', '.join(segments)}")
            for meeting_line in suggestion.get("import_hints", {}).get(
                "calendar_overlap", []
            ):
                lines.append(f"  Calendar overlap: {meeting_line.strip()}")
        elif suggestion_type == "import_linkable":
            lines.append(
                "Import linkable: "
                f"{suggestion['name']} ({suggestion['entity_id']}) "
                f"\u2014 mentioned in {suggestion['meetings_mentioned']} meetings"
            )
            lines.append(f"  Days: {', '.join(suggestion['meeting_days'])}")
        elif suggestion_type == "name_variant":
            lines.append(
                "Name variant: "
                f'"{suggestion["entity_a"]["name"]}" '
                f'\u2194 "{suggestion["entity_b"]["name"]}" '
                f"(similarity: {suggestion['similarity']:.2f})"
            )
        elif suggestion_type == "low_confidence_review":
            seg_info = suggestion.get("segment_key", "")
            lines.append(
                "Low confidence review: "
                f"{suggestion['day']}/{seg_info} \u2014 "
                f"{suggestion['medium_or_null_count']} of "
                f"{suggestion['total_labels']} labels are medium/unresolved"
            )

    return "\n".join(lines)
