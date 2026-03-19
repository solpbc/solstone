# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Speaker ID status aggregation — read-only state inspection.

Aggregates speaker identification state from disk into a structured
JSON dashboard. No new computations — just reads existing files.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from think.utils import day_dirs, day_path, get_journal, iter_segments, segment_path

logger = logging.getLogger(__name__)

VALID_SECTIONS = {"embeddings", "owner", "speakers", "clusters", "imports", "attribution"}


def _routes_helpers():
    """Load speakers route helpers lazily to avoid import cycles."""
    from apps.speakers.routes import (
        _load_embeddings_file,
        _load_entity_voiceprints_file,
        _load_speaker_labels,
        _normalize_embedding,
        _scan_segment_embeddings,
    )

    return (
        _load_embeddings_file,
        _load_entity_voiceprints_file,
        _load_speaker_labels,
        _normalize_embedding,
        _scan_segment_embeddings,
    )


def _has_audio_embeddings(seg_path: Path) -> bool:
    """Return True if the segment has audio embedding NPZ files."""
    for p in seg_path.glob("*.npz"):
        if p.stem.endswith("_audio") or p.stem == "audio":
            return True
    return False


def get_embeddings_status() -> dict[str, Any]:
    """Aggregate embedding coverage statistics."""
    (
        load_embeddings_file,
        _,
        _,
        _,
        scan_segment_embeddings,
    ) = _routes_helpers()

    total_segments = 0
    segments_with_embeddings = 0
    total_embeddings = 0
    streams: dict[str, dict[str, int]] = {}
    days_with: set[str] = set()

    for day_name in sorted(day_dirs().keys()):
        segments = list(iter_segments(day_name))
        total_segments += len(segments)

        for segment in scan_segment_embeddings(day_name):
            segments_with_embeddings += 1
            stream = segment["stream"]
            seg_key = segment["key"]
            seg_dir = segment_path(day_name, seg_key, stream)

            if stream not in streams:
                streams[stream] = {"segments": 0, "embeddings": 0}
            streams[stream]["segments"] += 1
            days_with.add(day_name)

            for source in segment["sources"]:
                emb_data = load_embeddings_file(seg_dir / f"{source}.npz")
                if emb_data is not None:
                    count = len(emb_data[0])
                    total_embeddings += count
                    streams[stream]["embeddings"] += count

    sorted_days = sorted(days_with) if days_with else []

    return {
        "total_segments": total_segments,
        "segments_with_embeddings": segments_with_embeddings,
        "total_embeddings": total_embeddings,
        "coverage_pct": round(
            100.0 * segments_with_embeddings / total_segments, 1
        )
        if total_segments > 0
        else 0.0,
        "date_range": [sorted_days[0], sorted_days[-1]] if sorted_days else [],
        "days_with_embeddings": len(days_with),
        "streams": dict(sorted(streams.items(), key=lambda x: x[1]["embeddings"], reverse=True)),
    }


def get_owner_status() -> dict[str, Any]:
    """Aggregate owner centroid status."""
    from apps.speakers.owner import load_owner_centroid
    from think.entities.journal import get_journal_principal, journal_entity_memory_path

    principal = get_journal_principal()
    if not principal:
        return {"exists": False}

    centroid_data = load_owner_centroid()
    if centroid_data is None:
        return {"exists": False}

    # Load full centroid metadata
    centroid_path = journal_entity_memory_path(principal["id"]) / "owner_centroid.npz"
    result: dict[str, Any] = {"exists": True}

    try:
        data = np.load(centroid_path, allow_pickle=False)
        cluster_size = data.get("cluster_size")
        threshold = data.get("threshold")
        version = data.get("version")

        if cluster_size is not None:
            result["cluster_size"] = int(np.asarray(cluster_size).item())
        if threshold is not None:
            result["threshold"] = round(float(np.asarray(threshold).item()), 2)
        if version is not None:
            result["version"] = str(np.asarray(version).item())
    except Exception:
        pass

    # Estimate coverage: count how many embeddings match the owner centroid
    _, centroid_threshold = centroid_data
    owner_centroid = centroid_data[0]

    (
        load_embeddings_file,
        _,
        _,
        normalize_embedding,
        scan_segment_embeddings,
    ) = _routes_helpers()

    total_embeddings = 0
    owner_matches = 0
    streams_represented: set[str] = set()

    for day_name in day_dirs().keys():
        for segment in scan_segment_embeddings(day_name):
            stream = segment["stream"]
            seg_dir = segment_path(day_name, segment["key"], stream)
            for source in segment["sources"]:
                emb_data = load_embeddings_file(seg_dir / f"{source}.npz")
                if emb_data is None:
                    continue
                embeddings, _ = emb_data
                total_embeddings += len(embeddings)
                for emb in embeddings:
                    normalized = normalize_embedding(emb)
                    if normalized is not None:
                        score = float(np.dot(normalized, owner_centroid))
                        if score >= centroid_threshold:
                            owner_matches += 1
                            streams_represented.add(stream)

    result["streams_represented"] = sorted(streams_represented)
    result["coverage_estimate_pct"] = (
        round(100.0 * owner_matches / total_embeddings, 1) if total_embeddings > 0 else 0.0
    )

    return result


def get_speakers_status() -> dict[str, Any]:
    """Aggregate known speaker statistics."""
    from think.entities.journal import load_all_journal_entities

    _, load_entity_voiceprints_file, _, _, _ = _routes_helpers()

    journal_entities = load_all_journal_entities()
    speakers: list[dict[str, Any]] = []
    total_voiceprint_embeddings = 0

    for entity_id, entity in journal_entities.items():
        if entity.get("blocked") or entity.get("is_principal"):
            continue

        result = load_entity_voiceprints_file(entity_id)
        if result is None:
            continue

        embeddings, metadata_list = result
        embedding_count = len(embeddings)
        if embedding_count == 0:
            continue

        total_voiceprint_embeddings += embedding_count

        # Count unique segments and streams
        segments: set[tuple[str, str]] = set()
        streams: set[str] = set()
        for m in metadata_list:
            day = m.get("day", "")
            seg_key = m.get("segment_key", "")
            stream = m.get("stream", "")
            if day and seg_key:
                segments.add((day, seg_key))
            if stream:
                streams.add(stream)

        # Derive confidence rating
        stream_count = len(streams)
        if embedding_count >= 100 and stream_count >= 3:
            confidence = "strong"
        elif embedding_count >= 20 or stream_count >= 2:
            confidence = "moderate"
        else:
            confidence = "developing"

        speakers.append({
            "entity_id": entity_id,
            "name": entity.get("name", entity_id),
            "embeddings": embedding_count,
            "segments": len(segments),
            "streams": stream_count,
            "confidence": confidence,
        })

    # Sort by embedding count descending
    speakers.sort(key=lambda s: s["embeddings"], reverse=True)

    return {
        "total": len(speakers),
        "total_voiceprint_embeddings": total_voiceprint_embeddings,
        "top": speakers[:10],
    }


def get_clusters_status() -> dict[str, Any]:
    """Aggregate discovery cluster statistics from cache."""
    cache_path = Path(get_journal()) / "awareness" / "discovery_clusters.json"

    if not cache_path.exists():
        return {
            "total_unmatched": 0,
            "candidate_count": 0,
            "candidates": [],
        }

    try:
        with open(cache_path, encoding="utf-8") as f:
            cache_data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {
            "total_unmatched": 0,
            "candidate_count": 0,
            "candidates": [],
        }

    clusters = cache_data.get("clusters", {})
    total_unmatched = sum(len(members) for members in clusters.values())

    candidates: list[dict[str, Any]] = []
    for cluster_id, members in clusters.items():
        segment_set = {
            (m["day"], m["stream"], m["segment_key"]) for m in members
        }
        # Get a preview from the first member's text if possible
        preview = ""
        if members:
            first = members[0]
            seg_dir = segment_path(first["day"], first["segment_key"], first["stream"])
            jsonl_path = seg_dir / f"{first['source']}.jsonl"
            if jsonl_path.exists():
                try:
                    lines = jsonl_path.read_text(encoding="utf-8").splitlines()
                    sid = int(first.get("sentence_id", 0))
                    if 0 < sid < len(lines):
                        entry = json.loads(lines[sid])
                        preview = (entry.get("text") or "")[:80]
                except Exception:
                    pass

        candidates.append({
            "cluster_id": int(cluster_id),
            "size": len(members),
            "segment_count": len(segment_set),
            "preview": preview,
        })

    candidates.sort(key=lambda c: c["size"], reverse=True)

    return {
        "total_unmatched": total_unmatched,
        "candidate_count": len(candidates),
        "candidates": candidates,
    }


def get_imports_status() -> dict[str, Any]:
    """Aggregate import signal statistics."""
    from apps.speakers.attribution import (
        _extract_meeting_participants,
        _extract_screen_participants,
        _load_setting_field,
        _parse_setting_names,
    )

    settings_with_participants = 0
    meetings_with_attendees = 0
    screen_with_participants = 0

    seen_meeting_days: set[str] = set()

    for day_name in day_dirs().keys():
        for stream, seg_key, seg_path in iter_segments(day_name):
            # Check setting field
            setting = _load_setting_field(seg_path)
            if setting:
                names = _parse_setting_names(setting)
                if names:
                    settings_with_participants += 1

            # Check screen.md
            screen_names = _extract_screen_participants(seg_path)
            if screen_names:
                screen_with_participants += 1

            # Check meetings.md (once per day)
            if day_name not in seen_meeting_days:
                meeting_names = _extract_meeting_participants(day_name, seg_key)
                if meeting_names:
                    meetings_with_attendees += 1
                    seen_meeting_days.add(day_name)

    return {
        "settings_with_participants": settings_with_participants,
        "meetings_with_attendees": meetings_with_attendees,
        "screen_with_participants": screen_with_participants,
    }


def get_attribution_status() -> dict[str, Any]:
    """Aggregate attribution coverage statistics."""
    _, _, load_speaker_labels, _, scan_segment_embeddings = _routes_helpers()

    segments_with_embeddings = 0
    segments_with_labels = 0
    total_sentences = 0
    high_count = 0
    medium_count = 0
    null_count = 0
    method_breakdown: dict[str, int] = {}

    for day_name in day_dirs().keys():
        for segment in scan_segment_embeddings(day_name):
            segments_with_embeddings += 1
            seg_dir = segment_path(day_name, segment["key"], segment["stream"])
            labels_data = load_speaker_labels(seg_dir)
            if labels_data is None:
                continue

            segments_with_labels += 1
            for label in labels_data.get("labels", []):
                total_sentences += 1
                confidence = label.get("confidence")
                method = label.get("method") or "unmatched"

                if confidence == "high":
                    high_count += 1
                elif confidence == "medium":
                    medium_count += 1
                else:
                    null_count += 1

                method_breakdown[method] = method_breakdown.get(method, 0) + 1

    return {
        "segments_with_labels": segments_with_labels,
        "coverage_pct": round(
            100.0 * segments_with_labels / segments_with_embeddings, 1
        )
        if segments_with_embeddings > 0
        else 0.0,
        "total_sentences": total_sentences,
        "high": high_count,
        "medium": medium_count,
        "null": null_count,
        "needs_review": medium_count + null_count,
        "method_breakdown": dict(sorted(method_breakdown.items(), key=lambda x: x[1], reverse=True)),
    }


def get_status(section: str | None = None) -> dict[str, Any]:
    """Return full speaker ID status or a single section.

    Args:
        section: Optional section name to return. If None, returns all sections.

    Returns:
        Dict with all six sections, or just the requested section.
    """
    if section and section not in VALID_SECTIONS:
        return {"error": f"Unknown section: {section}. Valid: {', '.join(sorted(VALID_SECTIONS))}"}

    builders = {
        "embeddings": get_embeddings_status,
        "owner": get_owner_status,
        "speakers": get_speakers_status,
        "clusters": get_clusters_status,
        "imports": get_imports_status,
        "attribution": get_attribution_status,
    }

    if section:
        return {section: builders[section]()}

    return {name: builder() for name, builder in builders.items()}
