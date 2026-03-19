# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Speaker subsystem status aggregation."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from think.awareness import get_current
from think.utils import day_dirs, get_journal

logger = logging.getLogger(__name__)

SECTIONS = ("embeddings", "owner", "speakers", "clusters", "imports", "attribution")


def get_speakers_status(section: str | None = None) -> Any:
    """Aggregate speaker subsystem status.

    Args:
        section: Optional section name to return. If None, returns all sections.

    Returns:
        Dict with all sections, or a single section's value if section is specified.
    """
    builders = {
        "embeddings": _embeddings_section,
        "owner": _owner_section,
        "speakers": _speakers_section,
        "clusters": _clusters_section,
        "imports": _imports_section,
        "attribution": _attribution_section,
    }

    if section:
        builder = builders.get(section)
        if builder is None:
            return {
                "error": f"Unknown section '{section}'. Valid: {', '.join(SECTIONS)}"
            }
        return builder()

    return {name: builder() for name, builder in builders.items()}


def _embeddings_section() -> dict[str, Any]:
    from apps.speakers.routes import _scan_segment_embeddings

    segments = 0
    streams: dict[str, int] = {}
    days_seen: set[str] = set()

    for day in day_dirs().keys():
        day_segments = _scan_segment_embeddings(day)
        if day_segments:
            days_seen.add(day)
        for seg in day_segments:
            segments += 1
            stream = seg["stream"]
            streams[stream] = streams.get(stream, 0) + 1

    sorted_days = sorted(days_seen) if days_seen else []
    return {
        "segments": segments,
        "streams": streams,
        "days": len(sorted_days),
        "date_range": [sorted_days[0], sorted_days[-1]] if sorted_days else None,
    }


def _owner_section() -> dict[str, Any]:
    from apps.speakers.owner import load_owner_centroid

    voiceprint = get_current().get("voiceprint", {})
    status = voiceprint.get("status", "none")
    result: dict[str, Any] = {"status": status}

    if status == "candidate":
        result["cluster_size"] = voiceprint.get("cluster_size")
        result["detected_at"] = voiceprint.get("detected_at")
        result["streams_represented"] = voiceprint.get("streams_represented")
        result["recommendation"] = voiceprint.get("recommendation")
    elif status == "no_cluster":
        result["segments_checked"] = voiceprint.get("segments_checked")
        result["attempted_at"] = voiceprint.get("attempted_at")

    result["centroid_saved"] = load_owner_centroid() is not None
    return result


def _speakers_section() -> list[dict[str, Any]]:
    from apps.speakers.routes import _load_entity_voiceprints_file
    from think.entities.journal import scan_journal_entities

    speakers = []
    for entity in scan_journal_entities():
        entity_id = entity["id"]
        result = _load_entity_voiceprints_file(entity_id)
        if result is None:
            continue

        embeddings, metadata_list = result
        streams: set[str] = set()
        segments: set[tuple[str, str]] = set()
        for metadata in metadata_list:
            if "stream" in metadata:
                streams.add(metadata["stream"])
            segments.add((metadata.get("day", ""), metadata.get("segment_key", "")))

        speakers.append(
            {
                "entity_id": entity_id,
                "name": entity.get("name", entity_id),
                "embedding_count": len(embeddings),
                "segment_count": len(segments),
                "streams": sorted(streams),
            }
        )

    return speakers


def _clusters_section() -> dict[str, Any] | None:
    cache_path = Path(get_journal()) / "awareness" / "discovery_clusters.json"
    if not cache_path.exists():
        return None
    try:
        data = json.loads(cache_path.read_text())
        clusters = data.get("clusters", [])
        return {
            "cached_at": data.get("version"),
            "count": len(clusters),
            "clusters": clusters,
        }
    except Exception:
        logger.warning("Failed to read discovery cache", exc_info=True)
        return None


def _imports_section() -> dict[str, Any]:
    journal = Path(get_journal())
    meetings = 0
    screens = 0

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
                if (seg_dir / "meetings.md").exists():
                    meetings += 1
                if (seg_dir / "screen.md").exists():
                    screens += 1

    return {"meetings_files": meetings, "screen_files": screens}


def _attribution_section() -> dict[str, Any]:
    journal = Path(get_journal())
    total_files = 0
    total_labels = 0
    by_confidence: dict[str, int] = {}
    by_method: dict[str, int] = {}

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
                labels_file = seg_dir / "agents" / "speaker_labels.json"
                if not labels_file.exists():
                    continue
                try:
                    data = json.loads(labels_file.read_text())
                except Exception:
                    continue
                total_files += 1
                for label in data.get("labels", []):
                    total_labels += 1
                    confidence = label.get("confidence", "unknown")
                    method = label.get("method", "unknown")
                    by_confidence[confidence] = by_confidence.get(confidence, 0) + 1
                    by_method[method] = by_method.get(method, 0) + 1

    return {
        "files": total_files,
        "labels": total_labels,
        "by_confidence": by_confidence,
        "by_method": by_method,
    }
