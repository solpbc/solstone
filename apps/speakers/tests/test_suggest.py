# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for speaker suggestion generation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _write_discovery_cache(journal: Path, clusters: dict[str, list[dict]]) -> None:
    cache_dir = journal / "awareness"
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "discovery_clusters.json").write_text(
        json.dumps({"version": "2026-03-19T00:00:00", "clusters": clusters}),
        encoding="utf-8",
    )


def _write_events(journal: Path, facet: str, day: str, events: list[dict]) -> None:
    events_dir = journal / "facets" / facet / "events"
    events_dir.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(event) for event in events]
    (events_dir / f"{day}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_voiceprints(
    journal: Path, entity_id: str, vectors: list[np.ndarray]
) -> None:
    entity_dir = journal / "entities" / entity_id
    entity_dir.mkdir(parents=True, exist_ok=True)
    metadata = np.array(
        [
            json.dumps(
                {
                    "day": "20240101",
                    "segment_key": f"0900{idx:02d}_300",
                    "source": "audio",
                    "sentence_id": 1,
                    "added_at": 1700000000000,
                }
            )
            for idx, _ in enumerate(vectors, start=1)
        ],
        dtype=str,
    )
    np.savez_compressed(
        entity_dir / "voiceprints.npz",
        embeddings=np.array(vectors, dtype=np.float32),
        metadata=metadata,
    )


def test_suggest_empty(speakers_env):
    """No suggestions when journal has no relevant data."""
    speakers_env()

    from apps.speakers.suggest import suggest_speakers

    result = suggest_speakers()

    assert result == []


def test_suggest_unknown_recurring_from_cache(speakers_env):
    env = speakers_env()
    _write_discovery_cache(
        env.journal,
        {
            "7": [
                {
                    "day": "20240101",
                    "stream": "test",
                    "segment_key": "090000_1800",
                    "source": "audio",
                    "sentence_id": 1,
                },
                {
                    "day": "20240101",
                    "stream": "test",
                    "segment_key": "090000_1800",
                    "source": "audio",
                    "sentence_id": 2,
                },
                {
                    "day": "20240101",
                    "stream": "test",
                    "segment_key": "100000_1800",
                    "source": "audio",
                    "sentence_id": 3,
                },
            ]
        },
    )

    from apps.speakers.suggest import suggest_speakers

    result = suggest_speakers(limit=10)

    assert result[0]["type"] == "unknown_recurring"
    assert result[0]["cluster_id"] == 7
    assert result[0]["size"] == 3
    assert result[0]["segment_count"] == 2
    assert result[0]["segments"] == [
        "20240101/test/090000_1800",
        "20240101/test/100000_1800",
    ]
    assert "calendar_overlap" in result[0]["import_hints"]


def test_suggest_unknown_recurring_calendar_overlap(speakers_env):
    env = speakers_env()
    env.create_segment("20240101", "090000_1800", ["audio"])
    _write_discovery_cache(
        env.journal,
        {
            "2": [
                {
                    "day": "20240101",
                    "stream": "test",
                    "segment_key": "090000_1800",
                    "source": "audio",
                    "sentence_id": 1,
                }
            ]
        },
    )
    _write_events(
        env.journal,
        "testfacet",
        "20240101",
        [
            {
                "type": "meeting",
                "start": "09:15:00",
                "end": "09:45:00",
                "title": "Design Sync",
                "facet": "testfacet",
                "participants": ["Alice", "Bob"],
                "occurred": True,
            }
        ],
    )

    from apps.speakers.suggest import suggest_speakers

    result = suggest_speakers(limit=10)
    overlap = result[0]["import_hints"]["calendar_overlap"][0]

    assert overlap["title"] == "Design Sync"
    assert overlap["facet"] == "testfacet"
    assert overlap["participants"] == ["Alice", "Bob"]
    assert overlap["segments"] == ["20240101/test/090000_1800"]


def test_suggest_import_linkable(speakers_env):
    env = speakers_env()
    env.create_segment("20240101", "090000_1800", ["audio"])
    env.create_segment("20240101", "120000_1800", ["audio"])
    env.create_entity("Has Voiceprint")
    env.create_entity("Needs Import")
    _write_voiceprints(
        env.journal,
        "has_voiceprint",
        [env.create_embedding([1.0, 0.0])],
    )
    _write_events(
        env.journal,
        "work",
        "20240101",
        [
            {
                "type": "meeting",
                "start": "09:15:00",
                "end": "09:45:00",
                "title": "Planning",
                "facet": "work",
                "participants": ["Has Voiceprint", "Needs Import"],
                "occurred": True,
            },
            {
                "type": "meeting",
                "start": "12:15:00",
                "end": "12:45:00",
                "title": "Followup",
                "facet": "work",
                "participants": ["Needs Import"],
                "occurred": True,
            },
        ],
    )

    from apps.speakers.suggest import suggest_speakers

    result = suggest_speakers(limit=10)
    import_linkable = [s for s in result if s["type"] == "import_linkable"]

    assert len(import_linkable) == 1
    assert import_linkable[0]["name"] == "Needs Import"
    assert import_linkable[0]["meetings_count"] == 2
    assert import_linkable[0]["has_voiceprint"] is False
    assert import_linkable[0]["overlapping_segments"] == [
        "20240101/test/090000_1800",
        "20240101/test/120000_1800",
    ]


def test_suggest_name_variant(speakers_env):
    env = speakers_env()
    env.create_entity("Owner Person", is_principal=True)
    env.create_entity("Alice", voiceprints=[("20240101", "090000_300", "audio", 1)])
    env.create_entity(
        "Alice Johnson",
        voiceprints=[("20240101", "090000_300", "audio", 1)],
    )
    shared = env.create_embedding([1.0, 0.0, 0.0])
    _write_voiceprints(env.journal, "alice", [shared, shared])
    _write_voiceprints(env.journal, "alice_johnson", [shared, shared])

    from apps.speakers.suggest import suggest_speakers

    result = suggest_speakers(limit=10)
    variants = [s for s in result if s["type"] == "name_variant"]

    assert len(variants) == 1
    assert variants[0]["names"] == ["Alice", "Alice Johnson"]
    assert variants[0]["entity_ids"] == ["alice", "alice_johnson"]
    assert variants[0]["similarity"] >= 0.9


def test_suggest_low_confidence_review(speakers_env):
    env = speakers_env()
    env.create_segment("20240102", "090000_1800", ["audio"], num_sentences=12)
    labels = []
    for idx in range(1, 7):
        labels.append(
            {
                "sentence_id": idx,
                "speaker": f"speaker_{idx}",
                "confidence": "medium",
                "method": "acoustic",
            }
        )
    for idx in range(7, 13):
        labels.append(
            {
                "sentence_id": idx,
                "speaker": None,
                "confidence": None,
                "method": None,
            }
        )
    env.create_speaker_labels("20240102", "090000_1800", labels)

    from apps.speakers.suggest import suggest_speakers

    result = suggest_speakers(limit=10)
    review = [s for s in result if s["type"] == "low_confidence_review"]

    assert len(review) == 1
    assert review[0]["day"] == "20240102"
    assert review[0]["medium_count"] == 6
    assert review[0]["null_count"] == 6
    assert review[0]["segments_needing_review"] == ["20240102/test/090000_1800"]


def test_suggest_limit_truncation(speakers_env):
    env = speakers_env()
    _write_discovery_cache(
        env.journal,
        {
            "1": [
                {
                    "day": "20240101",
                    "stream": "test",
                    "segment_key": "090000_1800",
                    "source": "audio",
                    "sentence_id": 1,
                }
            ]
        },
    )
    env.create_segment("20240102", "090000_1800", ["audio"], num_sentences=12)
    labels = [
        {
            "sentence_id": idx,
            "speaker": None,
            "confidence": None,
            "method": None,
        }
        for idx in range(1, 13)
    ]
    env.create_speaker_labels("20240102", "090000_1800", labels)

    from apps.speakers.suggest import suggest_speakers

    result = suggest_speakers(limit=1)

    assert len(result) == 1
    assert result[0]["type"] == "unknown_recurring"


def test_suggest_priority_ordering(speakers_env):
    env = speakers_env()
    env.create_segment("20240101", "090000_1800", ["audio"])
    _write_discovery_cache(
        env.journal,
        {
            "4": [
                {
                    "day": "20240101",
                    "stream": "test",
                    "segment_key": "090000_1800",
                    "source": "audio",
                    "sentence_id": 1,
                }
            ]
        },
    )
    env.create_entity("Import Target")
    _write_events(
        env.journal,
        "work",
        "20240101",
        [
            {
                "type": "meeting",
                "start": "09:15:00",
                "end": "09:45:00",
                "title": "Planning",
                "facet": "work",
                "participants": ["Import Target"],
                "occurred": True,
            }
        ],
    )
    env.create_entity("Owner Person", is_principal=True)
    env.create_entity("Bob", voiceprints=[("20240101", "090000_300", "audio", 1)])
    env.create_entity(
        "Bob Smith",
        voiceprints=[("20240101", "090000_300", "audio", 1)],
    )
    shared = env.create_embedding([0.0, 1.0, 0.0])
    _write_voiceprints(env.journal, "bob", [shared, shared])
    _write_voiceprints(env.journal, "bob_smith", [shared, shared])
    env.create_segment("20240103", "090000_1800", ["audio"], num_sentences=12)
    review_labels = [
        {
            "sentence_id": idx,
            "speaker": None,
            "confidence": None,
            "method": None,
        }
        for idx in range(1, 13)
    ]
    env.create_speaker_labels("20240103", "090000_1800", review_labels)

    from apps.speakers.suggest import suggest_speakers

    result = suggest_speakers(limit=50)
    types = [item["type"] for item in result]

    assert types.index("unknown_recurring") < types.index("import_linkable")
    assert types.index("import_linkable") < types.index("name_variant")
    assert types.index("name_variant") < types.index("low_confidence_review")
