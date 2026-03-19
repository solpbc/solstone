# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for the speakers status command."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from think.entities import entity_slug


def _normalized(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)


def test_status_empty_journal(speakers_env):
    from apps.speakers.status import get_status

    speakers_env()
    result = get_status()

    assert "embeddings" in result
    assert "owner" in result
    assert "speakers" in result
    assert "clusters" in result
    assert "imports" in result
    assert "attribution" in result

    assert result["embeddings"]["total_segments"] == 0
    assert result["embeddings"]["segments_with_embeddings"] == 0
    assert result["owner"]["exists"] is False
    assert result["speakers"]["total"] == 0
    assert result["clusters"]["candidate_count"] == 0
    assert result["attribution"]["segments_with_labels"] == 0


def test_status_section_filter(speakers_env):
    from apps.speakers.status import get_status

    speakers_env()
    result = get_status(section="embeddings")

    assert "embeddings" in result
    assert "owner" not in result


def test_status_invalid_section(speakers_env):
    from apps.speakers.status import get_status

    speakers_env()
    result = get_status(section="nonexistent")

    assert "error" in result


def test_status_embeddings(speakers_env):
    from apps.speakers.status import get_status

    env = speakers_env()
    env.create_segment("20240101", "090000_300", ["mic_audio"], num_sentences=10)
    env.create_segment("20240101", "100000_300", ["mic_audio"], num_sentences=5)
    env.create_segment("20240102", "090000_300", ["mic_audio"], num_sentences=8)

    result = get_status(section="embeddings")
    emb = result["embeddings"]

    assert emb["segments_with_embeddings"] == 3
    assert emb["total_embeddings"] == 23
    assert emb["days_with_embeddings"] == 2
    assert len(emb["date_range"]) == 2
    assert emb["date_range"][0] == "20240101"
    assert emb["date_range"][1] == "20240102"
    assert "test" in emb["streams"]
    assert emb["streams"]["test"]["segments"] == 3
    assert emb["streams"]["test"]["embeddings"] == 23


def test_status_owner_no_centroid(speakers_env):
    from apps.speakers.status import get_status

    speakers_env()
    result = get_status(section="owner")
    assert result["owner"]["exists"] is False


def test_status_owner_with_centroid(speakers_env):
    from apps.speakers.owner import OWNER_THRESHOLD
    from apps.speakers.status import get_status

    env = speakers_env()
    # Create principal entity with owner centroid
    principal_dir = env.create_entity("Self Person", is_principal=True)
    centroid = _normalized(np.array([1.0] + [0.0] * 255, dtype=np.float32))
    np.savez_compressed(
        principal_dir / "owner_centroid.npz",
        centroid=centroid,
        cluster_size=np.array(100, dtype=np.int32),
        threshold=np.array(OWNER_THRESHOLD, dtype=np.float32),
        version=np.array("2026-03-15T10:30:00Z"),
    )

    # Create a segment with embeddings similar to owner
    close_embs = np.tile(
        _normalized(np.array([0.95, 0.05] + [0.0] * 254, dtype=np.float32)),
        (5, 1),
    )
    env.create_segment("20240101", "090000_300", ["mic_audio"], embeddings=close_embs)

    result = get_status(section="owner")
    owner = result["owner"]

    assert owner["exists"] is True
    assert owner["cluster_size"] == 100
    assert owner["threshold"] == OWNER_THRESHOLD
    assert owner["version"] == "2026-03-15T10:30:00Z"


def test_status_speakers(speakers_env):
    from apps.speakers.status import get_status

    env = speakers_env()

    # Create entities with voiceprints
    env.create_entity(
        "Alice Test",
        voiceprints=[
            ("20240101", "090000_300", "mic_audio", 1),
            ("20240101", "090000_300", "mic_audio", 2),
            ("20240101", "090000_300", "mic_audio", 3),
        ],
    )
    env.create_entity(
        "Bob Test",
        voiceprints=[
            ("20240101", "100000_300", "mic_audio", 1),
        ],
    )

    result = get_status(section="speakers")
    spk = result["speakers"]

    assert spk["total"] == 2
    assert spk["total_voiceprint_embeddings"] == 4
    assert len(spk["top"]) == 2
    assert spk["top"][0]["name"] == "Alice Test"
    assert spk["top"][0]["embeddings"] == 3
    assert spk["top"][0]["confidence"] == "developing"


def test_status_clusters_empty(speakers_env):
    from apps.speakers.status import get_status

    speakers_env()
    result = get_status(section="clusters")

    assert result["clusters"]["total_unmatched"] == 0
    assert result["clusters"]["candidate_count"] == 0


def test_status_clusters_with_cache(speakers_env):
    from apps.speakers.status import get_status

    env = speakers_env()
    # Create a segment so transcript text lookup doesn't crash
    env.create_segment("20240101", "090000_300", ["mic_audio"], num_sentences=5)

    # Write a discovery cache file
    cache_dir = env.journal / "awareness"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_data = {
        "version": "2026-03-15T10:00:00",
        "clusters": {
            "0": [
                {
                    "day": "20240101",
                    "stream": "test",
                    "segment_key": "090000_300",
                    "source": "mic_audio",
                    "sentence_id": 1,
                },
                {
                    "day": "20240101",
                    "stream": "test",
                    "segment_key": "090000_300",
                    "source": "mic_audio",
                    "sentence_id": 2,
                },
            ],
        },
    }
    with open(cache_dir / "discovery_clusters.json", "w") as f:
        json.dump(cache_data, f)

    result = get_status(section="clusters")
    clusters = result["clusters"]

    assert clusters["total_unmatched"] == 2
    assert clusters["candidate_count"] == 1
    assert clusters["candidates"][0]["cluster_id"] == 0
    assert clusters["candidates"][0]["size"] == 2


def test_status_attribution(speakers_env):
    from apps.speakers.status import get_status

    env = speakers_env()
    env.create_segment("20240101", "090000_300", ["mic_audio"], num_sentences=5)
    env.create_segment("20240101", "100000_300", ["mic_audio"], num_sentences=3)

    # Create labels for one segment
    env.create_speaker_labels(
        "20240101",
        "090000_300",
        [
            {"sentence_id": 1, "speaker": "alice", "confidence": "high", "method": "owner_centroid"},
            {"sentence_id": 2, "speaker": "alice", "confidence": "high", "method": "owner_centroid"},
            {"sentence_id": 3, "speaker": "bob", "confidence": "medium", "method": "acoustic"},
            {"sentence_id": 4, "speaker": None, "confidence": None, "method": None},
            {"sentence_id": 5, "speaker": "alice", "confidence": "high", "method": "structural_single_speaker"},
        ],
    )

    result = get_status(section="attribution")
    attr = result["attribution"]

    assert attr["segments_with_labels"] == 1
    assert attr["total_sentences"] == 5
    assert attr["high"] == 3
    assert attr["medium"] == 1
    assert attr["null"] == 1
    assert attr["needs_review"] == 2
    assert "owner_centroid" in attr["method_breakdown"]


def test_status_full_returns_all_sections(speakers_env):
    from apps.speakers.status import get_status

    env = speakers_env()
    env.create_segment("20240101", "090000_300", ["mic_audio"])

    result = get_status()

    assert len(result) == 6
    for section in ("embeddings", "owner", "speakers", "clusters", "imports", "attribution"):
        assert section in result


def test_status_json_serializable(speakers_env):
    """Ensure the full status output can be serialized to JSON."""
    from apps.speakers.status import get_status

    env = speakers_env()
    env.create_segment("20240101", "090000_300", ["mic_audio"])

    result = get_status()
    # This will raise if any value is not JSON-serializable
    serialized = json.dumps(result)
    assert isinstance(serialized, str)
    parsed = json.loads(serialized)
    assert "embeddings" in parsed
