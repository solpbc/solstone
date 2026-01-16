# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for speakers app - sentence-based embeddings."""

import numpy as np


def test_normalize_embedding():
    """Test L2 normalization of embeddings."""
    from apps.speakers.routes import _normalize_embedding

    emb = np.array([3.0, 4.0, 0.0] + [0.0] * 253, dtype=np.float32)
    normalized = _normalize_embedding(emb)

    assert normalized is not None
    assert np.isclose(np.linalg.norm(normalized), 1.0)
    # 3-4-5 right triangle, normalized to unit vector
    assert np.isclose(normalized[0], 0.6)
    assert np.isclose(normalized[1], 0.8)


def test_normalize_embedding_zero_vector():
    """Test that zero vector returns None."""
    from apps.speakers.routes import _normalize_embedding

    emb = np.zeros(256, dtype=np.float32)
    normalized = _normalize_embedding(emb)

    assert normalized is None


def test_parse_time_to_seconds():
    """Test time string parsing."""
    from apps.speakers.routes import _parse_time_to_seconds

    assert _parse_time_to_seconds("00:00:00") == 0
    assert _parse_time_to_seconds("00:01:30") == 90
    assert _parse_time_to_seconds("01:00:00") == 3600
    assert _parse_time_to_seconds("14:30:22") == 52222


def test_scan_segment_embeddings_empty(speakers_env):
    """Test scanning when no embeddings exist."""
    from apps.speakers.routes import _scan_segment_embeddings

    env = speakers_env()

    # Create a day dir but no segments
    day_dir = env.journal / "20240101"
    day_dir.mkdir()

    segments = _scan_segment_embeddings("20240101")
    assert segments == []


def test_scan_segment_embeddings_with_data(speakers_env):
    """Test scanning when embeddings and speakers exist."""
    from apps.speakers.routes import _scan_segment_embeddings

    env = speakers_env()
    env.create_segment("20240101", "143022_300", ["mic_audio", "sys_audio"])
    env.create_speakers_json("20240101", "143022_300", ["Alice", "Bob"])

    segments = _scan_segment_embeddings("20240101")
    assert len(segments) == 1
    assert segments[0]["key"] == "143022_300"
    assert segments[0]["start"] == "14:30"
    assert segments[0]["end"] == "14:35"
    assert segments[0]["duration"] == 300
    assert set(segments[0]["sources"]) == {"mic_audio", "sys_audio"}
    assert segments[0]["speakers"] == ["Alice", "Bob"]
    assert segments[0]["speaker_count"] == 2


def test_scan_segment_embeddings_plain_audio(speakers_env):
    """Test scanning finds plain 'audio' source (not just *_audio pattern)."""
    from apps.speakers.routes import _scan_segment_embeddings

    env = speakers_env()
    env.create_segment("20240101", "143022_300", ["audio"])
    env.create_speakers_json("20240101", "143022_300", ["Alice", "Bob"])

    segments = _scan_segment_embeddings("20240101")
    assert len(segments) == 1
    assert segments[0]["sources"] == ["audio"]


def test_load_sentences(speakers_env):
    """Test loading sentences with embeddings."""
    from apps.speakers.routes import _load_sentences

    env = speakers_env()
    env.create_segment("20240101", "143022_300", ["mic_audio"], num_sentences=3)

    sentences, emb_data = _load_sentences("20240101", "143022_300", "mic_audio")

    assert len(sentences) == 3
    assert sentences[0]["id"] == 1
    assert sentences[0]["text"] == "This is sentence 1."
    assert sentences[0]["offset"] == 0
    assert sentences[0]["has_embedding"] is True

    assert emb_data is not None
    embeddings, segment_ids = emb_data
    assert embeddings.shape == (3, 256)
    assert len(segment_ids) == 3


def test_load_sentences_no_transcript(speakers_env):
    """Test loading sentences when no transcript exists."""
    from apps.speakers.routes import _load_sentences

    env = speakers_env()

    # Create day dir but no segment
    day_dir = env.journal / "20240101" / "143022_300"
    day_dir.mkdir(parents=True)

    sentences, emb_data = _load_sentences("20240101", "143022_300", "mic_audio")
    assert sentences == []
    assert emb_data is None


def test_get_sentence_embedding(speakers_env):
    """Test getting a specific sentence's embedding."""
    from apps.speakers.routes import _get_sentence_embedding

    env = speakers_env()
    env.create_segment("20240101", "143022_300", ["mic_audio"], num_sentences=5)

    # Get embedding for sentence 3
    emb = _get_sentence_embedding("20240101", "143022_300", "mic_audio", 3)

    assert emb is not None
    assert emb.shape == (256,)
    assert np.isclose(np.linalg.norm(emb), 1.0)


def test_get_sentence_embedding_not_found(speakers_env):
    """Test getting embedding for non-existent sentence."""
    from apps.speakers.routes import _get_sentence_embedding

    env = speakers_env()
    env.create_segment("20240101", "143022_300", ["mic_audio"], num_sentences=3)

    # Try to get embedding for sentence that doesn't exist
    emb = _get_sentence_embedding("20240101", "143022_300", "mic_audio", 99)
    assert emb is None


def test_compute_best_match():
    """Test best match algorithm."""
    from apps.speakers.routes import _compute_best_match

    # Create segment embedding
    seg_emb = np.array([1.0, 0.0, 0.0] + [0.0] * 253, dtype=np.float32)

    # Create known voiceprints with varying similarity
    known = {
        "Alice": np.array(
            [1.0, 0.0, 0.0] + [0.0] * 253, dtype=np.float32
        ),  # Perfect match
        "Bob": np.array(
            [0.7, 0.7, 0.0] + [0.0] * 253, dtype=np.float32
        ),  # Partial match
        "Charlie": np.array(
            [0.0, 1.0, 0.0] + [0.0] * 253, dtype=np.float32
        ),  # No match
    }
    # Normalize
    for name in known:
        known[name] = known[name] / np.linalg.norm(known[name])

    match = _compute_best_match(seg_emb, known)

    # Alice should be best match (perfect 1.0)
    assert match is not None
    assert match["entity"] == "Alice"
    assert match["score"] >= 0.99


def test_compute_best_match_below_threshold():
    """Test that no match is returned when all below threshold."""
    from apps.speakers.routes import _compute_best_match

    seg_emb = np.array([1.0, 0.0, 0.0] + [0.0] * 253, dtype=np.float32)

    # All orthogonal to segment embedding
    known = {
        "Alice": np.array([0.0, 1.0, 0.0] + [0.0] * 253, dtype=np.float32),
        "Bob": np.array([0.0, 0.0, 1.0] + [0.0] * 253, dtype=np.float32),
    }
    for name in known:
        known[name] = known[name] / np.linalg.norm(known[name])

    match = _compute_best_match(seg_emb, known)
    assert match is None


def test_scan_entity_voiceprints(speakers_env):
    """Test scanning voiceprints averages multiple embeddings."""
    from apps.speakers.routes import _scan_entity_voiceprints

    env = speakers_env()
    env.create_entity(
        "test",
        "Alice Test",
        voiceprints=[
            ("20240101", "120000_300", "mic_audio", 1),
            ("20240102", "130000_300", "mic_audio", 2),
        ],
    )

    voiceprints = _scan_entity_voiceprints("test")

    assert "Alice Test" in voiceprints
    avg_emb = voiceprints["Alice Test"]
    assert avg_emb.shape == (256,)
    assert np.isclose(np.linalg.norm(avg_emb), 1.0)


def test_load_entity_voiceprints_file(speakers_env):
    """Test loading voiceprints from consolidated file."""
    from apps.speakers.routes import _load_entity_voiceprints_file

    env = speakers_env()
    env.create_entity(
        "test",
        "Bob Test",
        voiceprints=[
            ("20240101", "120000_300", "mic_audio", 1),
            ("20240102", "130000_300", "audio", 2),
        ],
    )

    result = _load_entity_voiceprints_file("test", "Bob Test")

    assert result is not None
    embeddings, metadata_list = result
    assert embeddings.shape == (2, 256)
    assert len(metadata_list) == 2
    assert metadata_list[0]["day"] == "20240101"
    assert metadata_list[1]["day"] == "20240102"
    assert metadata_list[0]["source"] == "mic_audio"
    assert metadata_list[1]["source"] == "audio"


def test_load_entity_voiceprints_file_not_found(speakers_env):
    """Test loading voiceprints for non-existent entity returns None."""
    from apps.speakers.routes import _load_entity_voiceprints_file

    env = speakers_env()

    # Create facet but no entity
    facet_dir = env.journal / "facets" / "test"
    facet_dir.mkdir(parents=True)

    result = _load_entity_voiceprints_file("test", "Nobody")
    assert result is None


def test_save_voiceprint(speakers_env):
    """Test saving voiceprint to consolidated voiceprints.npz."""
    import json

    from apps.speakers.routes import _save_voiceprint

    env = speakers_env()

    # Create facet
    facet_dir = env.journal / "facets" / "test"
    facet_dir.mkdir(parents=True)

    emb = np.array([1.0, 0.0, 0.0] + [0.0] * 253, dtype=np.float32)

    path = _save_voiceprint(
        "test", "John Doe", emb, "20240101", "143022_300", "mic_audio", 5
    )

    assert path.exists()
    assert path.name == "voiceprints.npz"
    assert "john_doe" in str(path.parent)

    # Verify format content
    data = np.load(path)
    assert "embeddings" in data
    assert "metadata" in data
    assert data["embeddings"].shape == (1, 256)

    # Verify metadata
    metadata = json.loads(data["metadata"][0])
    assert metadata["day"] == "20240101"
    assert metadata["segment_key"] == "143022_300"
    assert metadata["source"] == "mic_audio"
    assert metadata["sentence_id"] == 5
    assert "added_at" in metadata


def test_save_voiceprint_appends(speakers_env):
    """Test saving multiple voiceprints appends to existing file."""
    import json

    from apps.speakers.routes import _save_voiceprint

    env = speakers_env()

    # Create facet
    facet_dir = env.journal / "facets" / "test"
    facet_dir.mkdir(parents=True)

    emb1 = np.array([1.0, 0.0, 0.0] + [0.0] * 253, dtype=np.float32)
    emb2 = np.array([0.0, 1.0, 0.0] + [0.0] * 253, dtype=np.float32)

    # Save first voiceprint
    path = _save_voiceprint(
        "test", "John Doe", emb1, "20240101", "143022_300", "mic_audio", 5
    )

    # Save second voiceprint
    path2 = _save_voiceprint(
        "test", "John Doe", emb2, "20240102", "150000_300", "audio", 3
    )

    assert path == path2  # Same file

    # Verify both are in the file
    data = np.load(path)
    assert data["embeddings"].shape == (2, 256)
    assert len(data["metadata"]) == 2

    meta1 = json.loads(data["metadata"][0])
    meta2 = json.loads(data["metadata"][1])
    assert meta1["day"] == "20240101"
    assert meta2["day"] == "20240102"


def test_load_embeddings_file(speakers_env):
    """Test loading embeddings from NPZ file."""
    from apps.speakers.routes import _load_embeddings_file

    env = speakers_env()
    env.create_segment("20240101", "143022_300", ["mic_audio"], num_sentences=3)

    npz_path = env.journal / "20240101" / "143022_300" / "mic_audio.npz"
    result = _load_embeddings_file(npz_path)

    assert result is not None
    embeddings, segment_ids = result
    assert embeddings.shape == (3, 256)
    assert len(segment_ids) == 3


def test_load_embeddings_file_not_found():
    """Test loading non-existent embeddings file returns None."""
    from pathlib import Path

    from apps.speakers.routes import _load_embeddings_file

    result = _load_embeddings_file(Path("/nonexistent/file.npz"))

    assert result is None


def test_load_segment_speakers(speakers_env):
    """Test loading speakers from speakers.json."""
    from apps.speakers.routes import _load_segment_speakers

    env = speakers_env()
    env.create_segment("20240101", "143022_300", ["mic_audio"])
    env.create_speakers_json("20240101", "143022_300", ["Alice", "Bob", "Charlie"])

    segment_dir = env.journal / "20240101" / "143022_300"
    speakers = _load_segment_speakers(segment_dir)

    assert speakers == ["Alice", "Bob", "Charlie"]


def test_load_segment_speakers_not_found(speakers_env):
    """Test loading speakers returns empty list when file missing."""
    from apps.speakers.routes import _load_segment_speakers

    env = speakers_env()
    env.create_segment("20240101", "143022_300", ["mic_audio"])
    # No speakers.json created

    segment_dir = env.journal / "20240101" / "143022_300"
    speakers = _load_segment_speakers(segment_dir)

    assert speakers == []


def test_load_segment_speakers_invalid_json(speakers_env):
    """Test loading speakers returns empty list for invalid JSON."""
    from apps.speakers.routes import _load_segment_speakers

    env = speakers_env()
    segment_dir = env.journal / "20240101" / "143022_300"
    segment_dir.mkdir(parents=True)

    # Write invalid JSON
    speakers_path = segment_dir / "speakers.json"
    speakers_path.write_text("not valid json")

    speakers = _load_segment_speakers(segment_dir)
    assert speakers == []


def test_load_segment_speakers_not_list(speakers_env):
    """Test loading speakers returns empty list when JSON is not a list."""
    import json

    from apps.speakers.routes import _load_segment_speakers

    env = speakers_env()
    segment_dir = env.journal / "20240101" / "143022_300"
    segment_dir.mkdir(parents=True)

    # Write object instead of list
    speakers_path = segment_dir / "speakers.json"
    speakers_path.write_text(json.dumps({"speaker": "Alice"}))

    speakers = _load_segment_speakers(segment_dir)
    assert speakers == []


def test_scan_segment_embeddings_requires_speakers(speakers_env):
    """Test that segments without speakers.json are filtered out."""
    from apps.speakers.routes import _scan_segment_embeddings

    env = speakers_env()
    # Create segment with embeddings but NO speakers.json
    env.create_segment("20240101", "143022_300", ["mic_audio"])

    segments = _scan_segment_embeddings("20240101")
    assert segments == []


def test_scan_segment_embeddings_requires_two_speakers(speakers_env):
    """Test that segments with <2 speakers are filtered out."""
    from apps.speakers.routes import _scan_segment_embeddings

    env = speakers_env()
    env.create_segment("20240101", "143022_300", ["mic_audio"])
    env.create_speakers_json("20240101", "143022_300", ["OnlyAlice"])  # Just 1 speaker

    segments = _scan_segment_embeddings("20240101")
    assert segments == []


def test_scan_segment_embeddings_includes_speaker_data(speakers_env):
    """Test that segments include speaker names and count."""
    from apps.speakers.routes import _scan_segment_embeddings

    env = speakers_env()
    env.create_segment("20240101", "143022_300", ["mic_audio"])
    env.create_speakers_json("20240101", "143022_300", ["Alice", "Bob"])

    segments = _scan_segment_embeddings("20240101")

    assert len(segments) == 1
    assert segments[0]["speakers"] == ["Alice", "Bob"]
    assert segments[0]["speaker_count"] == 2
