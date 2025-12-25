"""Tests for speakers app."""

import numpy as np


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
    """Test scanning when embeddings exist."""
    from apps.speakers.routes import _scan_segment_embeddings

    env = speakers_env()
    env.create_segment("20240101", "143022_300", ["Speaker 1", "Speaker 2"])

    segments = _scan_segment_embeddings("20240101")
    assert len(segments) == 1
    assert segments[0]["key"] == "143022_300"
    assert segments[0]["start"] == "14:30"
    assert segments[0]["end"] == "14:35"
    assert segments[0]["duration"] == 300
    assert set(segments[0]["speakers"]) == {"Speaker 1", "Speaker 2"}


def test_load_segment_speaker_embedding(speakers_env):
    """Test loading a speaker embedding."""
    from apps.speakers.routes import _load_segment_speaker_embedding

    env = speakers_env()

    # Create segment with specific embedding
    audio_dir = env.journal / "20240101" / "143022_300" / "audio"
    audio_dir.mkdir(parents=True)
    emb = np.array([1.0, 0.0, 0.0] + [0.0] * 253, dtype=np.float32)
    np.savez_compressed(audio_dir / "Speaker 1.npz", embedding=emb)

    loaded = _load_segment_speaker_embedding("20240101", "143022_300", "Speaker 1")
    assert loaded is not None
    assert loaded.shape == (256,)
    assert np.isclose(np.linalg.norm(loaded), 1.0)


def test_compute_matches():
    """Test matching algorithm."""
    from apps.speakers.routes import _compute_matches

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

    matches = _compute_matches(seg_emb, known)

    # Alice should have perfect match (1.0)
    assert "Alice" in matches
    assert matches["Alice"] >= 0.99

    # Bob should have partial match (~0.7)
    assert "Bob" in matches
    assert 0.65 <= matches["Bob"] <= 0.75

    # Charlie should be below threshold (0.4), not included
    assert "Charlie" not in matches


def test_scan_entity_voiceprints_averaging(speakers_env):
    """Test that multiple voiceprints are averaged."""
    from apps.speakers.routes import _scan_entity_voiceprints
    from think.entities import normalize_entity_name

    env = speakers_env()

    # Create facet with entity
    facet_dir = env.journal / "facets" / "test"
    facet_dir.mkdir(parents=True)

    # Create entities.jsonl
    (facet_dir / "entities.jsonl").write_text(
        '{"type": "Person", "name": "Alice Test", "description": "Test"}\n'
    )

    # Create entity folder with multiple voiceprints
    entity_dir = facet_dir / "entities" / normalize_entity_name("Alice Test")
    entity_dir.mkdir(parents=True)

    # Create two voiceprints
    emb1 = np.array([1.0, 0.0, 0.0] + [0.0] * 253, dtype=np.float32)
    emb2 = np.array([0.8, 0.6, 0.0] + [0.0] * 253, dtype=np.float32)
    emb2 = emb2 / np.linalg.norm(emb2)

    np.savez_compressed(entity_dir / "20240101_120000_300.npz", embedding=emb1)
    np.savez_compressed(entity_dir / "20240102_130000_300.npz", embedding=emb2)

    voiceprints = _scan_entity_voiceprints("test")

    assert "Alice Test" in voiceprints
    avg_emb = voiceprints["Alice Test"]
    assert avg_emb.shape == (256,)
    assert np.isclose(np.linalg.norm(avg_emb), 1.0)  # Should be normalized


def test_save_voiceprint_to_entity(speakers_env):
    """Test saving voiceprint to entity folder."""
    from apps.speakers.routes import _save_voiceprint_to_entity

    env = speakers_env()

    # Create facet
    facet_dir = env.journal / "facets" / "test"
    facet_dir.mkdir(parents=True)

    emb = np.array([1.0, 0.0, 0.0] + [0.0] * 253, dtype=np.float32)

    path = _save_voiceprint_to_entity("test", "John Doe", "20240101", "143022_300", emb)

    assert path.exists()
    assert path.name == "20240101_143022_300.npz"
    assert "john_doe" in str(path.parent)

    # Verify content
    data = np.load(path)
    assert "embedding" in data
    assert data["embedding"].shape == (256,)
