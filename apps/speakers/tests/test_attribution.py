# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for speaker attribution engine."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from apps.speakers.owner import OWNER_THRESHOLD

# Test stream name (matches conftest.STREAM)
STREAM = "test"


def _normalized(vector: list[float]) -> np.ndarray:
    emb = np.array(vector + [0.0] * (256 - len(vector)), dtype=np.float32)
    return emb / np.linalg.norm(emb)


def _setup_owner(env, name: str = "Self Person") -> tuple[Path, np.ndarray]:
    """Create a principal entity with confirmed owner centroid."""
    principal_dir = env.create_entity(name, is_principal=True)
    centroid = _normalized([1.0, 0.0])
    np.savez_compressed(
        principal_dir / "owner_centroid.npz",
        centroid=centroid,
        cluster_size=np.array(70, dtype=np.int32),
        threshold=np.array(OWNER_THRESHOLD, dtype=np.float32),
        version=np.array("2026-03-15T12:00:00"),
    )
    return principal_dir, centroid


def _write_controlled_segment(
    env,
    day: str,
    segment_key: str,
    embeddings: np.ndarray,
    source: str = "mic_audio",
) -> Path:
    """Write a segment with specific embeddings."""
    seg_dir = env.journal / day / STREAM / segment_key
    seg_dir.mkdir(parents=True, exist_ok=True)

    # NPZ
    statement_ids = np.arange(1, len(embeddings) + 1, dtype=np.int32)
    np.savez_compressed(
        seg_dir / f"{source}.npz",
        embeddings=embeddings.astype(np.float32),
        statement_ids=statement_ids,
    )

    # JSONL transcript
    time_part = segment_key.split("_")[0]
    base_h, base_m, base_s = int(time_part[:2]), int(time_part[2:4]), int(time_part[4:6])
    base_seconds = base_h * 3600 + base_m * 60 + base_s

    lines = [json.dumps({"raw": f"{source}.flac", "model": "medium.en"})]
    for idx in range(len(embeddings)):
        t = base_seconds + idx * 5
        h, m, s = t // 3600, (t % 3600) // 60, t % 60
        lines.append(json.dumps({"start": f"{h:02d}:{m:02d}:{s:02d}", "text": f"Sentence {idx + 1} text."}))
    (seg_dir / f"{source}.jsonl").write_text("\n".join(lines) + "\n")
    (seg_dir / f"{source}.flac").write_bytes(b"")

    return seg_dir


# ---------------------------------------------------------------------------
# Setting name parser tests
# ---------------------------------------------------------------------------


def test_parse_setting_names():
    from apps.speakers.attribution import _parse_setting_names

    assert _parse_setting_names("Jer and Jack at coffee") == ["Jack"]
    assert _parse_setting_names("Meeting with Perry and Thomas") == ["Perry", "Thomas"]
    assert _parse_setting_names("Lunch with John Borthwick") == ["John Borthwick"]
    assert _parse_setting_names("") == []
    assert _parse_setting_names("Call with Ryan") == ["Ryan"]


# ---------------------------------------------------------------------------
# Layer 1: Owner separation
# ---------------------------------------------------------------------------


def test_attribute_no_owner_centroid(speakers_env):
    from apps.speakers.attribution import attribute_segment

    env = speakers_env()
    env.create_segment("20240101", "090000_300", ["mic_audio"])

    result = attribute_segment("20240101", STREAM, "090000_300")

    assert result.get("error") == "no_owner_centroid"


def test_attribute_no_embeddings(speakers_env):
    from apps.speakers.attribution import attribute_segment

    env = speakers_env()
    _setup_owner(env)
    # Create empty segment directory (no npz files)
    seg_dir = env.journal / "20240101" / STREAM / "090000_300"
    seg_dir.mkdir(parents=True, exist_ok=True)

    result = attribute_segment("20240101", STREAM, "090000_300")

    assert result["labels"] == []
    assert result["unmatched"] == []


def test_layer1_owner_classification(speakers_env):
    from apps.speakers.attribution import attribute_segment

    env = speakers_env()
    _setup_owner(env)

    # Sentence 1: close to owner centroid [1,0,...], sentence 2: far from it
    owner_emb = _normalized([0.95, 0.05])
    other_emb = _normalized([0.1, 0.99])
    embeddings = np.vstack([owner_emb, other_emb])

    _write_controlled_segment(env, "20240101", "090000_300", embeddings)

    result = attribute_segment("20240101", STREAM, "090000_300")
    labels = result["labels"]

    assert len(labels) == 2
    # First sentence should be owner
    assert labels[0]["speaker"] == "self_person"
    assert labels[0]["method"] == "owner_centroid"
    assert labels[0]["confidence"] == "high"
    # Second sentence: unmatched (no speakers.json, no voiceprints)
    assert labels[1]["speaker"] is None


# ---------------------------------------------------------------------------
# Layer 2: Structural heuristics — single speaker
# ---------------------------------------------------------------------------


def test_layer2_single_speaker(speakers_env):
    from apps.speakers.attribution import attribute_segment

    env = speakers_env()
    _setup_owner(env)
    env.create_entity("Ryan Bennett")

    owner_emb = _normalized([0.95, 0.05])
    other_emb = _normalized([0.1, 0.99])
    embeddings = np.vstack([owner_emb, other_emb, other_emb])

    seg_dir = _write_controlled_segment(env, "20240101", "090000_300", embeddings)

    # speakers.json with exactly 1 speaker
    agents_dir = seg_dir / "agents"
    agents_dir.mkdir(parents=True, exist_ok=True)
    (agents_dir / "speakers.json").write_text(json.dumps(["Ryan Bennett"]))

    result = attribute_segment("20240101", STREAM, "090000_300")
    labels = result["labels"]

    assert labels[0]["method"] == "owner_centroid"  # sentence 1: owner
    assert labels[1]["speaker"] == "ryan_bennett"   # sentence 2: Ryan
    assert labels[1]["method"] == "structural_single_speaker"
    assert labels[1]["confidence"] == "high"
    assert labels[2]["speaker"] == "ryan_bennett"   # sentence 3: Ryan
    assert result["unmatched"] == []


# ---------------------------------------------------------------------------
# Layer 2: Setting field
# ---------------------------------------------------------------------------


def test_layer2_setting_field(speakers_env):
    from apps.speakers.attribution import attribute_segment

    env = speakers_env()
    _setup_owner(env)
    env.create_entity("Jack Andersohn")

    owner_emb = _normalized([0.95, 0.05])
    other_emb = _normalized([0.1, 0.99])
    embeddings = np.vstack([owner_emb, other_emb])

    seg_dir = _write_controlled_segment(
        env, "20240101", "090000_300", embeddings, source="imported_audio"
    )

    # Write imported_audio.jsonl with setting field
    jsonl_path = seg_dir / "imported_audio.jsonl"
    header = {"raw": "imported_audio.flac", "model": "medium.en", "setting": "Jer and Jack at coffee"}
    lines = [json.dumps(header)]
    lines.append(json.dumps({"start": "09:00:00", "text": "Owner talking"}))
    lines.append(json.dumps({"start": "09:00:05", "text": "Jack talking"}))
    jsonl_path.write_text("\n".join(lines) + "\n")

    result = attribute_segment("20240101", STREAM, "090000_300")
    labels = result["labels"]

    assert labels[0]["method"] == "owner_centroid"
    assert labels[1]["speaker"] == "jack_andersohn"
    assert labels[1]["method"] == "structural_setting"


# ---------------------------------------------------------------------------
# Layer 3: Acoustic matching
# ---------------------------------------------------------------------------


def test_layer3_acoustic_matching(speakers_env):
    from apps.speakers.attribution import attribute_segment

    env = speakers_env()
    _setup_owner(env)

    # Create entity with voiceprints similar to [0, 1, 0, ...]
    vp_emb = _normalized([0.0, 1.0])
    entity_dir = env.create_entity("Alice Test")
    np.savez_compressed(
        entity_dir / "voiceprints.npz",
        embeddings=np.vstack([vp_emb] * 10).astype(np.float32),
        metadata=np.array(
            [json.dumps({"day": "20240101", "segment_key": f"09{i:02d}00_300",
                         "source": "mic_audio", "sentence_id": 1,
                         "stream": STREAM, "added_at": 1700000000000})
             for i in range(10)],
            dtype=str,
        ),
    )

    owner_emb = _normalized([0.95, 0.05])
    alice_emb = _normalized([0.05, 0.95])  # similar to voiceprint
    embeddings = np.vstack([owner_emb, alice_emb])

    _write_controlled_segment(env, "20240101", "090000_300", embeddings)
    # No speakers.json, so Layer 2 can't resolve — falls through to Layer 3

    result = attribute_segment("20240101", STREAM, "090000_300")
    labels = result["labels"]

    assert labels[0]["method"] == "owner_centroid"
    assert labels[1]["speaker"] == "alice_test"
    assert labels[1]["method"] == "acoustic"
    assert labels[1]["confidence"] == "high"


# ---------------------------------------------------------------------------
# Graceful degradation: unmatched → null
# ---------------------------------------------------------------------------


def test_unmatched_sentences_get_null(speakers_env):
    from apps.speakers.attribution import attribute_segment

    env = speakers_env()
    _setup_owner(env)

    owner_emb = _normalized([0.95, 0.05])
    unknown_emb = _normalized([0.1, 0.5, 0.5])  # no matching voiceprint
    embeddings = np.vstack([owner_emb, unknown_emb])

    _write_controlled_segment(env, "20240101", "090000_300", embeddings)

    result = attribute_segment("20240101", STREAM, "090000_300")

    assert result["labels"][1]["speaker"] is None
    assert result["labels"][1]["confidence"] is None
    assert result["labels"][1]["method"] is None
    assert 2 in result["unmatched"]
    assert 2 in result["unmatched_texts"]


# ---------------------------------------------------------------------------
# Output: save_speaker_labels
# ---------------------------------------------------------------------------


def test_save_speaker_labels(tmp_path):
    from apps.speakers.attribution import save_speaker_labels

    labels = [
        {"sentence_id": 1, "speaker": "owner", "confidence": "high", "method": "owner_centroid"},
        {"sentence_id": 2, "speaker": "alice", "confidence": "high", "method": "acoustic"},
    ]
    metadata = {
        "owner_centroid_version": "2026-03-15T12:00:00",
        "voiceprint_versions": {"alice": 10},
    }

    path = save_speaker_labels(tmp_path, labels, metadata)

    assert path.name == "speaker_labels.json"
    data = json.loads(path.read_text())
    assert len(data["labels"]) == 2
    assert data["owner_centroid_version"] == "2026-03-15T12:00:00"
    assert data["voiceprint_versions"]["alice"] == 10


# ---------------------------------------------------------------------------
# Voiceprint accumulation
# ---------------------------------------------------------------------------


def test_accumulate_voiceprints_saves(speakers_env):
    from apps.speakers.attribution import accumulate_voiceprints

    env = speakers_env()
    _setup_owner(env)
    env.create_entity("Bob Smith")

    # Create segment with one non-owner embedding
    other_emb = _normalized([0.1, 0.99])
    _write_controlled_segment(env, "20240101", "090000_300", np.vstack([other_emb]))

    labels = [
        {
            "sentence_id": 1,
            "speaker": "bob_smith",
            "confidence": "high",
            "method": "structural_single_speaker",
        }
    ]

    saved = accumulate_voiceprints("20240101", STREAM, "090000_300", labels, "mic_audio")

    assert "bob_smith" in saved
    assert saved["bob_smith"] == 1

    # Verify voiceprints.npz was written
    vp_path = env.journal / "entities" / "bob_smith" / "voiceprints.npz"
    assert vp_path.exists()
    data = np.load(vp_path, allow_pickle=False)
    assert len(data["embeddings"]) == 1


def test_accumulate_idempotent(speakers_env):
    from apps.speakers.attribution import accumulate_voiceprints

    env = speakers_env()
    _setup_owner(env)
    env.create_entity("Bob Smith")

    other_emb = _normalized([0.1, 0.99])
    _write_controlled_segment(env, "20240101", "090000_300", np.vstack([other_emb]))

    labels = [
        {
            "sentence_id": 1,
            "speaker": "bob_smith",
            "confidence": "high",
            "method": "structural_single_speaker",
        }
    ]

    # Run twice
    accumulate_voiceprints("20240101", STREAM, "090000_300", labels, "mic_audio")
    saved = accumulate_voiceprints("20240101", STREAM, "090000_300", labels, "mic_audio")

    # Second run should save nothing (idempotent)
    assert saved == {}

    # Still only 1 embedding in voiceprints
    vp_path = env.journal / "entities" / "bob_smith" / "voiceprints.npz"
    data = np.load(vp_path, allow_pickle=False)
    assert len(data["embeddings"]) == 1


def test_accumulate_contamination_guard(speakers_env):
    from apps.speakers.attribution import accumulate_voiceprints

    env = speakers_env()
    _setup_owner(env)
    env.create_entity("Bob Smith")

    # Embedding very similar to owner centroid [1, 0, ...]
    owner_like = _normalized([0.99, 0.01])
    _write_controlled_segment(env, "20240101", "090000_300", np.vstack([owner_like]))

    labels = [
        {
            "sentence_id": 1,
            "speaker": "bob_smith",
            "confidence": "high",
            "method": "structural_single_speaker",
        }
    ]

    saved = accumulate_voiceprints("20240101", STREAM, "090000_300", labels, "mic_audio")

    # Should not save — embedding is too similar to owner
    assert saved == {}


def test_accumulate_skips_medium_confidence(speakers_env):
    from apps.speakers.attribution import accumulate_voiceprints

    env = speakers_env()
    _setup_owner(env)
    env.create_entity("Bob Smith")

    other_emb = _normalized([0.1, 0.99])
    _write_controlled_segment(env, "20240101", "090000_300", np.vstack([other_emb]))

    labels = [
        {
            "sentence_id": 1,
            "speaker": "bob_smith",
            "confidence": "medium",  # Not high — should not accumulate
            "method": "acoustic",
        }
    ]

    saved = accumulate_voiceprints("20240101", STREAM, "090000_300", labels, "mic_audio")

    assert saved == {}


def test_accumulate_skips_contextual_method(speakers_env):
    from apps.speakers.attribution import accumulate_voiceprints

    env = speakers_env()
    _setup_owner(env)
    env.create_entity("Bob Smith")

    other_emb = _normalized([0.1, 0.99])
    _write_controlled_segment(env, "20240101", "090000_300", np.vstack([other_emb]))

    labels = [
        {
            "sentence_id": 1,
            "speaker": "bob_smith",
            "confidence": "high",
            "method": "contextual",  # Layer 4 — should not accumulate
        }
    ]

    saved = accumulate_voiceprints("20240101", STREAM, "090000_300", labels, "mic_audio")

    assert saved == {}
