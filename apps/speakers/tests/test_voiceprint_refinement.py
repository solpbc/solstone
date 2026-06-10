# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for Step 5 voiceprint refinements: temporal decay, outlier rejection,
and retroactive confirmation.

Each test is designed so the expected behaviour only holds with the refinement
active — the comment above each assertion notes what would happen without it.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from apps.speakers.attribution import (
    _VP_DECAY_LAMBDA,
    _VP_OUTLIER_MIN_SAMPLES,
    _VP_OUTLIER_MIN_SIMILARITY,
    accumulate_voiceprints,
    attribute_segment,
)
from apps.speakers.encoder_config import ACOUSTIC_HIGH, ACOUSTIC_MEDIUM, OWNER_THRESHOLD
from think.entities.voiceprints import save_voiceprints_safely

STREAM = "test"


# ---------------------------------------------------------------------------
# Helpers (mirrors test_attribution.py pattern)
# ---------------------------------------------------------------------------


def _unit(v: list[float]) -> np.ndarray:
    emb = np.array(v + [0.0] * (256 - len(v)), dtype=np.float32)
    return emb / np.linalg.norm(emb)


def _setup_owner(env) -> np.ndarray:
    principal_dir = env.create_entity("Self Person", is_principal=True)
    centroid = _unit([1.0, 0.0])
    np.savez_compressed(
        principal_dir / "owner_centroid.npz",
        centroid=centroid,
        cluster_size=np.array(70, dtype=np.int32),
        threshold=np.array(OWNER_THRESHOLD, dtype=np.float32),
        version=np.array("2026-01-01T00:00:00"),
    )
    return centroid


def _write_voiceprints(
    entity_dir: Path,
    embeddings: list[np.ndarray],
    added_ats: list[int],
    stream: str = STREAM,
) -> None:
    """Write voiceprints.npz directly with controlled timestamps."""
    metadata = [
        {
            "day": "20260101",
            "segment_key": "090000_300",
            "source": "mic_audio",
            "stream": stream,
            "sentence_id": i + 1,
            "added_at": ts,
        }
        for i, ts in enumerate(added_ats)
    ]
    save_voiceprints_safely(
        npz_path=entity_dir / "voiceprints.npz",
        embeddings=np.stack(embeddings).astype(np.float32),
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# 1. Temporal decay
# ---------------------------------------------------------------------------


def test_temporal_decay_weights_recent_voiceprints(speakers_env):
    """Recent voiceprints should dominate even when outnumbered by old ones.

    Setup:
      - Alice has 10 voiceprints from 365 days ago at dir_old = [0,1,0,...]
      - Alice has 2 voiceprints from today     at dir_new = [0,0,1,...]
      - Test sentence embedding is exactly at dir_new

    Without decay:
      centroid = normalize(10*dir_old + 2*dir_new) ≈ [0, 0.981, 0.196]
      cosine with dir_new = 0.196 < ACOUSTIC_MEDIUM (0.22) → sentence unmatched

    With decay (half-life 120d, age=365d → weight≈0.121):
      weighted centroid ≈ [0, 0.519, 0.854]
      cosine with dir_new = 0.854 > ACOUSTIC_HIGH (0.36) → high-confidence attribution
    """
    env = speakers_env()
    _setup_owner(env)
    alice_dir = env.create_entity("Alice Test")

    dir_old = _unit([0.0, 1.0])  # orthogonal to owner [1,0,...] and dir_new
    dir_new = _unit([0.0, 0.0, 1.0])  # target direction

    now_ts = int(time.time() * 1000)
    old_ts = int((time.time() - 365 * 86400) * 1000)

    _write_voiceprints(
        alice_dir,
        embeddings=[dir_old] * 10 + [dir_new] * 2,
        added_ats=[old_ts] * 10 + [now_ts] * 2,
    )

    # Confirm the decay math: unweighted centroid would score below ACOUSTIC_MEDIUM
    unweighted = np.mean(np.stack([dir_old] * 10 + [dir_new] * 2), axis=0)
    unweighted /= np.linalg.norm(unweighted)
    assert float(np.dot(unweighted, dir_new)) < ACOUSTIC_MEDIUM, (
        "test setup: unweighted centroid must score below ACOUSTIC_MEDIUM"
    )

    # Confirm decay gives a centroid that scores above ACOUSTIC_HIGH
    import math

    weights = np.array(
        [math.exp(-_VP_DECAY_LAMBDA * 365)] * 10 + [1.0] * 2, dtype=np.float32
    )
    embs = np.stack([dir_old] * 10 + [dir_new] * 2)
    weighted_mean = np.dot(weights, embs) / weights.sum()
    weighted_mean /= np.linalg.norm(weighted_mean)
    assert float(np.dot(weighted_mean, dir_new)) > ACOUSTIC_HIGH, (
        "test setup: decay-weighted centroid must score above ACOUSTIC_HIGH"
    )

    # Create segment with a single sentence at dir_new
    env.create_segment(
        "20260101",
        "090000_300",
        ["mic_audio"],
        stream=STREAM,
        embeddings=dir_new.reshape(1, -1),
    )

    result = attribute_segment("20260101", STREAM, "090000_300")

    attributed = [lbl for lbl in result["labels"] if lbl.get("speaker") == "alice_test"]
    assert len(attributed) == 1, "sentence should be attributed to Alice via decay"
    assert attributed[0]["confidence"] == "high"
    assert attributed[0]["sentence_id"] not in result["unmatched"]


# ---------------------------------------------------------------------------
# 2. Outlier rejection
# ---------------------------------------------------------------------------


def test_outlier_rejection_blocks_inconsistent_embeddings(speakers_env):
    """Embeddings that disagree with the existing centroid should be rejected.

    Setup:
      - Alice has 5 existing voiceprints at dir_a = [0,1,0,...] (centroid = dir_a)
      - We try to accumulate 3 new embeddings:
          sid 1: at dir_a          (cosine ≈ 1.0) → should be saved
          sid 2: orthogonal to a   (cosine = 0.0) → outlier, should be rejected
          sid 3: opposite to dir_a (cosine = -1.0) → outlier, should be rejected
    """
    env = speakers_env()
    _setup_owner(env)
    alice_dir = env.create_entity("Alice Test")

    dir_a = _unit([0.0, 1.0])
    dir_ortho = _unit([0.0, 0.0, 1.0])
    dir_opp = _unit([0.0, -1.0])

    now_ts = int(time.time() * 1000)
    _write_voiceprints(
        alice_dir,
        embeddings=[dir_a] * _VP_OUTLIER_MIN_SAMPLES,
        added_ats=[now_ts] * _VP_OUTLIER_MIN_SAMPLES,
    )

    # All three new embeddings use a different segment so idempotency won't block them
    env.create_segment(
        "20260102",
        "090000_300",
        ["mic_audio"],
        stream=STREAM,
        embeddings=np.stack([dir_a, dir_ortho, dir_opp]),
    )

    labels = [
        {
            "sentence_id": 1,
            "speaker": "alice_test",
            "confidence": "high",
            "method": "structural_single_speaker",
        },
        {
            "sentence_id": 2,
            "speaker": "alice_test",
            "confidence": "high",
            "method": "structural_single_speaker",
        },
        {
            "sentence_id": 3,
            "speaker": "alice_test",
            "confidence": "high",
            "method": "structural_single_speaker",
        },
    ]

    # Verify test data: outlier check should apply (entity has enough samples)
    assert float(np.dot(dir_a, dir_a)) >= _VP_OUTLIER_MIN_SIMILARITY
    assert float(np.dot(dir_ortho, dir_a)) < _VP_OUTLIER_MIN_SIMILARITY
    assert float(np.dot(dir_opp, dir_a)) < _VP_OUTLIER_MIN_SIMILARITY

    saved = accumulate_voiceprints(
        "20260102", STREAM, "090000_300", labels, "mic_audio"
    )

    assert "alice_test" in saved
    assert saved["alice_test"] == 1, "only the consistent embedding should be saved"

    vp_path = alice_dir / "voiceprints.npz"
    data = np.load(vp_path, allow_pickle=False)
    assert len(data["embeddings"]) == _VP_OUTLIER_MIN_SAMPLES + 1


def test_outlier_rejection_not_applied_below_min_samples(speakers_env):
    """Outlier check is skipped when entity has < _VP_OUTLIER_MIN_SAMPLES voiceprints."""
    env = speakers_env()
    _setup_owner(env)
    alice_dir = env.create_entity("Alice Test")

    dir_a = _unit([0.0, 1.0])
    dir_ortho = _unit([0.0, 0.0, 1.0])

    # Only 3 voiceprints — below minimum to trigger outlier check
    assert _VP_OUTLIER_MIN_SAMPLES > 3
    now_ts = int(time.time() * 1000)
    _write_voiceprints(
        alice_dir,
        embeddings=[dir_a] * 3,
        added_ats=[now_ts] * 3,
    )

    env.create_segment(
        "20260102",
        "090000_300",
        ["mic_audio"],
        stream=STREAM,
        embeddings=dir_ortho.reshape(1, -1),
    )

    labels = [
        {
            "sentence_id": 1,
            "speaker": "alice_test",
            "confidence": "high",
            "method": "structural_single_speaker",
        },
    ]

    saved = accumulate_voiceprints(
        "20260102", STREAM, "090000_300", labels, "mic_audio"
    )

    # Outlier check not applied → embedding accepted despite being orthogonal
    assert saved.get("alice_test") == 1


# ---------------------------------------------------------------------------
# 3. Retroactive confirmation
# ---------------------------------------------------------------------------


def _write_labeled_segment(
    seg_dir: Path,
    source: str,
    cluster_label: int,
    embeddings: np.ndarray,
) -> None:
    """Write JSONL + NPZ with integer cluster labels for retroactive_confirm tests."""
    lines = [json.dumps({"raw": f"{source}.flac", "model": "parakeet"})]
    for i in range(len(embeddings)):
        lines.append(
            json.dumps(
                {
                    "start": "09:00:00",
                    "text": f"sentence {i + 1}",
                    "speaker": cluster_label,
                }
            )
        )
    (seg_dir / f"{source}.jsonl").write_text("\n".join(lines) + "\n")
    np.savez_compressed(
        seg_dir / f"{source}.npz",
        embeddings=embeddings.astype(np.float32),
        statement_ids=np.arange(1, len(embeddings) + 1, dtype=np.int32),
        durations_s=np.full(len(embeddings), 3.0, dtype=np.float32),
    )


def test_retroactive_confirm_backfills_voiceprints(speakers_env, tmp_path):
    """Confirming a candidate should save its source embeddings to the entity."""
    from apps.speakers.candidate_tracker import CandidateTracker

    env = speakers_env()

    rng = np.random.default_rng(42)
    base = _unit(rng.normal(0, 1, 256).astype(np.float32).tolist()[:256])

    # Create source segment with 4 cluster-1 embeddings
    seg_dir = env.journal / "chronicle" / "20260101" / "field.audio" / "090000_300"
    seg_dir.mkdir(parents=True)
    embs = np.stack(
        [base + rng.normal(0, 0.02, 256).astype(np.float32) for _ in range(4)]
    )
    embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)
    _write_labeled_segment(seg_dir, "audio", cluster_label=1, embeddings=embs)

    # Build a candidate that points to this segment
    store = tmp_path / "cands.json"
    tracker = CandidateTracker(store_path=store)
    tracker.load()
    tracker.process_segment(seg_dir, "audio", "20260101", "090000_300")
    cand_id = list(tracker._candidates.keys())[0]

    # Confirm the candidate as entity "alice_test"
    alice_dir = env.create_entity("Alice Test")
    tracker.confirm(cand_id, "alice_test")

    result = tracker.retroactive_confirm(
        cand_id, "alice_test", journal_root=env.journal
    )

    assert result["status"] == "ok"
    assert result["n_added"] > 0
    assert result["n_skipped_missing"] == 0

    # Verify voiceprints were written to the entity
    vp_path = alice_dir / "voiceprints.npz"
    assert vp_path.exists()
    data = np.load(vp_path, allow_pickle=False)
    assert len(data["embeddings"]) == result["n_added"]


def test_retroactive_confirm_applies_outlier_guard(speakers_env, tmp_path):
    """Embeddings that disagree with the entity's existing centroid are skipped."""
    from apps.speakers.candidate_tracker import CandidateTracker

    env = speakers_env()

    dir_a = _unit([0.0, 1.0])
    dir_b = _unit([0.0, 0.0, 1.0])  # orthogonal to dir_a

    # Candidate's source segments contain embeddings at dir_b
    seg_dir = env.journal / "chronicle" / "20260101" / "field.audio" / "090000_300"
    seg_dir.mkdir(parents=True)
    _write_labeled_segment(
        seg_dir,
        "audio",
        cluster_label=1,
        embeddings=np.stack([dir_b] * 4),
    )

    store = tmp_path / "cands.json"
    tracker = CandidateTracker(store_path=store)
    tracker.load()
    tracker.process_segment(seg_dir, "audio", "20260101", "090000_300")
    cand_id = list(tracker._candidates.keys())[0]

    # Entity has existing voiceprints at dir_a (orthogonal → outlier for dir_b)
    alice_dir = env.create_entity("Alice Test")
    now_ts = int(time.time() * 1000)
    _write_voiceprints(
        alice_dir,
        embeddings=[dir_a] * _VP_OUTLIER_MIN_SAMPLES,
        added_ats=[now_ts] * _VP_OUTLIER_MIN_SAMPLES,
    )

    tracker.confirm(cand_id, "alice_test")
    result = tracker.retroactive_confirm(
        cand_id, "alice_test", journal_root=env.journal
    )

    assert result["status"] == "ok"
    # All candidate embeddings are orthogonal to the entity centroid → all rejected
    assert result["n_added"] == 0
    assert result["n_skipped_outlier"] > 0


def test_retroactive_confirm_skips_missing_segment(speakers_env, tmp_path):
    """Missing source segments are counted and don't crash retroactive_confirm."""
    from apps.speakers.candidate_tracker import CandidateProfile, CandidateTracker

    env = speakers_env()
    env.create_entity("Alice Test")

    store = tmp_path / "cands.json"
    tracker = CandidateTracker(store_path=store)
    # Manually insert a candidate pointing to a non-existent segment

    cand_id = "cand_0"
    cand = CandidateProfile(
        cand_id=cand_id,
        centroid=[0.0] * 256,
        n_segments=1,
        n_intervals=3,
        total_duration_s=30.0,
        source_segments=[
            {
                "day": "20260199",  # doesn't exist
                "segment": "090000_300",
                "cluster_label": 1,
                "n_intervals": 3,
                "duration_s": 9.0,
            }
        ],
        confirmed_entity="alice_test",
        status="confirmed",
    )
    tracker._candidates[cand_id] = cand
    tracker._next_id = 1

    result = tracker.retroactive_confirm(
        cand_id, "alice_test", journal_root=env.journal
    )

    assert result["status"] == "ok"
    assert result["n_added"] == 0
    assert result["n_skipped_missing"] == 1
