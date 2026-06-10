# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for CandidateTracker in apps/speakers/candidate_tracker.py."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from apps.speakers.candidate_tracker import (
    CONFIRM_MIN_DURATION_S,
    CONFIRM_MIN_INTERVALS,
    CONFIRM_MIN_SEGMENTS,
    MERGE_THRESHOLD,
    SPLIT_THRESHOLD,
    STABILITY_THRESHOLD,
    CandidateTracker,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def _make_tight_cluster(base: np.ndarray, n: int, noise: float = 0.02) -> np.ndarray:
    """n embeddings close to base (low std, passes stability check)."""
    rng = np.random.default_rng(0)
    embs = base[None] + rng.normal(0, noise, (n, 256)).astype(np.float32)
    return np.stack([_unit(e) for e in embs])


def _make_noisy_cluster(n: int) -> np.ndarray:
    """n embeddings from two distinct directions (mixed-speaker, fails stability check)."""
    rng = np.random.default_rng(1)
    a = _unit(rng.normal(0, 1, 256).astype(np.float32))
    b = _unit(rng.normal(0, 1, 256).astype(np.float32))
    # Ensure a and b are actually far apart
    b = _unit(b - np.dot(b, a) * a)  # make b orthogonal to a
    half = n // 2
    embs_a = [
        _unit(a + rng.normal(0, 0.02, 256).astype(np.float32)) for _ in range(half)
    ]
    embs_b = [
        _unit(b + rng.normal(0, 0.02, 256).astype(np.float32)) for _ in range(n - half)
    ]
    return np.stack(embs_a + embs_b)


def _write_labeled_segment(
    seg_dir: Path,
    source: str,
    cluster_embeddings: dict[int, np.ndarray],  # cluster_label → (n, 256) array
) -> None:
    """Write JSONL + NPZ for a segment with integer speaker cluster labels."""
    all_embs = []
    all_sids = []
    all_durs = []
    jsonl_lines = [json.dumps({"raw": f"{source}.flac", "model": "parakeet"})]

    sid = 1
    for cluster_label, embs in cluster_embeddings.items():
        for emb in embs:
            all_embs.append(emb)
            all_sids.append(sid)
            all_durs.append(3.0)  # 3s per sentence
            jsonl_lines.append(
                json.dumps(
                    {"start": "09:00:00", "text": "test", "speaker": cluster_label}
                )
            )
            sid += 1

    (seg_dir / f"{source}.jsonl").write_text("\n".join(jsonl_lines) + "\n")
    np.savez_compressed(
        seg_dir / f"{source}.npz",
        embeddings=np.stack(all_embs).astype(np.float32),
        statement_ids=np.array(all_sids, dtype=np.int32),
        durations_s=np.array(all_durs, dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_missing_files(speakers_env, tmp_path):
    env = speakers_env()
    seg_dir = env.journal / "20260101" / "audio" / "090000_300"
    seg_dir.mkdir(parents=True)

    tracker = CandidateTracker(store_path=tmp_path / "cands.json")
    result = tracker.process_segment(seg_dir, "audio", "20260101", "090000_300")
    assert result["status"] == "missing_files"


def test_no_speaker_labels(speakers_env, tmp_path):
    env = speakers_env()
    seg_dir = env.create_segment("20260101", "090000_300", ["audio"])

    tracker = CandidateTracker(store_path=tmp_path / "cands.json")
    result = tracker.process_segment(seg_dir, "audio", "20260101", "090000_300")
    assert result["status"] == "no_clusters"


def test_creates_candidates_from_labeled_segment(speakers_env, tmp_path):
    env = speakers_env()
    seg_dir = env.journal / "chronicle" / "20260101" / "audio" / "090000_300"
    seg_dir.mkdir(parents=True)

    rng = np.random.default_rng(42)
    base_a = _unit(rng.normal(0, 1, 256).astype(np.float32))
    base_b = _unit(rng.normal(0, 1, 256).astype(np.float32))

    _write_labeled_segment(
        seg_dir,
        "audio",
        {
            1: _make_tight_cluster(base_a, 4),
            2: _make_tight_cluster(base_b, 4),
        },
    )

    tracker = CandidateTracker(store_path=tmp_path / "cands.json")
    result = tracker.process_segment(seg_dir, "audio", "20260101", "090000_300")

    assert result["status"] == "ok"
    assert result["n_clusters_found"] == 2
    assert result["n_new"] == 2
    assert result["n_merges"] == 0
    assert result["total_candidates"] == 2


def test_stability_check_rejects_noisy_cluster(speakers_env, tmp_path):
    env = speakers_env()
    seg_dir = env.journal / "chronicle" / "20260101" / "audio" / "090000_300"
    seg_dir.mkdir(parents=True)

    rng = np.random.default_rng(42)
    base_a = _unit(rng.normal(0, 1, 256).astype(np.float32))

    _write_labeled_segment(
        seg_dir,
        "audio",
        {
            1: _make_tight_cluster(base_a, 4),  # stable — should pass
            2: _make_noisy_cluster(6),  # noisy — should be rejected
        },
    )

    # Confirm cluster 2 actually fails the stability threshold
    noisy = _make_noisy_cluster(6)
    stacked = noisy
    c = stacked.mean(axis=0)
    c = c / np.linalg.norm(c)
    stability = float(np.mean(1.0 - stacked @ c))
    assert stability >= STABILITY_THRESHOLD, (
        f"test data must actually be noisy: {stability:.4f}"
    )

    tracker = CandidateTracker(store_path=tmp_path / "cands.json")
    result = tracker.process_segment(seg_dir, "audio", "20260101", "090000_300")

    assert result["status"] == "ok"
    assert result["n_unstable"] == 1
    assert result["n_new"] == 1
    assert result["total_candidates"] == 1


def test_merge_same_speaker_across_segments(speakers_env, tmp_path):
    """Same speaker (similar embeddings) across two segments → one merged candidate."""
    env = speakers_env()
    rng = np.random.default_rng(7)
    base = _unit(rng.normal(0, 1, 256).astype(np.float32))

    store = tmp_path / "cands.json"

    for day, seg_key in [("20260101", "090000_300"), ("20260102", "090000_300")]:
        seg_dir = env.journal / "chronicle" / day / "audio" / seg_key
        seg_dir.mkdir(parents=True)
        _write_labeled_segment(
            seg_dir,
            "audio",
            {
                1: _make_tight_cluster(base, 4, noise=0.01),
            },
        )

        tracker = CandidateTracker(store_path=store)
        tracker.load()
        result = tracker.process_segment(seg_dir, "audio", day, seg_key)
        assert result["status"] == "ok"

    tracker = CandidateTracker(store_path=store)
    tracker.load()
    s = tracker.summary()
    # Should have merged into one candidate, not created two
    assert s["total_candidates"] == 1
    cand = list(tracker._candidates.values())[0]
    assert cand.n_segments == 2


def test_creates_separate_candidates_for_distinct_speakers(speakers_env, tmp_path):
    """Distinct speakers (low similarity) across segments → separate candidates."""
    env = speakers_env()
    rng = np.random.default_rng(99)
    store = tmp_path / "cands.json"

    bases = [_unit(rng.normal(0, 1, 256).astype(np.float32)) for _ in range(2)]
    # Ensure the two bases are actually below SPLIT_THRESHOLD
    sim = float(np.dot(bases[0], bases[1]))
    assert sim < SPLIT_THRESHOLD, f"test bases too similar: {sim:.3f}"

    for i, (day, seg_key) in enumerate(
        [
            ("20260101", "090000_300"),
            ("20260102", "090000_300"),
        ]
    ):
        seg_dir = env.journal / "chronicle" / day / "audio" / seg_key
        seg_dir.mkdir(parents=True)
        _write_labeled_segment(
            seg_dir,
            "audio",
            {
                1: _make_tight_cluster(bases[i], 4),
            },
        )
        tracker = CandidateTracker(store_path=store)
        tracker.load()
        tracker.process_segment(seg_dir, "audio", day, seg_key)

    tracker = CandidateTracker(store_path=store)
    tracker.load()
    assert tracker.summary()["total_candidates"] == 2


def test_ambiguous_score_not_merged_or_created(speakers_env, tmp_path):
    """Embedding with similarity between SPLIT and MERGE thresholds → ambiguous."""
    rng = np.random.default_rng(5)
    base_existing = _unit(rng.normal(0, 1, 256).astype(np.float32))

    # Build an ambiguous vector: target sim between SPLIT_THRESHOLD and MERGE_THRESHOLD
    target_sim = (SPLIT_THRESHOLD + MERGE_THRESHOLD) / 2
    perp = _unit(rng.normal(0, 1, 256).astype(np.float32))
    perp = _unit(perp - np.dot(perp, base_existing) * base_existing)
    ambiguous_base = _unit(
        target_sim * base_existing + np.sqrt(1 - target_sim**2) * perp
    )

    actual_sim = float(np.dot(base_existing, ambiguous_base))
    assert SPLIT_THRESHOLD < actual_sim < MERGE_THRESHOLD, (
        f"ambiguous_base not in ambiguous zone: {actual_sim:.3f}"
    )

    env = speakers_env()
    store = tmp_path / "cands.json"

    # Segment 1: seed an existing candidate from base_existing
    seg1 = env.journal / "chronicle" / "20260101" / "audio" / "090000_300"
    seg1.mkdir(parents=True)
    _write_labeled_segment(seg1, "audio", {1: _make_tight_cluster(base_existing, 4)})
    tracker = CandidateTracker(store_path=store)
    tracker.load()
    tracker.process_segment(seg1, "audio", "20260101", "090000_300")

    # Segment 2: present the ambiguous cluster
    seg2 = env.journal / "chronicle" / "20260102" / "audio" / "090000_300"
    seg2.mkdir(parents=True)
    _write_labeled_segment(seg2, "audio", {1: _make_tight_cluster(ambiguous_base, 4)})
    tracker = CandidateTracker(store_path=store)
    tracker.load()
    result = tracker.process_segment(seg2, "audio", "20260102", "090000_300")

    assert result["n_ambiguous"] == 1
    assert result["n_merges"] == 0
    assert result["n_new"] == 0


def test_confirmation_queue_empty_below_thresholds(speakers_env, tmp_path):
    env = speakers_env()
    seg_dir = env.journal / "chronicle" / "20260101" / "audio" / "090000_300"
    seg_dir.mkdir(parents=True)

    rng = np.random.default_rng(3)
    base = _unit(rng.normal(0, 1, 256).astype(np.float32))
    _write_labeled_segment(seg_dir, "audio", {1: _make_tight_cluster(base, 3)})

    tracker = CandidateTracker(store_path=tmp_path / "cands.json")
    tracker.process_segment(seg_dir, "audio", "20260101", "090000_300")

    # After one segment with 3 intervals (9s total) — not enough for confirmation
    assert tracker.get_confirmation_queue() == []


def test_confirmation_queue_ready_after_sufficient_evidence(speakers_env, tmp_path):
    env = speakers_env()
    rng = np.random.default_rng(13)
    base = _unit(rng.normal(0, 1, 256).astype(np.float32))
    store = tmp_path / "cands.json"

    for i in range(CONFIRM_MIN_SEGMENTS + 1):
        day = f"2026010{i + 1}"
        seg_key = "090000_300"
        seg_dir = env.journal / "chronicle" / day / "audio" / seg_key
        seg_dir.mkdir(parents=True)
        # Each segment contributes CONFIRM_MIN_INTERVALS intervals at 3s each
        _write_labeled_segment(
            seg_dir,
            "audio",
            {
                1: _make_tight_cluster(base, CONFIRM_MIN_INTERVALS, noise=0.01),
            },
        )
        tracker = CandidateTracker(store_path=store)
        tracker.load()
        tracker.process_segment(seg_dir, "audio", day, seg_key)

    tracker = CandidateTracker(store_path=store)
    tracker.load()
    queue = tracker.get_confirmation_queue()
    assert len(queue) == 1
    cand = queue[0]
    assert cand.n_segments >= CONFIRM_MIN_SEGMENTS
    assert cand.n_intervals >= CONFIRM_MIN_INTERVALS
    assert cand.total_duration_s >= CONFIRM_MIN_DURATION_S


def test_persist_and_reload(speakers_env, tmp_path):
    env = speakers_env()
    seg_dir = env.journal / "chronicle" / "20260101" / "audio" / "090000_300"
    seg_dir.mkdir(parents=True)

    rng = np.random.default_rng(21)
    base = _unit(rng.normal(0, 1, 256).astype(np.float32))
    _write_labeled_segment(seg_dir, "audio", {1: _make_tight_cluster(base, 4)})

    store = tmp_path / "cands.json"
    tracker = CandidateTracker(store_path=store)
    tracker.process_segment(seg_dir, "audio", "20260101", "090000_300")

    reloaded = CandidateTracker(store_path=store)
    reloaded.load()
    assert reloaded.summary()["total_candidates"] == 1
    cand = list(reloaded._candidates.values())[0]
    assert cand.n_intervals == 4


def test_confirm_and_reject(speakers_env, tmp_path):
    env = speakers_env()
    rng = np.random.default_rng(77)
    store = tmp_path / "cands.json"

    for i in range(2):
        base = _unit(rng.normal(0, 1, 256).astype(np.float32))
        seg_dir = env.journal / "chronicle" / f"2026010{i + 1}" / "audio" / "090000_300"
        seg_dir.mkdir(parents=True)
        _write_labeled_segment(seg_dir, "audio", {i + 1: _make_tight_cluster(base, 4)})
        tracker = CandidateTracker(store_path=store)
        tracker.load()
        tracker.process_segment(seg_dir, "audio", f"2026010{i + 1}", "090000_300")

    tracker = CandidateTracker(store_path=store)
    tracker.load()
    cand_ids = list(tracker._candidates.keys())
    assert len(cand_ids) == 2

    tracker.confirm(cand_ids[0], "alice")
    tracker.reject(cand_ids[1])

    reloaded = CandidateTracker(store_path=store)
    reloaded.load()
    s = reloaded.summary()
    assert s["confirmed"] == 1
    assert s["rejected"] == 1
    assert reloaded._candidates[cand_ids[0]].confirmed_entity == "alice"
    assert reloaded._candidates[cand_ids[1]].status == "rejected"
