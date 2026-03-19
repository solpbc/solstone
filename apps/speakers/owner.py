# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Owner voice identification helpers for the speakers app."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import HDBSCAN

from think.awareness import update_state
from think.entities.journal import get_journal_principal, journal_entity_memory_path
from think.utils import day_dirs, get_journal, segment_path

logger = logging.getLogger(__name__)

MAX_EMBEDDINGS = 30000
OWNER_THRESHOLD = 0.82


def _mark_no_cluster(segment_count: int) -> None:
    """Record that detection ran but did not produce a usable cluster."""
    update_state(
        "voiceprint",
        {
            "status": "no_cluster",
            "segments_checked": segment_count,
            "attempted_at": _iso_now(),
        },
    )


def _routes_helpers():
    """Load speakers route helpers lazily to avoid import cycles."""
    from apps.speakers.routes import (
        _load_embeddings_file,
        _normalize_embedding,
        _scan_segment_embeddings,
    )

    return _load_embeddings_file, _normalize_embedding, _scan_segment_embeddings


def _owner_candidate_path() -> Path:
    """Return the temporary owner candidate NPZ path."""
    awareness_dir = Path(get_journal()) / "awareness"
    awareness_dir.mkdir(parents=True, exist_ok=True)
    return awareness_dir / "owner_candidate.npz"


def _iso_now() -> str:
    """Return a timestamp string for persisted metadata."""
    return datetime.now().isoformat()


def _audio_url(day: str, stream: str, segment_key: str, source: str) -> str:
    """Build the existing speakers audio-serving URL for a sample."""
    return f"/app/speakers/api/serve_audio/{day}/{stream}__{segment_key}__{source}.flac"


def count_segments_with_embeddings() -> int:
    """Count all journal segments that contain audio embedding files."""
    _, _, scan_segment_embeddings = _routes_helpers()

    total = 0
    for day in day_dirs().keys():
        total += len(scan_segment_embeddings(day))
    return total


def _subsample_embeddings(
    embeddings: np.ndarray, provenance: list[dict[str, Any]]
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Subsample embeddings proportionally across streams when over the limit."""
    total = len(embeddings)
    if total <= MAX_EMBEDDINGS:
        return embeddings, provenance

    rng = np.random.default_rng(42)
    stream_indices: dict[str, list[int]] = {}
    for idx, record in enumerate(provenance):
        stream_indices.setdefault(record["stream"], []).append(idx)

    allocations: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    allocated = 0

    for stream, indices in stream_indices.items():
        count = len(indices)
        proportional = MAX_EMBEDDINGS * count / total
        allocation = min(count, int(proportional))
        allocations[stream] = allocation
        allocated += allocation
        remainders.append((proportional - allocation, stream))

    remaining = MAX_EMBEDDINGS - allocated
    for _, stream in sorted(remainders, reverse=True):
        if remaining <= 0:
            break
        available = len(stream_indices[stream]) - allocations[stream]
        if available <= 0:
            continue
        allocations[stream] += 1
        remaining -= 1

    selected_indices: list[int] = []
    for stream, indices in stream_indices.items():
        take = allocations[stream]
        if take <= 0:
            continue
        if take >= len(indices):
            selected_indices.extend(indices)
            continue
        sampled = rng.choice(indices, size=take, replace=False)
        selected_indices.extend(int(idx) for idx in sampled)

    selected_indices.sort()
    sampled_embeddings = embeddings[selected_indices]
    sampled_provenance = [provenance[idx] for idx in selected_indices]
    return sampled_embeddings, sampled_provenance


def detect_owner_candidate() -> dict[str, Any]:
    """Detect a likely owner voice centroid from journal embeddings."""
    load_embeddings_file, normalize_embedding, scan_segment_embeddings = (
        _routes_helpers()
    )

    segment_count = count_segments_with_embeddings()

    embedding_chunks: list[np.ndarray] = []
    provenance: list[dict[str, Any]] = []

    for day in day_dirs().keys():
        for segment in scan_segment_embeddings(day):
            stream = segment["stream"]
            segment_key = segment["key"]
            segment_dir = segment_path(day, segment_key, stream)

            for source in segment["sources"]:
                emb_data = load_embeddings_file(segment_dir / f"{source}.npz")
                if emb_data is None:
                    continue

                embeddings, statement_ids = emb_data
                if len(embeddings) == 0:
                    continue

                embedding_chunks.append(embeddings.astype(np.float32))
                provenance.extend(
                    {
                        "day": day,
                        "stream": stream,
                        "segment_key": segment_key,
                        "source": source,
                        "sentence_id": int(sid),
                    }
                    for sid in statement_ids
                )

    if not embedding_chunks:
        _mark_no_cluster(segment_count)
        return {
            "status": "no_embeddings",
            "segments_available": segment_count,
            "embeddings_available": 0,
            "recommendation": "no_embeddings",
        }

    embeddings_matrix = np.vstack(embedding_chunks)
    embeddings_matrix, provenance = _subsample_embeddings(embeddings_matrix, provenance)

    if len(embeddings_matrix) < 50:
        _mark_no_cluster(segment_count)
        return {
            "status": "low_data",
            "segments_available": segment_count,
            "embeddings_available": int(len(embeddings_matrix)),
            "recommendation": "low_data",
        }

    clusterer = HDBSCAN(
        min_cluster_size=50,
        min_samples=10,
        metric="euclidean",
    )
    clusterer.fit(embeddings_matrix)
    labels = clusterer.labels_

    valid_labels = labels[labels != -1]
    if len(valid_labels) == 0:
        _mark_no_cluster(segment_count)
        return {
            "status": "no_clusters",
            "segments_available": segment_count,
            "embeddings_available": int(len(embeddings_matrix)),
            "recommendation": "no_clusters",
        }

    largest_label = int(np.bincount(valid_labels).argmax())
    cluster_indices = np.flatnonzero(labels == largest_label)
    if len(cluster_indices) == 0:
        _mark_no_cluster(segment_count)
        return {
            "status": "no_clusters",
            "segments_available": segment_count,
            "embeddings_available": int(len(embeddings_matrix)),
            "recommendation": "no_clusters",
        }

    cluster_embeddings = embeddings_matrix[cluster_indices]
    centroid = normalize_embedding(np.mean(cluster_embeddings, axis=0))
    if centroid is None:
        _mark_no_cluster(segment_count)
        return {
            "status": "no_clusters",
            "segments_available": segment_count,
            "embeddings_available": int(len(embeddings_matrix)),
            "recommendation": "no_clusters",
        }

    cluster_size = int(len(cluster_indices))
    cluster_streams = {provenance[int(i)]["stream"] for i in cluster_indices}
    streams_represented = len(cluster_streams)
    recommendation = "ready" if streams_represented > 1 else "single_stream"
    similarities = np.dot(cluster_embeddings, centroid)
    sorted_cluster_positions = np.argsort(similarities)[::-1]

    samples: list[dict[str, Any]] = []
    seen_segments: set[tuple[str, str, str]] = set()

    for position in sorted_cluster_positions:
        record = provenance[int(cluster_indices[position])]
        segment_triplet = (record["day"], record["stream"], record["segment_key"])
        if segment_triplet in seen_segments:
            continue
        seen_segments.add(segment_triplet)
        samples.append(
            {
                **record,
                "audio_url": _audio_url(
                    record["day"],
                    record["stream"],
                    record["segment_key"],
                    record["source"],
                ),
            }
        )
        if len(samples) == 3:
            break

    if len(samples) < 3:
        for position in sorted_cluster_positions:
            record = provenance[int(cluster_indices[position])]
            sample = {
                **record,
                "audio_url": _audio_url(
                    record["day"],
                    record["stream"],
                    record["segment_key"],
                    record["source"],
                ),
            }
            if sample in samples:
                continue
            samples.append(sample)
            if len(samples) == 3:
                break

    version = _iso_now()
    np.savez_compressed(
        _owner_candidate_path(),
        centroid=centroid.astype(np.float32),
        cluster_size=np.array(cluster_size, dtype=np.int32),
        threshold=np.array(OWNER_THRESHOLD, dtype=np.float32),
        version=np.array(version),
    )

    update_state(
        "voiceprint",
        {
            "status": "candidate",
            "cluster_size": cluster_size,
            "streams_represented": streams_represented,
            "recommendation": recommendation,
            "samples": samples,
            "detected_at": version,
        },
    )

    return {
        "status": "candidate",
        "cluster_size": cluster_size,
        "streams_represented": streams_represented,
        "recommendation": recommendation,
        "samples": samples,
    }


def load_owner_centroid() -> tuple[np.ndarray, float] | None:
    """Load the confirmed owner centroid and threshold for the principal entity."""
    principal = get_journal_principal()
    if not principal:
        return None

    centroid_path = journal_entity_memory_path(principal["id"]) / "owner_centroid.npz"
    if not centroid_path.exists():
        return None

    try:
        data = np.load(centroid_path, allow_pickle=False)
        centroid = data.get("centroid")
        threshold = data.get("threshold")
        if centroid is None or threshold is None:
            return None

        normalized = centroid.astype(np.float32).reshape(-1)
        norm = np.linalg.norm(normalized)
        if norm == 0:
            return None
        normalized = normalized / norm
        return normalized, float(np.asarray(threshold).item())
    except Exception as exc:
        logger.warning("Failed to load owner centroid %s: %s", centroid_path, exc)
        return None


def classify_sentences(
    day: str, stream: str, segment_key: str, source: str
) -> list[dict[str, Any]]:
    """Classify segment sentences against the confirmed owner centroid."""
    load_embeddings_file, normalize_embedding, _ = _routes_helpers()

    centroid_data = load_owner_centroid()
    if centroid_data is None:
        return []

    centroid, threshold = centroid_data
    emb_data = load_embeddings_file(
        segment_path(day, segment_key, stream) / f"{source}.npz"
    )
    if emb_data is None:
        return []

    embeddings, statement_ids = emb_data
    results = []
    for embedding, statement_id in zip(embeddings, statement_ids):
        normalized = normalize_embedding(embedding)
        if normalized is None:
            continue
        score = float(np.dot(normalized, centroid))
        results.append(
            {
                "sentence_id": int(statement_id),
                "is_owner": score >= threshold,
                "score": round(score, 4),
            }
        )
    return results
