# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Owner voice identification helpers for the speakers app."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import HDBSCAN

from apps.speakers.encoder_config import (
    OWNER_BOOTSTRAP_MIN_INTRA_COSINE_P25,
    OWNER_BOOTSTRAP_MIN_MEDIAN_DURATION_S,
    OWNER_BOOTSTRAP_MIN_STMTS,
    OWNER_THRESHOLD,
)
from think.awareness import update_state
from think.entities.journal import get_journal_principal, journal_entity_memory_path
from think.utils import day_dirs, get_journal, segment_path

logger = logging.getLogger(__name__)

MAX_EMBEDDINGS = 30000


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


def _mark_low_quality(
    reason: str, observed: float, threshold: float, segment_count: int
) -> None:
    """Record that detection found a cluster, but it failed quality gates."""
    update_state(
        "voiceprint",
        {
            "status": "low_quality",
            "low_quality_reason": reason,
            "observed_value": float(observed),
            "threshold_value": float(threshold),
            "segments_checked": int(segment_count),
            "attempted_at": _iso_now(),
        },
    )


def _bail_low_quality(
    reason: str,
    observed: float,
    threshold: float,
    segment_count: int,
    embeddings_count: int,
) -> dict[str, Any]:
    """Record and return a locked low-quality owner detection result."""
    _mark_low_quality(reason, observed, threshold, segment_count)
    return {
        "status": "low_quality",
        "recommendation": "low_quality",
        "segments_available": int(segment_count),
        "embeddings_available": int(embeddings_count),
        "low_quality_reason": reason,
        "observed_value": float(observed),
        "threshold_value": float(threshold),
    }


def _pairwise_cosines(embeddings: np.ndarray) -> np.ndarray:
    """Return pairwise cosine similarities for a cluster of embeddings."""
    n = embeddings.shape[0]
    if n < 2:
        return np.empty(0, dtype=np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms < 1e-9, 1.0, norms)
    e_norm = embeddings / norms
    if n > 5000:
        rng = np.random.default_rng(seed=0)
        i = rng.integers(0, n, size=1000)
        j = rng.integers(0, n, size=1000)
        mask = i != j
        i = i[mask]
        j = j[mask]
        return np.einsum("ij,ij->i", e_norm[i], e_norm[j]).astype(
            np.float32, copy=False
        )
    sim = e_norm @ e_norm.T
    iu = np.triu_indices(n, k=1)
    return sim[iu].astype(np.float32, copy=False)


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
    return f"/app/speakers/api/serve_audio/{day}/{stream}/{segment_key}/{source}.flac"


def _fallback_statement_durations(jsonl_path: Path) -> dict[int, float | None]:
    """Estimate statement durations from adjacent transcript start times."""
    if not jsonl_path.exists():
        return {}

    starts: list[tuple[int, int]] = []
    try:
        with open(jsonl_path, encoding="utf-8") as f:
            lines = f.readlines()
    except OSError:
        return {}

    for sentence_id, line in enumerate(lines[1:], start=1):
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        start = entry.get("start")
        if not isinstance(start, str):
            continue
        try:
            hours, minutes, seconds = (int(part) for part in start.split(":", 2))
        except ValueError:
            continue
        starts.append((sentence_id, hours * 3600 + minutes * 60 + seconds))

    durations: dict[int, float | None] = {}
    for idx, (sentence_id, start_seconds) in enumerate(starts):
        next_start = starts[idx + 1][1] if idx + 1 < len(starts) else None
        # Why: older transcript JSONL files only persist statement starts, so
        # we estimate legacy durations from adjacent sentence boundaries.
        durations[sentence_id] = (
            None if next_start is None else float(next_start - start_seconds)
        )
    return durations


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

                embeddings, statement_ids, durations_data = emb_data
                if len(embeddings) == 0:
                    continue

                fallback_durations = (
                    {}
                    if durations_data is not None
                    else _fallback_statement_durations(segment_dir / f"{source}.jsonl")
                )
                embedding_chunks.append(embeddings.astype(np.float32))
                provenance.extend(
                    {
                        "day": day,
                        "stream": stream,
                        "segment_key": segment_key,
                        "source": source,
                        "sentence_id": int(sid),
                        "duration_s": (
                            float(durations_data[idx])
                            if durations_data is not None
                            else fallback_durations.get(int(sid))
                        ),
                    }
                    for idx, sid in enumerate(statement_ids)
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
    cluster_size = int(len(cluster_indices))
    embeddings_count = int(embeddings_matrix.shape[0])

    if cluster_size < OWNER_BOOTSTRAP_MIN_STMTS:
        return _bail_low_quality(
            "too_few_stmts",
            observed=cluster_size,
            threshold=OWNER_BOOTSTRAP_MIN_STMTS,
            segment_count=segment_count,
            embeddings_count=embeddings_count,
        )

    cluster_durations = [
        provenance[int(i)]["duration_s"]
        for i in cluster_indices
        if provenance[int(i)].get("duration_s") is not None
    ]
    if not cluster_durations:
        median_duration = 0.0
    else:
        median_duration = float(np.median(cluster_durations))
    if median_duration < OWNER_BOOTSTRAP_MIN_MEDIAN_DURATION_S:
        return _bail_low_quality(
            "median_duration_too_short",
            observed=median_duration,
            threshold=OWNER_BOOTSTRAP_MIN_MEDIAN_DURATION_S,
            segment_count=segment_count,
            embeddings_count=embeddings_count,
        )

    intra_cosines = _pairwise_cosines(cluster_embeddings)
    if intra_cosines.size == 0:
        intra_p25 = 0.0
    else:
        intra_p25 = float(np.percentile(intra_cosines, 25))
    if intra_p25 < OWNER_BOOTSTRAP_MIN_INTRA_COSINE_P25:
        return _bail_low_quality(
            "cluster_too_diffuse",
            observed=intra_p25,
            threshold=OWNER_BOOTSTRAP_MIN_INTRA_COSINE_P25,
            segment_count=segment_count,
            embeddings_count=embeddings_count,
        )

    centroid = normalize_embedding(np.mean(cluster_embeddings, axis=0))
    if centroid is None:
        _mark_no_cluster(segment_count)
        return {
            "status": "no_clusters",
            "segments_available": segment_count,
            "embeddings_available": embeddings_count,
            "recommendation": "no_clusters",
        }

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
        segment_path(day, segment_key, stream, create=False) / f"{source}.npz"
    )
    if emb_data is None:
        return []

    embeddings, statement_ids, _ = emb_data
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


def confirm_owner_candidate() -> dict[str, Any]:
    """Confirm the current owner voice candidate and persist the centroid.

    Moves the candidate centroid from awareness/ to the principal entity's
    memory directory as owner_centroid.npz. Updates awareness state to
    "confirmed".

    Returns a dict with status and principal_id on success, or an error key.
    """
    from think.entities import entity_slug
    from think.entities.core import get_identity_names
    from think.entities.journal import (
        create_journal_entity,
        ensure_journal_entity_memory,
        load_journal_entity,
    )

    candidate_path = _owner_candidate_path()
    if not candidate_path.exists():
        return {"error": "No candidate available"}

    try:
        data = np.load(candidate_path, allow_pickle=False)
        centroid = data["centroid"]
        cluster_size = int(np.asarray(data["cluster_size"]).item())
        threshold = float(np.asarray(data["threshold"]).item())
        version = str(np.asarray(data["version"]).item())
    except Exception as e:
        logger.warning("Failed to load owner candidate %s: %s", candidate_path, e)
        return {"error": "No candidate available"}

    principal = get_journal_principal()
    if principal is None:
        identity_names = get_identity_names()
        if not identity_names:
            return {"error": "No principal entity found"}
        principal_name = identity_names[0]
        principal_id = entity_slug(principal_name)
        principal = load_journal_entity(principal_id) or create_journal_entity(
            entity_id=principal_id,
            name=principal_name,
            entity_type="Person",
        )

    owner_path = ensure_journal_entity_memory(principal["id"]) / "owner_centroid.npz"
    np.savez_compressed(
        owner_path,
        centroid=np.asarray(centroid, dtype=np.float32).reshape(-1),
        cluster_size=np.array(cluster_size, dtype=np.int32),
        threshold=np.array(threshold, dtype=np.float32),
        version=np.array(version),
    )
    candidate_path.unlink(missing_ok=True)

    update_state(
        "voiceprint",
        {
            "status": "confirmed",
            "cluster_size": cluster_size,
            "confirmed_at": _iso_now(),
        },
    )

    return {
        "status": "confirmed",
        "principal_id": principal["id"],
        "cluster_size": cluster_size,
    }


def reject_owner_candidate() -> dict[str, Any]:
    """Reject the current owner voice candidate and enter cooldown.

    Deletes the candidate file and records rejection with timestamp in
    awareness state. The timestamp enables 14-day cooldown enforcement.

    Returns a dict with the updated status.
    """
    candidate_path = _owner_candidate_path()
    candidate_path.unlink(missing_ok=True)
    update_state(
        "voiceprint",
        {
            "status": "rejected",
            "rejected_at": _iso_now(),
        },
    )
    return {"status": "rejected"}
