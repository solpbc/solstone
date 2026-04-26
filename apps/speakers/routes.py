# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Speaker voiceprint management app - sentence-based embeddings.

Voiceprints are stored at the journal level (not per-facet) since a person's
voice is the same regardless of which facet they appear in.
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
from flask import (
    Blueprint,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    url_for,
)

from apps.speakers.discovery import discover_unknown_speakers, identify_cluster
from apps.speakers.encoder_config import OWNER_THRESHOLD
from apps.speakers.owner import (
    bootstrap_owner_from_manual_tags,
    classify_sentences,
    confirm_owner_candidate,
    detect_owner_candidate,
    load_owner_bootstrap_diagnostics,
    load_owner_provisional_centroid,
    reject_owner_candidate,
)
from apps.utils import log_app_action
from convey import state
from convey.utils import DATE_RE, error_response, format_date, success_response
from think.awareness import get_current
from think.entities import find_matching_entity
from think.entities.journal import (
    ensure_journal_entity_memory,
    get_journal_principal,
    journal_entity_memory_path,
    load_all_journal_entities,
    load_journal_entity,
)
from think.utils import (
    day_dirs,
    day_path,
    get_journal,
    iter_segments,
    now_ms,
    segment_parse,
)
from think.utils import segment_key as validate_segment_key
from think.utils import segment_path as get_segment_path

logger = logging.getLogger(__name__)

speakers_bp = Blueprint(
    "app:speakers",
    __name__,
    url_prefix="/app/speakers",
)


def _normalize_embedding(emb: np.ndarray) -> np.ndarray | None:
    from think.entities import normalize_embedding

    return normalize_embedding(emb)


def _parse_time_to_seconds(time_str: str) -> int:
    """Parse HH:MM:SS time string to seconds."""
    parts = time_str.split(":")
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])


def _time_to_seconds(t) -> int:
    """Convert datetime.time to seconds since midnight."""
    return t.hour * 3600 + t.minute * 60 + t.second


def _load_embeddings_file(
    npz_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None] | None:
    """Load embeddings, statement_ids, and optional durations from NPZ file.

    Returns tuple of (embeddings, statement_ids, durations_s) or None if file is invalid.
    """
    if not npz_path.exists():
        return None

    try:
        data = np.load(npz_path)
        embeddings = data.get("embeddings")
        statement_ids = data.get("statement_ids")
        durations_s = data.get("durations_s")

        if embeddings is None or statement_ids is None:
            return None

        return embeddings, statement_ids, durations_s
    except Exception as e:
        logger.warning("Failed to load embeddings %s: %s", npz_path, e)
        return None


def _load_segment_speakers(segment_dir: Path) -> list[str]:
    """Load speaker names from segment's speakers.json.

    Args:
        segment_dir: Path to segment directory

    Returns:
        List of speaker name strings, or empty list if not found/invalid.
    """
    speakers_path = segment_dir / "talents" / "speakers.json"
    if not speakers_path.exists():
        return []

    try:
        with open(speakers_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Must be a list of strings
        if not isinstance(data, list):
            return []

        # Filter to only strings
        return [name for name in data if isinstance(name, str) and name.strip()]
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to load speakers.json from %s: %s", segment_dir, e)
        return []


def _load_entity_voiceprints_file(
    entity_id: str,
) -> tuple[np.ndarray, list[dict]] | None:
    from think.entities import load_entity_voiceprints_file

    return load_entity_voiceprints_file(entity_id)


def _save_voiceprint(
    entity_id: str,
    embedding: np.ndarray,
    day: str,
    segment_key: str,
    source: str,
    sentence_id: int,
    stream: str | None = None,
) -> Path:
    """Save a voiceprint to the entity's journal-level voiceprints.npz.

    Voiceprints are stored at entities/<id>/voiceprints.npz since a person's
    voice is the same across all facets.

    Args:
        entity_id: Entity ID (slug)
        embedding: Normalized embedding vector (256-dim)
        day: Day string (YYYYMMDD)
        segment_key: Segment directory name
        source: Audio source stem
        sentence_id: Sentence ID within transcript

    Returns:
        Path to the voiceprints.npz file
    """
    folder = ensure_journal_entity_memory(entity_id)
    npz_path = folder / "voiceprints.npz"

    # Build metadata for this voiceprint
    metadata = {
        "day": day,
        "segment_key": segment_key,
        "source": source,
        "sentence_id": sentence_id,
        "added_at": now_ms(),
    }
    if stream:
        metadata["stream"] = stream
    metadata_json = json.dumps(metadata)

    # Load existing or initialize empty
    if npz_path.exists():
        try:
            data = np.load(npz_path, allow_pickle=False)
            existing_embeddings = data["embeddings"]
            existing_metadata = data["metadata"]
        except Exception:
            existing_embeddings = np.empty((0, 256), dtype=np.float32)
            existing_metadata = np.array([], dtype=str)
    else:
        existing_embeddings = np.empty((0, 256), dtype=np.float32)
        existing_metadata = np.array([], dtype=str)

    # Append new voiceprint
    new_embeddings = np.vstack([existing_embeddings, embedding.reshape(1, -1)])
    new_metadata = np.append(existing_metadata, metadata_json)

    # Write back (atomic: temp file + rename)
    tmp_path = npz_path.with_name(npz_path.stem + ".tmp.npz")
    np.savez_compressed(tmp_path, embeddings=new_embeddings, metadata=new_metadata)
    tmp_path.rename(npz_path)
    return npz_path


def _remove_voiceprint(
    entity_id: str,
    day: str,
    segment_key: str,
    source: str,
    sentence_id: int,
) -> Path | None:
    """Remove a specific voiceprint entry from an entity's voiceprints.npz.

    Matches by (day, segment_key, source, sentence_id) metadata key.
    Returns the unlinked NPZ path if the file was removed (all entries filtered
    out), or None if the entry was rewritten or not found.
    """
    try:
        folder = journal_entity_memory_path(entity_id)
    except (RuntimeError, ValueError):
        return None

    npz_path = folder / "voiceprints.npz"
    if not npz_path.exists():
        return None

    try:
        data = np.load(npz_path, allow_pickle=False)
        embeddings = data.get("embeddings")
        metadata_arr = data.get("metadata")
        if embeddings is None or metadata_arr is None:
            return None
    except Exception:
        return None

    keep = []
    for i, m_str in enumerate(metadata_arr):
        try:
            m = json.loads(m_str)
            if (
                m.get("day") == day
                and m.get("segment_key") == segment_key
                and m.get("source") == source
                and m.get("sentence_id") == sentence_id
            ):
                continue
        except (json.JSONDecodeError, TypeError):
            pass
        keep.append(i)

    if len(keep) == len(metadata_arr):
        return None

    if not keep:
        npz_path.unlink()
        return npz_path

    new_embeddings = embeddings[keep]
    new_metadata = metadata_arr[keep]
    tmp_path = npz_path.with_name(npz_path.stem + ".tmp.npz")
    np.savez_compressed(tmp_path, embeddings=new_embeddings, metadata=new_metadata)
    tmp_path.rename(npz_path)
    return None


def _load_speaker_labels(segment_dir: Path) -> dict | None:
    """Load speaker_labels.json from a segment's talents/ directory.

    Returns the parsed JSON dict, or None if not found/invalid.
    """
    labels_path = segment_dir / "talents" / "speaker_labels.json"
    if not labels_path.is_file():
        return None
    try:
        with open(labels_path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _save_speaker_labels(segment_dir: Path, labels_data: dict) -> None:
    """Atomically write speaker_labels.json to a segment's talents/ directory."""
    talents_dir = segment_dir / "talents"
    talents_dir.mkdir(parents=True, exist_ok=True)
    out_path = talents_dir / "speaker_labels.json"
    tmp_path = out_path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(labels_data, f, indent=2)
    tmp_path.rename(out_path)


def _load_speaker_corrections(segment_dir: Path) -> list[dict]:
    """Load speaker_corrections.json from a segment's talents/ directory.

    Returns list of correction entries, or empty list if not found.
    """
    corr_path = segment_dir / "talents" / "speaker_corrections.json"
    if not corr_path.is_file():
        return []
    try:
        with open(corr_path) as f:
            data = json.load(f)
        return data.get("corrections", [])
    except (json.JSONDecodeError, OSError):
        return []


def _append_speaker_correction(segment_dir: Path, correction: dict) -> None:
    """Append a correction entry to speaker_corrections.json (atomic write)."""
    corrections = _load_speaker_corrections(segment_dir)
    corrections.append(correction)
    talents_dir = segment_dir / "talents"
    talents_dir.mkdir(parents=True, exist_ok=True)
    out_path = talents_dir / "speaker_corrections.json"
    tmp_path = out_path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump({"corrections": corrections}, f, indent=2)
    tmp_path.rename(out_path)


def _check_owner_contamination(embedding: np.ndarray) -> bool:
    """Check if an embedding is too close to the owner centroid.

    Returns True if the embedding is contaminated (should NOT be saved
    to a non-owner entity's voiceprints).
    """
    from apps.speakers.owner import load_owner_centroid

    centroid_data = load_owner_centroid()
    if centroid_data is not None:
        owner_centroid, owner_threshold = centroid_data
    else:
        principal_id = _principal_id_or_none()
        if principal_id is None:
            return False
        owner_centroid = load_owner_provisional_centroid(principal_id)
        if owner_centroid is None:
            return False
        owner_threshold = OWNER_THRESHOLD
    score = float(np.dot(embedding, owner_centroid))
    return score >= owner_threshold


def _principal_id_or_none() -> str | None:
    """Return the current journal principal id if one exists."""
    principal = get_journal_principal()
    if principal is None:
        return None
    return str(principal["id"])


def _owner_bootstrap_status_fields() -> dict[str, Any]:
    """Return shared owner bootstrap diagnostics for status surfaces."""
    diagnostics = load_owner_bootstrap_diagnostics(_principal_id_or_none())
    return {
        **diagnostics,
        "segments_with_embeddings": diagnostics["segments_available"],
    }


def _maybe_bootstrap_owner_from_attestation(
    principal_id: str | None, speaker_id: str | None
) -> None:
    """Refresh manual owner bootstrap state after a principal attestation."""
    if principal_id is None or speaker_id != principal_id:
        return
    try:
        result = bootstrap_owner_from_manual_tags()
        if "error" in result:
            logger.warning(
                "owner manual bootstrap failed after attestation: %s",
                result["error"],
            )
    except Exception:
        logger.exception("owner manual bootstrap failed after attestation")


def _resolve_entity_display(
    entity_id: str,
    entity_cache: dict,
    principal_id: str | None,
) -> dict:
    """Resolve an entity ID to display info."""
    if entity_id not in entity_cache:
        entity_cache[entity_id] = load_journal_entity(entity_id)
    entity = entity_cache[entity_id]
    name = entity["name"] if entity else entity_id
    return {
        "name": name,
        "entity_id": entity_id,
        "is_owner": entity_id == principal_id,
    }


def _scan_segment_embeddings(day: str) -> list[dict]:
    """Scan a day for segments with audio embeddings.

    Only includes segments that have audio embedding NPZ files.
    Segments with a speakers.json file will include speaker names;
    segments without speakers.json will have an empty speakers list.

    Returns list of segment info dicts with keys:
        - key: segment directory name (HHMMSS_LEN)
        - start: formatted start time (HH:MM)
        - end: formatted end time (HH:MM)
        - duration: duration in seconds
        - sources: list of audio sources (e.g., ["mic_audio", "sys_audio"])
        - speakers: list of speaker names from speakers.json
        - speaker_count: number of speakers
    """
    segments = []
    for s_stream, s_key, s_path in iter_segments(day):
        # Validate segment key format
        parsed = segment_parse(s_key)
        if parsed[0] is None:
            continue

        start_time, end_time = parsed

        # Find embedding files at segment root (new format: <stem>.npz)
        npz_files = list(s_path.glob("*.npz"))
        if not npz_files:
            continue

        # Filter to audio sources (exclude other npz files if any)
        # Accept both "<source>_audio" pattern and plain "audio"
        sources = [
            f.stem for f in npz_files if f.stem.endswith("_audio") or f.stem == "audio"
        ]
        if not sources:
            continue

        # Load speakers.json (may be empty if not yet processed)
        speakers = _load_segment_speakers(s_path)

        # Calculate duration from start and end times
        duration = _time_to_seconds(end_time) - _time_to_seconds(start_time)

        segments.append(
            {
                "key": s_key,
                "stream": s_stream,
                "start": f"{start_time.hour:02d}:{start_time.minute:02d}",
                "end": f"{end_time.hour:02d}:{end_time.minute:02d}",
                "duration": duration,
                "sources": sorted(sources),
                "speakers": speakers,
                "speaker_count": len(speakers),
            }
        )

    return segments


def _load_sentences(
    day: str, segment_key: str, source: str, stream: str | None = None
) -> tuple[list[dict], tuple[np.ndarray, np.ndarray, np.ndarray | None] | None]:
    """Load transcript sentences and their embeddings for an audio source.

    Args:
        day: Day string (YYYYMMDD)
        segment_key: Segment directory name (HHMMSS_LEN)
        source: Audio source stem (e.g., "mic_audio")
        stream: Stream name for path resolution

    Returns:
        Tuple of (sentences, emb_data):
        - sentences: List of dicts with id, offset, text, has_embedding
        - emb_data: Tuple of (embeddings, statement_ids, durations_s) or None if no embeddings
    """
    if stream:
        segment_dir = get_segment_path(day, segment_key, stream, create=False)
    else:
        segment_dir = day_path(day) / segment_key

    # Load JSONL transcript
    jsonl_path = segment_dir / f"{source}.jsonl"
    if not jsonl_path.exists():
        return [], None

    sentences = []
    with open(jsonl_path) as f:
        lines = f.readlines()

    if not lines:
        return [], None

    # Get segment start time to compute relative offsets
    # JSONL contains absolute wall-clock times (e.g., "14:30:22")
    # Audio files start at time 0, so we need relative offset
    parsed = segment_parse(segment_key)
    segment_start_seconds = _time_to_seconds(parsed[0]) if parsed[0] else 0

    # First line is metadata, skip it
    # Remaining lines are sentences indexed by line number (1-based segment ID)
    for i, line in enumerate(lines[1:], start=1):
        try:
            entry = json.loads(line)
            abs_seconds = _parse_time_to_seconds(entry.get("start", "00:00:00"))
            # Convert absolute time to relative offset from segment start
            offset = abs_seconds - segment_start_seconds
            sentences.append(
                {
                    "id": i,
                    "offset": offset,
                    "text": entry.get("text", ""),
                }
            )
        except (json.JSONDecodeError, ValueError, IndexError):
            continue

    # Load embeddings
    npz_path = segment_dir / f"{source}.npz"
    emb_data = _load_embeddings_file(npz_path)

    if emb_data is not None:
        embeddings, statement_ids, _ = emb_data
        emb_map = {int(sid): True for sid in statement_ids}

        # Mark which sentences have embeddings
        for sentence in sentences:
            sentence["has_embedding"] = sentence["id"] in emb_map

    return sentences, emb_data


def _get_sentence_embedding(
    day: str, segment_key: str, source: str, sentence_id: int, stream: str | None = None
) -> np.ndarray | None:
    """Get a specific sentence's embedding, normalized."""
    if stream:
        segment_dir = get_segment_path(day, segment_key, stream, create=False)
    else:
        segment_dir = day_path(day) / segment_key
    npz_path = segment_dir / f"{source}.npz"

    emb_data = _load_embeddings_file(npz_path)
    if emb_data is None:
        return None

    embeddings, statement_ids, _ = emb_data

    # Find the embedding for this sentence
    for i, sid in enumerate(statement_ids):
        if int(sid) == sentence_id:
            return _normalize_embedding(embeddings[i])

    return None


@speakers_bp.route("/")
def index() -> Any:
    """Redirect to today's view."""
    today = date.today().strftime("%Y%m%d")
    return redirect(url_for("app:speakers.speakers_day", day=today))


@speakers_bp.route("/<day>")
def speakers_day(day: str) -> str:
    """Render speaker management view for a specific day."""
    if not DATE_RE.fullmatch(day):
        return "", 404

    title = format_date(day)
    return render_template("app.html", title=title)


@speakers_bp.route("/api/stats/<month>")
def api_stats(month: str) -> Any:
    """Return segment counts for each day in a month.

    Used by calendar heatmap to show days with embedding segments.
    """
    if not re.fullmatch(r"\d{6}", month):
        return error_response("Invalid month format, expected YYYYMM", 400)

    stats: dict[str, int] = {}

    for day_name in day_dirs().keys():
        if not day_name.startswith(month):
            continue

        segments = _scan_segment_embeddings(day_name)
        if segments:
            stats[day_name] = len(segments)

    return jsonify(stats)


@speakers_bp.route("/api/segments/<day>")
def api_segments(day: str) -> Any:
    """Return segments with audio embeddings for a day."""
    if not DATE_RE.fullmatch(day):
        return error_response("Invalid day format", 400)

    try:
        limit = max(0, int(request.args.get("limit", 20)))
        offset = max(0, int(request.args.get("offset", 0)))
    except (ValueError, TypeError):
        return error_response("Invalid limit/offset parameter", 400)

    segments = _scan_segment_embeddings(day)
    segments.sort(key=lambda s: s["key"])
    total = len(segments)
    segments = segments[offset : offset + limit]

    principal = get_journal_principal()
    principal_id = principal["id"] if principal else None
    for seg in segments:
        seg_dir = get_segment_path(day, seg["key"], seg["stream"], create=False)
        labels_data = _load_speaker_labels(seg_dir)
        if labels_data:
            labels = labels_data.get("labels", [])
            seg["attribution_total"] = len(labels)
            seg["attribution_needs_review"] = sum(
                1
                for label in labels
                if label.get("confidence") == "medium" or not label.get("speaker")
            )
            seg["attribution_null"] = sum(
                1 for label in labels if not label.get("speaker")
            )
            owner_count = sum(
                1
                for label in labels
                if label.get("speaker") and label.get("speaker") == principal_id
            )
            seg["attribution_non_owner_total"] = len(labels) - owner_count
        else:
            seg["attribution_total"] = 0
            seg["attribution_needs_review"] = 0
            seg["attribution_null"] = 0
            seg["attribution_non_owner_total"] = 0

    return jsonify({"segments": segments, "total": total})


@speakers_bp.route("/api/speakers/<day>/<stream>/<segment_key>")
def api_segment_speakers(day: str, stream: str, segment_key: str) -> Any:
    """Return speaker names with entity matching for a segment.

    Matches detected speaker names against all journal entities.
    """
    if not DATE_RE.fullmatch(day):
        return error_response("Invalid day format", 400)

    if not validate_segment_key(segment_key):
        return error_response("Invalid segment key", 400)

    # Load speakers from speakers.json
    segment_dir = get_segment_path(day, segment_key, stream, create=False)
    speakers = _load_segment_speakers(segment_dir)
    if not speakers:
        return jsonify({"matched": [], "unmatched": []})

    # Load all journal entities for matching
    journal_entities = load_all_journal_entities()
    entities_list = [e for e in journal_entities.values() if not e.get("blocked")]

    # Match each speaker name to an entity
    matched = []
    unmatched = []

    for speaker_name in speakers:
        entity = find_matching_entity(speaker_name, entities_list)
        if entity:
            matched.append(
                {
                    "detected_name": speaker_name,
                    "entity_name": entity.get("name"),
                    "entity_type": entity.get("type"),
                }
            )
        else:
            unmatched.append(speaker_name)

    return jsonify(
        {
            "matched": matched,
            "unmatched": unmatched,
        }
    )


@speakers_bp.route("/api/review/<day>/<stream>/<segment_key>/<source>")
def api_review(day: str, stream: str, segment_key: str, source: str) -> Any:
    """Return sentences with pre-computed speaker labels for review."""
    if not DATE_RE.fullmatch(day):
        return error_response("Invalid day format", 400)
    if not validate_segment_key(segment_key):
        return error_response("Invalid segment key", 400)

    sentences, _ = _load_sentences(day, segment_key, source, stream=stream)
    if not sentences:
        return error_response("No transcript found", 404)

    segment_dir = get_segment_path(day, segment_key, stream, create=False)
    labels_data = _load_speaker_labels(segment_dir)
    label_map: dict[int, dict] = {}
    if labels_data:
        for label in labels_data.get("labels", []):
            sid = label.get("sentence_id")
            if sid is not None:
                label_map[int(sid)] = label

    corrections = _load_speaker_corrections(segment_dir)
    correction_map: dict[int, dict] = {}
    for correction in corrections:
        sid = correction.get("sentence_id")
        if sid is not None:
            correction_map[int(sid)] = correction

    principal = get_journal_principal()
    principal_id = principal["id"] if principal else None
    entity_cache: dict[str, dict | None] = {}

    review_sentences = [s for s in sentences if s.get("has_embedding")]
    needs_review_count = 0
    corrections_count = 0

    for sentence in review_sentences:
        sid = sentence["id"]
        label = label_map.get(sid)
        if label:
            entity_id = label.get("speaker")
            confidence = label.get("confidence")
            method = label.get("method")
            if entity_id:
                info = _resolve_entity_display(entity_id, entity_cache, principal_id)
                sentence["speaker_entity_id"] = entity_id
                sentence["speaker_name"] = info["name"]
                sentence["is_owner"] = info["is_owner"]
            else:
                sentence["speaker_entity_id"] = None
                sentence["speaker_name"] = None
                sentence["is_owner"] = False

            sentence["confidence"] = confidence
            sentence["method"] = method
            sentence["needs_review"] = confidence == "medium" or not entity_id
        else:
            sentence["speaker_entity_id"] = None
            sentence["speaker_name"] = None
            sentence["confidence"] = None
            sentence["method"] = None
            sentence["is_owner"] = False
            sentence["needs_review"] = True if labels_data else False

        correction = correction_map.get(sid)
        sentence["is_correction"] = sentence.get("method") in {
            "user_corrected",
            "user_assigned",
        }
        if correction and sentence["is_correction"]:
            orig_speaker = correction.get("original_speaker")
            if orig_speaker:
                orig_info = _resolve_entity_display(
                    orig_speaker,
                    entity_cache,
                    principal_id,
                )
                sentence["original_speaker_entity_id"] = orig_speaker
                sentence["original_speaker_name"] = orig_info["name"]
            else:
                sentence["original_speaker_entity_id"] = None
                sentence["original_speaker_name"] = None
            corrections_count += 1
        else:
            sentence["original_speaker_entity_id"] = None
            sentence["original_speaker_name"] = None

        if sentence.get("needs_review"):
            needs_review_count += 1

    journal_entities = load_all_journal_entities()
    all_entities = []
    for eid, entity in journal_entities.items():
        if entity.get("blocked"):
            continue
        all_entities.append(
            {
                "entity_id": eid,
                "name": entity.get("name", eid),
                "is_principal": bool(entity.get("is_principal")),
            }
        )
    all_entities.sort(key=lambda x: (not x["is_principal"], x["name"].lower()))

    audio_file = None
    audio_path = segment_dir / f"{source}.flac"
    if audio_path.exists():
        rel_path = f"{stream}/{segment_key}/{source}.flac"
        audio_file = f"/app/speakers/api/serve_audio/{day}/{rel_path}"

    parsed = segment_parse(segment_key)
    start_time, end_time = parsed if parsed[0] else (None, None)

    return jsonify(
        {
            "segment": {
                "key": segment_key,
                "start": (
                    f"{start_time.hour:02d}:{start_time.minute:02d}"
                    if start_time
                    else ""
                ),
                "end": (
                    f"{end_time.hour:02d}:{end_time.minute:02d}" if end_time else ""
                ),
            },
            "source": source,
            "sentences": review_sentences,
            "all_entities": all_entities,
            "audio_file": audio_file,
            "has_labels": labels_data is not None,
            "summary": {
                "total": len(review_sentences),
                "needs_review": needs_review_count,
                "corrections": corrections_count,
            },
        }
    )


@speakers_bp.route("/api/confirm-attribution", methods=["POST"])
def api_confirm_attribution() -> Any:
    """Confirm a medium-confidence speaker attribution."""
    data = request.get_json()
    if not data:
        return error_response("No data provided", 400)

    day = data.get("day")
    stream = data.get("stream")
    segment_key = data.get("segment_key")
    source = data.get("source")
    sentence_id = data.get("sentence_id")

    if not all([day, stream, segment_key, source, sentence_id is not None]):
        return error_response("Missing required fields", 400)
    if not DATE_RE.fullmatch(day):
        return error_response("Invalid day format", 400)
    if not validate_segment_key(segment_key):
        return error_response("Invalid segment key", 400)

    segment_dir = get_segment_path(day, segment_key, stream)
    labels_data = _load_speaker_labels(segment_dir)
    if not labels_data:
        return error_response("No speaker labels found", 404)

    label = None
    label_idx = None
    for i, item in enumerate(labels_data.get("labels", [])):
        if item.get("sentence_id") == sentence_id:
            label = item
            label_idx = i
            break

    if label is None or label_idx is None:
        return error_response("Sentence not found in labels", 404)

    speaker = label.get("speaker")
    if not speaker:
        return error_response(
            "Cannot confirm unattributed sentence — use assign instead",
            400,
        )

    confidence = label.get("confidence")
    if confidence == "high" and label.get("method") == "user_confirmed":
        return success_response({"status": "already_confirmed"})
    if confidence != "medium":
        return error_response("Can only confirm medium-confidence attributions", 400)

    emb = _get_sentence_embedding(day, segment_key, source, sentence_id, stream=stream)
    if emb is None:
        return error_response("Sentence embedding not found", 404)

    principal_id = _principal_id_or_none()
    if speaker != principal_id and _check_owner_contamination(emb):
        return error_response("Embedding too similar to owner voice — cannot save", 400)

    _save_voiceprint(speaker, emb, day, segment_key, source, sentence_id, stream=stream)

    old_method = label.get("method")
    labels_data["labels"][label_idx]["confidence"] = "high"
    labels_data["labels"][label_idx]["method"] = "user_confirmed"
    _save_speaker_labels(segment_dir, labels_data)

    _append_speaker_correction(
        segment_dir,
        {
            "sentence_id": sentence_id,
            "original_speaker": speaker,
            "corrected_speaker": speaker,
            "original_method": old_method,
            "timestamp": now_ms(),
        },
    )

    log_app_action(
        app="speakers",
        facet=None,
        action="attribution_confirm",
        params={
            "day": day,
            "stream": stream,
            "segment_key": segment_key,
            "source": source,
            "sentence_id": sentence_id,
            "speaker": speaker,
        },
    )
    _maybe_bootstrap_owner_from_attestation(principal_id, speaker)

    return success_response({"status": "confirmed", "speaker": speaker})


@speakers_bp.route("/api/correct-attribution", methods=["POST"])
def api_correct_attribution() -> Any:
    """Correct a speaker attribution to a different entity."""
    data = request.get_json()
    if not data:
        return error_response("No data provided", 400)

    day = data.get("day")
    stream = data.get("stream")
    segment_key = data.get("segment_key")
    source = data.get("source")
    sentence_id = data.get("sentence_id")
    new_speaker = data.get("new_speaker")

    if not all(
        [day, stream, segment_key, source, sentence_id is not None, new_speaker]
    ):
        return error_response("Missing required fields", 400)
    if not DATE_RE.fullmatch(day):
        return error_response("Invalid day format", 400)
    if not validate_segment_key(segment_key):
        return error_response("Invalid segment key", 400)

    target_entity = load_journal_entity(new_speaker)
    if not target_entity:
        return error_response(f"Entity '{new_speaker}' not found", 404)
    if target_entity.get("blocked"):
        return error_response(f"Entity '{new_speaker}' is blocked", 400)

    segment_dir = get_segment_path(day, segment_key, stream)
    labels_data = _load_speaker_labels(segment_dir)
    if not labels_data:
        return error_response("No speaker labels found", 404)

    label = None
    label_idx = None
    for i, item in enumerate(labels_data.get("labels", [])):
        if item.get("sentence_id") == sentence_id:
            label = item
            label_idx = i
            break

    if label is None or label_idx is None:
        return error_response("Sentence not found in labels", 404)

    old_speaker = label.get("speaker")
    old_method = label.get("method")
    if old_speaker == new_speaker:
        return success_response({"status": "already_correct"})

    emb = _get_sentence_embedding(day, segment_key, source, sentence_id, stream=stream)
    if emb is None:
        return error_response("Sentence embedding not found", 404)

    principal_id = _principal_id_or_none()
    if new_speaker != principal_id and _check_owner_contamination(emb):
        return error_response("Embedding too similar to owner voice — cannot save", 400)

    auto_accumulated_methods = {"acoustic", "context", "contextual"}
    removed_voiceprint_path = None
    if old_speaker and old_method in auto_accumulated_methods:
        removed_voiceprint_path = _remove_voiceprint(
            old_speaker, day, segment_key, source, sentence_id
        )

    voiceprints_removed: list[str] = []
    if removed_voiceprint_path is not None:
        journal_root = Path(get_journal())
        voiceprints_removed = [str(removed_voiceprint_path.relative_to(journal_root))]

    _save_voiceprint(
        new_speaker,
        emb,
        day,
        segment_key,
        source,
        sentence_id,
        stream=stream,
    )

    labels_data["labels"][label_idx]["speaker"] = new_speaker
    labels_data["labels"][label_idx]["confidence"] = "high"
    labels_data["labels"][label_idx]["method"] = "user_corrected"
    _save_speaker_labels(segment_dir, labels_data)

    _append_speaker_correction(
        segment_dir,
        {
            "sentence_id": sentence_id,
            "original_speaker": old_speaker,
            "corrected_speaker": new_speaker,
            "original_method": old_method,
            "timestamp": now_ms(),
        },
    )

    log_app_action(
        app="speakers",
        facet=None,
        action="attribution_correct",
        params={
            "day": day,
            "stream": stream,
            "segment_key": segment_key,
            "source": source,
            "sentence_id": sentence_id,
            "old_speaker": old_speaker,
            "new_speaker": new_speaker,
            "voiceprints_removed": voiceprints_removed,
        },
    )
    _maybe_bootstrap_owner_from_attestation(principal_id, new_speaker)

    return success_response(
        {
            "status": "corrected",
            "old_speaker": old_speaker,
            "new_speaker": new_speaker,
        }
    )


@speakers_bp.route("/api/assign-attribution", methods=["POST"])
def api_assign_attribution() -> Any:
    """Assign a speaker to an unattributed sentence."""
    data = request.get_json()
    if not data:
        return error_response("No data provided", 400)

    day = data.get("day")
    stream = data.get("stream")
    segment_key = data.get("segment_key")
    source = data.get("source")
    sentence_id = data.get("sentence_id")
    speaker = data.get("speaker")

    if not all([day, stream, segment_key, source, sentence_id is not None, speaker]):
        return error_response("Missing required fields", 400)
    if not DATE_RE.fullmatch(day):
        return error_response("Invalid day format", 400)
    if not validate_segment_key(segment_key):
        return error_response("Invalid segment key", 400)

    target_entity = load_journal_entity(speaker)
    if not target_entity:
        return error_response(f"Entity '{speaker}' not found", 404)
    if target_entity.get("blocked"):
        return error_response(f"Entity '{speaker}' is blocked", 400)

    segment_dir = get_segment_path(day, segment_key, stream)
    labels_data = _load_speaker_labels(segment_dir)
    if not labels_data:
        return error_response("No speaker labels found", 404)

    label = None
    label_idx = None
    for i, item in enumerate(labels_data.get("labels", [])):
        if item.get("sentence_id") == sentence_id:
            label = item
            label_idx = i
            break

    if label is None or label_idx is None:
        return error_response("Sentence not found in labels", 404)

    existing_speaker = label.get("speaker")
    if existing_speaker == speaker and label.get("method") == "user_assigned":
        return success_response({"status": "already_assigned"})
    if existing_speaker:
        return error_response(
            "Sentence already has a speaker — use correct instead", 400
        )

    emb = _get_sentence_embedding(day, segment_key, source, sentence_id, stream=stream)
    if emb is None:
        return error_response("Sentence embedding not found", 404)

    principal_id = _principal_id_or_none()
    if speaker != principal_id and _check_owner_contamination(emb):
        return error_response("Embedding too similar to owner voice — cannot save", 400)

    _save_voiceprint(speaker, emb, day, segment_key, source, sentence_id, stream=stream)

    labels_data["labels"][label_idx]["speaker"] = speaker
    labels_data["labels"][label_idx]["confidence"] = "high"
    labels_data["labels"][label_idx]["method"] = "user_assigned"
    _save_speaker_labels(segment_dir, labels_data)

    _append_speaker_correction(
        segment_dir,
        {
            "sentence_id": sentence_id,
            "original_speaker": None,
            "corrected_speaker": speaker,
            "original_method": label.get("method"),
            "timestamp": now_ms(),
        },
    )

    log_app_action(
        app="speakers",
        facet=None,
        action="attribution_assign",
        params={
            "day": day,
            "stream": stream,
            "segment_key": segment_key,
            "source": source,
            "sentence_id": sentence_id,
            "speaker": speaker,
        },
    )
    _maybe_bootstrap_owner_from_attestation(principal_id, speaker)

    return success_response({"status": "assigned", "speaker": speaker})


@speakers_bp.route("/api/owner/status")
def api_owner_status() -> Any:
    """Return the current owner voiceprint confirmation state."""
    voiceprint = get_current().get("voiceprint", {})
    status = voiceprint.get("status", "none")
    diagnostics = _owner_bootstrap_status_fields()

    if status == "confirmed":
        return jsonify({"status": "confirmed"})

    if status == "candidate":
        return jsonify(
            {
                "status": "candidate",
                "cluster_size": voiceprint.get("cluster_size"),
                "samples": voiceprint.get("samples", []),
            }
        )

    if status == "low_quality":
        return jsonify(
            {
                "status": "low_quality",
                "source": voiceprint.get("source", "hdbscan"),
                "low_quality_reason": voiceprint.get("low_quality_reason", ""),
                "observed_value": voiceprint.get("observed_value", 0.0),
                "threshold_value": voiceprint.get("threshold_value", 0.0),
                **diagnostics,
            }
        )

    if status == "no_cluster":
        return jsonify({"status": "no_cluster"})

    if status in {"none", "rejected"}:
        if diagnostics["segments_available"] > 0:
            return jsonify(
                {
                    "status": "needs_detection",
                    **diagnostics,
                }
            )
        return jsonify(
            {
                "status": "none",
                **diagnostics,
            }
        )

    return jsonify({"status": "none", **diagnostics})


@speakers_bp.route("/api/owner/detect", methods=["POST"])
def api_owner_detect() -> Any:
    """Run owner voice candidate detection."""
    result = detect_owner_candidate()
    return jsonify(result)


@speakers_bp.route("/api/owner/build-from-tags", methods=["POST"])
def api_owner_build_from_tags() -> Any:
    """Build a confirmed owner centroid directly from validated manual tags."""
    result = bootstrap_owner_from_manual_tags()
    if "error" in result:
        return error_response(result["error"], 400)
    if result.get("status") == "confirmed":
        log_app_action(
            app="speakers",
            facet=None,
            action="owner_voiceprint_build_from_tags",
            params={
                "principal_id": result["principal_id"],
                "cluster_size": result.get("cluster_size"),
            },
        )
    return jsonify(result)


@speakers_bp.route("/api/owner/confirm", methods=["POST"])
def api_owner_confirm() -> Any:
    """Confirm the current owner voice candidate and persist the centroid."""
    result = confirm_owner_candidate()
    if "error" in result:
        code = 404 if "No candidate" in result["error"] else 400
        return error_response(result["error"], code)

    log_app_action(
        app="speakers",
        facet=None,
        action="owner_voiceprint_confirm",
        params={
            "principal_id": result["principal_id"],
            "cluster_size": result["cluster_size"],
        },
    )

    return jsonify({"status": "confirmed", "principal_id": result["principal_id"]})


@speakers_bp.route("/api/owner/reject", methods=["POST"])
def api_owner_reject() -> Any:
    """Reject the current owner voice candidate."""
    reject_owner_candidate()
    return jsonify({"status": "needs_detection"})


@speakers_bp.route("/api/owner/classify", methods=["POST"])
def api_owner_classify() -> Any:
    """Classify segment sentences against the confirmed owner centroid."""
    data = request.get_json()
    if not data:
        return error_response("No data provided", 400)

    day = data.get("day")
    stream = data.get("stream")
    segment_key = data.get("segment_key")
    source = data.get("source")

    if not all([day, stream, segment_key, source]):
        return error_response("Missing required fields", 400)
    if not DATE_RE.fullmatch(day):
        return error_response("Invalid day format", 400)
    if not validate_segment_key(segment_key):
        return error_response("Invalid segment key", 400)

    return jsonify(
        {
            "sentences": classify_sentences(day, stream, segment_key, source),
        }
    )


@speakers_bp.route("/api/discovery/scan", methods=["POST"])
def api_discovery_scan() -> Any:
    """Scan for recurring unknown speaker clusters."""
    result = discover_unknown_speakers()
    return jsonify(result)


@speakers_bp.route("/api/discovery/identify", methods=["POST"])
def api_discovery_identify() -> Any:
    """Identify a discovered unknown speaker cluster by naming it."""
    data = request.get_json(silent=True) or {}
    cluster_id = data.get("cluster_id")
    name = data.get("name", "").strip()

    if cluster_id is None:
        return error_response("cluster_id is required", 400)
    if not name:
        return error_response("name is required", 400)

    try:
        cluster_id = int(cluster_id)
    except (TypeError, ValueError):
        return error_response("cluster_id must be an integer", 400)

    result = identify_cluster(cluster_id, name)
    if "error" in result:
        return error_response(result["error"], 400)

    log_app_action(
        app="speakers",
        facet=None,
        action="speaker_identified",
        params={
            "entity_id": result.get("entity_id"),
            "entity_name": result.get("entity_name"),
            "cluster_id": cluster_id,
            "voiceprints_saved": result.get("voiceprints_saved"),
            "segments_updated": result.get("segments_updated"),
        },
    )

    return jsonify(result)


@speakers_bp.route("/api/serve_audio/<day>/<path:rel_path>")
def serve_audio(day: str, rel_path: str) -> Any:
    """Serve audio files for playback."""
    if not DATE_RE.fullmatch(day):
        return error_response("Day not found", 404)

    try:
        full_path = os.path.join(state.journal_root, day, rel_path)
        day_dir = str(day_path(day))
        if not os.path.commonpath([full_path, day_dir]) == day_dir:
            return error_response("Invalid file path", 403)
        if not os.path.isfile(full_path):
            return error_response("File not found", 404)
    except (OSError, ValueError):
        logger.warning(
            "serve_audio path validation failed for %s/%s",
            day,
            rel_path,
            exc_info=True,
        )
        return error_response("Failed to serve file", 404)

    return send_file(full_path, mimetype="audio/flac")
