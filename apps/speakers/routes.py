# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Speaker voiceprint management app - sentence-based embeddings."""

from __future__ import annotations

import json
import logging
import os
import re
import time
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

from apps.utils import log_app_action
from convey import state
from convey.utils import DATE_RE, error_response, format_date, success_response
from think.entities import (
    ensure_entity_memory,
    entity_memory_path,
    find_matching_attached_entity,
    load_entities,
    save_entities,
)
from think.utils import day_dirs, day_path
from think.utils import segment_key as validate_segment_key
from think.utils import segment_parse

logger = logging.getLogger(__name__)

speakers_bp = Blueprint(
    "app:speakers",
    __name__,
    url_prefix="/app/speakers",
)


def _normalize_embedding(emb: np.ndarray) -> np.ndarray | None:
    """L2-normalize an embedding vector. Returns None if norm is zero."""
    emb = emb.astype(np.float32)
    norm = np.linalg.norm(emb)
    if norm > 0:
        return emb / norm
    return None


def _parse_time_to_seconds(time_str: str) -> int:
    """Parse HH:MM:SS time string to seconds."""
    parts = time_str.split(":")
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])


def _time_to_seconds(t) -> int:
    """Convert datetime.time to seconds since midnight."""
    return t.hour * 3600 + t.minute * 60 + t.second


def _load_embeddings_file(npz_path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Load embeddings and statement_ids from NPZ file.

    Returns tuple of (embeddings, statement_ids) or None if file is invalid.
    """
    if not npz_path.exists():
        return None

    try:
        data = np.load(npz_path)
        embeddings = data.get("embeddings")
        statement_ids = data.get("statement_ids")

        if embeddings is None or statement_ids is None:
            return None

        return embeddings, statement_ids
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
    speakers_path = segment_dir / "speakers.json"
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
    facet: str, entity_name: str
) -> tuple[np.ndarray, list[dict]] | None:
    """Load voiceprints for an entity from consolidated voiceprints.npz.

    Returns tuple of (embeddings, metadata_list) or None if not found.
    - embeddings: (N, 256) float32 array
    - metadata_list: List of dicts parsed from JSON metadata strings
    """
    try:
        folder = entity_memory_path(facet, entity_name)
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

        # Parse JSON metadata strings
        metadata_list = [json.loads(m) for m in metadata_arr]
        return embeddings, metadata_list
    except Exception as e:
        logger.warning("Failed to load voiceprints for %s: %s", entity_name, e)
        return None


def _save_voiceprint(
    facet: str,
    entity_name: str,
    embedding: np.ndarray,
    day: str,
    segment_key: str,
    source: str,
    sentence_id: int,
) -> Path:
    """Save a voiceprint to the entity's consolidated voiceprints.npz.

    Appends to existing file or creates new one.
    """
    folder = ensure_entity_memory(facet, entity_name)
    npz_path = folder / "voiceprints.npz"

    # Build metadata for this voiceprint
    metadata = {
        "day": day,
        "segment_key": segment_key,
        "source": source,
        "sentence_id": sentence_id,
        "added_at": int(time.time() * 1000),
    }
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

    # Write back
    np.savez_compressed(npz_path, embeddings=new_embeddings, metadata=new_metadata)
    return npz_path


def _scan_segment_embeddings(day: str) -> list[dict]:
    """Scan a day for segments with embeddings and 2+ speakers.

    Only includes segments that have:
    1. Audio embedding NPZ files
    2. A speakers.json file with 2 or more speaker names

    Returns list of segment info dicts with keys:
        - key: segment directory name (HHMMSS_LEN)
        - start: formatted start time (HH:MM)
        - end: formatted end time (HH:MM)
        - duration: duration in seconds
        - sources: list of audio sources (e.g., ["mic_audio", "sys_audio"])
        - speakers: list of speaker names from speakers.json
        - speaker_count: number of speakers
    """
    day_dir = day_path(day)
    if not day_dir.is_dir():
        return []

    segments = []
    for item in sorted(os.listdir(day_dir)):
        item_path = day_dir / item
        if not item_path.is_dir():
            continue

        # Validate segment key format
        parsed = segment_parse(item)
        if parsed[0] is None:
            continue

        start_time, end_time = parsed

        # Find embedding files at segment root (new format: <stem>.npz)
        npz_files = list(item_path.glob("*.npz"))
        if not npz_files:
            continue

        # Filter to audio sources (exclude other npz files if any)
        # Accept both "<source>_audio" pattern and plain "audio"
        sources = [
            f.stem for f in npz_files if f.stem.endswith("_audio") or f.stem == "audio"
        ]
        if not sources:
            continue

        # Load speakers.json - require 2+ speakers
        speakers = _load_segment_speakers(item_path)
        if len(speakers) < 2:
            continue

        # Calculate duration from start and end times
        duration = _time_to_seconds(end_time) - _time_to_seconds(start_time)

        segments.append(
            {
                "key": item,
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
    day: str, segment_key: str, source: str
) -> tuple[list[dict], tuple[np.ndarray, np.ndarray] | None]:
    """Load transcript sentences and their embeddings for an audio source.

    Args:
        day: Day string (YYYYMMDD)
        segment_key: Segment directory name (HHMMSS_LEN)
        source: Audio source stem (e.g., "mic_audio")

    Returns:
        Tuple of (sentences, emb_data):
        - sentences: List of dicts with id, offset, text, has_embedding
        - emb_data: Tuple of (embeddings, statement_ids) or None if no embeddings
    """
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
        embeddings, statement_ids = emb_data
        emb_map = {int(sid): True for sid in statement_ids}

        # Mark which sentences have embeddings
        for sentence in sentences:
            sentence["has_embedding"] = sentence["id"] in emb_map

    return sentences, emb_data


def _get_sentence_embedding(
    day: str, segment_key: str, source: str, sentence_id: int
) -> np.ndarray | None:
    """Get a specific sentence's embedding, normalized."""
    segment_dir = day_path(day) / segment_key
    npz_path = segment_dir / f"{source}.npz"

    emb_data = _load_embeddings_file(npz_path)
    if emb_data is None:
        return None

    embeddings, statement_ids = emb_data

    # Find the embedding for this sentence
    for i, sid in enumerate(statement_ids):
        if int(sid) == sentence_id:
            return _normalize_embedding(embeddings[i])

    return None


def _scan_entity_voiceprints(facet: str) -> dict[str, np.ndarray]:
    """Scan entities in a facet for voiceprints.

    Returns dict mapping entity name to averaged embedding.
    """
    try:
        entities = load_entities(facet)
    except RuntimeError:
        return {}

    voiceprints = {}
    for entity in entities:
        name = entity.get("name", "")
        if not name:
            continue

        # Load voiceprints from consolidated file
        result = _load_entity_voiceprints_file(facet, name)
        if result is None:
            continue

        embeddings, _ = result
        if len(embeddings) == 0:
            continue

        # Normalize and average all embeddings
        all_normalized = []
        for emb in embeddings:
            normalized = _normalize_embedding(emb)
            if normalized is not None:
                all_normalized.append(normalized)

        if all_normalized:
            avg_emb = _normalize_embedding(np.mean(all_normalized, axis=0))
            if avg_emb is not None:
                voiceprints[name] = avg_emb

    return voiceprints


def _compute_best_match(
    emb: np.ndarray, known_embs: dict[str, np.ndarray], threshold: float = 0.4
) -> dict | None:
    """Find the best matching voiceprint for an embedding.

    Returns dict with 'entity' and 'score', or None if no match above threshold.
    """
    if not known_embs:
        return None

    best_entity = None
    best_score = threshold

    for name, known_emb in known_embs.items():
        score = float(np.dot(emb, known_emb))
        if score > best_score:
            best_score = score
            best_entity = name

    if best_entity:
        return {"entity": best_entity, "score": round(best_score, 4)}
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

    Used by calendar heatmap to show days with multi-speaker segments.
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
    """Return segments with embeddings and 2+ speakers for a day."""
    if not DATE_RE.fullmatch(day):
        return error_response("Invalid day format", 400)

    segments = _scan_segment_embeddings(day)
    return jsonify({"segments": segments})


@speakers_bp.route("/api/speakers/<day>/<segment_key>")
def api_segment_speakers(day: str, segment_key: str) -> Any:
    """Return speaker names with entity matching for a segment.

    Requires a facet (via query param or cookie) for entity matching.
    Returns matched and unmatched speakers.
    """
    if not DATE_RE.fullmatch(day):
        return error_response("Invalid day format", 400)

    if not validate_segment_key(segment_key):
        return error_response("Invalid segment key", 400)

    # Require facet for entity matching (query param takes precedence over cookie)
    selected_facet = request.args.get("facet") or request.cookies.get("selectedFacet")
    if not selected_facet:
        return error_response("Select a facet to view speaker details", 400)

    # Load speakers from speakers.json
    segment_dir = day_path(day) / segment_key
    speakers = _load_segment_speakers(segment_dir)
    if not speakers:
        return error_response("No speakers found for segment", 404)

    # Load attached entities for matching
    try:
        attached_entities = load_entities(selected_facet)
    except RuntimeError:
        attached_entities = []

    # Match each speaker name to an entity
    matched = []
    unmatched = []

    for speaker_name in speakers:
        entity = find_matching_attached_entity(speaker_name, attached_entities)
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
            "facet": selected_facet,
        }
    )


@speakers_bp.route("/api/sentences/<day>/<segment_key>/<source>")
def api_sentences(day: str, segment_key: str, source: str) -> Any:
    """Return sentences with embeddings and matches for an audio source."""
    if not DATE_RE.fullmatch(day):
        return error_response("Invalid day format", 400)

    if not validate_segment_key(segment_key):
        return error_response("Invalid segment key", 400)

    # Get selected facet from cookie (optional - sentences work without it)
    selected_facet = request.cookies.get("selectedFacet")

    # Load sentences and embeddings
    sentences, emb_data = _load_sentences(day, segment_key, source)
    if not sentences:
        return error_response("No transcript found", 404)

    # Load known voiceprints for matching (only if facet selected)
    known_voiceprints = {}
    if selected_facet:
        known_voiceprints = _scan_entity_voiceprints(selected_facet)

    # Compute best match for each sentence with embedding
    if emb_data is not None:
        embeddings, statement_ids = emb_data
        emb_map = {int(sid): emb for sid, emb in zip(statement_ids, embeddings)}

        for sentence in sentences:
            if sentence.get("has_embedding"):
                emb = emb_map.get(sentence["id"])
                if emb is not None:
                    normalized = _normalize_embedding(emb)
                    if normalized is not None:
                        match = _compute_best_match(normalized, known_voiceprints)
                        if match:
                            sentence["match"] = match

    # Filter to only sentences with embeddings
    sentences = [s for s in sentences if s.get("has_embedding")]

    # Load ALL entities in the facet for dropdown (only if facet selected)
    all_entity_names = []
    if selected_facet:
        try:
            all_entities = load_entities(selected_facet)
            all_entity_names = [e.get("name") for e in all_entities if e.get("name")]
        except RuntimeError:
            pass

    # Get audio file URL
    segment_dir = day_path(day) / segment_key
    audio_file = None
    audio_path = segment_dir / f"{source}.flac"
    if audio_path.exists():
        rel_path = f"{segment_key}/{source}.flac"
        audio_file = (
            f"/app/speakers/api/serve_audio/{day}/{rel_path.replace('/', '__')}"
        )

    # Get segment times
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
            "sentences": sentences,
            "all_entities": all_entity_names,
            "audio_file": audio_file,
            "facet": selected_facet,
        }
    )


@speakers_bp.route("/api/serve_audio/<day>/<path:encoded_path>")
def serve_audio(day: str, encoded_path: str) -> Any:
    """Serve audio files for playback."""
    if not DATE_RE.fullmatch(day):
        return "", 404

    try:
        rel_path = encoded_path.replace("__", "/")
        full_path = os.path.join(state.journal_root, day, rel_path)

        day_dir = str(day_path(day))
        if not os.path.commonpath([full_path, day_dir]) == day_dir:
            return "", 403

        if not os.path.isfile(full_path):
            return "", 404

        return send_file(full_path)

    except Exception as e:
        logger.warning("Error serving audio %s/%s: %s", day, encoded_path, e)
        return "", 404


@speakers_bp.route("/api/save-voiceprint", methods=["POST"])
def api_save_voiceprint() -> Any:
    """Save a sentence voiceprint to an existing entity."""
    data = request.get_json()
    if not data:
        return error_response("No data provided", 400)

    facet = data.get("facet")
    entity_name = data.get("entity_name")
    day = data.get("day")
    segment_key = data.get("segment_key")
    source = data.get("source")
    sentence_id = data.get("sentence_id")

    if not all([facet, entity_name, day, segment_key, source, sentence_id]):
        return error_response("Missing required fields", 400)

    # Validate formats
    if not DATE_RE.fullmatch(day):
        return error_response("Invalid day format", 400)
    if not validate_segment_key(segment_key):
        return error_response("Invalid segment key", 400)

    # Validate entity exists
    entities = load_entities(facet)
    entity_names = [e.get("name") for e in entities]
    if entity_name not in entity_names:
        return error_response(
            f"Entity '{entity_name}' not found in facet '{facet}'", 404
        )

    # Load sentence embedding
    emb = _get_sentence_embedding(day, segment_key, source, sentence_id)
    if emb is None:
        return error_response("Sentence embedding not found", 404)

    # Save voiceprint
    try:
        emb_path = _save_voiceprint(
            facet, entity_name, emb, day, segment_key, source, sentence_id
        )

        log_app_action(
            app="speakers",
            facet=facet,
            action="voiceprint_save",
            params={
                "entity_name": entity_name,
                "day": day,
                "segment_key": segment_key,
                "source": source,
                "sentence_id": sentence_id,
            },
        )

        return success_response({"path": str(emb_path)})
    except Exception as e:
        logger.exception("Failed to save voiceprint for %s", entity_name)
        return error_response(f"Failed to save voiceprint: {e}", 500)


@speakers_bp.route("/api/create-entity-voiceprint", methods=["POST"])
def api_create_entity_voiceprint() -> Any:
    """Create a new entity with a voiceprint."""
    data = request.get_json()
    if not data:
        return error_response("No data provided", 400)

    facet = data.get("facet")
    entity_type = data.get("type", "Person")
    entity_name = data.get("name")
    entity_description = data.get("description", "")
    day = data.get("day")
    segment_key = data.get("segment_key")
    source = data.get("source")
    sentence_id = data.get("sentence_id")

    if not all([facet, entity_name, day, segment_key, source, sentence_id]):
        return error_response("Missing required fields", 400)

    # Validate formats
    if not DATE_RE.fullmatch(day):
        return error_response("Invalid day format", 400)
    if not validate_segment_key(segment_key):
        return error_response("Invalid segment key", 400)

    # Check entity doesn't already exist
    entities = load_entities(facet, include_detached=True)
    entity_names = [e.get("name") for e in entities]
    if entity_name in entity_names:
        return error_response(
            f"Entity '{entity_name}' already exists in facet '{facet}'", 409
        )

    # Load sentence embedding
    emb = _get_sentence_embedding(day, segment_key, source, sentence_id)
    if emb is None:
        return error_response("Sentence embedding not found", 404)

    # Create new entity
    new_entity = {
        "type": entity_type,
        "name": entity_name,
        "description": entity_description,
        "attached_at": int(time.time() * 1000),
    }
    entities.append(new_entity)
    save_entities(facet, entities)

    # Save voiceprint
    try:
        emb_path = _save_voiceprint(
            facet, entity_name, emb, day, segment_key, source, sentence_id
        )

        log_app_action(
            app="speakers",
            facet=facet,
            action="entity_voiceprint_create",
            params={
                "entity_name": entity_name,
                "entity_type": entity_type,
                "day": day,
                "segment_key": segment_key,
                "source": source,
                "sentence_id": sentence_id,
            },
        )

        return success_response(
            {"entity": new_entity, "voiceprint_path": str(emb_path)}
        )
    except Exception as e:
        logger.exception("Failed to save voiceprint for new entity %s", entity_name)
        return error_response(f"Failed to save voiceprint: {e}", 500)
