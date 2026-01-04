# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Speaker voiceprint management app."""

from __future__ import annotations

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
    ensure_entity_folder,
    entity_folder_path,
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


def _scan_segment_embeddings(day: str) -> list[dict]:
    """Scan a day for segments with speaker embeddings.

    Returns list of segment info dicts with keys:
        - key: segment directory name (HHMMSS_LEN)
        - start: formatted start time (HH:MM)
        - end: formatted end time (HH:MM)
        - duration: duration in seconds
        - speakers: list of speaker labels
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

        # Check for audio embeddings subdir
        audio_dir = item_path / "audio"
        if not audio_dir.is_dir():
            continue

        # Find speaker embedding files
        npz_files = list(audio_dir.glob("*.npz"))
        if not npz_files:
            continue

        speakers = [f.stem for f in npz_files]

        # Calculate duration from start and end times
        start_seconds = (
            start_time.hour * 3600 + start_time.minute * 60 + start_time.second
        )
        end_seconds = end_time.hour * 3600 + end_time.minute * 60 + end_time.second
        duration = end_seconds - start_seconds

        segments.append(
            {
                "key": item,
                "start": f"{start_time.hour:02d}:{start_time.minute:02d}",
                "end": f"{end_time.hour:02d}:{end_time.minute:02d}",
                "duration": duration,
                "speakers": speakers,
            }
        )

    return segments


def _load_segment_speaker_embedding(
    day: str, segment_key: str, speaker: str
) -> np.ndarray | None:
    """Load a speaker's embedding from a segment."""
    emb_path = day_path(day) / segment_key / "audio" / f"{speaker}.npz"
    if not emb_path.exists():
        return None

    data = np.load(emb_path)
    return _normalize_embedding(data["embedding"])


def _scan_entity_voiceprints(facet: str) -> dict[str, np.ndarray]:
    """Scan entities in a facet for voiceprints.

    Returns dict mapping entity name to averaged embedding.
    Each entity may have multiple voiceprint files (day_segment.npz).
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

        try:
            folder = entity_folder_path(facet, name)
        except (RuntimeError, ValueError):
            continue

        if not folder.is_dir():
            continue

        # Find all voiceprint files (pattern: YYYYMMDD_HHMMSS_LEN.npz)
        npz_files = list(folder.glob("*_*.npz"))
        if not npz_files:
            continue

        # Load and average all voiceprints
        embeddings = []
        for npz_path in npz_files:
            try:
                data = np.load(npz_path)
                emb = _normalize_embedding(data["embedding"])
                if emb is not None:
                    embeddings.append(emb)
            except Exception as e:
                logger.warning("Failed to load voiceprint %s: %s", npz_path, e)
                continue

        if embeddings:
            # Average all embeddings and re-normalize
            avg_emb = _normalize_embedding(np.mean(embeddings, axis=0))
            if avg_emb is not None:
                voiceprints[name] = avg_emb

    return voiceprints


def _compute_matches(
    segment_emb: np.ndarray, known_embs: dict[str, np.ndarray]
) -> dict[str, float]:
    """Compute cosine similarity between segment embedding and known voiceprints."""
    if not known_embs:
        return {}

    matches = {}
    for name, known_emb in known_embs.items():
        # Cosine similarity via dot product (both are L2-normalized)
        score = float(np.dot(segment_emb, known_emb))
        if score >= 0.4:  # Only include matches above threshold
            matches[name] = round(score, 4)

    return matches


def _save_voiceprint_to_entity(
    facet: str, entity_name: str, day: str, segment_key: str, embedding: np.ndarray
) -> Path:
    """Save a voiceprint embedding to an entity's folder."""
    folder = ensure_entity_folder(facet, entity_name)
    filename = f"{day}_{segment_key}.npz"
    emb_path = folder / filename
    np.savez_compressed(emb_path, embedding=embedding)
    return emb_path


@speakers_bp.route("/")
def index() -> Any:
    """Redirect to today's view."""
    today = date.today().strftime("%Y%m%d")
    return redirect(url_for("app:speakers.speakers_day", day=today))


@speakers_bp.route("/<day>")
def speakers_day(day: str) -> str:
    """Render speaker management view for a specific day."""
    if not re.fullmatch(DATE_RE.pattern, day):
        return "", 404

    title = format_date(day)
    return render_template("app.html", title=title)


@speakers_bp.route("/api/stats/<month>")
def api_stats(month: str) -> Any:
    """Return segment counts for each day in a month.

    Used by calendar heatmap to show days with speaker embeddings.
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
    """Return segments with speaker embeddings for a day."""
    if not re.fullmatch(DATE_RE.pattern, day):
        return error_response("Invalid day format", 400)

    segments = _scan_segment_embeddings(day)
    return jsonify({"segments": segments})


@speakers_bp.route("/api/segment/<day>/<segment_key>")
def api_segment_detail(day: str, segment_key: str) -> Any:
    """Return segment detail with speaker match results."""
    if not re.fullmatch(DATE_RE.pattern, day):
        return error_response("Invalid day format", 400)

    if not validate_segment_key(segment_key):
        return error_response("Invalid segment key", 400)

    # Get selected facet from cookie
    selected_facet = request.cookies.get("selected_facet")
    if not selected_facet:
        return error_response("No facet selected", 400)

    # Load ALL entities in the facet for dropdown
    try:
        all_entities = load_entities(selected_facet)
        all_entity_names = [e.get("name") for e in all_entities if e.get("name")]
    except RuntimeError:
        all_entity_names = []

    # Load known voiceprints for matching
    known_voiceprints = _scan_entity_voiceprints(selected_facet)

    # Load segment speaker embeddings
    audio_dir = day_path(day) / segment_key / "audio"
    if not audio_dir.is_dir():
        return error_response("Segment has no speaker embeddings", 404)

    speakers = []
    for npz_path in sorted(audio_dir.glob("*.npz")):
        speaker_label = npz_path.stem
        emb = _load_segment_speaker_embedding(day, segment_key, speaker_label)
        if emb is None:
            continue

        matches = _compute_matches(emb, known_voiceprints)
        speakers.append(
            {
                "label": speaker_label,
                "matches": matches,
            }
        )

    # Get audio file URL if available
    audio_file = None
    segment_dir = day_path(day) / segment_key
    audio_files = list(segment_dir.glob("*audio.flac"))
    if audio_files:
        rel_path = f"{segment_key}/{audio_files[0].name}"
        audio_file = (
            f"/app/speakers/api/serve_audio/{day}/{rel_path.replace('/', '__')}"
        )

    return jsonify(
        {
            "speakers": speakers,
            "all_entities": all_entity_names,
            "audio_file": audio_file,
            "facet": selected_facet,
        }
    )


@speakers_bp.route("/api/serve_audio/<day>/<path:encoded_path>")
def serve_audio(day: str, encoded_path: str) -> Any:
    """Serve audio files for playback."""
    if not re.fullmatch(DATE_RE.pattern, day):
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
    """Save a voiceprint from a segment speaker to an existing entity."""
    data = request.get_json()
    if not data:
        return error_response("No data provided", 400)

    facet = data.get("facet")
    entity_name = data.get("entity_name")
    day = data.get("day")
    segment_key = data.get("segment_key")
    speaker_label = data.get("speaker_label")

    if not all([facet, entity_name, day, segment_key, speaker_label]):
        return error_response("Missing required fields", 400)

    # Validate day and segment_key formats
    if not re.fullmatch(DATE_RE.pattern, day):
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

    # Load speaker embedding
    emb = _load_segment_speaker_embedding(day, segment_key, speaker_label)
    if emb is None:
        return error_response("Speaker embedding not found", 404)

    # Save voiceprint
    try:
        emb_path = _save_voiceprint_to_entity(facet, entity_name, day, segment_key, emb)

        # Log action (facet-scoped since voiceprints are tied to entities)
        log_app_action(
            app="speakers",
            facet=facet,
            action="voiceprint_save",
            params={
                "entity_name": entity_name,
                "day": day,
                "segment_key": segment_key,
                "speaker_label": speaker_label,
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
    speaker_label = data.get("speaker_label")

    if not all([facet, entity_name, day, segment_key, speaker_label]):
        return error_response("Missing required fields", 400)

    # Validate day and segment_key formats
    if not re.fullmatch(DATE_RE.pattern, day):
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

    # Load speaker embedding
    emb = _load_segment_speaker_embedding(day, segment_key, speaker_label)
    if emb is None:
        return error_response("Speaker embedding not found", 404)

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
        emb_path = _save_voiceprint_to_entity(facet, entity_name, day, segment_key, emb)

        # Log action (facet-scoped since entities are facet-scoped)
        log_app_action(
            app="speakers",
            facet=facet,
            action="entity_voiceprint_create",
            params={
                "entity_name": entity_name,
                "entity_type": entity_type,
                "day": day,
                "segment_key": segment_key,
                "speaker_label": speaker_label,
            },
        )

        return success_response(
            {"entity": new_entity, "voiceprint_path": str(emb_path)}
        )
    except Exception as e:
        logger.exception("Failed to save voiceprint for new entity %s", entity_name)
        return error_response(f"Failed to save voiceprint: {e}", 500)
