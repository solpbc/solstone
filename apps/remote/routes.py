# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Remote app - manage remote observer connections.

Provides endpoints for:
- Managing remote observer registrations (UI)
- Receiving file uploads from remote observers (ingest)
- Relaying events from remote observers to local Callosum
"""

from __future__ import annotations

import base64
import json
import logging
import re
import secrets
import time
from pathlib import Path
from typing import Any

from flask import Blueprint, jsonify, request
from werkzeug.utils import secure_filename

from apps.utils import get_app_storage_path
from convey import emit
from think.utils import day_path

logger = logging.getLogger(__name__)

remote_bp = Blueprint(
    "app:remote",
    __name__,
    url_prefix="/app/remote",
)

# Key length in bytes (256 bits = 32 bytes)
KEY_BYTES = 32


def _generate_key() -> str:
    """Generate a URL-safe key for remote authentication."""
    return base64.urlsafe_b64encode(secrets.token_bytes(KEY_BYTES)).decode().rstrip("=")


def _get_remotes_dir() -> Path:
    """Get the remotes storage directory."""
    return get_app_storage_path("remote", "remotes", ensure_exists=True)


def _load_remote(key: str) -> dict | None:
    """Load remote metadata by key."""
    remotes_dir = _get_remotes_dir()
    # Use first 8 chars of key as filename for readability
    remote_path = remotes_dir / f"{key[:8]}.json"
    if not remote_path.exists():
        return None
    try:
        with open(remote_path) as f:
            data = json.load(f)
        # Verify full key matches
        if data.get("key") != key:
            return None
        return data
    except (json.JSONDecodeError, OSError):
        return None


def _save_remote(data: dict) -> bool:
    """Save remote metadata."""
    key = data.get("key")
    if not key:
        return False
    remotes_dir = _get_remotes_dir()
    remote_path = remotes_dir / f"{key[:8]}.json"
    try:
        with open(remote_path, "w") as f:
            json.dump(data, f, indent=2)
        return True
    except OSError:
        return False


def _delete_remote(key: str) -> bool:
    """Delete remote by key."""
    remotes_dir = _get_remotes_dir()
    remote_path = remotes_dir / f"{key[:8]}.json"
    try:
        if remote_path.exists():
            remote_path.unlink()
            return True
        return False
    except OSError:
        return False


def _list_remotes() -> list[dict]:
    """List all registered remotes."""
    remotes_dir = _get_remotes_dir()
    remotes = []
    for remote_path in remotes_dir.glob("*.json"):
        try:
            with open(remote_path) as f:
                data = json.load(f)
            remotes.append(data)
        except (json.JSONDecodeError, OSError):
            continue
    # Sort by created_at descending
    remotes.sort(key=lambda x: x.get("created_at", 0), reverse=True)
    return remotes


# === Management API (session-protected) ===


@remote_bp.route("/api/list")
def api_list() -> Any:
    """List all registered remotes."""
    remotes = _list_remotes()
    # Sanitize output - don't expose full keys
    result = []
    for r in remotes:
        result.append(
            {
                "key_prefix": r.get("key", "")[:8],
                "name": r.get("name", ""),
                "created_at": r.get("created_at", 0),
                "last_seen": r.get("last_seen"),
                "last_segment": r.get("last_segment"),
                "enabled": r.get("enabled", True),
                "stats": r.get("stats", {}),
            }
        )
    return jsonify(result)


@remote_bp.route("/api/create", methods=["POST"])
def api_create() -> Any:
    """Create a new remote registration."""
    data = request.get_json(force=True) if request.is_json else {}
    name = data.get("name", "").strip()
    if not name:
        return jsonify({"error": "Name is required"}), 400

    # Generate key
    key = _generate_key()

    # Create remote record
    remote_data = {
        "key": key,
        "name": name,
        "created_at": int(time.time() * 1000),
        "last_seen": None,
        "last_segment": None,
        "enabled": True,
        "stats": {
            "segments_received": 0,
            "bytes_received": 0,
        },
    }

    if not _save_remote(remote_data):
        return jsonify({"error": "Failed to save remote"}), 500

    # Build ingest URL
    ingest_url = f"/app/remote/ingest/{key}"

    return jsonify(
        {
            "key": key,
            "key_prefix": key[:8],
            "name": name,
            "ingest_url": ingest_url,
        }
    )


@remote_bp.route("/api/<key_prefix>", methods=["DELETE"])
def api_delete(key_prefix: str) -> Any:
    """Delete/revoke a remote by key prefix."""
    # Find remote by prefix
    remotes_dir = _get_remotes_dir()
    remote_path = remotes_dir / f"{key_prefix}.json"
    if not remote_path.exists():
        return jsonify({"error": "Remote not found"}), 404

    try:
        with open(remote_path) as f:
            data = json.load(f)
        key = data.get("key", "")
    except (json.JSONDecodeError, OSError):
        return jsonify({"error": "Failed to read remote"}), 500

    if not _delete_remote(key):
        return jsonify({"error": "Failed to delete remote"}), 500

    return jsonify({"status": "ok"})


# === Ingest API (key-protected) ===


@remote_bp.route("/ingest/<key>", methods=["POST"])
def ingest_upload(key: str) -> Any:
    """Receive file uploads from remote observer.

    Expects multipart form with:
    - segment: Segment key (HHMMSS_LEN)
    - day: Day string (YYYYMMDD)
    - files: One or more media files

    Writes files to journal and emits observe.observing event.
    """
    # Validate key
    remote = _load_remote(key)
    if not remote:
        return jsonify({"error": "Invalid key"}), 401

    if not remote.get("enabled", True):
        return jsonify({"error": "Remote disabled"}), 403

    # Get segment, day, and host info from form
    segment = request.form.get("segment", "").strip()
    day = request.form.get("day", "").strip()
    host = request.form.get("host", "").strip()
    platform = request.form.get("platform", "").strip()

    if not segment:
        return jsonify({"error": "Missing segment"}), 400
    if not day:
        return jsonify({"error": "Missing day"}), 400

    # Validate segment format (HHMMSS_LEN)
    if not re.match(r"^\d{6}_\d+$", segment):
        return jsonify({"error": "Invalid segment format"}), 400

    # Validate day format (YYYYMMDD)
    if not re.match(r"^\d{8}$", day):
        return jsonify({"error": "Invalid day format"}), 400

    # Get uploaded files
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "No files uploaded"}), 400

    # Ensure day directory exists
    target_dir = day_path(day)
    target_dir.mkdir(parents=True, exist_ok=True)

    # Save files
    saved_files = []
    total_bytes = 0

    for upload in files:
        if not upload.filename:
            continue

        # Secure the filename
        filename = secure_filename(upload.filename)
        if not filename:
            continue

        target_path = target_dir / filename

        # Save file
        try:
            upload.save(target_path)
            saved_files.append(filename)
            total_bytes += target_path.stat().st_size
            logger.info(f"Saved {filename} to {target_dir}")
        except OSError as e:
            logger.error(f"Failed to save {filename}: {e}")
            return jsonify({"error": f"Failed to save {filename}"}), 500

    if not saved_files:
        return jsonify({"error": "No valid files saved"}), 400

    # Update remote stats
    remote["last_seen"] = int(time.time() * 1000)
    remote["last_segment"] = segment
    remote["stats"]["segments_received"] = (
        remote["stats"].get("segments_received", 0) + 1
    )
    remote["stats"]["bytes_received"] = (
        remote["stats"].get("bytes_received", 0) + total_bytes
    )
    _save_remote(remote)

    # Emit observe.observing event to local Callosum
    # Include host/platform from remote observer if provided
    event_fields = {
        "segment": segment,
        "day": day,
        "files": saved_files,
        "remote": remote.get("name", "unknown"),
    }
    if host:
        event_fields["host"] = host
    if platform:
        event_fields["platform"] = platform
    emit("observe", "observing", **event_fields)

    logger.info(
        f"Received {len(saved_files)} files for {day}/{segment} from {remote.get('name')}"
    )

    return jsonify(
        {
            "status": "ok",
            "files": saved_files,
            "bytes": total_bytes,
        }
    )


@remote_bp.route("/ingest/<key>/event", methods=["POST"])
def ingest_event(key: str) -> Any:
    """Receive events from remote observer and relay to local Callosum.

    Expects JSON body with:
    - tract: Event tract
    - event: Event name
    - ...additional fields
    """
    # Validate key
    remote = _load_remote(key)
    if not remote:
        return jsonify({"error": "Invalid key"}), 401

    if not remote.get("enabled", True):
        return jsonify({"error": "Remote disabled"}), 403

    # Parse event
    data = request.get_json(force=True) if request.is_json else {}

    tract = data.get("tract")
    event = data.get("event")

    if not tract or not event:
        return jsonify({"error": "Missing tract or event"}), 400

    # Add remote identifier
    data["remote"] = remote.get("name", "unknown")

    # Relay to local Callosum
    emit(tract, event, **{k: v for k, v in data.items() if k not in ("tract", "event")})

    # Update last_seen on status events
    if tract == "observe" and event == "status":
        remote["last_seen"] = int(time.time() * 1000)
        _save_remote(remote)

    return jsonify({"status": "ok"})
