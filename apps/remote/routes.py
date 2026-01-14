# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Remote app - manage remote observer connections.

Provides endpoints for:
- Managing remote observer registrations (UI)
- Receiving file uploads from remote observers (ingest)
- Relaying events from remote observers to local Callosum
- Retrieving segment upload history for sync verification
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

from apps.utils import get_app_storage_path, log_app_action
from convey import emit
from observe.utils import (
    MAX_SEGMENT_ATTEMPTS,
    compute_bytes_sha256,
    find_available_segment,
)
from think.utils import day_path

from .utils import (
    append_history_record,
    find_segment_by_sha256,
    get_remotes_dir,
    list_remotes,
    load_history,
    load_remote,
    save_remote,
)

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


def _revoke_remote(key: str) -> bool:
    """Revoke remote by key (soft-delete)."""
    remote = load_remote(key)
    if not remote:
        return False
    remote["revoked"] = True
    remote["revoked_at"] = int(time.time() * 1000)
    return save_remote(remote)


# === Management API (session-protected) ===


@remote_bp.route("/api/list")
def api_list() -> Any:
    """List all registered remotes."""
    remotes = list_remotes()
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
                "revoked": r.get("revoked", False),
                "revoked_at": r.get("revoked_at"),
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

    if not save_remote(remote_data):
        return jsonify({"error": "Failed to save remote"}), 500

    # Log observer creation (journal-level, no facet)
    log_app_action(
        app="remote",
        facet=None,
        action="observer_create",
        params={"name": name, "key_prefix": key[:8]},
    )

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
    """Revoke a remote by key prefix (soft-delete)."""
    # Find remote by prefix
    remotes_dir = get_remotes_dir()
    remote_path = remotes_dir / f"{key_prefix}.json"
    if not remote_path.exists():
        return jsonify({"error": "Remote not found"}), 404

    try:
        with open(remote_path) as f:
            data = json.load(f)
        key = data.get("key", "")
        name = data.get("name", "")
    except (json.JSONDecodeError, OSError):
        return jsonify({"error": "Failed to read remote"}), 500

    if not _revoke_remote(key):
        return jsonify({"error": "Failed to revoke remote"}), 500

    # Log observer revocation (journal-level, no facet)
    log_app_action(
        app="remote",
        facet=None,
        action="observer_revoke",
        params={"name": name, "key_prefix": key_prefix},
    )

    return jsonify({"status": "ok"})


@remote_bp.route("/api/<key_prefix>/key")
def api_get_key(key_prefix: str) -> Any:
    """Get full key and ingest URL for a remote."""
    # Find remote by prefix
    remotes_dir = get_remotes_dir()
    remote_path = remotes_dir / f"{key_prefix}.json"
    if not remote_path.exists():
        return jsonify({"error": "Remote not found"}), 404

    try:
        with open(remote_path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return jsonify({"error": "Failed to read remote"}), 500

    key = data.get("key", "")
    return jsonify(
        {
            "key": key,
            "name": data.get("name", ""),
            "ingest_url": f"/app/remote/ingest/{key}",
        }
    )


# === Sync history helpers ===


def _find_by_inode(day_dir: Path, inode: int) -> Path | None:
    """Find a file by inode in the day directory.

    Searches recursively for a file with the given inode.

    Args:
        day_dir: Path to day directory
        inode: Inode number to search for

    Returns:
        Path to file if found, None otherwise
    """
    try:
        for path in day_dir.rglob("*"):
            if path.is_file():
                try:
                    if path.stat().st_ino == inode:
                        return path
                except OSError:
                    continue
    except OSError:
        pass
    return None


# === Segment collision helpers ===


def _strip_segment_prefix(filename: str, segment: str) -> str:
    """Strip segment prefix from filename if present.

    Handles old-style prefixed filenames (e.g., "143022_300_audio.flac")
    and returns simple names (e.g., "audio.flac").

    Args:
        filename: Original filename (may have segment prefix)
        segment: Segment key (HHMMSS_LEN)

    Returns:
        Simple filename without segment prefix
    """
    prefix = f"{segment}_"
    if filename.startswith(prefix):
        return filename[len(prefix) :]
    return filename


def _save_to_failed(
    day_dir: Path, file_data: list[tuple[str, str, bytes, str]], segment: str
) -> Path:
    """Save files to failed directory for manual review.

    Files are saved with their original segment key (not adjusted) since
    the collision resolution failed.

    Args:
        day_dir: Path to day directory
        file_data: List of (submitted_filename, simple_filename, content, sha256) tuples
        segment: Original segment key (used in directory name)

    Returns:
        Path to the failed directory where files were saved
    """
    # Use segment in path for easier identification of failed uploads
    failed_dir = day_dir / "remote" / "failed" / segment / str(int(time.time() * 1000))
    failed_dir.mkdir(parents=True, exist_ok=True)

    for submitted_filename, _simple_filename, content, _sha256 in file_data:
        target_path = failed_dir / submitted_filename
        target_path.write_bytes(content)

    return failed_dir


# === Ingest API (key-protected) ===


@remote_bp.route("/ingest/<key>", methods=["POST"])
def ingest_upload(key: str) -> Any:
    """Receive file uploads from remote observer.

    Expects multipart form with:
    - segment: Segment key (HHMMSS_LEN)
    - day: Day string (YYYYMMDD)
    - files: One or more media files

    Writes files to journal and emits observe.observing event.

    Returns status:
    - "ok": New segment accepted
    - "duplicate": All files already received (no processing triggered)
    - "collision": New segment saved with adjusted key (directory conflict)
    """
    # Validate key
    remote = load_remote(key)
    if not remote:
        return jsonify({"error": "Invalid key"}), 401

    if remote.get("revoked", False):
        return jsonify({"error": "Remote revoked"}), 403

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

    key_prefix = key[:8]

    # Read file contents into memory and compute SHA256 before saving
    # This allows duplicate detection without writing to disk
    file_data = []  # List of (submitted_filename, simple_filename, content, sha256)
    for upload in files:
        if not upload.filename:
            continue

        submitted_filename = secure_filename(upload.filename)
        if not submitted_filename:
            continue

        # Strip segment prefix from filename if present
        simple_filename = _strip_segment_prefix(submitted_filename, segment)

        # Read content and compute SHA256
        content = upload.read()
        sha256 = compute_bytes_sha256(content)

        file_data.append((submitted_filename, simple_filename, content, sha256))

    if not file_data:
        return jsonify({"error": "No valid files uploaded"}), 400

    # Check for duplicate submission by SHA256
    incoming_sha256s = {fd[3] for fd in file_data}
    existing_segment, matched_sha256s = find_segment_by_sha256(
        key_prefix, day, incoming_sha256s
    )

    if existing_segment:
        # Full duplicate - all files already exist in an existing segment
        logger.info(
            f"Duplicate segment rejected: {day}/{segment} from {remote.get('name')} "
            f"(matches existing {existing_segment})"
        )

        # Update last_seen and increment duplicates_rejected stat
        remote["last_seen"] = int(time.time() * 1000)
        remote["stats"]["duplicates_rejected"] = (
            remote["stats"].get("duplicates_rejected", 0) + 1
        )
        save_remote(remote)

        return jsonify(
            {
                "status": "duplicate",
                "existing_segment": existing_segment,
                "message": "All files already received",
            }
        )

    # Log partial match context if some files already exist
    partial_match = bool(matched_sha256s)

    # Ensure day directory exists
    day_dir = day_path(day)
    day_dir.mkdir(parents=True, exist_ok=True)

    # Find available segment key (may differ from original if collision)
    original_segment = segment
    available_segment = find_available_segment(day_dir, segment)

    if available_segment is None:
        # Exhausted attempts, save to failed directory
        logger.error(
            f"No available segment slot for {day}/{segment} from "
            f"{remote.get('name')} after {MAX_SEGMENT_ATTEMPTS} attempts"
        )
        failed_dir = _save_to_failed(day_dir, file_data, segment)
        return (
            jsonify(
                {
                    "status": "failed",
                    "error": f"No available segment slot after {MAX_SEGMENT_ATTEMPTS} attempts",
                    "failed_path": str(failed_dir.relative_to(day_dir.parent)),
                }
            ),
            507,
        )  # Insufficient Storage

    segment = available_segment
    if segment != original_segment:
        logger.info(
            f"Segment collision resolved: {original_segment} -> {segment} "
            f"for remote {remote.get('name')}"
        )

    # Create segment directory for files
    segment_dir = day_dir / segment
    segment_dir.mkdir(parents=True, exist_ok=True)

    # Save files from memory to disk
    saved_files = []
    file_records = []
    total_bytes = 0

    for submitted_filename, simple_filename, content, sha256 in file_data:
        target_path = segment_dir / simple_filename

        try:
            target_path.write_bytes(content)
            stat = target_path.stat()
            file_size = stat.st_size
            file_inode = stat.st_ino

            saved_files.append(simple_filename)
            total_bytes += file_size

            file_records.append(
                {
                    "submitted": submitted_filename,
                    "written": simple_filename,
                    "inode": file_inode,
                    "size": file_size,
                    "sha256": sha256,
                }
            )

            logger.info(f"Saved {simple_filename} to {segment_dir}")
        except OSError as e:
            logger.error(f"Failed to save {simple_filename}: {e}")
            return jsonify({"error": f"Failed to save {simple_filename}"}), 500

    if not saved_files:
        return jsonify({"error": "No valid files saved"}), 400

    # Write sync history record
    sync_record = {
        "ts": int(time.time() * 1000),
        "segment": segment,
        "files": file_records,
    }
    if segment != original_segment:
        sync_record["segment_original"] = original_segment
    if partial_match:
        # Log which SHA256s matched existing files (for debugging/audit)
        sync_record["partial_match_sha256s"] = list(matched_sha256s)
    append_history_record(key_prefix, day, sync_record)

    # Update remote stats
    remote["last_seen"] = int(time.time() * 1000)
    remote["last_segment"] = segment
    remote["stats"]["segments_received"] = (
        remote["stats"].get("segments_received", 0) + 1
    )
    remote["stats"]["bytes_received"] = (
        remote["stats"].get("bytes_received", 0) + total_bytes
    )
    save_remote(remote)

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

    # Determine response status
    if segment != original_segment:
        status = "collision"
    else:
        status = "ok"

    return jsonify(
        {
            "status": status,
            "segment": segment,
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
    remote = load_remote(key)
    if not remote:
        return jsonify({"error": "Invalid key"}), 401

    if remote.get("revoked", False):
        return jsonify({"error": "Remote revoked"}), 403

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
        save_remote(remote)

    return jsonify({"status": "ok"})


@remote_bp.route("/ingest/<key>/segments/<day>")
def ingest_segments(key: str, day: str) -> Any:
    """List uploaded segments for a day with file verification.

    Returns JSON array of segments with file status:
    - present: File exists at recorded path
    - relocated: File found at different path (by inode)
    - missing: File not found

    Args:
        key: Remote authentication key
        day: Day string (YYYYMMDD)
    """
    # Validate key
    remote = load_remote(key)
    if not remote:
        return jsonify({"error": "Invalid key"}), 401

    if remote.get("revoked", False):
        return jsonify({"error": "Remote revoked"}), 403

    if not remote.get("enabled", True):
        return jsonify({"error": "Remote disabled"}), 403

    # Validate day format (YYYYMMDD)
    if not re.match(r"^\d{8}$", day):
        return jsonify({"error": "Invalid day format"}), 400

    # Load sync history for this remote/day
    key_prefix = key[:8]
    records = load_history(key_prefix, day)

    if not records:
        return jsonify([])

    # Get day directory for file verification
    day_dir = day_path(day)

    # Build response grouped by segment, deduplicating by sha256
    # Later records overwrite earlier ones (most recent upload wins)
    segments: dict[str, dict] = {}
    observed_segments: set[str] = set()  # Track which segments have been observed

    for record in records:
        # Handle "observed" record type (from event handler)
        record_type = record.get("type", "upload")
        if record_type == "observed":
            observed_segments.add(record.get("segment", ""))
            continue

        segment = record.get("segment", "")
        segment_original = record.get("segment_original")

        if segment not in segments:
            segments[segment] = {
                "key": segment,
                "files_by_sha": {},  # Keyed by sha256 for deduplication
            }
            if segment_original:
                segments[segment]["original_key"] = segment_original

        # Check each file's status
        for file_rec in record.get("files", []):
            written = file_rec.get("written", "")
            submitted = file_rec.get("submitted", "")
            inode = file_rec.get("inode")
            size = file_rec.get("size", 0)
            sha256 = file_rec.get("sha256", "")

            file_info = {
                "name": written,
                "size": size,
                "sha256": sha256,
            }

            # Include submitted_name only if different
            if submitted != written:
                file_info["submitted_name"] = submitted

            # Check file status - files are now in segment directories
            segment_dir = day_dir / segment
            recorded_path = segment_dir / written
            if recorded_path.exists():
                file_info["status"] = "present"
            elif inode and day_dir.exists():
                # Try to find by inode
                relocated = _find_by_inode(day_dir, inode)
                if relocated:
                    file_info["status"] = "relocated"
                    file_info["current_path"] = str(relocated.relative_to(day_dir))
                else:
                    file_info["status"] = "missing"
            else:
                file_info["status"] = "missing"

            # Deduplicate by sha256 - later uploads overwrite earlier
            segments[segment]["files_by_sha"][sha256] = file_info

    # Convert files_by_sha dicts to lists and sort by segment key
    result = []
    for segment_data in sorted(segments.values(), key=lambda s: s["key"]):
        segment_key = segment_data["key"]
        entry = {
            "key": segment_key,
            "observed": segment_key in observed_segments,
            "files": list(segment_data["files_by_sha"].values()),
        }
        if "original_key" in segment_data:
            entry["original_key"] = segment_data["original_key"]
        result.append(entry)
    return jsonify(result)
