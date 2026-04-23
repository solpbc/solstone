# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Observer app - manage observer connections.

Provides endpoints for:
- Managing observer registrations (UI)
- Receiving file uploads from observers (ingest)
- Receiving transferred segments from other instances (transfer ingest)
- Serving segment manifests for transfer diffing
- Relaying events from observers to local Callosum
- Retrieving segment upload history for sync verification
"""

from __future__ import annotations

import base64
import json
import logging
import platform
import re
import secrets
from pathlib import Path
from typing import Any

from flask import Blueprint, jsonify, request
from werkzeug.utils import secure_filename

from apps.utils import log_app_action
from convey import emit
from observe.utils import (
    MAX_SEGMENT_ATTEMPTS,
    compute_bytes_sha256,
    compute_file_sha256,
    find_available_segment,
)
from think.streams import stream_name, update_stream, write_segment_stream
from think.utils import day_path, iter_segments, now_ms, segment_path

from .utils import (
    append_history_record,
    find_segment_by_sha256,
    get_hist_dir,
    get_observers_dir,
    list_observers,
    load_history,
    load_observer,
    save_observer,
)

logger = logging.getLogger(__name__)

observer_bp = Blueprint(
    "app:observer",
    __name__,
    url_prefix="/app/observer",
)

# Key length in bytes (256 bits = 32 bytes)
KEY_BYTES = 32
ACTIVE_THRESHOLD_MS = 30_000
STALE_THRESHOLD_MS = 120_000
FUTURE_CLOCK_DRIFT_TOLERANCE_MS = 5 * 60 * 1000

OBSERVER_STATE_LABELS = {
    "connected": "Connected",
    "stale": "Stale",
    "disconnected": "Disconnected",
    "revoked": "Revoked",
}


def _get_key(url_key: str | None = None) -> str | None:
    """Extract auth key from Authorization: Bearer header (primary) or URL path (legacy)."""
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        bearer = auth[7:].strip()
        if bearer:
            return bearer
    return url_key or None


def _generate_key() -> str:
    """Generate a URL-safe key for observer authentication."""
    return base64.urlsafe_b64encode(secrets.token_bytes(KEY_BYTES)).decode().rstrip("=")


def _classify_observer_freshness(
    last_seen_ms: int | None,
    revoked: bool,
    now_ms: int,
) -> dict[str, object]:
    """Classify a registered observer's freshness.

    Returns keys: state, group, elapsed_ms, clock_skew.
    """
    if revoked:
        return {
            "state": "revoked",
            "group": "inactive",
            "elapsed_ms": None,
            "clock_skew": False,
        }
    if last_seen_ms is None:
        return {
            "state": "disconnected",
            "group": "inactive",
            "elapsed_ms": None,
            "clock_skew": False,
        }
    elapsed = now_ms - last_seen_ms
    if elapsed < -FUTURE_CLOCK_DRIFT_TOLERANCE_MS:
        return {
            "state": "disconnected",
            "group": "inactive",
            "elapsed_ms": elapsed,
            "clock_skew": True,
        }
    if elapsed < 0:
        return {
            "state": "connected",
            "group": "active",
            "elapsed_ms": 0,
            "clock_skew": False,
        }
    if elapsed < ACTIVE_THRESHOLD_MS:
        return {
            "state": "connected",
            "group": "active",
            "elapsed_ms": elapsed,
            "clock_skew": False,
        }
    if elapsed < STALE_THRESHOLD_MS:
        return {
            "state": "stale",
            "group": "stale",
            "elapsed_ms": elapsed,
            "clock_skew": False,
        }
    return {
        "state": "disconnected",
        "group": "inactive",
        "elapsed_ms": elapsed,
        "clock_skew": False,
    }


def _serialize_observer(observer: dict[str, Any], current_now: int) -> dict[str, Any]:
    """Serialize a registered observer for management API consumers."""
    freshness = _classify_observer_freshness(
        observer.get("last_seen"),
        observer.get("revoked", False),
        current_now,
    )
    return {
        "key_prefix": observer.get("key", "")[:8],
        "name": observer.get("name", ""),
        "created_at": observer.get("created_at", 0),
        "last_seen": observer.get("last_seen"),
        "last_segment": observer.get("last_segment"),
        "enabled": observer.get("enabled", True),
        "revoked": observer.get("revoked", False),
        "revoked_at": observer.get("revoked_at"),
        "stats": observer.get("stats", {}),
        **freshness,
        "label": OBSERVER_STATE_LABELS[str(freshness["state"])],
    }


def _revoke_observer(key: str) -> bool:
    """Revoke observer by key (soft-delete)."""
    observer = load_observer(key)
    if not observer:
        return False
    observer["revoked"] = True
    observer["revoked_at"] = now_ms()
    return save_observer(observer)


# === Management API (session-protected) ===


@observer_bp.route("/api/list")
def api_list() -> Any:
    """List all registered observers."""
    current_now = now_ms()
    observers = list_observers()
    # Sanitize output - don't expose full keys
    result = [_serialize_observer(observer, current_now) for observer in observers]

    group_order = {"active": 0, "stale": 1, "inactive": 2}
    result.sort(
        key=lambda observer: (
            group_order[observer.get("group", "inactive")],
            1 if observer.get("last_seen") is None else 0,
            -(observer.get("last_seen") or 0),
            observer.get("key_prefix", ""),
        )
    )

    return jsonify(
        {
            "thresholds": {
                "active_ms": ACTIVE_THRESHOLD_MS,
                "stale_ms": STALE_THRESHOLD_MS,
            },
            "observers": result,
        }
    )


@observer_bp.route("/api/create", methods=["POST"])
def api_create() -> Any:
    """Create a new observer registration."""
    data = request.get_json(force=True) if request.is_json else {}
    name = data.get("name", "").strip()
    if not name:
        return jsonify({"error": "Name is required"}), 400

    # Generate key
    key = _generate_key()

    # Create observer record
    observer_data = {
        "key": key,
        "name": name,
        "created_at": now_ms(),
        "last_seen": None,
        "last_segment": None,
        "enabled": True,
        "stats": {
            "segments_received": 0,
            "bytes_received": 0,
        },
    }

    if not save_observer(observer_data):
        return jsonify({"error": "Failed to save observer"}), 500

    # Log observer creation (journal-level, no facet)
    log_app_action(
        app="observer",
        facet=None,
        action="observer_create",
        params={"name": name, "key_prefix": key[:8]},
    )

    # Build ingest URL
    ingest_url = f"/app/observer/ingest/{key}"

    return jsonify(
        {
            "key": key,
            "key_prefix": key[:8],
            "name": name,
            "ingest_url": ingest_url,
        }
    )


@observer_bp.route("/api/<key_prefix>", methods=["DELETE"])
def api_delete(key_prefix: str) -> Any:
    """Revoke an observer by key prefix (soft-delete)."""
    # Find observer by prefix
    observers_dir = get_observers_dir()
    observer_path = observers_dir / f"{key_prefix}.json"
    if not observer_path.exists():
        return jsonify({"error": "Observer not found"}), 404

    try:
        with open(observer_path) as f:
            data = json.load(f)
        key = data.get("key", "")
        name = data.get("name", "")
    except (json.JSONDecodeError, OSError):
        return jsonify({"error": "Failed to read observer"}), 500

    if not _revoke_observer(key):
        return jsonify({"error": "Failed to revoke observer"}), 500

    # Log observer revocation (journal-level, no facet)
    log_app_action(
        app="observer",
        facet=None,
        action="observer_revoke",
        params={"name": name, "key_prefix": key_prefix},
    )

    return jsonify({"status": "ok"})


@observer_bp.route("/api/<key_prefix>/key")
def api_get_key(key_prefix: str) -> Any:
    """Get full key and ingest URL for an observer."""
    # Find observer by prefix
    observers_dir = get_observers_dir()
    observer_path = observers_dir / f"{key_prefix}.json"
    if not observer_path.exists():
        return jsonify({"error": "Observer not found"}), 404

    try:
        with open(observer_path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return jsonify({"error": "Failed to read observer"}), 500

    if data.get("revoked", False):
        return jsonify({"error": "key unavailable — observer revoked"}), 403

    log_app_action(
        app="observer",
        facet=None,
        action="observer_key_view",
        params={"name": data.get("name", ""), "key_prefix": key_prefix},
    )

    key = data.get("key", "")
    return jsonify(
        {
            "key": key,
            "name": data.get("name", ""),
            "ingest_url": f"/app/observer/ingest/{key}",
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
    failed_dir = day_dir / "observer" / "failed" / segment / str(now_ms())
    failed_dir.mkdir(parents=True, exist_ok=True)

    for submitted_filename, _simple_filename, content, _sha256 in file_data:
        target_path = failed_dir / submitted_filename
        target_path.write_bytes(content)

    return failed_dir


# === Ingest API (key-protected) ===


def _process_ingest_files(
    observer: dict,
    key_prefix: str,
    segment: str,
    day: str,
    stream: str,
    uploaded_files,
    *,
    source: str | None = None,
) -> tuple[dict, int]:
    """Shared ingest pipeline: read/hash files, dedup, deconflict, save, record history, update stats.

    Parameters
    ----------
    observer : dict
        Observer metadata dict (must include 'stats', 'name', 'last_seen', etc.)
    key_prefix : str
        First 8 chars of observer key.
    segment : str
        Requested segment key (HHMMSS_LEN format).
    day : str
        Day string (YYYYMMDD format).
    stream : str
        Stream name (already resolved by caller).
    uploaded_files : list
        List of Flask FileStorage objects from request.files.getlist("files").
    source : str or None
        If provided, added as "source" field to history record (e.g., "transfer").

    Returns
    -------
    tuple of (dict, int)
        Response body dict and HTTP status code.
    """
    # Read file contents into memory and compute SHA256 before saving
    # This allows duplicate detection without writing to disk
    file_data = []  # List of (submitted_filename, simple_filename, content, sha256)
    for upload in uploaded_files:
        if not upload.filename:
            continue

        submitted_filename = secure_filename(upload.filename)
        if not submitted_filename:
            continue

        # Strip segment prefix from filename if present
        simple_filename = _strip_segment_prefix(submitted_filename, segment)

        # Read content and compute SHA256
        content = upload.read()
        if len(content) == 0:
            logger.warning(f"Skipping 0-byte file: {submitted_filename}")
            continue
        sha256 = compute_bytes_sha256(content)

        file_data.append((submitted_filename, simple_filename, content, sha256))

    if not file_data:
        return {"error": "No valid files uploaded"}, 400

    # Check for duplicate submission by SHA256
    incoming_sha256s = {fd[3] for fd in file_data}
    existing_segment, matched_sha256s = find_segment_by_sha256(
        key_prefix, day, incoming_sha256s
    )

    if existing_segment:
        logger.info(
            f"Duplicate segment rejected: {day}/{segment} from {observer.get('name')} "
            f"(matches existing {existing_segment})"
        )

        observer["last_seen"] = now_ms()
        observer["stats"]["duplicates_rejected"] = (
            observer["stats"].get("duplicates_rejected", 0) + 1
        )
        save_observer(observer)

        return (
            {
                "status": "duplicate",
                "existing_segment": existing_segment,
                "message": "All files already received",
            },
            200,
        )

    partial_match = bool(matched_sha256s)

    # Ensure day directory exists
    day_dir = day_path(day)
    day_dir.mkdir(parents=True, exist_ok=True)

    # Find available segment key within the stream directory
    stream_dir = day_dir / stream
    stream_dir.mkdir(parents=True, exist_ok=True)

    original_segment = segment
    available_segment = find_available_segment(stream_dir, segment)

    if available_segment is None:
        logger.error(
            f"No available segment slot for {day}/{stream}/{segment} from "
            f"{observer.get('name', 'unknown')} after {MAX_SEGMENT_ATTEMPTS} attempts"
        )
        failed_dir = _save_to_failed(day_dir, file_data, segment)
        return (
            {
                "status": "failed",
                "error": f"No available segment slot after {MAX_SEGMENT_ATTEMPTS} attempts",
                "failed_path": str(failed_dir.relative_to(day_dir.parent)),
            },
            507,
        )

    segment = available_segment
    if segment != original_segment:
        logger.info(
            f"Segment collision resolved: {original_segment} -> {segment} "
            f"for observer {observer.get('name', 'unknown')}"
        )

    # Create segment directory for files (under stream)
    segment_dir = segment_path(day, segment, stream)
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
            return {"error": f"Failed to save {simple_filename}"}, 500

    if not saved_files:
        return {"error": "No valid files saved"}, 400

    sync_record = {
        "ts": now_ms(),
        "segment": segment,
        "stream": stream,
        "files": file_records,
    }
    if segment != original_segment:
        sync_record["segment_original"] = original_segment
    if partial_match:
        sync_record["partial_match_sha256s"] = list(matched_sha256s)
    if source:
        sync_record["source"] = source
    append_history_record(key_prefix, day, sync_record)

    observer["last_seen"] = now_ms()
    observer["last_segment"] = segment
    observer["stats"]["segments_received"] = (
        observer["stats"].get("segments_received", 0) + 1
    )
    observer["stats"]["bytes_received"] = (
        observer["stats"].get("bytes_received", 0) + total_bytes
    )
    save_observer(observer)

    status = "collision" if segment != original_segment else "ok"
    return {
        "status": status,
        "segment": segment,
        "files": saved_files,
        "bytes": total_bytes,
    }, 200


@observer_bp.route("/ingest", methods=["POST"])
@observer_bp.route("/ingest/<key>", methods=["POST"])
def ingest_upload(key: str | None = None) -> Any:
    """Receive file uploads from observer.

    Expects multipart form with:
    - segment: Segment key (HHMMSS_LEN)
    - day: Day string (YYYYMMDD)
    - files: One or more media files
    - host: (optional) Hostname of observer
    - platform: (optional) Platform of observer
    - meta: (optional) JSON-encoded metadata dict (facet, setting, etc.)

    Writes files to journal and emits observe.observing event.
    Host/platform are merged into meta (meta values take precedence).

    Returns status:
    - "ok": New segment accepted
    - "duplicate": All files already received (no processing triggered)
    - "collision": New segment saved with adjusted key (directory conflict)
    """
    # Extract key from Bearer header (primary) or URL path (legacy)
    auth_key = _get_key(key)
    if not auth_key:
        return jsonify({"error": "Authorization required"}), 401

    # Validate key
    observer = load_observer(auth_key)
    if not observer:
        return jsonify({"error": "Invalid key"}), 401

    if observer.get("revoked", False):
        return jsonify({"error": "Observer revoked"}), 403

    if not observer.get("enabled", True):
        return jsonify({"error": "Observer disabled"}), 403

    # Get segment, day, and host info from form
    segment = request.form.get("segment", "").strip()
    day = request.form.get("day", "").strip()
    host = request.form.get("host", "").strip()
    platform = request.form.get("platform", "").strip()
    meta_str = request.form.get("meta", "").strip()

    # Parse meta JSON and merge host/platform (meta values take precedence)
    meta: dict = {}
    if meta_str:
        try:
            meta = json.loads(meta_str)
        except json.JSONDecodeError:
            logger.warning(f"Invalid meta JSON from observer: {meta_str[:100]}")
    if host and "host" not in meta:
        meta["host"] = host
    if platform and "platform" not in meta:
        meta["platform"] = platform

    # Warn if client hostname differs from registered observer name
    effective_host = meta.get("host", host)
    observer_name = observer.get("name", "")
    if effective_host and effective_host != observer_name:
        logger.warning(
            f"Observer '{observer_name}' ({auth_key[:8]}) connecting from host "
            f"'{effective_host}' — hostname differs from registered name. "
            f"Use `sol observer rename` to update if the host was renamed."
        )

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

    key_prefix = auth_key[:8]

    # Determine stream name: trust client-provided stream in meta if valid,
    # otherwise derive from observer registration name.
    # Deriving from observer name via stream_name(observer=...) calls _strip_hostname,
    # which strips qualifiers like ".tmux" — so "fedora.tmux" becomes "fedora",
    # colliding both observers into one stream.
    client_stream = meta.get("stream", "").strip()
    observer_name = observer.get("name", "unknown")
    if client_stream and re.match(r"^[a-z0-9][a-z0-9._-]*$", client_stream):
        stream = client_stream
    else:
        stream = stream_name(observer=observer_name)

    body, status = _process_ingest_files(
        observer, key_prefix, segment, day, stream, files
    )
    if status != 200 or body.get("status") == "duplicate":
        return jsonify(body), status

    segment = body["segment"]
    saved_files = body["files"]
    segment_dir = segment_path(day, segment, stream)

    # Write stream identity for this segment
    try:
        result = update_stream(stream, day, segment, type="observer")
        write_segment_stream(
            segment_dir,
            stream,
            result["prev_day"],
            result["prev_segment"],
            result["seq"],
        )
    except Exception as e:
        logger.warning(f"Failed to write stream identity: {e}")

    # Add stream to meta for downstream handlers
    meta["stream"] = stream

    # Emit observe.observing event to local Callosum
    # Include meta dict with host/platform and any client-provided metadata
    event_fields: dict[str, Any] = {
        "segment": segment,
        "day": day,
        "files": saved_files,
        "observer": observer_name,
        "stream": stream,
    }
    if meta:
        event_fields["meta"] = meta
    emit("observe", "observing", **event_fields)

    logger.info(
        f"Received {len(saved_files)} files for {day}/{segment} from {observer.get('name')}"
    )
    return jsonify(body), status


@observer_bp.route("/ingest/<key>/transfer", methods=["POST"])
def ingest_transfer(key: str) -> Any:
    """Receive transferred file uploads from another solstone instance."""
    auth_key = _get_key(key)
    if not auth_key:
        return jsonify({"error": "Authorization required"}), 401

    observer = load_observer(auth_key)
    if not observer:
        return jsonify({"error": "Invalid key"}), 401

    if observer.get("revoked", False):
        return jsonify({"error": "Observer revoked"}), 403

    if not observer.get("enabled", True):
        return jsonify({"error": "Observer disabled"}), 403

    segment = request.form.get("segment", "").strip()
    day = request.form.get("day", "").strip()
    stream = request.form.get("stream", "").strip()
    host = request.form.get("host", "").strip()
    platform_name = request.form.get("platform", "").strip()
    meta_str = request.form.get("meta", "").strip()

    meta: dict = {}
    if meta_str:
        try:
            meta = json.loads(meta_str)
        except json.JSONDecodeError:
            logger.warning(f"Invalid meta JSON from observer: {meta_str[:100]}")
    if host and "host" not in meta:
        meta["host"] = host
    if platform_name and "platform" not in meta:
        meta["platform"] = platform_name

    if not segment:
        return jsonify({"error": "Missing segment"}), 400
    if not day:
        return jsonify({"error": "Missing day"}), 400
    if not stream:
        return jsonify({"error": "Missing stream"}), 400
    if not re.match(r"^\d{6}_\d+$", segment):
        return jsonify({"error": "Invalid segment format"}), 400
    if not re.match(r"^\d{8}$", day):
        return jsonify({"error": "Invalid day format"}), 400
    if not re.match(r"^[a-z0-9][a-z0-9._-]*$", stream):
        return jsonify({"error": "Invalid stream format"}), 400

    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "No files uploaded"}), 400

    key_prefix = auth_key[:8]
    body, status = _process_ingest_files(
        observer,
        key_prefix,
        segment,
        day,
        stream,
        files,
        source="transfer",
    )
    if status != 200 or body.get("status") == "duplicate":
        return jsonify(body), status

    observer_name = observer.get("name", "")
    event_fields: dict[str, Any] = {
        "segment": body["segment"],
        "day": day,
        "files": body["files"],
        "observer": observer_name,
        "stream": stream,
    }
    if meta:
        event_fields["meta"] = meta
    emit("observe", "transferred", **event_fields)

    return jsonify(body), status


@observer_bp.route("/ingest/<key>/manifest", methods=["GET"])
def ingest_manifest(key: str) -> Any:
    """List available manifest days for an observer."""
    auth_key = _get_key(key)
    if not auth_key:
        return jsonify({"error": "Authorization required"}), 401

    observer = load_observer(auth_key)
    if not observer:
        return jsonify({"error": "Invalid key"}), 401

    if observer.get("revoked", False):
        return jsonify({"error": "Observer revoked"}), 403

    if not observer.get("enabled", True):
        return jsonify({"error": "Observer disabled"}), 403

    key_prefix = auth_key[:8]
    hist_dir = get_hist_dir(key_prefix, ensure_exists=False)
    if not hist_dir.exists():
        return jsonify({"days": {}})

    days: dict[str, dict[str, int]] = {}
    for hist_path in sorted(hist_dir.glob("*.jsonl")):
        records = load_history(key_prefix, hist_path.stem)
        segments = {
            record.get("segment", "")
            for record in records
            if not record.get("type") and record.get("segment")
        }
        days[hist_path.stem] = {"segments": len(segments)}

    return jsonify({"days": days})


@observer_bp.route("/ingest/<key>/manifest/<day>", methods=["GET"])
def ingest_manifest_day(key: str, day: str) -> Any:
    """Return a transfer manifest for all segments on a given day."""
    auth_key = _get_key(key)
    if not auth_key:
        return jsonify({"error": "Authorization required"}), 401

    observer = load_observer(auth_key)
    if not observer:
        return jsonify({"error": "Invalid key"}), 401

    if observer.get("revoked", False):
        return jsonify({"error": "Observer revoked"}), 403

    if not observer.get("enabled", True):
        return jsonify({"error": "Observer disabled"}), 403

    if not re.match(r"^\d{8}$", day):
        return jsonify({"error": "Invalid day format"}), 400

    manifest = {
        "version": 1,
        "day": day,
        "created_at": now_ms(),
        "host": platform.node() or "unknown",
        "segments": {},
    }

    for stream, seg_key, seg_path in iter_segments(day):
        arc_key = f"{stream}/{seg_key}"
        files = []
        for file_path in sorted(seg_path.iterdir()):
            if file_path.is_file():
                files.append(
                    {
                        "name": file_path.name,
                        "sha256": compute_file_sha256(file_path),
                        "size": file_path.stat().st_size,
                    }
                )
        manifest["segments"][arc_key] = {"files": files}

    return jsonify(manifest)


@observer_bp.route("/ingest/event", methods=["POST"])
@observer_bp.route("/ingest/<key>/event", methods=["POST"])
def ingest_event(key: str | None = None) -> Any:
    """Receive events from observer and relay to local Callosum.

    Expects JSON body with:
    - tract: Event tract
    - event: Event name
    - ...additional fields
    """
    # Extract key from Bearer header (primary) or URL path (legacy)
    auth_key = _get_key(key)
    if not auth_key:
        return jsonify({"error": "Authorization required"}), 401

    # Validate key
    observer = load_observer(auth_key)
    if not observer:
        return jsonify({"error": "Invalid key"}), 401

    if observer.get("revoked", False):
        return jsonify({"error": "Observer revoked"}), 403

    if not observer.get("enabled", True):
        return jsonify({"error": "Observer disabled"}), 403

    # Parse event
    data = request.get_json(force=True) if request.is_json else {}

    tract = data.get("tract")
    event = data.get("event")

    if not tract or not event:
        return jsonify({"error": "Missing tract or event"}), 400

    # Add observer identifier
    data["observer"] = observer.get("name", "unknown")

    # Relay to local Callosum
    emit(tract, event, **{k: v for k, v in data.items() if k not in ("tract", "event")})

    # Update last_seen on status events
    if tract == "observe" and event == "status":
        observer["last_seen"] = now_ms()
        save_observer(observer)

    return jsonify({"status": "ok"})


@observer_bp.route("/ingest/segments/<day>")
@observer_bp.route("/ingest/<key>/segments/<day>")
def ingest_segments(day: str, key: str | None = None) -> Any:
    """List uploaded segments for a day with file verification.

    Returns JSON array of segments with file status:
    - present: File exists at recorded path
    - relocated: File found at different path (by inode)
    - missing: File not found

    Args:
        day: Day string (YYYYMMDD)
        key: Observer authentication key (from URL path, legacy)
    """
    # Extract key from Bearer header (primary) or URL path (legacy)
    auth_key = _get_key(key)
    if not auth_key:
        return jsonify({"error": "Authorization required"}), 401

    # Validate key
    observer = load_observer(auth_key)
    if not observer:
        return jsonify({"error": "Invalid key"}), 401

    if observer.get("revoked", False):
        return jsonify({"error": "Observer revoked"}), 403

    if not observer.get("enabled", True):
        return jsonify({"error": "Observer disabled"}), 403

    # Validate day format (YYYYMMDD)
    if not re.match(r"^\d{8}$", day):
        return jsonify({"error": "Invalid day format"}), 400

    # Load sync history for this observer/day
    key_prefix = auth_key[:8]
    records = load_history(key_prefix, day)

    if not records:
        return jsonify([])

    # Get day directory for file verification
    day_dir = day_path(day)

    # Determine stream: trust client-provided query param if valid,
    # otherwise derive from observer name (same logic as ingest_upload).
    client_stream = request.args.get("stream", "").strip()
    observer_name = observer.get("name", "unknown")
    if client_stream and re.match(r"^[a-z0-9][a-z0-9._-]*$", client_stream):
        fallback_stream = client_stream
    else:
        fallback_stream = stream_name(observer=observer_name)

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
        stream = record.get("stream", fallback_stream)
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

            # Check file status - files are in stream/segment directories
            segment_dir = day_dir / stream / segment
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
