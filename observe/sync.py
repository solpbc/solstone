# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Sync service for remote observer uploads.

Listens for observe.observing events and uploads segments to a remote server.
Processes one segment at a time: upload, confirm via sha256 match, cleanup.

State is persisted in YYYYMMDD/health/sync.jsonl files for crash recovery.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import queue
import socket
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

from think.callosum import CallosumConnection
from think.utils import day_path, setup_cli

from .utils import compute_file_sha256

logger = logging.getLogger(__name__)

# Host identification
HOST = socket.gethostname()
PLATFORM = platform.system().lower()

# Retry configuration for RemoteClient
MAX_RETRIES = 3
RETRY_BACKOFF = [1, 5, 15]  # seconds
UPLOAD_TIMEOUT = 300  # 5 minutes for large files
HEALTH_CHECK_TIMEOUT = 10  # seconds for startup health check


def check_remote_health(
    remote_url: str, timeout: float = HEALTH_CHECK_TIMEOUT
) -> tuple[bool, str]:
    """Check if remote server is reachable and key is valid.

    Hits the segments endpoint to verify connectivity and authentication.
    Intended to be called at startup before launching sync service.

    Args:
        remote_url: Full URL to remote ingest endpoint (including key)
                   e.g., "https://server:5000/app/remote/ingest/abc123..."
        timeout: Request timeout in seconds (default: 10)

    Returns:
        Tuple of (success, message):
        - (True, "Connected to host:port (key: prefix)") on success
        - (False, "error description") on failure
    """
    # Parse URL for readable logging
    remote_url = remote_url.rstrip("/")
    try:
        parsed = urlparse(remote_url)
        host = parsed.netloc or parsed.hostname or "unknown"
        # Extract key from path: /app/remote/ingest/KEY -> KEY[:8]
        path_parts = parsed.path.split("/")
        key_prefix = path_parts[-1][:8] if path_parts else "unknown"
    except Exception:
        host = "unknown"
        key_prefix = "unknown"

    # Build segments endpoint URL with today's date
    today = datetime.now().strftime("%Y%m%d")
    segments_url = f"{remote_url}/segments/{today}"

    try:
        response = requests.get(segments_url, timeout=timeout)

        if response.status_code == 200:
            return (True, f"Connected to {host} (key: {key_prefix})")
        elif response.status_code == 401:
            return (False, "Invalid key (401) - check remote URL")
        elif response.status_code == 403:
            try:
                error = response.json().get("error", "forbidden")
            except Exception:
                error = "forbidden"
            return (False, f"Key rejected (403): {error}")
        else:
            return (False, f"Unexpected response ({response.status_code}) from {host}")

    except requests.exceptions.Timeout:
        return (False, f"Connection timeout after {timeout}s to {host}")
    except requests.exceptions.ConnectionError as e:
        # Extract cleaner error message
        err_str = str(e)
        if "Connection refused" in err_str:
            return (False, f"Connection refused: {host}")
        elif "Name or service not known" in err_str or "getaddrinfo failed" in err_str:
            return (False, f"Host not found: {host}")
        else:
            return (False, f"Connection error: {host}")
    except Exception as e:
        return (False, f"Health check failed: {e}")


class RemoteClient:
    """Client for uploading segment files to a remote server."""

    def __init__(self, remote_url: str):
        """Initialize remote client.

        Args:
            remote_url: Full URL to remote ingest endpoint (including key)
                       e.g., "https://server:5000/app/remote/ingest/abc123..."
        """
        self.remote_url = remote_url.rstrip("/")
        self.session = requests.Session()

    def upload_segment(
        self,
        day: str,
        segment: str,
        files: list[Path],
        meta: dict | None = None,
    ) -> bool:
        """Upload segment files to remote server.

        Args:
            day: Day string (YYYYMMDD)
            segment: Segment key (HHMMSS_LEN)
            files: List of file paths to upload
            meta: Optional metadata dict (host, platform, facet, setting, etc.)
                  to include in the segment. Will be JSON-encoded. If meta
                  doesn't contain host/platform, they're sent as top-level
                  fields and the server merges them into meta.

        Returns:
            True if upload succeeded, False otherwise
        """
        if not files:
            logger.warning("No files to upload")
            return False

        for attempt, delay in enumerate(RETRY_BACKOFF):
            # Open file handles and ensure they're closed
            file_handles = []
            files_data = []
            try:
                # Build files list for requests
                for path in files:
                    if not path.exists():
                        logger.warning(f"File not found, skipping: {path}")
                        continue
                    fh = open(path, "rb")
                    file_handles.append(fh)
                    files_data.append(
                        ("files", (path.name, fh, "application/octet-stream"))
                    )

                if not files_data:
                    logger.error("No valid files to upload")
                    return False

                # Build request data
                data: dict[str, Any] = {
                    "day": day,
                    "segment": segment,
                }
                # Only send top-level host/platform if not already in meta
                # (avoids redundant data; server merges them into meta if missing)
                if not meta or "host" not in meta:
                    data["host"] = HOST
                if not meta or "platform" not in meta:
                    data["platform"] = PLATFORM
                if meta:
                    data["meta"] = json.dumps(meta)

                response = self.session.post(
                    self.remote_url,
                    data=data,
                    files=files_data,
                    timeout=UPLOAD_TIMEOUT,
                )

                if response.status_code == 200:
                    result = response.json()
                    logger.info(
                        f"Uploaded {len(result.get('files', []))} files "
                        f"({result.get('bytes', 0)} bytes) for {day}/{segment}"
                    )
                    return True

                logger.warning(f"Upload failed: {response.status_code} {response.text}")

            except requests.RequestException as e:
                logger.warning(f"Upload attempt {attempt + 1} failed: {e}")

            finally:
                # Always close file handles
                for fh in file_handles:
                    try:
                        fh.close()
                    except Exception:
                        pass

            if attempt < len(RETRY_BACKOFF) - 1:
                logger.info(f"Retrying upload in {delay}s...")
                time.sleep(delay)

        logger.error(f"Upload failed after {MAX_RETRIES} attempts: {day}/{segment}")
        return False


# Confirmation polling configuration
CONFIRM_POLL_INTERVAL = 5  # seconds between confirmation checks
CONFIRM_MAX_ATTEMPTS = 12  # 5s * 12 = 60s max wait before retry


@dataclass
class SegmentInfo:
    """Info about a segment to sync."""

    day: str
    segment: str
    files: list[dict]  # [{name, sha256}, ...]
    meta: dict | None = None  # Optional metadata (host, platform, facet, etc.)


def get_sync_state_path(day: str) -> Path:
    """Get path to sync state file for a day."""
    health_dir = day_path(day) / "health"
    health_dir.mkdir(parents=True, exist_ok=True)
    return health_dir / "sync.jsonl"


def append_sync_record(day: str, record: dict) -> None:
    """Append a record to the sync state file."""
    state_path = get_sync_state_path(day)
    with open(state_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_sync_state(day: str) -> list[dict]:
    """Load sync state records for a day."""
    state_path = get_sync_state_path(day)
    if not state_path.exists():
        return []

    records = []
    try:
        with open(state_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Failed to load sync state {state_path}: {e}")
    return records


def get_pending_segments(days_back: int = 7) -> list[SegmentInfo]:
    """Scan journal for pending segments that need sync.

    Finds segments with 'pending' status but no 'confirmed' status.

    Args:
        days_back: Number of days to scan back

    Returns:
        List of SegmentInfo for pending segments
    """
    from datetime import datetime, timedelta

    pending = []
    today = datetime.now()

    for i in range(days_back):
        day_date = today - timedelta(days=i)
        day = day_date.strftime("%Y%m%d")

        # Use day_path() for consistency with load_sync_state
        day_dir = day_path(day)
        if not day_dir.exists():
            continue

        # Load state and find pending without confirmed
        records = load_sync_state(day)

        # Track status per segment
        segment_status: dict[str, dict] = {}
        for record in records:
            seg = record.get("segment", "")
            status = record.get("status", "")

            if seg not in segment_status:
                segment_status[seg] = {"pending": None, "confirmed": False}

            if status == "pending":
                segment_status[seg]["pending"] = record
            elif status == "confirmed":
                segment_status[seg]["confirmed"] = True

        # Collect pending segments
        for seg, info in segment_status.items():
            if info["pending"] and not info["confirmed"]:
                pending_record = info["pending"]
                pending.append(
                    SegmentInfo(
                        day=day,
                        segment=seg,
                        files=pending_record.get("files", []),
                        meta=pending_record.get("meta"),
                    )
                )

    return pending


class SyncService:
    """Service for syncing segments to a remote server."""

    def __init__(self, remote_url: str, days_back: int = 7):
        """Initialize sync service.

        Args:
            remote_url: Full URL to remote ingest endpoint (including key)
            days_back: Number of days to scan back on startup
        """
        self.remote_url = remote_url
        self.days_back = days_back

        self._client = RemoteClient(remote_url)
        self._callosum: CallosumConnection | None = None

        # Segment queue
        self._queue: queue.Queue[SegmentInfo] = queue.Queue()

        # Worker thread
        self._worker_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        # Status tracking
        self._current_segment: SegmentInfo | None = None
        self._current_state: str | None = None  # "uploading" | "confirming"
        self._confirm_attempt = 0
        self._last_confirmed: str | None = None
        self._last_status_emit = 0.0
        self._lock = threading.Lock()

    def start(self) -> None:
        """Start the sync service."""
        # Start Callosum connection
        self._callosum = CallosumConnection()
        self._callosum.start(callback=self._handle_message)

        # Scan for pending segments
        pending = get_pending_segments(self.days_back)
        if pending:
            logger.info(f"Found {len(pending)} pending segment(s) from previous run")
            for seg_info in pending:
                self._queue.put(seg_info)

        # Start worker thread
        self._stop_event.clear()
        self._worker_thread = threading.Thread(target=self._sync_worker, daemon=True)
        self._worker_thread.start()

        logger.info(f"Sync service started: {self.remote_url[:50]}...")

    def stop(self) -> None:
        """Stop the sync service."""
        self._stop_event.set()

        if self._worker_thread:
            self._worker_thread.join(timeout=5.0)

        if self._callosum:
            self._callosum.stop()
            self._callosum = None

        logger.info("Sync service stopped")

    def _handle_message(self, message: dict[str, Any]) -> None:
        """Handle incoming Callosum messages."""
        tract = message.get("tract")
        event = message.get("event")

        if tract != "observe" or event != "observing":
            return

        day = message.get("day")
        segment = message.get("segment")
        files = message.get("files", [])

        if not day or not segment or not files:
            logger.warning(f"Invalid observing event: {message}")
            return

        logger.info(f"Received observing event: {day}/{segment} ({len(files)} files)")

        # Build metadata dict from message fields
        # Observers emit host/platform as top-level fields, and may include a meta dict
        meta: dict[str, Any] = {}
        if message.get("host"):
            meta["host"] = message["host"]
        if message.get("platform"):
            meta["platform"] = message["platform"]
        # Merge any explicit meta dict (its values take precedence)
        if message.get("meta"):
            meta.update(message["meta"])

        # Compute sha256 for all files
        segment_dir = day_path(day) / segment
        file_info = []
        for filename in files:
            file_path = segment_dir / filename
            if file_path.exists():
                sha = compute_file_sha256(file_path)
                file_info.append({"name": filename, "sha256": sha})
            else:
                logger.warning(f"File not found: {file_path}")

        if not file_info:
            logger.error(f"No valid files for segment {day}/{segment}")
            return

        # Write pending record (include meta for crash recovery)
        record: dict[str, Any] = {
            "ts": int(time.time() * 1000),
            "segment": segment,
            "status": "pending",
            "files": file_info,
        }
        if meta:
            record["meta"] = meta
        append_sync_record(day, record)

        # Add to queue
        seg_info = SegmentInfo(
            day=day, segment=segment, files=file_info, meta=meta or None
        )
        self._queue.put(seg_info)

    def _sync_worker(self) -> None:
        """Worker thread: upload segments one at a time."""
        while not self._stop_event.is_set():
            # Emit status periodically
            now = time.time()
            if now - self._last_status_emit >= 5:
                self._emit_status()
                self._last_status_emit = now

            # Try to get next segment
            try:
                seg_info = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            # Process this segment
            self._process_segment(seg_info)

    def _process_segment(self, seg_info: SegmentInfo) -> None:
        """Process a single segment: upload, confirm, cleanup."""
        day = seg_info.day
        segment = seg_info.segment
        expected_sha256s = {f["name"]: f["sha256"] for f in seg_info.files}

        logger.info(f"Processing segment: {day}/{segment}")

        # Check if already confirmed on server before uploading
        # This handles crash recovery where we have a pending record but server already has it
        with self._lock:
            self._current_segment = seg_info
            self._current_state = "checking"
            self._confirm_attempt = 0

        if self._check_confirmation(day, segment, expected_sha256s):
            logger.info(
                f"Segment already confirmed on server: {day}/{segment}, skipping upload"
            )
            # Write confirmed record and cleanup
            record = {
                "ts": int(time.time() * 1000),
                "segment": segment,
                "status": "confirmed",
            }
            append_sync_record(day, record)

            segment_dir = day_path(day) / segment
            file_paths = [segment_dir / f["name"] for f in seg_info.files]
            existing_files = [p for p in file_paths if p.exists()]
            self._cleanup_segment(day, segment, existing_files)

            with self._lock:
                self._last_confirmed = f"{day}/{segment}"
                self._current_segment = None
                self._current_state = None
            return

        while not self._stop_event.is_set():
            # Upload
            with self._lock:
                self._current_segment = seg_info
                self._current_state = "uploading"
                self._confirm_attempt = 0

            segment_dir = day_path(day) / segment
            file_paths = [segment_dir / f["name"] for f in seg_info.files]

            # Filter to existing files
            existing_files = [p for p in file_paths if p.exists()]
            if not existing_files:
                logger.warning(f"No files found for segment {day}/{segment}, skipping")
                break

            success = self._client.upload_segment(
                day, segment, existing_files, meta=seg_info.meta
            )
            if not success:
                logger.error(f"Upload failed for {day}/{segment}, will retry")
                time.sleep(CONFIRM_POLL_INTERVAL)
                continue

            logger.info(f"Upload complete for {day}/{segment}, confirming...")

            # Confirm via sha256 match
            with self._lock:
                self._current_state = "confirming"
                self._confirm_attempt = 0

            confirmed = False
            for attempt in range(CONFIRM_MAX_ATTEMPTS):
                if self._stop_event.is_set():
                    return

                with self._lock:
                    self._confirm_attempt = attempt + 1

                if self._check_confirmation(day, segment, expected_sha256s):
                    confirmed = True
                    break

                time.sleep(CONFIRM_POLL_INTERVAL)

            if confirmed:
                # Write confirmed record
                record = {
                    "ts": int(time.time() * 1000),
                    "segment": segment,
                    "status": "confirmed",
                }
                append_sync_record(day, record)

                # Cleanup local files
                self._cleanup_segment(day, segment, existing_files)

                with self._lock:
                    self._last_confirmed = f"{day}/{segment}"

                logger.info(f"Segment confirmed and cleaned up: {day}/{segment}")
                break
            else:
                logger.warning(
                    f"Confirmation timeout for {day}/{segment}, retrying upload"
                )
                # Loop back to upload

        with self._lock:
            self._current_segment = None
            self._current_state = None
            self._confirm_attempt = 0

    def _check_confirmation(
        self, day: str, segment: str, expected: dict[str, str]
    ) -> bool:
        """Check if segment is confirmed on server via sha256 match.

        Args:
            day: Day string
            segment: Segment key
            expected: Dict of {filename: sha256} expected values

        Returns:
            True if all files confirmed with matching sha256
        """
        # Build segments endpoint URL
        # remote_url is like: https://server/app/remote/ingest/KEY
        # segments endpoint is: https://server/app/remote/ingest/KEY/segments/DAY
        segments_url = f"{self.remote_url}/segments/{day}"

        try:
            response = self._client.session.get(segments_url, timeout=30)
            if response.status_code != 200:
                logger.debug(f"Segments check failed: {response.status_code}")
                return False

            data = response.json()
        except Exception as e:
            logger.debug(f"Segments check error: {e}")
            return False

        # Find our segment in the response
        for seg_data in data:
            if seg_data.get("key") == segment:
                # Check all files have matching sha256
                server_files = {
                    f["name"]: f.get("sha256", "") for f in seg_data.get("files", [])
                }

                for name, expected_sha in expected.items():
                    server_sha = server_files.get(name, "")
                    if server_sha != expected_sha:
                        logger.debug(
                            f"SHA256 mismatch for {name}: "
                            f"expected {expected_sha[:8]}..., got {server_sha[:8]}..."
                        )
                        return False

                return True

        return False

    def _cleanup_segment(self, day: str, segment: str, file_paths: list[Path]) -> None:
        """Delete local segment files after confirmation."""
        for path in file_paths:
            try:
                if path.exists():
                    path.unlink()
                    logger.debug(f"Deleted: {path}")
            except OSError as e:
                logger.warning(f"Failed to delete {path}: {e}")

        # Try to remove segment directory if empty
        segment_dir = day_path(day) / segment
        try:
            segment_dir.rmdir()
            logger.debug(f"Removed empty segment directory: {segment_dir}")
        except OSError:
            pass  # Directory not empty or other error

    def _emit_status(self) -> None:
        """Emit sync.status event."""
        if not self._callosum:
            return

        with self._lock:
            status = {
                "queue_size": self._queue.qsize(),
                "host": HOST,
                "platform": PLATFORM,
            }

            if self._current_segment:
                status["segment"] = (
                    f"{self._current_segment.day}/{self._current_segment.segment}"
                )
                status["state"] = self._current_state
                if self._current_state == "confirming":
                    status["confirm_attempt"] = self._confirm_attempt

            if self._last_confirmed:
                status["last_confirmed"] = self._last_confirmed

        self._callosum.emit("sync", "status", **status)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Sync service for remote observer uploads"
    )
    parser.add_argument(
        "--remote",
        type=str,
        required=True,
        help="Remote server URL (e.g., https://server:5000/app/remote/ingest/KEY)",
    )
    parser.add_argument(
        "--days-back",
        type=int,
        default=7,
        help="Number of days to scan for pending segments on startup (default: 7)",
    )
    args = setup_cli(parser)

    service = SyncService(
        remote_url=args.remote,
        days_back=args.days_back,
    )

    logger.info("Starting sync service...")
    try:
        service.start()

        # Main loop - just wait for stop
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        logger.info("Shutting down...")
        service.stop()


if __name__ == "__main__":
    main()
