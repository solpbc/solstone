# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Plaud audio recorder API utilities and syncable backend."""

import datetime as dt
import json
import logging
import os
import pathlib
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

API_BASE = "https://api.plaud.ai"

# Skip recordings shorter than this (milliseconds)
MIN_DURATION_MS = 30_000


def make_session() -> requests.Session:
    """Create a requests session with sane retries."""
    s = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "HEAD"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s


def get_temp_url(
    session: requests.Session, token: str, file_hash: str
) -> Optional[str]:
    """Call Plaud API to get a time-limited S3 URL for the given hash."""
    url = f"{API_BASE}/file/temp-url/{file_hash}"
    headers = {
        "accept": "application/json, text/plain, */*",
        "authorization": f"bearer {token}",
        "app-platform": "web",  # mirrors the curl; likely not strictly required
        "user-agent": "fetch.py/1.0",
    }
    resp = session.get(url, headers=headers, timeout=20)
    if resp.status_code != 200:
        logger.warning(
            "[%s] API error %s: %s",
            file_hash,
            resp.status_code,
            resp.text[:200],
        )
        return None

    try:
        data = resp.json()
    except json.JSONDecodeError:
        logger.warning("[%s] Failed to parse JSON from API.", file_hash)
        return None

    # Expected shape: {"status":0,"temp_url":"https://...mp3?...","temp_url_opus":null}
    if data.get("status") != 0:
        logger.warning("[%s] API returned non-zero status: %s", file_hash, data)
        return None

    temp_url = data.get("temp_url")
    if not temp_url:
        logger.warning("[%s] No temp_url in response: %s", file_hash, data)
        return None

    return temp_url


def list_files(session: requests.Session, token: str) -> Optional[List[Dict[str, Any]]]:
    """Fetch the list of all files from Plaud API."""
    url = f"{API_BASE}/file/simple/web"
    params = {
        "skip": 0,
        "limit": 99999,
        "is_trash": 2,
        "sort_by": "start_time",
        "is_desc": "true",
    }
    headers = {
        "accept": "application/json, text/plain, */*",
        "authorization": f"bearer {token}",
        "app-platform": "web",
        "user-agent": "plaud.py/1.0",
    }

    try:
        resp = session.get(url, headers=headers, params=params, timeout=30)
        if resp.status_code != 200:
            logger.warning("API error %s: %s", resp.status_code, resp.text[:200])
            return None

        data = resp.json()
        if data.get("status") != 0:
            logger.warning("API returned non-zero status: %s", data)
            return None

        file_list = data.get("data_file_list", [])
        total = data.get("data_file_total", len(file_list))
        logger.info("Found %s files in Plaud account", total)
        return file_list

    except Exception as e:
        logger.warning("Error fetching file list: %s", e)
        return None


def sanitize_filename(filename: str) -> str:
    """Convert a filename to a safe filesystem name."""
    # Replace problematic characters with underscores
    safe = re.sub(r'[<>:"/\\|?*]', "_", filename)
    # Collapse multiple spaces/underscores
    safe = re.sub(r"[\s_]+", "_", safe)
    # Remove leading/trailing underscores
    safe = safe.strip("_")
    return safe or "unnamed"


def download_to_file(
    session: requests.Session, url: str, dest_path: pathlib.Path
) -> bool:
    """Stream-download URL to dest_path atomically."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with session.get(url, stream=True, timeout=60) as r:
        if r.status_code != 200:
            logger.warning(
                "[%s] Download error %s: %s",
                dest_path.stem,
                r.status_code,
                r.text[:200],
            )
            return False
        total = int(r.headers.get("Content-Length", "0")) or None
        # Write to a temp file then atomically move
        with tempfile.NamedTemporaryFile(
            dir=str(dest_path.parent), delete=False
        ) as tmp:
            tmp_path = pathlib.Path(tmp.name)
            try:
                downloaded = 0
                for chunk in r.iter_content(chunk_size=1024 * 256):
                    if not chunk:
                        continue
                    tmp.write(chunk)
                    downloaded += len(chunk)
                tmp.flush()
                os.fsync(tmp.fileno())
            except Exception as e:
                tmp.close()
                tmp_path.unlink(missing_ok=True)
                logger.warning("[%s] Error while writing file: %s", dest_path.stem, e)
                return False

    tmp_path.replace(dest_path)
    size_info = f" ({total} bytes)" if total else ""
    logger.info("[%s] Saved -> %s%s", dest_path.stem, dest_path, size_info)
    return True


def format_size(bytes_size: int) -> str:
    """Format bytes as human-readable size."""
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f}{unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f}TB"


def timestamp_from_start_time(start_time: int | float) -> str:
    """Convert Plaud epoch milliseconds to YYYYMMDD_HHMMSS timestamp."""
    # Plaud start_time is milliseconds since epoch
    if start_time > 1e12:
        start_time = start_time / 1000
    d = dt.datetime.fromtimestamp(start_time)
    return d.strftime("%Y%m%d_%H%M%S")


def match_existing_imports(
    journal_root: Path, plaud_files: list[dict[str, Any]]
) -> dict[str, str]:
    """Match Plaud files against existing imports by filename.

    Scans all imports/*/import.json for original_filename and matches against
    Plaud filenames using exact and sanitized comparison.

    Returns:
        Dict mapping plaud file ID -> import timestamp for matches.
    """
    imports_dir = journal_root / "imports"
    if not imports_dir.exists():
        return {}

    # Build index: normalized filename stem -> import timestamp
    filename_index: dict[str, str] = {}
    for import_dir in imports_dir.iterdir():
        if not import_dir.is_dir():
            continue
        meta_path = import_dir / "import.json"
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            orig = meta.get("original_filename", "")
            if orig:
                # Index by exact name and by stem (without extension)
                filename_index[orig] = import_dir.name
                stem = pathlib.Path(orig).stem
                if stem:
                    filename_index[stem] = import_dir.name
                # Also index sanitized form
                sanitized = sanitize_filename(stem) if stem else ""
                if sanitized and sanitized != stem:
                    filename_index[sanitized] = import_dir.name
        except (json.JSONDecodeError, OSError):
            continue

    # Match each Plaud file against the index
    matches: dict[str, str] = {}
    for file_info in plaud_files:
        file_id = file_info.get("id", "")
        filename = file_info.get("filename", "")
        fullname = file_info.get("fullname", "")

        # Try matching strategies in priority order
        candidates = [
            filename,  # exact display name
            pathlib.Path(fullname).stem if fullname else "",  # hash stem
            sanitize_filename(filename) if filename else "",  # sanitized display name
        ]
        ext = pathlib.Path(fullname).suffix if fullname else ".opus"
        if filename:
            candidates.append(f"{filename}{ext}")  # display name + extension
            candidates.append(f"{sanitize_filename(filename)}{ext}")  # sanitized + ext

        for candidate in candidates:
            if candidate and candidate in filename_index:
                matches[file_id] = filename_index[candidate]
                break

    return matches


class PlaudBackend:
    """Syncable backend for Plaud audio recorder service."""

    name: str = "plaud"

    def sync(self, journal_root: Path, *, dry_run: bool = True) -> dict[str, Any]:
        """Sync catalog from Plaud service.

        Fetches the file list from the Plaud API, matches against existing
        imports, and saves sync state. When dry_run=False, downloads and
        imports new files through the import pipeline.

        Returns:
            Summary dict with total, imported, available, downloaded, errors.
        """
        from think.importers.sync import load_sync_state, save_sync_state

        token = os.getenv("PLAUD_ACCESS_TOKEN")
        if not token:
            raise ValueError(
                "PLAUD_ACCESS_TOKEN not configured — set in Settings > API Keys"
            )

        session = make_session()

        # Fetch current file list from Plaud API
        file_list = list_files(session, token)
        if file_list is None:
            raise RuntimeError("Failed to fetch file list from Plaud API")

        # Load existing sync state
        state = load_sync_state(journal_root, "plaud") or {
            "backend": "plaud",
            "files": {},
        }
        known_files: dict[str, dict] = state.get("files", {})

        # Collect files that need matching: new files + still-available files
        # (re-check available in case they were imported manually since last sync)
        needs_matching = [
            f
            for f in file_list
            if f.get("id") not in known_files
            or known_files.get(f.get("id", ""), {}).get("status") == "available"
        ]
        matches = match_existing_imports(journal_root, needs_matching)

        # Merge into state
        for file_info in file_list:
            file_id = file_info.get("id", "")
            if not file_id:
                continue

            if file_id in known_files:
                # Preserve existing status, update metadata
                entry = known_files[file_id]
                entry["filename"] = file_info.get("filename", entry.get("filename", ""))
                entry["filesize"] = file_info.get("filesize", entry.get("filesize", 0))
                # Promote available -> imported if matched since last sync
                if entry.get("status") == "available" and file_id in matches:
                    entry["status"] = "imported"
                    entry["import_timestamp"] = matches[file_id]
                    entry["matched_at"] = dt.datetime.now().isoformat()
                continue

            # New file — build entry with full metadata
            duration = file_info.get("duration", 0)
            is_trash = file_info.get("is_trash", False)

            entry: dict[str, Any] = {
                "filename": file_info.get("filename", "unnamed"),
                "fullname": file_info.get("fullname", ""),
                "filesize": file_info.get("filesize", 0),
                "start_time": file_info.get("start_time", 0),
                "duration": duration,
                "is_trash": is_trash,
            }

            if file_id in matches:
                entry["status"] = "imported"
                entry["import_timestamp"] = matches[file_id]
                entry["matched_at"] = dt.datetime.now().isoformat()
            elif is_trash:
                entry["status"] = "skipped"
                entry["skip_reason"] = "trashed"
            elif duration and duration < MIN_DURATION_MS:
                entry["status"] = "skipped"
                entry["skip_reason"] = "too_short"
            else:
                entry["status"] = "available"

            known_files[file_id] = entry

        # Compute summary
        total = len(known_files)
        imported = sum(1 for f in known_files.values() if f.get("status") == "imported")
        available = sum(
            1 for f in known_files.values() if f.get("status") == "available"
        )
        skipped = sum(1 for f in known_files.values() if f.get("status") == "skipped")

        result: dict[str, Any] = {
            "total": total,
            "imported": imported,
            "available": available,
            "skipped": skipped,
            "downloaded": 0,
            "errors": [],
        }

        # Download and import if not dry-run
        if not dry_run and available > 0:
            to_process = [
                (fid, info)
                for fid, info in known_files.items()
                if info.get("status") == "available"
            ]
            downloaded = 0
            errors: list[str] = []

            for idx, (file_id, info) in enumerate(to_process, 1):
                filename = info.get("filename", "unnamed")
                filesize = info.get("filesize", 0)
                start_time = info.get("start_time", 0)

                # Derive timestamp from Plaud recording start time
                if start_time:
                    ts = timestamp_from_start_time(start_time)
                else:
                    logger.warning(
                        "  [%s/%s] %s — skipping (no start_time)",
                        idx,
                        len(to_process),
                        filename,
                    )
                    errors.append(f"{filename}: no start_time")
                    continue

                fullname = info.get("fullname", f"{file_id}.opus")
                ext = pathlib.Path(fullname).suffix or ".opus"
                safe_name = f"{sanitize_filename(filename)}{ext}"

                logger.info(
                    "  [%s/%s] %s (%s)",
                    idx,
                    len(to_process),
                    filename,
                    format_size(filesize),
                )

                # Download to imports/{timestamp}/
                import_dir = journal_root / "imports" / ts
                import_dir.mkdir(parents=True, exist_ok=True)
                dest_path = import_dir / safe_name

                # Get temp URL and download
                temp_url = get_temp_url(session, token, file_id)
                if not temp_url:
                    msg = f"{filename}: failed to get download URL"
                    logger.warning("    FAILED — %s", msg)
                    errors.append(msg)
                    continue

                if not download_to_file(session, temp_url, dest_path):
                    msg = f"{filename}: download failed"
                    logger.warning("    FAILED — %s", msg)
                    errors.append(msg)
                    continue

                logger.info("    Downloaded -> %s", dest_path.name)

                # Run through import pipeline
                import_cmd = [
                    "sol",
                    "import",
                    str(dest_path),
                    "--timestamp",
                    ts,
                    "--source",
                    "plaud",
                    "--auto",
                ]
                logger.info("    Importing %s...", ts)
                try:
                    proc = subprocess.run(
                        import_cmd,
                        capture_output=True,
                        text=True,
                        timeout=300,
                    )
                    if proc.returncode == 0:
                        info["status"] = "imported"
                        info["import_timestamp"] = ts
                        info["imported_at"] = dt.datetime.now().isoformat()
                        downloaded += 1
                        logger.info("    Imported successfully")
                    else:
                        stderr_tail = (
                            proc.stderr.strip().split("\n")[-1] if proc.stderr else ""
                        )
                        msg = f"{filename}: import failed — {stderr_tail}"
                        logger.warning("    FAILED — import error")
                        logger.warning(
                            "Import failed for %s: %s", filename, proc.stderr
                        )
                        errors.append(msg)
                except subprocess.TimeoutExpired:
                    msg = f"{filename}: import timed out"
                    logger.warning("    FAILED — timed out")
                    errors.append(msg)

            result["downloaded"] = downloaded
            result["errors"] = errors
            # Update available count after processing
            result["imported"] = sum(
                1 for f in known_files.values() if f.get("status") == "imported"
            )
            result["available"] = sum(
                1 for f in known_files.values() if f.get("status") == "available"
            )

        # Save updated state
        state["files"] = known_files
        state["last_sync"] = dt.datetime.now().isoformat()
        save_sync_state(journal_root, "plaud", state)

        return result


# Module-level backend instance for discovery
backend = PlaudBackend()
