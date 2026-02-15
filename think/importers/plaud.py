# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Plaud audio recorder API utilities and syncable backend."""

import json
import os
import pathlib
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

API_BASE = "https://api.plaud.ai"


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
        print(
            f"[{file_hash}] API error {resp.status_code}: {resp.text[:200]}",
            file=sys.stderr,
        )
        return None

    try:
        data = resp.json()
    except json.JSONDecodeError:
        print(f"[{file_hash}] Failed to parse JSON from API.", file=sys.stderr)
        return None

    # Expected shape: {"status":0,"temp_url":"https://...mp3?...","temp_url_opus":null}
    if data.get("status") != 0:
        print(f"[{file_hash}] API returned non-zero status: {data}", file=sys.stderr)
        return None

    temp_url = data.get("temp_url")
    if not temp_url:
        print(f"[{file_hash}] No temp_url in response: {data}", file=sys.stderr)
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
            print(f"API error {resp.status_code}: {resp.text[:200]}", file=sys.stderr)
            return None

        data = resp.json()
        if data.get("status") != 0:
            print(f"API returned non-zero status: {data}", file=sys.stderr)
            return None

        file_list = data.get("data_file_list", [])
        total = data.get("data_file_total", len(file_list))
        print(f"Found {total} files in Plaud account")
        return file_list

    except Exception as e:
        print(f"Error fetching file list: {e}", file=sys.stderr)
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
            print(
                f"[{dest_path.stem}] Download error {r.status_code}: {r.text[:200]}",
                file=sys.stderr,
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
                print(
                    f"[{dest_path.stem}] Error while writing file: {e}", file=sys.stderr
                )
                return False

    tmp_path.replace(dest_path)
    size_info = f" ({total} bytes)" if total else ""
    print(f"[{dest_path.stem}] Saved -> {dest_path}{size_info}")
    return True


def format_size(bytes_size: int) -> str:
    """Format bytes as human-readable size."""
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f}{unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f}TB"


def sync_files(
    session: requests.Session,
    token: str,
    target_dir: pathlib.Path,
    dry_run: bool = True,
) -> int:
    """
    Sync files from Plaud API to local directory.

    Returns the number of files that were (or would be) downloaded.
    """
    file_list = list_files(session, token)
    if file_list is None:
        return -1

    target_dir.mkdir(parents=True, exist_ok=True)

    to_download = []
    already_exist = []

    print(f"\nChecking local files in: {target_dir}")
    print("=" * 70)

    for file_info in file_list:
        file_id = file_info.get("id")
        filename = file_info.get("filename", "unnamed")
        filesize = file_info.get("filesize", 0)
        fullname = file_info.get("fullname", f"{file_id}.opus")

        # Use the fullname (e.g., "hash.opus") or sanitize the filename
        # Let's use the fullname which includes extension
        safe_name = sanitize_filename(filename)
        # Get extension from fullname
        ext = pathlib.Path(fullname).suffix or ".opus"
        local_filename = f"{safe_name}{ext}"
        local_path = target_dir / local_filename

        if local_path.exists():
            already_exist.append((file_info, local_path))
        else:
            to_download.append((file_info, local_path))

    print(f"✓ {len(already_exist)} files already exist locally")
    print(f"⬇ {len(to_download)} files need to be downloaded")

    if not to_download:
        print("\n✓ All files are already synced!")
        return 0

    if dry_run:
        print(f"\n{'DRY RUN MODE':-^70}")
        print("The following files would be downloaded:\n")
        total_size = 0
        for file_info, local_path in to_download:
            filename = file_info.get("filename", "unnamed")
            filesize = file_info.get("filesize", 0)
            total_size += filesize
            print(f"  • {local_path.name}")
            print(f"    Size: {format_size(filesize)}, Original: {filename}")

        print(f"\nTotal download size: {format_size(total_size)}")
        print("\nTo actually download these files, run with --save flag")
        return len(to_download)

    # Actually download files
    print(f"\n{'DOWNLOADING FILES':-^70}\n")
    downloaded = 0
    failed = 0

    for idx, (file_info, local_path) in enumerate(to_download, 1):
        file_id = file_info.get("id")
        filename = file_info.get("filename", "unnamed")
        filesize = file_info.get("filesize", 0)

        print(f"[{idx}/{len(to_download)}] {local_path.name} ({format_size(filesize)})")

        # Get temp URL
        temp_url = get_temp_url(session, token, file_id)
        if not temp_url:
            print("  ✗ Failed to get download URL", file=sys.stderr)
            failed += 1
            continue

        # Download file
        if download_to_file(session, temp_url, local_path):
            downloaded += 1
            print("  ✓ Downloaded")
        else:
            failed += 1
            print("  ✗ Download failed", file=sys.stderr)

    print(f"\n{'SUMMARY':-^70}")
    print(f"✓ Downloaded: {downloaded}")
    if failed > 0:
        print(f"✗ Failed: {failed}")
    print(f"Total synced: {len(already_exist) + downloaded}/{len(file_list)}")

    return downloaded


class PlaudBackend:
    """Syncable backend for Plaud audio recorder service."""

    name: str = "plaud"

    def sync(self, journal_root: Path, *, dry_run: bool = True) -> dict[str, Any]:
        """Sync files from Plaud service.

        Not yet implemented — Phase 2 will wire up actual sync.
        """
        token = os.getenv("PLAUD_ACCESS_TOKEN")
        if not token:
            raise ValueError(
                "PLAUD_ACCESS_TOKEN not configured — set in Settings > API Keys"
            )
        raise NotImplementedError("Plaud sync execution is not yet implemented")


# Module-level backend instance for discovery
backend = PlaudBackend()
