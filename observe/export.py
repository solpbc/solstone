# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Export journal data to a remote solstone instance.

Usage:
    sol export --to HOST --key KEY [--only segments] [--dry-run] [--day YYYYMMDD]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import requests

from observe.transfer import (
    RETRY_BACKOFF,
    _build_segment_manifest,
    _normalize_url,
    _parse_day_spec,
)
from think.utils import get_journal, iter_segments, setup_cli

logger = logging.getLogger(__name__)

UPLOAD_TIMEOUT = 300


def _query_manifest(session: requests.Session, base_url: str, key: str) -> dict[str, Any]:
    key_prefix = key[:8]
    url = f"{base_url}/app/import/journal/{key_prefix}/manifest/segments"
    response = session.get(url, timeout=UPLOAD_TIMEOUT)
    if response.status_code == 401:
        raise ValueError("Authentication failed: invalid or missing API key")
    if response.status_code == 403:
        raise ValueError("Authentication failed: journal source revoked or disabled")
    if response.status_code != 200:
        raise ValueError(f"Manifest query failed: {response.status_code} {response.text}")
    return response.json()


def _upload_segment(
    session: requests.Session,
    base_url: str,
    key: str,
    day: str,
    stream_name: str,
    segment_key: str,
    segment_path: Path,
) -> tuple[str, int]:
    files = [
        file_path
        for file_path in sorted(segment_path.iterdir())
        if file_path.is_file() and file_path.name != "stream.json"
    ]
    if not files:
        return ("skip", 0)

    bytes_sent = sum(file_path.stat().st_size for file_path in files)
    metadata = {
        "segments": [
            {
                "day": day,
                "stream": stream_name,
                "segment_key": segment_key,
                "files": [file_path.name for file_path in files],
            }
        ]
    }
    key_prefix = key[:8]
    url = f"{base_url}/app/import/journal/{key_prefix}/ingest/segments"

    for attempt, delay in enumerate(RETRY_BACKOFF):
        file_handles = []
        files_data = []
        try:
            for file_path in files:
                fh = open(file_path, "rb")
                file_handles.append(fh)
                files_data.append(
                    ("files_0", (file_path.name, fh, "application/octet-stream"))
                )

            response = session.post(
                url,
                data={"metadata": json.dumps(metadata)},
                files=files_data,
                timeout=UPLOAD_TIMEOUT,
            )
            if response.status_code == 200:
                return ("sent", bytes_sent)
            if response.status_code == 401:
                return ("auth_invalid", 0)
            if response.status_code == 403:
                return ("auth_revoked", 0)
            if 500 <= response.status_code <= 599:
                logger.warning(
                    "Upload attempt %s failed for %s/%s/%s: %s %s",
                    attempt + 1,
                    day,
                    stream_name,
                    segment_key,
                    response.status_code,
                    response.text,
                )
            else:
                logger.warning(
                    "Upload rejected for %s/%s/%s: %s %s",
                    day,
                    stream_name,
                    segment_key,
                    response.status_code,
                    response.text,
                )
                return ("error", 0)
        except (requests.RequestException, OSError) as e:
            logger.warning(
                "Upload attempt %s failed for %s/%s/%s: %s",
                attempt + 1,
                day,
                stream_name,
                segment_key,
                e,
            )
        finally:
            for fh in file_handles:
                try:
                    fh.close()
                except Exception:
                    pass

        if attempt < len(RETRY_BACKOFF) - 1:
            time.sleep(delay)

    return ("error", 0)


def export_segments(base_url: str, key: str, days: list[str], dry_run: bool) -> None:
    session = requests.Session()
    session.headers["Authorization"] = f"Bearer {key}"

    sent = 0
    skipped = 0
    failed = 0
    bytes_total = 0

    try:
        try:
            remote_manifest = _query_manifest(session, base_url, key)
        except requests.ConnectionError:
            print(f"Connection failed: could not reach {base_url}")
            return
        except ValueError as e:
            print(str(e))
            return

        journal = get_journal()
        for day in days:
            day_dir = Path(journal) / day
            if not day_dir.exists():
                continue

            segment_entries = iter_segments(day_dir)
            if not segment_entries:
                continue

            day_sent = 0
            day_bytes = 0

            for stream_name, seg_key, seg_path in segment_entries:
                manifest = _build_segment_manifest(seg_path)
                local_files = {
                    file_info["name"]: file_info["sha256"]
                    for file_info in manifest["files"]
                    if file_info["name"] != "stream.json"
                }
                if not local_files:
                    skipped += 1
                    continue

                remote_entry = remote_manifest.get(day, {}).get(
                    f"{stream_name}/{seg_key}", {}
                )
                remote_files = {
                    file_info["name"]: file_info["sha256"]
                    for file_info in remote_entry.get("files", [])
                }
                if local_files == remote_files:
                    skipped += 1
                    logger.info(f"  [skip] {day}/{stream_name}/{seg_key}")
                    continue

                if dry_run:
                    seg_bytes = sum(
                        file_info["size"]
                        for file_info in manifest["files"]
                        if file_info["name"] != "stream.json"
                    )
                    logger.info(f"  [would send] {day}/{stream_name}/{seg_key}")
                    day_sent += 1
                    day_bytes += seg_bytes
                    continue

                status, segment_bytes = _upload_segment(
                    session,
                    base_url,
                    key,
                    day,
                    stream_name,
                    seg_key,
                    seg_path,
                )
                if status == "sent":
                    logger.info(
                        f"  [sent] {day}/{stream_name}/{seg_key} ({segment_bytes} bytes)"
                    )
                    sent += 1
                    bytes_total += segment_bytes
                elif status == "skip":
                    skipped += 1
                elif status == "auth_invalid":
                    print("Authentication failed: invalid or missing API key")
                    return
                elif status == "auth_revoked":
                    print("Authentication failed: journal source revoked or disabled")
                    return
                else:
                    logger.info(f"  [FAILED] {day}/{stream_name}/{seg_key}")
                    failed += 1

            if dry_run:
                sent += day_sent
                if day_sent > 0:
                    print(f"  {day}: {day_sent} segment(s), {day_bytes} bytes")

        total = sent + skipped + failed
        if total == 0:
            print("No segments found to export")
            return
        if dry_run:
            print(f"\nDry run: would send {sent}, skip {skipped}")
            return

        print(
            f"\nExport complete: {sent} sent, {skipped} skipped, "
            f"{failed} failed, {bytes_total} bytes transferred"
        )
        if sent == 0 and skipped > 0 and failed == 0:
            print("Nothing to send - remote is up to date")
    finally:
        session.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export journal data to a remote solstone instance"
    )
    parser.add_argument(
        "--to",
        required=True,
        help="Remote instance URL (e.g., host:port or https://host)",
    )
    parser.add_argument(
        "--key",
        required=True,
        help="API key for the remote journal source",
    )
    parser.add_argument(
        "--only",
        default=None,
        help="Export only specific area (segments)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be exported without sending",
    )
    parser.add_argument(
        "--day",
        default=None,
        help="Day or range (YYYYMMDD or YYYYMMDD-YYYYMMDD)",
    )
    args = setup_cli(parser)

    if args.only is not None and args.only != "segments":
        print(f"Export of '{args.only}' is not yet implemented")
        sys.exit(0)

    base_url = _normalize_url(args.to)
    days = _parse_day_spec(args.day, Path(get_journal()))
    export_segments(base_url, args.key, days, args.dry_run)
