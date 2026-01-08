# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Shared utilities for the remote app.

Provides common helpers for remote metadata management and sync history
that are used by both routes.py and events.py.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from apps.utils import get_app_storage_path

logger = logging.getLogger(__name__)


def get_remotes_dir() -> Path:
    """Get the remotes storage directory."""
    return get_app_storage_path("remote", "remotes", ensure_exists=True)


def get_hist_dir(key_prefix: str, ensure_exists: bool = True) -> Path:
    """Get the history directory for a remote.

    Args:
        key_prefix: First 8 chars of remote key
        ensure_exists: Create directory if it doesn't exist (default: True)

    Returns:
        Path to apps/remote/remotes/<key_prefix>/hist/
    """
    return get_app_storage_path(
        "remote", "remotes", key_prefix, "hist", ensure_exists=ensure_exists
    )


def load_remote(key: str) -> dict | None:
    """Load remote metadata by key.

    Args:
        key: Full remote authentication key

    Returns:
        Remote metadata dict if found and key matches, None otherwise
    """
    remotes_dir = get_remotes_dir()
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


def save_remote(data: dict) -> bool:
    """Save remote metadata.

    Args:
        data: Remote metadata dict (must contain 'key' field)

    Returns:
        True if saved successfully, False otherwise
    """
    key = data.get("key")
    if not key:
        return False
    remotes_dir = get_remotes_dir()
    remote_path = remotes_dir / f"{key[:8]}.json"
    try:
        with open(remote_path, "w") as f:
            json.dump(data, f, indent=2)
        return True
    except OSError:
        return False


def list_remotes() -> list[dict]:
    """List all registered remotes.

    Returns:
        List of remote metadata dicts, sorted by created_at descending
    """
    remotes_dir = get_remotes_dir()
    remotes = []
    for remote_path in remotes_dir.glob("*.json"):
        try:
            with open(remote_path) as f:
                data = json.load(f)
            remotes.append(data)
        except (json.JSONDecodeError, OSError):
            continue
    remotes.sort(key=lambda x: x.get("created_at", 0), reverse=True)
    return remotes


def find_remote_by_name(name: str) -> dict | None:
    """Find remote metadata by name.

    Args:
        name: Remote name to search for

    Returns:
        Remote metadata dict if found, None otherwise
    """
    for remote in list_remotes():
        if remote.get("name") == name:
            return remote
    return None


def append_history_record(key_prefix: str, day: str, record: dict) -> None:
    """Append a record to the sync history file.

    Args:
        key_prefix: First 8 chars of remote key
        day: Day string (YYYYMMDD)
        record: Record to append (will be JSON-serialized)
    """
    hist_dir = get_hist_dir(key_prefix)
    hist_path = hist_dir / f"{day}.jsonl"
    with open(hist_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_history(key_prefix: str, day: str) -> list[dict]:
    """Load sync history for a remote on a given day.

    Args:
        key_prefix: First 8 chars of remote key
        day: Day string (YYYYMMDD)

    Returns:
        List of history records, empty if file doesn't exist
    """
    hist_dir = get_hist_dir(key_prefix, ensure_exists=False)
    hist_path = hist_dir / f"{day}.jsonl"
    if not hist_path.exists():
        return []

    records = []
    try:
        with open(hist_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Failed to load sync history {hist_path}: {e}")
    return records


def increment_stat(key_prefix: str, stat_name: str) -> None:
    """Increment a stat counter for a remote.

    Args:
        key_prefix: First 8 chars of remote key
        stat_name: Name of the stat to increment (e.g., 'segments_observed')
    """
    remotes_dir = get_remotes_dir()
    remote_path = remotes_dir / f"{key_prefix}.json"
    if not remote_path.exists():
        return

    try:
        with open(remote_path) as f:
            data = json.load(f)

        data["stats"][stat_name] = data["stats"].get(stat_name, 0) + 1

        with open(remote_path, "w") as f:
            json.dump(data, f, indent=2)
    except (json.JSONDecodeError, OSError, KeyError) as e:
        logger.warning(f"Failed to update {stat_name} for {key_prefix}: {e}")


def find_segment_by_sha256(
    key_prefix: str, day: str, file_sha256s: set[str]
) -> tuple[str | None, set[str]]:
    """Find existing segment with matching file SHA256 signatures.

    Searches history records for the given day to find a segment where
    all provided SHA256 hashes match existing files.

    Args:
        key_prefix: First 8 chars of remote key
        day: Day string (YYYYMMDD)
        file_sha256s: Set of SHA256 hashes to match

    Returns:
        Tuple of (segment_key, matched_sha256s):
        - If full match: (segment_key, all sha256s)
        - If partial match: (None, set of matching sha256s)
        - If no match: (None, empty set)
    """
    records = load_history(key_prefix, day)
    if not records:
        return None, set()

    # Build map of sha256 -> segment for all upload records
    sha256_to_segment: dict[str, str] = {}
    segment_sha256s: dict[str, set[str]] = {}

    for record in records:
        # Skip non-upload records (e.g., "observed" type)
        if record.get("type"):
            continue

        segment = record.get("segment", "")
        if not segment:
            continue

        if segment not in segment_sha256s:
            segment_sha256s[segment] = set()

        for file_rec in record.get("files", []):
            sha256 = file_rec.get("sha256", "")
            if sha256:
                sha256_to_segment[sha256] = segment
                segment_sha256s[segment].add(sha256)

    # Check for full match - all incoming sha256s exist in a single segment
    for segment, existing_sha256s in segment_sha256s.items():
        if file_sha256s and file_sha256s.issubset(existing_sha256s):
            return segment, file_sha256s

    # Check for partial match - some sha256s already exist
    matched = file_sha256s & set(sha256_to_segment.keys())
    return None, matched
