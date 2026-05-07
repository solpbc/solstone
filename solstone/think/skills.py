# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc
"""Owner-wide skills storage and helpers.

Sole write-owner of:
  journal/skills/patterns.jsonl
  journal/skills/edit_requests.jsonl
  journal/skills/{slug}.md
"""

from __future__ import annotations

import fcntl
import json
import logging
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from solstone.think.entities.core import atomic_write
from solstone.think.utils import get_journal

logger = logging.getLogger(__name__)


def skills_dir() -> Path:
    """Return the owner-wide skills directory, creating it if needed."""
    path = Path(get_journal()) / "skills"
    path.mkdir(parents=True, exist_ok=True)
    return path


def patterns_path() -> Path:
    """Return the owner-wide skills patterns JSONL path."""
    return skills_dir() / "patterns.jsonl"


def edit_requests_path() -> Path:
    """Return the owner-wide skill edit requests JSONL path."""
    return skills_dir() / "edit_requests.jsonl"


def profile_path(slug: str) -> Path:
    """Return the markdown profile path for one skill slug."""
    return skills_dir() / f"{slug}.md"


def patterns_lock_path() -> Path:
    """Return the sibling lock path for patterns.jsonl."""
    return skills_dir() / ".patterns.lock"


def edit_requests_lock_path() -> Path:
    """Return the sibling lock path for edit_requests.jsonl."""
    return skills_dir() / ".edit_requests.lock"


def _load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    """Load JSONL rows from *path*, skipping blanks and malformed lines."""
    if not path.exists():
        return []

    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("skills: malformed JSONL line %s in %s", lineno, path)
                continue
            if not isinstance(data, dict):
                logger.warning(
                    "skills: non-object JSONL line %s in %s (got %s)",
                    lineno,
                    path,
                    type(data).__name__,
                )
                continue
            rows.append(data)
    return rows


def load_patterns() -> list[dict[str, Any]]:
    """Load owner-wide skill patterns from JSONL."""
    return _load_jsonl_rows(patterns_path())


def load_edit_requests() -> list[dict[str, Any]]:
    """Load owner-wide skill edit requests from JSONL."""
    return _load_jsonl_rows(edit_requests_path())


def load_profile(slug: str) -> str | None:
    """Load one markdown skill profile, returning None when absent."""
    path = profile_path(slug)
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None


def find_pattern(
    slug: str, patterns: list[dict[str, Any]] | None = None
) -> dict[str, Any] | None:
    """Return one pattern by slug, or None when not found."""
    rows = load_patterns() if patterns is None else patterns
    for row in rows:
        if row.get("slug") == slug:
            return row
    return None


def _save_jsonl_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write *rows* to *path* as JSONL using an atomic replace."""
    content = ""
    if rows:
        content = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n"
    atomic_write(path, content)


def save_patterns(rows: list[dict[str, Any]]) -> None:
    """Persist owner-wide skill patterns atomically."""
    _save_jsonl_rows(patterns_path(), rows)


def save_edit_requests(rows: list[dict[str, Any]]) -> None:
    """Persist owner-wide skill edit requests atomically."""
    _save_jsonl_rows(edit_requests_path(), rows)


def save_profile(slug: str, markdown: str) -> None:
    """Persist one markdown skill profile atomically."""
    atomic_write(profile_path(slug), markdown)


def rename_profile(old_slug: str, new_slug: str) -> bool:
    """Rename one skill profile file, returning False when the source is absent."""
    source = profile_path(old_slug)
    target = profile_path(new_slug)
    if not source.exists():
        return False
    if target.exists():
        raise FileExistsError(f"profile already exists for slug {new_slug}")
    source.rename(target)
    return True


def locked_modify_patterns(
    fn: Callable[[list[dict[str, Any]]], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Apply a locked read-modify-write cycle to patterns.jsonl."""
    skills_dir()
    lock_path = patterns_lock_path()
    # Lock file contents are irrelevant; opening with "w" matches the existing pattern.
    with open(lock_path, "w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            rows = load_patterns()
            new_rows = fn(rows)
            save_patterns(new_rows)
            return new_rows
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def locked_modify_edit_requests(
    fn: Callable[[list[dict[str, Any]]], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Apply a locked read-modify-write cycle to edit_requests.jsonl."""
    skills_dir()
    lock_path = edit_requests_lock_path()
    # Lock file contents are irrelevant; opening with "w" matches the existing pattern.
    with open(lock_path, "w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            rows = load_edit_requests()
            new_rows = fn(rows)
            save_edit_requests(new_rows)
            return new_rows
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def observation_key(slug: str, day: str, activity_ids: list[str]) -> str:
    """Return the deterministic idempotency key for one observation."""
    return f"{slug}|{day}|{','.join(sorted(activity_ids))}"


def _utc_compact() -> str:
    """Return a compact UTC timestamp for request ids."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")


def make_request_id() -> str:
    """Return a collision-resistant edit request id."""
    return f"req_{_utc_compact()}_{secrets.token_hex(6)}"


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string ending in Z."""
    return (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )


def touch_updated(pattern: dict[str, Any]) -> None:
    """Update a pattern row's updated_at timestamp in place."""
    pattern["updated_at"] = utc_now_iso()
