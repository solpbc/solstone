# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from think.importers.utils import save_import_file, write_import_metadata
from think.utils import day_path, get_journal, now_ms

if TYPE_CHECKING:
    from think.entities.core import EntityDict

logger = logging.getLogger(__name__)


def _get_relative_path(path: str) -> str:
    """Get path relative to journal, or return as-is if not under journal."""
    journal_path = get_journal()
    try:
        return os.path.relpath(path, journal_path)
    except ValueError:
        return path


def _write_import_jsonl(
    file_path: str,
    entries: list[dict],
    *,
    import_id: str,
    raw_filename: str | None = None,
    facet: str | None = None,
    setting: str | None = None,
    topics: str | None = None,
    detected_setting: str | None = None,
) -> None:
    """Write imported transcript entries in JSONL format.

    First line contains imported metadata, subsequent lines contain entries.
    Each entry gets source="import" added to match the imported_audio.jsonl convention.

    Args:
        file_path: Path to write JSONL file
        entries: List of transcript entries
        import_id: Import identifier
        raw_filename: Source filename (basename only, used to build relative path)
        facet: Optional facet name
        setting: Optional setting description
        topics: Optional LLM-detected topics (top-level metadata for format_audio)
        detected_setting: Optional LLM-detected setting context (top-level metadata)
    """
    imported_meta: dict[str, str] = {"id": import_id}
    if facet:
        imported_meta["facet"] = facet
    if setting:
        imported_meta["setting"] = setting

    # Build top-level metadata with imported info
    metadata: dict[str, object] = {"imported": imported_meta}

    # Add raw file reference (path relative from segment to imports directory)
    if raw_filename:
        metadata["raw"] = f"../../../imports/{import_id}/{raw_filename}"

    # Add LLM-detected enrichment at top level (displayed by format_audio header)
    if topics:
        metadata["topics"] = topics
    if detected_setting:
        metadata["setting"] = detected_setting

    # Write JSONL: metadata first, then entries with source field
    jsonl_lines = [json.dumps(metadata)]
    for entry in entries:
        # Add source field if not already present (skip metadata entries like topics/setting)
        if "text" in entry and "source" not in entry:
            entry = {**entry, "source": "import"}
        jsonl_lines.append(json.dumps(entry))

    with open(file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(jsonl_lines) + "\n")


# MIME type mapping for import metadata
_MIME_TYPES = {
    ".m4a": "audio/mp4",
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".flac": "audio/flac",
    ".ogg": "audio/ogg",
    ".opus": "audio/opus",
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".mov": "video/quicktime",
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".pdf": "application/pdf",
    ".ics": "text/calendar",
    ".dms": "application/zip",
}


def _is_in_imports(media_path: str) -> bool:
    """Check if file path is already under journal/imports/."""
    imports_dir = os.path.join(get_journal(), "imports")
    abs_media = os.path.abspath(media_path)
    abs_imports = os.path.abspath(imports_dir)
    return abs_media.startswith(abs_imports + os.sep)


def _setup_import(
    media_path: str,
    timestamp: str,
    facet: str | None,
    setting: str | None,
    detection_result: dict | None,
    force: bool = False,
) -> str:
    """Copy file to imports/ and write metadata. Returns new file path."""
    journal_root = Path(get_journal())
    import_dir = journal_root / "imports" / timestamp

    # Check for conflict
    if import_dir.exists():
        if force:
            logger.info(f"Removing existing import directory: {import_dir}")
            shutil.rmtree(import_dir)
        else:
            raise SystemExit(
                f"Error: Import already exists for timestamp {timestamp}\n"
                f"To re-import, use --force to delete existing data and start over"
            )

    # Copy file to imports/
    filename = os.path.basename(media_path)
    new_path = save_import_file(
        journal_root=journal_root,
        timestamp=timestamp,
        source_path=Path(media_path),
        filename=filename,
    )

    # Build metadata matching app structure
    upload_ts = now_ms()
    ext = os.path.splitext(filename)[1].lower()
    metadata = {
        "original_filename": filename,
        "upload_timestamp": upload_ts,
        "upload_datetime": dt.datetime.fromtimestamp(upload_ts / 1000).isoformat(),
        "detection_result": detection_result,
        "detected_timestamp": timestamp,
        "user_timestamp": timestamp,
        "file_size": new_path.stat().st_size if new_path.exists() else 0,
        "mime_type": _MIME_TYPES.get(ext, "application/octet-stream"),
        "facet": facet,
        "setting": setting,
        "file_path": str(new_path),
    }

    write_import_metadata(
        journal_root=journal_root,
        timestamp=timestamp,
        metadata=metadata,
    )

    logger.info(f"Copied to journal: {new_path}")
    return str(new_path)


def write_structured_import(
    source: str,
    entries: list[dict],
    *,
    import_id: str,
    facet: str | None = None,
) -> list[str]:
    """Write structured import entries to journal, grouped by day.

    Creates YYYYMMDD/import.{source}/imported.jsonl files.

    Each entry must have at minimum: type, ts (ISO 8601), content.
    Entries are grouped by day (from ts) and sorted by timestamp within each day.

    JSONL format per file:
    Line 1 (header): {"import": {"id": "...", "source": "ics", "facet": "..."}, "entry_count": N}
    Line 2+: {"type": "calendar_event", "ts": "2025-03-15T10:00:00", ...}

    Args:
        source: Import source name (e.g. "ics", "obsidian", "claude")
        entries: List of entry dicts, each with type, ts, content fields
        import_id: Import identifier (timestamp string)
        facet: Optional facet name

    Returns:
        List of created file paths (absolute)
    """
    import tempfile
    from collections import defaultdict

    # Group entries by day (YYYYMMDD extracted from ts)
    by_day: defaultdict[str, list[dict]] = defaultdict(list)
    for entry in entries:
        ts = entry["ts"]
        # Parse ISO 8601 timestamp to extract YYYYMMDD
        day = dt.datetime.fromisoformat(ts).strftime("%Y%m%d")
        by_day[day].append(entry)

    created: list[str] = []

    for day, day_entries in sorted(by_day.items()):
        # Sort entries by timestamp within the day
        day_entries.sort(key=lambda e: e["ts"])

        # Create directory: YYYYMMDD/import.{source}/
        import_dir = day_path(day) / f"import.{source}"
        import_dir.mkdir(parents=True, exist_ok=True)

        out_path = import_dir / "imported.jsonl"

        # Build header
        header: dict[str, object] = {
            "import": {"id": import_id, "source": source},
            "entry_count": len(day_entries),
        }
        if facet:
            header["import"]["facet"] = facet  # type: ignore[index]

        # Build JSONL content
        lines = [json.dumps(header)]
        for entry in day_entries:
            lines.append(json.dumps(entry))
        content = "\n".join(lines) + "\n"

        # Atomic write: write to temp file, then rename
        fd, tmp_path = tempfile.mkstemp(
            dir=str(import_dir), suffix=".tmp", prefix="imported_"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(content)
            os.replace(tmp_path, str(out_path))
        except BaseException:
            os.unlink(tmp_path)
            raise

        created.append(str(out_path))
        logger.info("Wrote %d entries to %s", len(day_entries), out_path)

    return created


def seed_entities(
    facet: str,
    day: str,
    entities: list[dict],
) -> list[EntityDict]:
    """Seed entities from structured imports.

    Each dict should have: name (required), type (default "Person"),
    email (optional), context (optional).

    Matches by email first, then name. Creates new entities for non-matches.

    Args:
        facet: Facet name for entity context
        day: Day string YYYYMMDD for activity tracking
        entities: List of entity dicts to seed

    Returns:
        List of resolved/created entity dicts
    """
    from think.entities.core import entity_slug
    from think.entities.journal import (
        get_or_create_journal_entity,
        load_all_journal_entities,
        save_journal_entity,
    )
    from think.entities.matching import find_entity_by_email, find_matching_entity

    # Load all journal entities for matching
    all_entities = load_all_journal_entities()
    entity_list = list(all_entities.values())

    resolved: list[EntityDict] = []

    for ent in entities:
        name = ent.get("name", "").strip()
        if not name:
            continue

        entity_type = ent.get("type", "Person")
        email = ent.get("email", "")

        matched = None

        # Try email match first
        if email:
            matched = find_entity_by_email(email, entity_list)

        # Fall back to name match
        if not matched:
            matched = find_matching_entity(name, entity_list)

        if matched:
            # Merge email into existing entity if new
            if email:
                existing_emails = set(e.lower() for e in matched.get("emails", []))
                if email.lower() not in existing_emails:
                    matched["emails"] = sorted(existing_emails | {email.lower()})
                    save_journal_entity(matched)
            resolved.append(matched)
        else:
            # Create new entity
            eid = entity_slug(name)
            emails = [email.lower()] if email else None
            new_entity = get_or_create_journal_entity(
                entity_id=eid,
                name=name,
                entity_type=entity_type,
                emails=emails,
            )
            entity_list.append(new_entity)  # Add to list for future matches
            resolved.append(new_entity)

    return resolved
