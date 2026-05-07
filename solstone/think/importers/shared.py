# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Callable

from solstone.think.importers.utils import save_import_file, write_import_metadata
from solstone.think.media import MIME_TYPES
from solstone.think.utils import day_path, get_journal, now_ms

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
    model: str | None = None,
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
    if model:
        metadata["model"] = model

    # Write JSONL: metadata first, then entries with source field
    jsonl_lines = [json.dumps(metadata)]
    for entry in entries:
        # Add source to transcript rows when it is not already present.
        if "text" in entry and "source" not in entry:
            entry = {**entry, "source": "import"}
        jsonl_lines.append(json.dumps(entry))

    with open(file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(jsonl_lines) + "\n")


def write_segment(
    day_dir: str,
    stream: str,
    segment_key: str,
    entries: list[dict],
    *,
    import_id: str,
    raw_filename: str | None = None,
    facet: str | None = None,
    setting: str | None = None,
    topics: str | None = None,
    detected_setting: str | None = None,
    model: str | None = None,
) -> str:
    """Write a single segment's conversation_transcript.jsonl file."""
    ts_dir = os.path.join(day_dir, stream, segment_key)
    os.makedirs(ts_dir, exist_ok=True)
    json_path = os.path.join(ts_dir, "conversation_transcript.jsonl")

    _write_import_jsonl(
        json_path,
        entries,
        import_id=import_id,
        raw_filename=raw_filename,
        facet=facet,
        setting=setting,
        topics=topics,
        detected_setting=detected_setting,
        model=model,
    )
    return json_path


def _window_messages(
    messages: list[dict[str, Any]],
    window_duration: int = 300,
) -> list[tuple[str, str, str | None, list[dict[str, Any]]]]:
    """Group sorted messages into fixed-duration windows per day."""
    if not messages:
        return []

    windows: list[tuple[str, str, str | None, list[dict[str, Any]]]] = []
    window_start: float | None = None
    window_day: str | None = None
    window_entries: list[dict[str, Any]] = []
    window_model: str | None = None

    for msg in messages:
        msg_dt = dt.datetime.fromtimestamp(msg["create_time"])
        msg_day = msg_dt.strftime("%Y%m%d")

        if (
            window_start is None
            or msg_day != window_day
            or msg["create_time"] - window_start >= window_duration
        ):
            if window_entries and window_day and window_start is not None:
                start_dt = dt.datetime.fromtimestamp(window_start)
                seg_key = f"{start_dt.strftime('%H%M%S')}_{window_duration}"
                windows.append((window_day, seg_key, window_model, window_entries))

            window_start = msg["create_time"]
            window_day = msg_day
            window_entries = []
            window_model = None

        offset = int(msg["create_time"] - window_start)
        h, remainder = divmod(offset, 3600)
        m, s = divmod(remainder, 60)
        window_entries.append(
            {
                "start": f"{h:02d}:{m:02d}:{s:02d}",
                "speaker": msg["speaker"],
                "text": msg["text"],
            }
        )

        if msg["model_slug"] and window_model is None:
            window_model = msg["model_slug"]

    if window_entries and window_day and window_start is not None:
        start_dt = dt.datetime.fromtimestamp(window_start)
        seg_key = f"{start_dt.strftime('%H%M%S')}_{window_duration}"
        windows.append((window_day, seg_key, window_model, window_entries))

    return windows


def window_items(
    items: list[dict[str, Any]],
    ts_key: str,
    *,
    window_duration: int = 300,
    tz: dt.timezone | None = dt.timezone.utc,
) -> list[tuple[str, str, list[dict[str, Any]]]]:
    """Group sorted items into fixed-duration windows per day.

    Parameters
    ----------
    items : list[dict]
        Items sorted by ts_key. The ts_key field must be a float epoch.
    ts_key : str
        Key name for the float epoch timestamp in each item.
    window_duration : int
        Window size in seconds (default 300 = 5 minutes).
    tz : timezone or None
        Timezone for day grouping and seg_key formatting.
        Use dt.timezone.utc for UTC timestamps, None for local time.

    Returns
    -------
    list[tuple[str, str, list[dict]]]
        (day_str, seg_key, items) tuples.
    """
    if not items:
        return []

    windows: list[tuple[str, str, list[dict[str, Any]]]] = []
    window_start: float | None = None
    window_day: str | None = None
    window_items_acc: list[dict[str, Any]] = []

    for item in items:
        ts = item[ts_key]
        item_dt = dt.datetime.fromtimestamp(ts, tz=tz)
        item_day = item_dt.strftime("%Y%m%d")

        if (
            window_start is None
            or item_day != window_day
            or ts - window_start >= window_duration
        ):
            if window_items_acc and window_day and window_start is not None:
                start_dt = dt.datetime.fromtimestamp(window_start, tz=tz)
                seg_key = f"{start_dt.strftime('%H%M%S')}_{window_duration}"
                windows.append((window_day, seg_key, window_items_acc))

            window_start = ts
            window_day = item_day
            window_items_acc = []

        window_items_acc.append(item)

    if window_items_acc and window_day and window_start is not None:
        start_dt = dt.datetime.fromtimestamp(window_start, tz=tz)
        seg_key = f"{start_dt.strftime('%H%M%S')}_{window_duration}"
        windows.append((window_day, seg_key, window_items_acc))

    return windows


def write_markdown_segments(
    source: str,
    windows: list[tuple[str, str, list[dict[str, Any]]]],
    render: Callable[[list[dict[str, Any]]], str],
    *,
    filename: str = "imported.md",
) -> tuple[list[str], list[tuple[str, str]]]:
    """Write markdown segments from windowed items.

    Parameters
    ----------
    source : str
        Import source name (used in path: ``import.{source}``).
    windows : list
        Output of ``window_items`` — (day, seg_key, items) tuples.
    render : callable
        Function taking list of items and returning markdown string.
    filename : str
        Output filename (default: ``imported.md`` for backward compat).
        New importers should use ``*_transcript.md`` convention.

    Returns
    -------
    tuple[list[str], list[tuple[str, str]]]
        (created_file_paths, segment_tuples)
    """
    created_files: list[str] = []
    segments: list[tuple[str, str]] = []

    for day, seg_key, items in windows:
        segment_dir = day_path(day) / f"import.{source}" / seg_key
        segment_dir.mkdir(parents=True, exist_ok=True)
        md_path = segment_dir / filename
        md_path.write_text(render(items) + "\n", encoding="utf-8")
        created_files.append(str(md_path))
        segments.append((day, seg_key))

    return created_files, segments


# MIME type mapping for import metadata
_MIME_TYPES = {
    **MIME_TYPES,
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


def _build_import_manifest(import_dir: Path) -> dict[str, Any]:
    """Return a hash manifest for files currently stored in an import directory."""
    files = sorted(path for path in import_dir.rglob("*") if path.is_file())
    entries: list[dict[str, Any]] = []
    total_bytes = 0

    for path in files:
        size = path.stat().st_size
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            while chunk := handle.read(64 * 1024):
                digest.update(chunk)
        entries.append(
            {
                "name": path.relative_to(import_dir).as_posix(),
                "bytes": size,
                "hash": digest.hexdigest(),
            }
        )
        total_bytes += size

    return {
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "import_dir": str(import_dir),
        "total_bytes": total_bytes,
        "file_count": len(entries),
        "files": entries,
    }


def _setup_import(
    media_path: str,
    timestamp: str,
    facet: str | None,
    setting: str | None,
    detection_result: dict | None,
    force: bool = False,
    dry_run: bool = False,
) -> str:
    """Copy file to imports/ and write metadata. Returns new file path."""
    journal_root = Path(get_journal())
    import_dir = journal_root / "imports" / timestamp
    filename = os.path.basename(media_path)
    dry_run_target_path = str(import_dir / filename)

    # Check for conflict
    if import_dir.exists():
        if force:
            from solstone.apps.utils import log_app_action

            manifest = _build_import_manifest(import_dir)
            log_app_action(
                app="import",
                facet=None,
                action="import_force_reimport",
                params={**manifest, "dry_run": dry_run},
            )
            if dry_run:
                return dry_run_target_path
            logger.info(f"Removing existing import directory: {import_dir}")
            shutil.rmtree(import_dir)
        else:
            raise SystemExit(
                f"Error: Import already exists for timestamp {timestamp}\n"
                f"To re-import, use --force to delete existing data and start over"
            )

    # Copy file to imports/
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


def _setup_file_import(import_id: str) -> Path:
    """Create imports/{import_id}/ directory for file importers."""
    journal_root = Path(get_journal())
    import_dir = journal_root / "imports" / import_id
    import_dir.mkdir(parents=True, exist_ok=True)
    return import_dir


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

        # Merge with existing entries if file already exists (entry-level dedup)
        if out_path.exists():
            existing_keys: set[str] = set()
            try:
                with open(out_path, "r", encoding="utf-8") as f:
                    for line_num, line in enumerate(f):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            existing = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        # Skip header (first line)
                        if line_num == 0 and "import" in existing:
                            continue
                        existing_keys.add(_entry_content_key(existing))
            except OSError:
                existing_keys = set()

            # Filter to only new entries
            new_entries = [
                e for e in day_entries if _entry_content_key(e) not in existing_keys
            ]
            if not new_entries:
                logger.info("No new entries for %s — skipping", out_path)
                created.append(str(out_path))
                continue
            # Combine: keep all existing plus new
            day_entries = _load_existing_entries(out_path) + new_entries
            day_entries.sort(key=lambda e: e.get("ts", ""))

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


def hash_source(path: Path) -> str:
    """Compute SHA-256 hash of an import source file or directory.

    For files: hash the file contents.
    For directories: hash a sorted listing of relative paths + sizes.
    """
    h = hashlib.sha256()
    if path.is_file():
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
    elif path.is_dir():
        # Hash sorted listing of relative paths + sizes (fast, catches changes)
        entries = []
        for child in sorted(path.rglob("*")):
            if child.is_file():
                rel = str(child.relative_to(path))
                entries.append(f"{rel}:{child.stat().st_size}")
        h.update("\n".join(entries).encode())
    return h.hexdigest()


def write_manifest(
    journal_root: Path,
    import_id: str,
    source_type: str,
    source_hash: str,
    entry_count: int,
    files_created: list[str],
) -> Path:
    """Write an import manifest for deduplication tracking.

    Returns path to the manifest file.
    """
    days_affected = sorted(
        {
            os.path.basename(os.path.dirname(os.path.dirname(f)))
            for f in files_created
            if os.path.basename(os.path.dirname(os.path.dirname(f))).isdigit()
        }
    )
    manifest = {
        "source_type": source_type,
        "source_hash": source_hash,
        "entry_count": entry_count,
        "days_affected": days_affected,
        "files_created": files_created,
        "imported_at": dt.datetime.now().isoformat(),
    }
    manifest_dir = journal_root / "imports" / import_id
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest_path


def find_manifest_by_hash(journal_root: Path, source_hash: str) -> dict | None:
    """Search existing import manifests for a matching source hash.

    Returns the manifest dict if found, None otherwise.
    """
    imports_dir = journal_root / "imports"
    if not imports_dir.is_dir():
        return None
    for entry in imports_dir.iterdir():
        if not entry.is_dir():
            continue
        manifest_path = entry / "manifest.json"
        if not manifest_path.exists():
            continue
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            if manifest.get("source_hash") == source_hash:
                return manifest
        except (json.JSONDecodeError, OSError):
            continue
    return None


def _load_existing_entries(out_path: Path) -> list[dict]:
    """Load content entries from an existing imported.jsonl, skipping header."""
    entries: list[dict] = []
    try:
        with open(out_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if line_num == 0 and "import" in entry:
                    continue
                entries.append(entry)
    except OSError:
        pass
    return entries


def _entry_content_key(entry: dict) -> str:
    """Compute a content key for dedup within a source type.

    Uses type + ts + title/book_title to identify unique entries.
    """
    parts = [
        entry.get("type", ""),
        entry.get("ts", ""),
        entry.get("title", entry.get("book_title", "")),
    ]
    return "|".join(parts)


def write_content_manifest(
    import_id: str,
    entries: list[dict[str, Any]],
) -> Path:
    """Write content_manifest.jsonl for an import."""
    journal_root = Path(get_journal())
    manifest_dir = journal_root / "imports" / import_id
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "content_manifest.jsonl"

    with open(manifest_path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    return manifest_path


def map_items_to_segments(
    timestamps: list[float],
    *,
    window_duration: int = 300,
    tz: dt.timezone | None = dt.timezone.utc,
) -> list[tuple[str, str]]:
    """Map sorted timestamps to the segments produced by windowing helpers."""
    result: list[tuple[str, str]] = []
    window_start: float | None = None
    window_day: str | None = None

    for ts in timestamps:
        ts_dt = dt.datetime.fromtimestamp(ts, tz=tz)
        ts_day = ts_dt.strftime("%Y%m%d")

        if (
            window_start is None
            or ts_day != window_day
            or ts - window_start >= window_duration
        ):
            window_start = ts
            window_day = ts_day

        start_dt = dt.datetime.fromtimestamp(window_start, tz=tz)
        seg_key = f"{start_dt.strftime('%H%M%S')}_{window_duration}"
        result.append((window_day, seg_key))

    return result
