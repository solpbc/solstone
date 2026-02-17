# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

import datetime as dt
import json
import logging
import os
import shutil
from pathlib import Path

from think.importers.utils import save_import_file, write_import_metadata
from think.utils import get_journal, now_ms

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
) -> None:
    """Write imported transcript entries in JSONL format.

    First line contains imported metadata, subsequent lines contain entries.
    Each entry gets source="import" added to match the imported_audio.jsonl convention.

    Args:
        file_path: Path to write JSONL file
        entries: List of transcript entries
        import_id: Import identifier
        raw_filename: Source file name (relative path from segment to imports/)
        facet: Optional facet name
        setting: Optional setting description
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
        metadata["raw"] = f"../../imports/{import_id}/{raw_filename}"

    # Write JSONL: metadata first, then entries with source field
    jsonl_lines = [json.dumps(metadata)]
    for entry in entries:
        # Add source field if not already present (skip metadata entries like topics/setting)
        if "text" in entry and "source" not in entry:
            entry = {**entry, "source": "import"}
        jsonl_lines.append(json.dumps(entry))

    with open(file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(jsonl_lines) + "\n")


def _run_import_summary(
    import_dir: Path,
    day: str,
    segments: list[str],
) -> bool:
    """Create a summary for imported segments using cortex generator.

    Args:
        import_dir: Directory where the summary will be saved
        day: Day string (YYYYMMDD format)
        segments: List of segment keys to summarize

    Returns:
        True if summary was created successfully, False otherwise
    """
    from think.cortex_client import cortex_request, wait_for_agents

    if not segments:
        logger.info("No segments to summarize")
        return False

    summary_path = import_dir / "summary.md"

    try:
        logger.info(f"Creating summary for {len(segments)} segments via cortex")

        # Spawn generator via cortex
        agent_id = cortex_request(
            prompt="",  # Generators don't use prompt
            name="importer",
            config={
                "day": day,
                "span": segments,
                "output": "md",
                "output_path": str(summary_path),
            },
        )

        if agent_id is None:
            logger.error("Failed to send cortex request for import summary")
            return False

        # Wait for completion
        completed, timed_out = wait_for_agents([agent_id], timeout=300)

        if timed_out:
            logger.error(f"Import summary timed out (ID: {agent_id})")
            return False

        if completed:
            end_state = completed.get(agent_id, "unknown")
            if end_state == "finish" and summary_path.exists():
                logger.info(f"Created import summary: {summary_path}")
                return True
            else:
                logger.warning(
                    f"Generator ended with state {end_state}, "
                    f"summary exists: {summary_path.exists()}"
                )
                return False

        logger.error("Generator did not complete")
        return False

    except Exception as e:
        logger.error(f"Failed to create summary: {e}")
        return False


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
