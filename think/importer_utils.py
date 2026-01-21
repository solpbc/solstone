# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Utility functions for import operations.

This module contains reusable logic for managing imports in the journal,
extracted from apps/import/routes.py to be usable in CLI tools and other contexts.
"""

from __future__ import annotations

import json
from pathlib import Path

# ============================================================================
# File Operations
# ============================================================================


def save_import_file(
    journal_root: Path,
    timestamp: str,
    source_path: Path,
    filename: str,
) -> Path:
    """Copy/move file into imports/{timestamp}/ directory.

    Args:
        journal_root: Root journal directory
        timestamp: Import timestamp (YYYYMMDD_HHMMSS format)
        source_path: Path to source file
        filename: Desired filename in import directory

    Returns:
        Final file path where file was saved
    """
    # Create import folder structure: imports/<timestamp>/<filename>
    import_dir = journal_root / "imports" / timestamp
    import_dir.mkdir(parents=True, exist_ok=True)

    # Save the file
    file_path = import_dir / filename
    if source_path != file_path:
        # Copy content if different paths
        file_path.write_bytes(source_path.read_bytes())

    return file_path


def save_import_text(
    journal_root: Path,
    timestamp: str,
    content: str,
    filename: str,
) -> Path:
    """Save text content to imports/{timestamp}/ directory.

    Args:
        journal_root: Root journal directory
        timestamp: Import timestamp (YYYYMMDD_HHMMSS format)
        content: Text content to save
        filename: Desired filename in import directory

    Returns:
        Final file path where content was saved
    """
    # Create import folder structure: imports/<timestamp>/<filename>
    import_dir = journal_root / "imports" / timestamp
    import_dir.mkdir(parents=True, exist_ok=True)

    # Save the text
    file_path = import_dir / filename
    file_path.write_text(content, encoding="utf-8")

    return file_path


# ============================================================================
# Metadata Operations
# ============================================================================


def write_import_metadata(
    journal_root: Path,
    timestamp: str,
    metadata: dict,
) -> None:
    """Write import.json with provided metadata dict.

    Args:
        journal_root: Root journal directory
        timestamp: Import timestamp
        metadata: Metadata dictionary to write
    """
    import_dir = journal_root / "imports" / timestamp
    import_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = import_dir / "import.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def read_import_metadata(
    journal_root: Path,
    timestamp: str,
) -> dict:
    """Read import.json for an import.

    Args:
        journal_root: Root journal directory
        timestamp: Import timestamp

    Returns:
        Metadata dictionary

    Raises:
        FileNotFoundError: If import metadata not found
    """
    import_dir = journal_root / "imports" / timestamp
    metadata_path = import_dir / "import.json"

    if metadata_path.exists():
        return json.loads(metadata_path.read_text(encoding="utf-8"))

    raise FileNotFoundError(f"Import metadata not found for {timestamp}")


def update_import_metadata_fields(
    journal_root: Path,
    timestamp: str,
    updates: dict,
) -> tuple[dict, bool]:
    """Update specific fields in import.json.

    Args:
        journal_root: Root journal directory
        timestamp: Import timestamp
        updates: Dict of fields to update (e.g., {"facet": "foo", "setting": "bar"})

    Returns:
        Tuple of (updated_metadata, was_modified)

    Raises:
        FileNotFoundError: If import metadata not found
    """
    import_dir = journal_root / "imports" / timestamp
    metadata_path = import_dir / "import.json"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Import metadata not found for {timestamp}")

    # Read current metadata
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    # Track if anything changed
    updated = False

    # Update each field
    for key, value in updates.items():
        # Check if field is missing or value changed
        field_missing = key not in metadata
        value_changed = metadata.get(key) != value

        if field_missing or value_changed:
            metadata[key] = value
            updated = True

    # Write back if modified
    if updated:
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return metadata, updated


# ============================================================================
# Reading Processing Results
# ============================================================================


def read_imported_results(
    journal_root: Path,
    timestamp: str,
) -> dict | None:
    """Read imported.json if exists, else None.

    Args:
        journal_root: Root journal directory
        timestamp: Import timestamp

    Returns:
        Imported results dict or None if not found
    """
    import_dir = journal_root / "imports" / timestamp
    imported_json = import_dir / "imported.json"

    if not imported_json.exists():
        return None

    try:
        with open(imported_json, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def has_summary(
    journal_root: Path,
    timestamp: str,
) -> bool:
    """Check if summary.md exists.

    Args:
        journal_root: Root journal directory
        timestamp: Import timestamp

    Returns:
        True if summary exists, False otherwise
    """
    import_dir = journal_root / "imports" / timestamp
    summary_path = import_dir / "summary.md"
    return summary_path.exists()


# ============================================================================
# Scanning and Status Logic
# ============================================================================


def list_import_timestamps(
    journal_root: Path,
) -> list[str]:
    """Get all valid import timestamps from imports/ directory.

    Args:
        journal_root: Root journal directory

    Returns:
        List of timestamp strings (YYYYMMDD_HHMMSS format)
    """
    imports_dir = journal_root / "imports"

    if not imports_dir.exists():
        return []

    timestamps = []
    for import_folder in imports_dir.iterdir():
        if not import_folder.is_dir():
            continue

        # Skip if it's not a timestamp folder (YYYYMMDD_HHMMSS format)
        if not (import_folder.name.count("_") == 1 and len(import_folder.name) == 15):
            continue

        timestamps.append(import_folder.name)

    return timestamps


def calculate_duration_from_files(
    files: list[str],
) -> int | None:
    """Calculate duration in minutes from imported file timestamps.

    Expects filenames like "120000_imported_audio.jsonl"
    Extracts HHMMSS, calculates start-to-end duration.

    Args:
        files: List of file paths

    Returns:
        Duration in minutes, or None if can't calculate
    """
    if not files:
        return None

    timestamps = []
    for file in files:
        # Extract timestamp from filename like "120000_imported_audio.jsonl"
        basename = Path(file).name
        if basename[:6].isdigit():
            timestamps.append(basename[:6])

    if not timestamps:
        return None

    timestamps.sort()
    start_time = timestamps[0]
    end_time = timestamps[-1]

    # Convert to minutes
    start_h, start_m = int(start_time[:2]), int(start_time[2:4])
    end_h, end_m = int(end_time[:2]), int(end_time[2:4])
    duration_minutes = (end_h * 60 + end_m) - (start_h * 60 + start_m)

    if duration_minutes > 0:
        return duration_minutes

    return None


def build_import_info(
    journal_root: Path,
    timestamp: str,
) -> dict:
    """Build complete info dict for one import.

    Args:
        journal_root: Root journal directory
        timestamp: Import timestamp

    Returns:
        Dict with all import information (without status - caller adds that)
    """
    import_dir = journal_root / "imports" / timestamp

    import_data = {
        "timestamp": timestamp,
        "created_at": import_dir.stat().st_ctime,
        "imported_at": import_dir.stat().st_ctime,  # Default, may be overridden
    }

    # Read import.json if it exists
    import_json = import_dir / "import.json"
    task_id = None
    if import_json.exists():
        try:
            with open(import_json, "r", encoding="utf-8") as f:
                import_meta = json.load(f)
                import_data["original_filename"] = import_meta.get(
                    "original_filename", "Unknown"
                )
                import_data["file_size"] = import_meta.get("file_size", 0)
                import_data["mime_type"] = import_meta.get("mime_type", "")
                import_data["facet"] = import_meta.get("facet")
                import_data["setting"] = import_meta.get("setting")
                import_data["user_timestamp"] = import_meta.get("user_timestamp")
                task_id = import_meta.get("task_id")
                import_data["task_id"] = task_id
                # Use upload_timestamp if available for better sorting
                if "upload_timestamp" in import_meta:
                    import_data["imported_at"] = (
                        import_meta["upload_timestamp"] / 1000
                    )  # Convert ms to seconds
        except Exception:
            pass

    # Read imported.json if it exists (processing results)
    import_data["processed"] = False
    import_data["error"] = None
    import_data["error_stage"] = None
    imported_json = import_dir / "imported.json"
    if imported_json.exists():
        try:
            with open(imported_json, "r", encoding="utf-8") as f:
                imported_meta = json.load(f)
                import_data["processed"] = True
                import_data["total_files_created"] = imported_meta.get(
                    "total_files_created", 0
                )
                import_data["target_day"] = imported_meta.get("target_day")

                # Check for error state
                if "error" in imported_meta:
                    import_data["error"] = imported_meta.get("error")
                    import_data["error_stage"] = imported_meta.get("error_stage")

                # Calculate duration from imported files
                if imported_meta.get("all_created_files"):
                    duration = calculate_duration_from_files(
                        imported_meta["all_created_files"]
                    )
                    if duration:
                        import_data["duration_minutes"] = duration
        except Exception:
            pass

    return import_data


# ============================================================================
# Detail View
# ============================================================================


def get_import_details(
    journal_root: Path,
    timestamp: str,
) -> dict:
    """Get all metadata files for import detail view.

    Args:
        journal_root: Root journal directory
        timestamp: Import timestamp

    Returns:
        Dict with all detail information

    Raises:
        FileNotFoundError: If import directory doesn't exist
    """
    import_dir = journal_root / "imports" / timestamp
    if not import_dir.exists():
        raise FileNotFoundError(f"Import not found: {timestamp}")

    result = {
        "timestamp": timestamp,
        "import_json": None,
        "imported_json": None,
        "has_summary": False,
    }

    # Read import.json
    import_json_path = import_dir / "import.json"
    if import_json_path.exists():
        try:
            with open(import_json_path, "r", encoding="utf-8") as f:
                result["import_json"] = json.load(f)
        except Exception:
            pass

    # Read imported.json
    imported_json_path = import_dir / "imported.json"
    if imported_json_path.exists():
        try:
            with open(imported_json_path, "r", encoding="utf-8") as f:
                result["imported_json"] = json.load(f)
        except Exception:
            pass

    # Check if summary.md exists
    summary_path = import_dir / "summary.md"
    if summary_path.exists():
        result["has_summary"] = True

    # Read segments.json
    segments_json_path = import_dir / "segments.json"
    if segments_json_path.exists():
        try:
            with open(segments_json_path, "r", encoding="utf-8") as f:
                result["segments_json"] = json.load(f)
        except Exception:
            pass

    return result


# ============================================================================
# Segment Tracking
# ============================================================================


def save_import_segments(
    journal_root: Path,
    timestamp: str,
    segments: list[str],
    day: str,
) -> None:
    """Save segment list for an import.

    Args:
        journal_root: Root journal directory
        timestamp: Import timestamp (YYYYMMDD_HHMMSS format)
        segments: List of segment keys (HHMMSS_LEN format)
        day: Day string (YYYYMMDD format)
    """
    import_dir = journal_root / "imports" / timestamp
    import_dir.mkdir(parents=True, exist_ok=True)

    segments_path = import_dir / "segments.json"
    data = {
        "segments": segments,
        "day": day,
    }
    segments_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_import_segments(
    journal_root: Path,
    timestamp: str,
) -> tuple[list[str], str] | None:
    """Load segment list for an import.

    Args:
        journal_root: Root journal directory
        timestamp: Import timestamp

    Returns:
        Tuple of (segments_list, day) or None if not found
    """
    import_dir = journal_root / "imports" / timestamp
    segments_path = import_dir / "segments.json"

    if not segments_path.exists():
        return None

    try:
        data = json.loads(segments_path.read_text(encoding="utf-8"))
        return data.get("segments", []), data.get("day", "")
    except Exception:
        return None
