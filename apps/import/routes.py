# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from flask import Blueprint, jsonify, render_template, request
from werkzeug.utils import secure_filename

from convey import emit, state
from think.detect_created import detect_created
from think.importer_utils import (
    build_import_info,
    get_import_details,
    list_import_timestamps,
    read_import_metadata,
    save_import_file,
    save_import_text,
    update_import_metadata_fields,
    write_import_metadata,
)
from think.utils import now_ms

import_bp = Blueprint(
    "app:import",
    __name__,
    url_prefix="/app/import",
)


@import_bp.route("/api/save", methods=["POST"])
def import_save() -> Any:
    from datetime import datetime

    upload = request.files.get("file")
    text = request.form.get("text", "").strip()
    facet = request.form.get("facet", "").strip() or None
    setting = request.form.get("setting", "").strip() or None

    # Generate timestamp for folder name
    timestamp_ms = now_ms()

    # Determine filename
    if upload and upload.filename:
        filename = secure_filename(upload.filename)
    elif text:
        filename = "paste.txt"
    else:
        return jsonify({"error": "No input"}), 400

    # Detect timestamp from content first (need temporary save for detection)
    ts = None
    detection_result = None

    # Create temporary file for detection if needed
    if upload:
        import tempfile

        # Preserve original filename structure in temp file name for timestamp detection
        # Use prefix to include original filename (minus extension)
        original_stem = Path(filename).stem
        suffix = Path(filename).suffix
        with tempfile.NamedTemporaryFile(
            delete=False, prefix=f"{original_stem}_", suffix=suffix
        ) as tmp:
            upload.save(tmp.name)
            temp_path = tmp.name
            upload.seek(0)  # Reset file pointer for later save
    else:
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as tmp:
            tmp.write(text)
            temp_path = tmp.name

    try:
        # Pass original filename for better timestamp detection
        original_name = upload.filename if upload else None
        detection_result = detect_created(temp_path, original_filename=original_name)
        if (
            detection_result
            and detection_result.get("day")
            and detection_result.get("time")
        ):
            ts = f"{detection_result['day']}_{detection_result['time']}"
    except Exception:
        ts = None
    finally:
        # Clean up temporary file
        Path(temp_path).unlink(missing_ok=True)

    # Use detected timestamp or fall back to upload timestamp
    folder_timestamp = (
        ts
        if ts
        else f"{datetime.fromtimestamp(timestamp_ms / 1000).strftime('%Y%m%d_%H%M%S')}"
    )

    # Save the actual file using utility function
    if upload:
        # Save uploaded file to temp location first, then move to import dir
        import tempfile

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            upload.save(tmp.name)
            temp_source = Path(tmp.name)

        try:
            file_path = save_import_file(
                journal_root=Path(state.journal_root),
                timestamp=folder_timestamp,
                source_path=temp_source,
                filename=filename,
            )
        finally:
            temp_source.unlink(missing_ok=True)
    else:
        file_path = save_import_text(
            journal_root=Path(state.journal_root),
            timestamp=folder_timestamp,
            content=text,
            filename=filename,
        )

    # Build metadata dict
    metadata = {
        "original_filename": upload.filename if upload else "paste.txt",
        "upload_timestamp": timestamp_ms,
        "upload_datetime": datetime.fromtimestamp(timestamp_ms / 1000).isoformat(),
        "detection_result": detection_result,
        "detected_timestamp": ts,
        "user_timestamp": folder_timestamp,  # The timestamp used for the folder
        "file_size": file_path.stat().st_size if file_path.exists() else 0,
        "mime_type": upload.content_type if upload else "text/plain",
        "facet": facet,  # Include selected facet
        "setting": setting,
        "file_path": str(file_path),  # Store the actual file path
    }

    # Write metadata using utility function
    write_import_metadata(
        journal_root=Path(state.journal_root),
        timestamp=folder_timestamp,
        metadata=metadata,
    )

    return jsonify(
        {
            "path": str(file_path),
            "timestamp": folder_timestamp,
            "facet": facet,
            "setting": setting,
        }
    )


@import_bp.route("/api/facet", methods=["POST"])
def import_update_metadata() -> Any:
    """Update stored metadata (facet/setting) for a saved import."""
    data = request.get_json(force=True)
    raw_path = data.get("path", "").strip()
    if not raw_path:
        return jsonify({"error": "Missing import path"}), 400

    facet = data.get("facet", "").strip() or None
    setting = data.get("setting", "").strip() or None

    # Extract timestamp from path
    # Path format: .../imports/{timestamp}/{filename}
    file_path = Path(raw_path)
    timestamp = file_path.parent.name

    try:
        # Use utility function to update metadata
        metadata, updated = update_import_metadata_fields(
            journal_root=Path(state.journal_root),
            timestamp=timestamp,
            updates={"facet": facet, "setting": setting},
        )
    except FileNotFoundError:
        return jsonify({"error": "Import metadata not found"}), 404
    except Exception as exc:
        return jsonify({"error": f"Failed to update metadata: {exc}"}), 500

    return jsonify(
        {
            "status": "ok",
            "facet": facet,
            "setting": setting,
            "updated": updated,
        }
    )


@import_bp.route("/api/list")
def import_list() -> Any:
    """Get list of all imports with their metadata."""
    # Get all import timestamps using utility function
    timestamps = list_import_timestamps(journal_root=Path(state.journal_root))

    # Build info for each import using utility function
    imports = []
    for timestamp in timestamps:
        import_data = build_import_info(
            journal_root=Path(state.journal_root),
            timestamp=timestamp,
        )

        # Calculate status based on processing state
        # Default status
        import_data["status"] = "pending"

        task_id = import_data.get("task_id")
        current_time = time.time()
        import_age_seconds = current_time - import_data.get("imported_at", current_time)
        import_timeout_seconds = 3600  # 1 hour

        # Check for error state first (imported.json exists with error field)
        if import_data.get("error"):
            import_data["status"] = "failed"
        # If we have processing results without error, it's successful
        elif import_data.get("processed"):
            import_data["status"] = "success"
        # If task was started but no results yet, check if it timed out
        elif task_id:
            if import_age_seconds > import_timeout_seconds:
                # Import is stuck/crashed - mark as failed
                import_data["status"] = "failed"
                import_data["error"] = "Import never completed"
                import_data["error_stage"] = "timeout"
            else:
                import_data["status"] = "running"

        imports.append(import_data)

    # Sort by imported_at (newest first)
    imports.sort(key=lambda x: x.get("imported_at", 0), reverse=True)

    return jsonify(imports)


@import_bp.route("/<timestamp>")
def import_detail(timestamp: str) -> str:
    """Show detailed view of a specific import."""
    import_dir = Path(state.journal_root) / "imports" / timestamp
    if not import_dir.exists():
        return render_template("error.html", error="Import not found"), 404

    return render_template(
        "app.html",
        view="detail",
        timestamp=timestamp,
    )


@import_bp.route("/api/<timestamp>")
def import_detail_api(timestamp: str) -> Any:
    """Get detailed data for a specific import."""
    try:
        # Use utility function to get all details
        result = get_import_details(
            journal_root=Path(state.journal_root),
            timestamp=timestamp,
        )
        return jsonify(result)
    except FileNotFoundError:
        return jsonify({"error": "Import not found"}), 404


@import_bp.route("/api/<timestamp>/summary")
def import_summary_api(timestamp: str) -> Any:
    """Get the summary HTML for a specific import."""
    import_dir = Path(state.journal_root) / "imports" / timestamp
    if not import_dir.exists():
        return jsonify({"error": "Import not found"}), 404

    summary_path = import_dir / "summary.md"
    if not summary_path.exists():
        return jsonify(
            {
                "html": "<div class='no-data'>No summary available</div>",
                "has_summary": False,
            }
        )

    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            summary_md = f.read()

        # Render markdown to HTML server-side
        import markdown  # type: ignore

        html_output = markdown.markdown(
            summary_md, extensions=["extra", "codehilite", "fenced_code", "tables"]
        )

        return jsonify({"html": html_output, "has_summary": True})
    except Exception as e:
        return jsonify(
            {
                "error": str(e),
                "html": "<div class='no-data'>Error loading summary</div>",
                "has_summary": False,
            }
        )


@import_bp.route("/api/start", methods=["POST"])
def import_start() -> Any:
    data = request.get_json(force=True)
    path = data.get("path")
    ts = data.get("timestamp")
    if not path or not ts:
        return jsonify({"error": "missing params"}), 400

    # Generate task ID
    task_id = str(now_ms())

    # Extract original timestamp from path and handle timestamp changes
    file_path = Path(path)
    original_timestamp = file_path.parent.name
    journal_root = Path(state.journal_root)

    # If timestamp changed, rename the import directory
    if original_timestamp != ts:
        old_import_dir = journal_root / "imports" / original_timestamp
        new_import_dir = journal_root / "imports" / ts

        # Check if old directory exists
        if not old_import_dir.exists():
            return (
                jsonify(
                    {"error": f"Import directory not found for {original_timestamp}"}
                ),
                404,
            )

        # Check if target directory already exists
        if new_import_dir.exists():
            return jsonify({"error": f"Import already exists for timestamp {ts}"}), 409

        # Rename the directory
        try:
            old_import_dir.rename(new_import_dir)
        except Exception as e:
            return (
                jsonify({"error": f"Failed to rename import directory: {str(e)}"}),
                500,
            )

        # Update path to point to new location
        path = str(new_import_dir / file_path.name)

        # Update file_path in metadata (need to update after reading)
        # We'll handle this after reading the metadata below

    # Read import metadata to get facet and setting
    try:
        metadata = read_import_metadata(journal_root=journal_root, timestamp=ts)
    except FileNotFoundError:
        return jsonify({"error": f"Import metadata not found for {ts}"}), 404
    except Exception as e:
        return jsonify({"error": f"Failed to read metadata: {str(e)}"}), 500

    # Update file_path in metadata if timestamp changed
    if original_timestamp != ts:
        try:
            update_import_metadata_fields(
                journal_root=journal_root,
                timestamp=ts,
                updates={"file_path": path},
            )
        except Exception as e:
            return (
                jsonify({"error": f"Failed to update file path in metadata: {str(e)}"}),
                500,
            )

    facet = metadata.get("facet")
    setting = metadata.get("setting")

    # Build command
    cmd = ["sol", "import", path, ts]
    if facet:
        cmd.extend(["--facet", facet])
    if setting:
        cmd.extend(["--setting", setting])

    # Store task_id in metadata
    try:
        update_import_metadata_fields(
            journal_root=journal_root,
            timestamp=ts,
            updates={"task_id": task_id},
        )
    except Exception as e:
        return jsonify({"error": f"Failed to update metadata: {str(e)}"}), 500

    # Emit task request to Callosum (non-blocking, drops if disconnected)
    emit("supervisor", "request", ref=task_id, cmd=cmd)

    return jsonify({"status": "ok", "task_id": task_id})
