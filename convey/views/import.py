from __future__ import annotations

import datetime
import time
from pathlib import Path
from typing import Any

from flask import Blueprint, jsonify, render_template, request
from werkzeug.utils import secure_filename

from think.detect_created import detect_created
from think.importer_utils import (
    archive_imported_results,
    build_import_info,
    get_import_details,
    list_import_timestamps,
    read_import_metadata,
    save_import_file,
    save_import_text,
    update_import_metadata_fields,
    write_import_metadata,
)

from .. import state
from ..task_runner import run_task

bp = Blueprint("import_view", __name__, template_folder="../templates")


@bp.route("/import")
def import_page() -> str:
    return render_template("import.html", active="import")


@bp.route("/import/api/save", methods=["POST"])
def import_save() -> Any:
    from datetime import datetime

    if not state.journal_root:
        resp = jsonify({"error": "JOURNAL_PATH not set"})
        resp.status_code = 500
        return resp

    upload = request.files.get("file")
    text = request.form.get("text", "").strip()
    domain = request.form.get("domain", "").strip() or None
    setting = request.form.get("setting", "").strip() or None

    # Generate timestamp for folder name
    timestamp_ms = int(time.time() * 1000)

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
        "domain": domain,  # Include selected domain
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
            "domain": domain,
            "setting": setting,
        }
    )


@bp.route("/import/api/domain", methods=["POST"])
def import_update_metadata() -> Any:
    """Update stored metadata (domain/setting) for a saved import."""
    if not state.journal_root:
        return jsonify({"error": "JOURNAL_PATH not set"}), 500

    data = request.get_json(force=True)
    raw_path = data.get("path", "").strip()
    if not raw_path:
        return jsonify({"error": "Missing import path"}), 400

    domain = data.get("domain", "").strip() or None
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
            updates={"domain": domain, "setting": setting},
        )
    except FileNotFoundError:
        return jsonify({"error": "Import metadata not found"}), 404
    except Exception as exc:
        return jsonify({"error": f"Failed to update metadata: {exc}"}), 500

    return jsonify(
        {
            "status": "ok",
            "domain": domain,
            "setting": setting,
            "updated": updated,
        }
    )


@bp.route("/import/api/list")
def import_list() -> Any:
    """Get list of all imports with their metadata."""
    if not state.journal_root:
        return jsonify([])

    # Get all import timestamps using utility function
    timestamps = list_import_timestamps(journal_root=Path(state.journal_root))

    # Build info for each import using utility function
    imports = []
    for timestamp in timestamps:
        import_data = build_import_info(
            journal_root=Path(state.journal_root),
            timestamp=timestamp,
        )

        # Calculate status based on processing state and task manager
        # Default status
        import_data["status"] = "pending"

        task_id = import_data.get("task_id")

        # If we have processing results, it's successful
        if import_data.get("processed"):
            import_data["status"] = "success"
        # If task was started but no results
        elif task_id:
            # Check for task exit code by looking at task history
            from convey.tasks import task_manager

            task = task_manager.tasks.get(task_id)
            if task:
                if task.exit_code is not None and task.exit_code != 0:
                    import_data["status"] = "failed"
                elif task.exit_code == 0:
                    import_data["status"] = "success"
                else:
                    import_data["status"] = "running"
            else:
                # Task was started but no longer in memory, likely failed
                import_data["status"] = "failed"

        imports.append(import_data)

    # Sort by imported_at (newest first)
    imports.sort(key=lambda x: x.get("imported_at", 0), reverse=True)

    return jsonify(imports)


@bp.route("/import/<timestamp>")
def import_detail(timestamp: str) -> str:
    """Show detailed view of a specific import."""
    if not state.journal_root:
        return render_template("error.html", error="JOURNAL_PATH not set"), 500

    import_dir = Path(state.journal_root) / "imports" / timestamp
    if not import_dir.exists():
        return render_template("error.html", error="Import not found"), 404

    return render_template("import_detail.html", timestamp=timestamp, active="import")


@bp.route("/import/api/<timestamp>")
def import_detail_api(timestamp: str) -> Any:
    """Get detailed data for a specific import."""
    if not state.journal_root:
        return jsonify({"error": "JOURNAL_PATH not set"}), 500

    try:
        # Use utility function to get all details
        result = get_import_details(
            journal_root=Path(state.journal_root),
            timestamp=timestamp,
        )
        return jsonify(result)
    except FileNotFoundError:
        return jsonify({"error": "Import not found"}), 404


@bp.route("/import/api/<timestamp>/summary")
def import_summary_api(timestamp: str) -> Any:
    """Get the summary HTML for a specific import."""
    if not state.journal_root:
        return jsonify({"error": "JOURNAL_PATH not set"}), 500

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


@bp.route("/import/api/start", methods=["POST"])
def import_start() -> Any:
    data = request.get_json(force=True)
    path = data.get("path")
    ts = data.get("timestamp")
    if not path or not ts:
        return jsonify({"error": "missing params"}), 400
    run_task("importer", f"{path}|{ts}")
    return jsonify({"status": "ok"})


@bp.route("/import/api/<timestamp>/rerun", methods=["POST"])
def import_rerun(timestamp: str) -> Any:
    """Re-run an import with optionally updated domain."""
    if not state.journal_root:
        return jsonify({"error": "JOURNAL_PATH not set"}), 500

    journal_root = Path(state.journal_root)

    # Check if import exists
    import_dir = journal_root / "imports" / timestamp
    if not import_dir.exists():
        return jsonify({"error": "Import not found"}), 404

    # Read import metadata using utility function
    try:
        metadata = read_import_metadata(journal_root=journal_root, timestamp=timestamp)
    except FileNotFoundError:
        return jsonify({"error": "Import metadata not found"}), 404
    except Exception as e:
        return jsonify({"error": f"Failed to read import metadata: {str(e)}"}), 500

    # Get file path from metadata
    file_path = metadata.get("file_path")
    if not file_path:
        return jsonify({"error": "File path not found in metadata"}), 500

    # Check if file still exists
    if not Path(file_path).exists():
        return jsonify({"error": "Original file no longer exists"}), 404

    # Get new domain/setting from request
    data = request.get_json(force=True)
    new_domain = data.get("domain", "").strip() or None
    new_setting = data.get("setting", "").strip() or None

    # Check if values changed
    domain_changed = new_domain != metadata.get("domain")
    setting_changed = new_setting != metadata.get("setting")

    # Update metadata with new values and rerun timestamp
    if domain_changed or setting_changed or "setting" not in metadata:
        updates = {
            "domain": new_domain,
            "setting": new_setting,
            "rerun_at": time.time() * 1000,
            "rerun_datetime": datetime.datetime.now().isoformat(),
        }
        try:
            update_import_metadata_fields(
                journal_root=journal_root,
                timestamp=timestamp,
                updates=updates,
            )
        except Exception as e:
            return jsonify({"error": f"Failed to update metadata: {str(e)}"}), 500

    # Archive previous processing results using utility function
    archive_imported_results(journal_root=journal_root, timestamp=timestamp)

    # Run the importer task with the same timestamp
    run_task("importer", f"{file_path}|{timestamp}")

    return jsonify({"status": "ok", "domain": new_domain, "setting": new_setting})
