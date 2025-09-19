from __future__ import annotations

import datetime
import time
from pathlib import Path
from typing import Any

from flask import Blueprint, jsonify, render_template, request
from werkzeug.utils import secure_filename

from think.detect_created import detect_created

from .. import state
from ..task_runner import run_task

bp = Blueprint("import_view", __name__, template_folder="../templates")


@bp.route("/import")
def import_page() -> str:
    return render_template("import.html", active="import")


@bp.route("/import/api/save", methods=["POST"])
def import_save() -> Any:
    import json
    from datetime import datetime

    if not state.journal_root:
        resp = jsonify({"error": "JOURNAL_PATH not set"})
        resp.status_code = 500
        return resp

    upload = request.files.get("file")
    text = request.form.get("text", "").strip()
    domain = request.form.get("domain", "").strip() or None

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

    # Create import folder structure: imports/<timestamp>/<filename>
    import_dir = Path(state.journal_root) / "imports" / folder_timestamp
    import_dir.mkdir(parents=True, exist_ok=True)

    # Save the actual file
    file_path = import_dir / filename
    if upload:
        upload.save(file_path)
    else:
        file_path.write_text(text, encoding="utf-8")

    # Save metadata to import.json in the same folder
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
        "file_path": str(file_path),  # Store the actual file path
    }

    metadata_path = import_dir / "import.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return jsonify({"path": str(file_path), "timestamp": folder_timestamp})


@bp.route("/import/api/domain", methods=["POST"])
def import_update_domain() -> Any:
    """Update the domain stored in import metadata for a saved file."""
    import json

    if not state.journal_root:
        return jsonify({"error": "JOURNAL_PATH not set"}), 500

    data = request.get_json(force=True)
    raw_path = data.get("path", "").strip()
    if not raw_path:
        return jsonify({"error": "Missing import path"}), 400

    domain = data.get("domain", "").strip() or None
    file_path = Path(raw_path)

    metadata_path = file_path.parent / "import.json"
    if not metadata_path.exists():
        legacy_path = Path(f"{raw_path}.json")
        if legacy_path.exists():
            metadata_path = legacy_path
        else:
            return jsonify({"error": "Import metadata not found"}), 404

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return jsonify({"error": f"Failed to read import metadata: {exc}"}), 500

    if metadata.get("domain") == domain:
        return jsonify({"status": "ok", "domain": domain, "updated": False})

    metadata["domain"] = domain

    try:
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    except Exception as exc:
        return jsonify({"error": f"Failed to update metadata: {exc}"}), 500

    return jsonify({"status": "ok", "domain": domain, "updated": True})


@bp.route("/import/api/list")
def import_list() -> Any:
    """Get list of all imports with their metadata."""
    import json
    from datetime import datetime

    imports = []

    if not state.journal_root:
        return jsonify([])

    imports_dir = Path(state.journal_root) / "imports"
    if not imports_dir.exists():
        return jsonify([])

    # Iterate through each import folder
    for import_folder in imports_dir.iterdir():
        if not import_folder.is_dir():
            continue

        # Skip if it's not a timestamp folder
        if not (import_folder.name.count("_") == 1 and len(import_folder.name) == 15):
            continue

        import_data = {
            "timestamp": import_folder.name,
            "created_at": import_folder.stat().st_ctime,
            "created_at_iso": datetime.fromtimestamp(
                import_folder.stat().st_ctime
            ).isoformat(),
        }

        # Read import.json if it exists
        import_json = import_folder / "import.json"
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
                    import_data["domain"] = import_meta.get("domain")
                    import_data["user_timestamp"] = import_meta.get("user_timestamp")
                    task_id = import_meta.get("task_id")
                    import_data["task_id"] = task_id
                    # Use upload_timestamp if available for better sorting
                    if "upload_timestamp" in import_meta:
                        import_data["imported_at"] = (
                            import_meta["upload_timestamp"] / 1000
                        )  # Convert ms to seconds
                    else:
                        import_data["imported_at"] = import_folder.stat().st_ctime
            except Exception:
                import_data["imported_at"] = import_folder.stat().st_ctime
        else:
            import_data["imported_at"] = import_folder.stat().st_ctime

        # Check task status if we have a task_id
        import_data["status"] = "pending"  # Default status

        # Read imported.json if it exists (processing results)
        imported_json = import_folder / "imported.json"
        if imported_json.exists():
            try:
                with open(imported_json, "r", encoding="utf-8") as f:
                    imported_meta = json.load(f)
                    import_data["processed"] = True
                    import_data["total_files_created"] = imported_meta.get(
                        "total_files_created", 0
                    )
                    import_data["target_day"] = imported_meta.get("target_day")
                    import_data["status"] = "success"  # Has imported.json = successful

                    # Calculate duration from imported files
                    if imported_meta.get("all_created_files"):
                        files = imported_meta["all_created_files"]
                        timestamps = []
                        for file in files:
                            # Extract timestamp from filename like "120000_imported_audio.json"
                            basename = Path(file).name
                            if basename[:6].isdigit():
                                timestamps.append(basename[:6])
                        if timestamps:
                            timestamps.sort()
                            start_time = timestamps[0]
                            end_time = timestamps[-1]
                            # Convert to minutes
                            start_h, start_m = int(start_time[:2]), int(start_time[2:4])
                            end_h, end_m = int(end_time[:2]), int(end_time[2:4])
                            duration_minutes = (end_h * 60 + end_m) - (
                                start_h * 60 + start_m
                            )
                            if duration_minutes > 0:
                                import_data["duration_minutes"] = duration_minutes
            except Exception:
                import_data["processed"] = False
                # If task was started but no imported.json, it likely failed
                if task_id:
                    import_data["status"] = "failed"
        else:
            import_data["processed"] = False
            # Check if task was started but didn't produce results
            if task_id:
                # Check for task exit code by looking at task history
                from dream.tasks import task_manager

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
    import json

    if not state.journal_root:
        return jsonify({"error": "JOURNAL_PATH not set"}), 500

    import_dir = Path(state.journal_root) / "imports" / timestamp
    if not import_dir.exists():
        return jsonify({"error": "Import not found"}), 404

    result = {
        "timestamp": timestamp,
        "import_json": None,
        "imported_json": None,
        "revai_json": None,
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

    # Read revai.json
    revai_json_path = import_dir / "revai.json"
    if revai_json_path.exists():
        try:
            with open(revai_json_path, "r", encoding="utf-8") as f:
                result["revai_json"] = json.load(f)
        except Exception:
            pass

    # Check if summary.md exists
    summary_path = import_dir / "summary.md"
    if summary_path.exists():
        result["has_summary"] = True

    return jsonify(result)


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
    import json

    if not state.journal_root:
        return jsonify({"error": "JOURNAL_PATH not set"}), 500

    import_dir = Path(state.journal_root) / "imports" / timestamp
    if not import_dir.exists():
        return jsonify({"error": "Import not found"}), 404

    # Read import metadata
    import_json_path = import_dir / "import.json"
    if not import_json_path.exists():
        return jsonify({"error": "Import metadata not found"}), 404

    try:
        with open(import_json_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except Exception as e:
        return jsonify({"error": f"Failed to read import metadata: {str(e)}"}), 500

    # Get file path from metadata
    file_path = metadata.get("file_path")
    if not file_path:
        return jsonify({"error": "File path not found in metadata"}), 500

    # Check if file still exists
    if not Path(file_path).exists():
        return jsonify({"error": "Original file no longer exists"}), 404

    # Get new domain from request
    data = request.get_json(force=True)
    new_domain = data.get("domain", "").strip() or None

    # Update metadata with new domain
    if new_domain != metadata.get("domain"):
        metadata["domain"] = new_domain
        metadata["rerun_at"] = time.time() * 1000
        metadata["rerun_datetime"] = datetime.datetime.now().isoformat()

        try:
            with open(import_json_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            return jsonify({"error": f"Failed to update metadata: {str(e)}"}), 500

    # Clear previous processing results to indicate re-run
    imported_json_path = import_dir / "imported.json"
    if imported_json_path.exists():
        try:
            # Archive the old results
            archive_path = import_dir / f"imported.{int(time.time())}.json.bak"
            imported_json_path.rename(archive_path)
        except Exception:
            pass  # Continue even if archiving fails

    # Run the importer task with the same timestamp
    run_task("importer", f"{file_path}|{timestamp}")

    return jsonify({"status": "ok", "domain": new_domain})
