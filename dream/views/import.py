from __future__ import annotations

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
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix) as tmp:
            upload.save(tmp.name)
            temp_path = tmp.name
            upload.seek(0)  # Reset file pointer for later save
    else:
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as tmp:
            tmp.write(text)
            temp_path = tmp.name

    try:
        detection_result = detect_created(temp_path)
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
    folder_timestamp = ts if ts else f"{datetime.fromtimestamp(timestamp_ms / 1000).strftime('%Y%m%d_%H%M%S')}"

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
        "optional_text": (
            text if text and upload else None
        ),  # Only include if file upload with text
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


@bp.route("/import/api/log")
def import_log() -> Any:
    entries: list[dict[str, Any]] = []
    if state.journal_root:
        # Check both old and new locations for backwards compatibility
        log_paths = [
            Path(state.journal_root) / "imports" / "task_log.txt",
            Path(state.journal_root) / "importer" / "task_log.txt",  # Legacy location
        ]
        for log_path in log_paths:
            if log_path.is_file():
                try:
                    with open(log_path, "r", encoding="utf-8") as f:
                        for line in f:
                            parts = line.strip().split("\t", 1)
                            if len(parts) != 2:
                                continue
                            try:
                                ts = int(parts[0])
                            except ValueError:
                                continue
                            entries.append({"time": ts, "message": parts[1]})
                except Exception:
                    pass
    entries.sort(key=lambda e: e["time"], reverse=True)
    return jsonify(entries)


@bp.route("/import/api/start", methods=["POST"])
def import_start() -> Any:
    data = request.get_json(force=True)
    path = data.get("path")
    ts = data.get("timestamp")
    if not path or not ts:
        return jsonify({"error": "missing params"}), 400
    run_task("importer", f"{path}|{ts}")
    return jsonify({"status": "ok"})
