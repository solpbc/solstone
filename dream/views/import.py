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
    imp_dir = Path(state.journal_root) / "importer"
    imp_dir.mkdir(parents=True, exist_ok=True)
    upload = request.files.get("file")
    text = request.form.get("text", "").strip()
    domain = request.form.get("domain", "").strip() or None

    # Generate timestamp for filename
    timestamp_ms = int(time.time() * 1000)

    if upload and upload.filename:
        filename = secure_filename(upload.filename)
        path = imp_dir / f"{timestamp_ms}_{filename}"
        upload.save(path)
    elif text:
        path = imp_dir / f"{timestamp_ms}_paste.txt"
        path.write_text(text, encoding="utf-8")
    else:
        return jsonify({"error": "No input"}), 400

    # Detect timestamp from content
    ts = None
    detection_result = None
    try:
        detection_result = detect_created(str(path))
        if (
            detection_result
            and detection_result.get("day")
            and detection_result.get("time")
        ):
            ts = f"{detection_result['day']}_{detection_result['time']}"
    except Exception:
        ts = None

    # Save metadata JSON file
    metadata = {
        "original_filename": upload.filename if upload else "paste.txt",
        "upload_timestamp": timestamp_ms,
        "upload_datetime": datetime.fromtimestamp(timestamp_ms / 1000).isoformat(),
        "optional_text": (
            text if text and upload else None
        ),  # Only include if file upload with text
        "detection_result": detection_result,
        "detected_timestamp": ts,
        "file_size": path.stat().st_size if path.exists() else 0,
        "mime_type": upload.content_type if upload else "text/plain",
        "domain": domain,  # Include selected domain
    }

    metadata_path = Path(str(path) + ".json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return jsonify({"path": str(path), "timestamp": ts})


@bp.route("/import/api/log")
def import_log() -> Any:
    entries: list[dict[str, Any]] = []
    if state.journal_root:
        log_path = Path(state.journal_root) / "importer" / "task_log.txt"
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
                entries = []
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
