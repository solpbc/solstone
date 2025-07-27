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
    if not state.journal_root:
        resp = jsonify({"error": "JOURNAL_PATH not set"})
        resp.status_code = 500
        return resp
    imp_dir = Path(state.journal_root) / "importer"
    imp_dir.mkdir(parents=True, exist_ok=True)
    upload = request.files.get("file")
    text = request.form.get("text", "").strip()
    if upload and upload.filename:
        filename = secure_filename(upload.filename)
        path = imp_dir / f"{int(time.time()*1000)}_{filename}"
        upload.save(path)
    elif text:
        path = imp_dir / f"{int(time.time()*1000)}_paste.txt"
        path.write_text(text, encoding="utf-8")
    else:
        return jsonify({"error": "No input"}), 400
    ts = None
    try:
        result = detect_created(str(path))
        if result and result.get("day") and result.get("time"):
            ts = f"{result['day']}_{result['time']}"
    except Exception:
        ts = None
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
