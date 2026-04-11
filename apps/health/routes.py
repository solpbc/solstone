# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

import re
import socket
from pathlib import Path

from flask import Blueprint, jsonify, request

from convey import state
from think.streams import stream_name

health_bp = Blueprint("app:health", __name__, url_prefix="/app/health")


@health_bp.get("/api/log")
def get_log():
    path = request.args.get("path")
    if not path:
        return jsonify(error="Missing path parameter"), 400

    if not re.fullmatch(r"\d{8}/health/[^/]+\.log", path):
        return jsonify(error="Invalid path"), 400

    journal_root = Path(state.journal_root).resolve()
    try:
        file_path = (Path(state.journal_root) / path).resolve()
    except ValueError:
        return jsonify(error="Invalid path"), 400
    try:
        file_path.relative_to(journal_root)
    except ValueError:
        return jsonify(error="Invalid path"), 400

    if not file_path.exists():
        return jsonify(error="Log file not found"), 404

    try:
        content = file_path.read_text(encoding="utf-8")
    except IOError:
        return jsonify(error="Failed to read log file"), 500

    return jsonify(content=content, path=path)


@health_bp.route("/api/info")
def api_info():
    return jsonify({"hostname": stream_name(host=socket.gethostname())})


@health_bp.post("/api/retry-import")
def retry_import():
    data = request.get_json(silent=True) or {}
    if not data.get("import_id"):
        return jsonify(error="Missing import_id"), 400
    return jsonify(
        status="not_implemented",
        message="Import retry will be available in a future update",
    ), 501
