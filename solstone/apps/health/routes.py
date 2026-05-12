# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

import re
import socket
from pathlib import Path

from flask import Blueprint, jsonify, request

from solstone.convey import state
from solstone.convey.reasons import (
    FILE_NOT_FOUND,
    FILE_READ_FAILED,
    INVALID_PATH,
    INVALID_REQUEST_VALUE,
    MISSING_REQUIRED_FIELD,
    OBSERVER_RESTART_FAILED,
)
from solstone.convey.utils import error_response
from solstone.think.callosum import callosum_send
from solstone.think.streams import stream_name

health_bp = Blueprint("app:health", __name__, url_prefix="/app/health")

# Supervisor currently registers one observer-facing processing service: "sense".
# Observer rows are per registration key, but reconnect restarts this shared worker.
# Keep this endpoint whitelist local until supervisor exposes a public service list.
OBSERVER_RESTART_SERVICES = {"sense"}


@health_bp.get("/api/log")
def get_log():
    path = request.args.get("path")
    if not path:
        return error_response(MISSING_REQUIRED_FIELD, detail="Missing path parameter")

    if not re.fullmatch(r"\d{8}/health/[^/]+\.log", path):
        return error_response(INVALID_PATH, detail="Invalid path")

    journal_root = Path(state.journal_root).resolve()
    try:
        file_path = (Path(state.journal_root) / path).resolve()
    except ValueError:
        return error_response(INVALID_PATH, detail="Invalid path")
    try:
        file_path.relative_to(journal_root)
    except ValueError:
        return error_response(INVALID_PATH, detail="Invalid path")

    if not file_path.exists():
        return error_response(FILE_NOT_FOUND, detail="Log file not found")

    try:
        content = file_path.read_text(encoding="utf-8")
    except IOError:
        return error_response(FILE_READ_FAILED, detail="Failed to read log file")

    return jsonify(content=content, path=path)


@health_bp.route("/api/info")
def api_info():
    return jsonify({"hostname": stream_name(host=socket.gethostname())})


@health_bp.post("/api/retry-import")
def retry_import():
    data = request.get_json(silent=True) or {}
    if not data.get("import_id"):
        return error_response(MISSING_REQUIRED_FIELD, detail="Missing import_id")
    stage = data.get("stage")
    message = "Import retry will be available in a future update"
    if stage:
        message = (
            f"Import retry from stage {stage} will be available in a future update"
        )
    return jsonify(
        status="not_implemented",
        message=message,
    ), 501


@health_bp.post("/api/restart-observer")
def restart_observer():
    data = request.get_json(silent=True) or {}
    service = data.get("service")
    if not service:
        return error_response(MISSING_REQUIRED_FIELD, detail="Missing service")
    if service not in OBSERVER_RESTART_SERVICES:
        return error_response(INVALID_REQUEST_VALUE, detail="Unknown observer service")

    ok = callosum_send("supervisor", "restart", service=service)
    if not ok:
        return error_response(
            OBSERVER_RESTART_FAILED,
            detail="Could not reach the supervisor",
        )

    return jsonify(status="restart_requested", service=service)
