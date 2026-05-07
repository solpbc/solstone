# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Root-level push API."""

from __future__ import annotations

import uuid
from typing import Any

from flask import Blueprint, jsonify, request
from werkzeug.exceptions import BadRequest

from solstone.think.push import triggers
from solstone.think.push.config import is_configured
from solstone.think.push.devices import (
    load_devices,
    register_device,
    remove_device,
    status_view,
)
from solstone.think.push.dispatch import CATEGORIES, CATEGORY_AGENT_ALERT

push_bp = Blueprint("push", __name__, url_prefix="/api/push")


def _error(message: str, status: int):
    return jsonify({"error": message}), status


def _optional_json_object() -> tuple[dict[str, Any], Any | None]:
    if not request.get_data(cache=True):
        return {}, None
    try:
        data = request.get_json(silent=False)
    except BadRequest:
        return {}, _error("request body must be valid JSON", 400)
    if not isinstance(data, dict):
        return {}, _error("request body must be a JSON object", 400)
    return data, None


def _required_json_object() -> tuple[dict[str, Any], Any | None]:
    try:
        data = request.get_json(silent=False)
    except BadRequest:
        return {}, _error("request body must be valid JSON", 400)
    if not isinstance(data, dict):
        return {}, _error("request body must be a JSON object", 400)
    return data, None


@push_bp.post("/register")
def register_push_device():
    body, error = _required_json_object()
    if error is not None:
        return error
    token = str(body.get("device_token") or "").strip()
    bundle_id = str(body.get("bundle_id") or "").strip()
    environment = str(body.get("environment") or "").strip()
    platform = str(body.get("platform") or "").strip()
    if not token:
        return _error("device_token is required", 400)
    if not bundle_id:
        return _error("bundle_id is required", 400)
    if environment not in {"development", "production"}:
        return _error("environment must be development or production", 400)
    if platform != "ios":
        return _error("platform must be ios", 400)
    count = register_device(
        token="".join(token.split()).lower(),
        bundle_id=bundle_id,
        environment=environment,
        platform=platform,
    )
    return jsonify({"registered": True, "device_count": count})


@push_bp.delete("/register")
def unregister_push_device():
    body, error = _required_json_object()
    if error is not None:
        return error
    token = str(body.get("device_token") or "").strip()
    if not token:
        return _error("device_token is required", 400)
    removed = remove_device("".join(token.split()).lower())
    return jsonify({"removed": removed, "device_count": len(load_devices())})


@push_bp.get("/status")
def push_status():
    devices = sorted(
        load_devices(),
        key=lambda device: int(device.get("registered_at", 0)),
        reverse=True,
    )
    return jsonify(
        {
            "configured": is_configured(),
            "device_count": len(devices),
            "devices": [status_view(device) for device in devices],
        }
    )


@push_bp.post("/test")
def send_push_test():
    body, error = _optional_json_object()
    if error is not None:
        return error
    if not is_configured():
        return _error("push not configured", 503)
    category = body.get("category", CATEGORY_AGENT_ALERT)
    if category not in CATEGORIES:
        return _error("category must be a known push category", 400)
    title = str(body.get("title") or "Push test")
    message = str(body.get("body") or "This is a test notification.")
    sent, failed = triggers.send_agent_alert(
        title=title,
        body=message,
        context_id=f"push-test-{uuid.uuid4().hex[:12]}",
        route="/app/home",
    )
    return jsonify({"sent": sent, "failed": failed})


__all__ = ["push_bp"]
