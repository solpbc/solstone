# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Root-level push API."""

from __future__ import annotations

import uuid
from typing import Any

from flask import Blueprint, jsonify, request
from werkzeug.exceptions import BadRequest

from solstone.convey.reasons import (
    FEATURE_UNAVAILABLE,
    INVALID_JSON_REQUEST,
    PUSH_REQUEST_INVALID,
)
from solstone.convey.utils import error_response
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


def _optional_json_object() -> tuple[dict[str, Any], Any | None]:
    if not request.get_data(cache=True):
        return {}, None
    try:
        data = request.get_json(silent=False)
    except BadRequest:
        return {}, error_response(
            INVALID_JSON_REQUEST,
            detail="request body must be valid JSON",
        )
    if not isinstance(data, dict):
        return {}, error_response(
            INVALID_JSON_REQUEST,
            detail="request body must be a JSON object",
        )
    return data, None


def _required_json_object() -> tuple[dict[str, Any], Any | None]:
    try:
        data = request.get_json(silent=False)
    except BadRequest:
        return {}, error_response(
            INVALID_JSON_REQUEST,
            detail="request body must be valid JSON",
        )
    if not isinstance(data, dict):
        return {}, error_response(
            INVALID_JSON_REQUEST,
            detail="request body must be a JSON object",
        )
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
        return error_response(PUSH_REQUEST_INVALID, detail="device_token is required")
    if not bundle_id:
        return error_response(PUSH_REQUEST_INVALID, detail="bundle_id is required")
    if environment not in {"development", "production"}:
        return error_response(
            PUSH_REQUEST_INVALID,
            detail="environment must be development or production",
        )
    if platform != "ios":
        return error_response(PUSH_REQUEST_INVALID, detail="platform must be ios")
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
        return error_response(PUSH_REQUEST_INVALID, detail="device_token is required")
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
        return error_response(
            FEATURE_UNAVAILABLE,
            status=503,
            detail="push not configured",
        )
    category = body.get("category", CATEGORY_AGENT_ALERT)
    if category not in CATEGORIES:
        return error_response(
            PUSH_REQUEST_INVALID,
            detail="category must be a known push category",
        )
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
