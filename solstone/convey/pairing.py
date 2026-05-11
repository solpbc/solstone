# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Root-level pairing API and UI."""

from __future__ import annotations

import logging
import time
from typing import Any
from urllib.parse import quote

from flask import Blueprint, g, jsonify, render_template, request
from werkzeug.exceptions import BadRequest

from solstone.convey.auth import (
    is_owner_authed,
    require_paired_device,
    resolve_paired_device,
)
from solstone.convey.reasons import (
    AUTH_REQUIRED,
    INVALID_JSON_REQUEST,
    PAIRED_DEVICE_NOT_FOUND,
    PAIRING_KEY_INVALID,
    PAIRING_REQUEST_INVALID,
    PAIRING_TOKEN_INVALID,
    PAIRING_TOKEN_UNAVAILABLE,
)
from solstone.convey.utils import error_response_with_reason
from solstone.think.pairing.config import (
    _detect_lan_ipv4,
    get_host_url,
    get_owner_identity,
)
from solstone.think.pairing.devices import (
    Device,
    load_devices,
    register_device,
    remove_device,
    status_view,
    touch_last_seen,
)
from solstone.think.pairing.keys import (
    generate_session_key,
    hash_session_key,
    mask_session_key,
    validate_public_key,
)
from solstone.think.pairing.tokens import consume_token, peek_token
from solstone.think.pairing.tokens import create_token as mint_pairing_token
from solstone.think.utils import get_config, get_journal

logger = logging.getLogger(__name__)

MAX_DEVICE_NAME_LENGTH = 128

pairing_bp = Blueprint("pairing", __name__, url_prefix="/api/pairing")
pairing_ui_bp = Blueprint("pairing_ui", __name__, url_prefix="/app/pairing")


def _optional_json_object() -> tuple[dict[str, Any], Any | None]:
    if not request.get_data(cache=True):
        return {}, None
    try:
        data = request.get_json(silent=False)
    except BadRequest:
        return {}, error_response_with_reason(
            INVALID_JSON_REQUEST,
            detail="request body must be valid JSON",
        )
    if not isinstance(data, dict):
        return {}, error_response_with_reason(
            PAIRING_REQUEST_INVALID,
            detail="request body must be a JSON object",
        )
    return data, None


def _required_json_object() -> tuple[dict[str, Any], Any | None]:
    try:
        data = request.get_json(silent=False)
    except BadRequest:
        return {}, error_response_with_reason(
            INVALID_JSON_REQUEST,
            detail="request body must be valid JSON",
        )
    if not isinstance(data, dict):
        return {}, error_response_with_reason(
            PAIRING_REQUEST_INVALID,
            detail="request body must be a JSON object",
        )
    return data, None


def _require_field(body: dict[str, Any], field: str) -> str | None:
    value = str(body.get(field) or "").strip()
    if value:
        return value
    return None


def _resolve_owner_or_paired_device() -> tuple[Device | None, Any | None]:
    device = resolve_paired_device()
    if device is not None:
        g.paired_device = device
        return device, None
    if is_owner_authed():
        return None, None
    return None, error_response_with_reason(
        AUTH_REQUIRED,
        detail="owner or paired device required",
    )


def _server_version() -> str:
    try:
        from solstone.think.version import __version__

        return __version__
    except Exception:
        return "unknown"


@pairing_bp.post("/create")
def create_token():
    _, error = _optional_json_object()
    if error is not None:
        return error
    token = mint_pairing_token()
    host_url = get_host_url()
    pairing_url = f"solstone://pair?token={token.token}&host={quote(host_url, safe='')}"
    logger.info("pairing token minted expires_at=%s", token.expires_at)
    return jsonify(
        {
            "token": token.token,
            "expires_at": token.expires_at,
            "pairing_url": pairing_url,
            "qr_data": pairing_url,
        }
    )


@pairing_bp.post("/confirm")
def confirm_pairing():
    body, error = _required_json_object()
    if error is not None:
        return error

    token = _require_field(body, "token")
    public_key = _require_field(body, "public_key")
    device_name = _require_field(body, "device_name")
    platform = _require_field(body, "platform")
    bundle_id = _require_field(body, "bundle_id")
    app_version = _require_field(body, "app_version")

    if token is None:
        return error_response_with_reason(
            PAIRING_REQUEST_INVALID,
            detail="token is required",
        )
    if public_key is None:
        return error_response_with_reason(
            PAIRING_REQUEST_INVALID,
            detail="public_key is required",
        )
    if device_name is None:
        return error_response_with_reason(
            PAIRING_REQUEST_INVALID,
            detail="device_name is required",
        )
    if len(device_name) > MAX_DEVICE_NAME_LENGTH:
        return error_response_with_reason(
            PAIRING_REQUEST_INVALID,
            detail="device_name is too long",
        )
    if platform != "ios":
        return error_response_with_reason(
            PAIRING_REQUEST_INVALID,
            detail="platform must be ios",
        )
    if bundle_id is None:
        return error_response_with_reason(
            PAIRING_REQUEST_INVALID,
            detail="bundle_id is required",
        )
    if app_version is None:
        return error_response_with_reason(
            PAIRING_REQUEST_INVALID,
            detail="app_version is required",
        )

    try:
        normalized_public_key = validate_public_key(public_key)
    except ValueError:
        return error_response_with_reason(
            PAIRING_KEY_INVALID,
            detail="public_key must be a valid ssh-ed25519 key",
        )

    now = int(time.time())
    entry = peek_token(token, now=now)
    if entry is None:
        return error_response_with_reason(
            PAIRING_TOKEN_INVALID,
            detail="pairing token is invalid",
        )
    if entry.expires_at <= now:
        return error_response_with_reason(
            PAIRING_TOKEN_UNAVAILABLE,
            detail="pairing token expired",
        )
    if entry.consumed_at is not None:
        return error_response_with_reason(
            PAIRING_TOKEN_UNAVAILABLE,
            detail="pairing token already used",
        )

    consumed = consume_token(token, now=now)
    if consumed is None:
        entry = peek_token(token, now=now)
        if entry is not None and entry.expires_at <= now:
            return error_response_with_reason(
                PAIRING_TOKEN_UNAVAILABLE,
                detail="pairing token expired",
            )
        if entry is not None and entry.consumed_at is not None:
            return error_response_with_reason(
                PAIRING_TOKEN_UNAVAILABLE,
                detail="pairing token already used",
            )
        return error_response_with_reason(
            PAIRING_TOKEN_INVALID,
            detail="pairing token is invalid",
        )

    session_key = generate_session_key()
    device = register_device(
        name=device_name,
        platform=platform,
        public_key=normalized_public_key,
        session_key_hash=hash_session_key(session_key),
        bundle_id=bundle_id,
        app_version=app_version,
    )
    logger.info(
        "pairing confirmed device_id=%s platform=%s session_key=%s",
        device["id"],
        device["platform"],
        mask_session_key(session_key),
    )
    return jsonify(
        {
            "session_key": session_key,
            "device_id": device["id"],
            "journal_root": str(get_journal()),
            "owner_identity": get_owner_identity(),
            "server_version": _server_version(),
        }
    )


@pairing_bp.post("/heartbeat")
@require_paired_device
def heartbeat():
    _, error = _optional_json_object()
    if error is not None:
        return error
    if not touch_last_seen(g.paired_device["id"]):
        return error_response_with_reason(
            PAIRED_DEVICE_NOT_FOUND,
            detail="paired device not found",
        )
    return jsonify({"ok": True})


@pairing_bp.get("/devices")
def list_devices():
    _, error = _resolve_owner_or_paired_device()
    if error is not None:
        return error
    return jsonify({"devices": [status_view(device) for device in load_devices()]})


@pairing_bp.delete("/devices/<device_id>")
def unpair_device(device_id: str):
    _, error = _resolve_owner_or_paired_device()
    if error is not None:
        return error
    if not remove_device(device_id):
        return error_response_with_reason(
            PAIRED_DEVICE_NOT_FOUND,
            detail="paired device not found",
        )
    return jsonify({"unpaired": True})


@pairing_ui_bp.get("/")
def index():
    from solstone.convey import copy as convey_copy

    config = get_config()
    convey_config = config.get("convey", {})
    allow_network_access = convey_config.get("allow_network_access", False)
    pairing_host_override = config.get("pairing", {}).get("host_url")
    has_override = isinstance(pairing_host_override, str) and bool(
        pairing_host_override.strip()
    )
    pairing_network_off = not allow_network_access
    pairing_lan_detect_failed = (
        allow_network_access and not has_override and _detect_lan_ipv4() is None
    )
    return render_template(
        "pairing.html",
        convey_copy=convey_copy,
        pairing_lan_detect_failed=pairing_lan_detect_failed,
        pairing_network_off=pairing_network_off,
    )


__all__ = ["pairing_bp", "pairing_ui_bp"]
