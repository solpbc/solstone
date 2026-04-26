# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Shared pairing auth helpers."""

from __future__ import annotations

import logging
from functools import wraps
from typing import Any, Callable, TypeVar, cast

from flask import g, jsonify, request, session
from werkzeug.security import check_password_hash

from think.pairing.devices import Device, find_device_by_session_key_hash
from think.pairing.keys import hash_session_key, mask_session_key
from think.utils import get_config

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def extract_bearer_token() -> str | None:
    header = request.headers.get("Authorization")
    if not isinstance(header, str):
        return None
    scheme, _, value = header.partition(" ")
    if scheme.lower() != "bearer":
        return None
    token = value.strip()
    return token or None


def resolve_paired_device() -> Device | None:
    token = extract_bearer_token()
    if token is None:
        return None
    masked = mask_session_key(token)
    session_key_hash = hash_session_key(token)
    device = find_device_by_session_key_hash(session_key_hash)
    if device is None:
        logger.debug("paired device not found session_key=%s", masked)
        return None
    logger.debug("paired device resolved id=%s session_key=%s", device["id"], masked)
    return device


def _check_basic_auth() -> bool:
    auth = request.authorization
    if not auth or auth.type != "basic":
        return False
    password_hash = str(get_config().get("convey", {}).get("password_hash", "") or "")
    if not password_hash:
        return False
    return check_password_hash(password_hash, auth.password or "")


def _is_setup_complete() -> bool:
    return bool(get_config().get("setup", {}).get("completed_at"))


def is_owner_authed() -> bool:
    if session.get("logged_in"):
        return True
    if _check_basic_auth():
        return True
    if not _is_setup_complete():
        return False
    config = get_config()
    if not config.get("convey", {}).get("trust_localhost", True):
        return False
    remote_addr = request.remote_addr
    is_localhost = remote_addr in ("127.0.0.1", "::1", "localhost")
    proxy_headers = (
        request.headers.get("X-Forwarded-For")
        or request.headers.get("X-Real-IP")
        or request.headers.get("X-Forwarded-Host")
    )
    return bool(is_localhost and not proxy_headers)


def require_paired_device(func: F) -> F:
    @wraps(func)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        device = resolve_paired_device()
        if device is None:
            return (
                jsonify(
                    {
                        "error": "paired device required",
                        "reason": "auth_required",
                    }
                ),
                401,
            )
        g.paired_device = device
        return func(*args, **kwargs)

    return cast(F, wrapped)


__all__ = [
    "extract_bearer_token",
    "is_owner_authed",
    "require_paired_device",
    "resolve_paired_device",
]
