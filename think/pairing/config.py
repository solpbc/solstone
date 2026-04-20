# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Pairing config readers."""

from __future__ import annotations

from typing import Any

from think.service import DEFAULT_SERVICE_PORT
from think.utils import get_config, read_service_port

DEFAULT_TOKEN_TTL_SECONDS = 600
MIN_TOKEN_TTL_SECONDS = 60
MAX_TOKEN_TTL_SECONDS = 3600


def _pairing_config() -> dict[str, Any]:
    config = get_config()
    pairing = config.get("pairing")
    return pairing if isinstance(pairing, dict) else {}


def _clean_str(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def get_host_url() -> str:
    configured = _clean_str(_pairing_config().get("host_url"))
    if configured is not None:
        return configured
    convey_port = read_service_port("convey") or DEFAULT_SERVICE_PORT
    return f"http://localhost:{convey_port}"


def get_token_ttl_seconds() -> int:
    configured = _pairing_config().get("token_ttl_seconds")
    try:
        ttl_seconds = int(configured)
    except (TypeError, ValueError):
        ttl_seconds = DEFAULT_TOKEN_TTL_SECONDS
    return max(MIN_TOKEN_TTL_SECONDS, min(MAX_TOKEN_TTL_SECONDS, ttl_seconds))


def get_owner_identity() -> str:
    config = get_config()
    identity = config.get("identity")
    if not isinstance(identity, dict):
        return ""
    preferred = _clean_str(identity.get("preferred"))
    if preferred is not None:
        return preferred
    name = _clean_str(identity.get("name"))
    return name or ""


__all__ = [
    "DEFAULT_TOKEN_TTL_SECONDS",
    "MAX_TOKEN_TTL_SECONDS",
    "MIN_TOKEN_TTL_SECONDS",
    "get_host_url",
    "get_owner_identity",
    "get_token_ttl_seconds",
]
