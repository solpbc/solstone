# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Push config readers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from solstone.think.utils import get_config

DEFAULT_ENVIRONMENT = "development"
_VALID_ENVIRONMENTS = {"development", "production"}


def _push_config() -> dict[str, Any]:
    config = get_config()
    push = config.get("push")
    return push if isinstance(push, dict) else {}


def _clean_str(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def get_apns_key_path() -> Path | None:
    configured = _clean_str(_push_config().get("apns_key_path"))
    return Path(configured) if configured else None


def get_apns_key_id() -> str | None:
    return _clean_str(_push_config().get("apns_key_id"))


def get_apns_team_id() -> str | None:
    return _clean_str(_push_config().get("apns_team_id"))


def get_bundle_id() -> str | None:
    return _clean_str(_push_config().get("bundle_id"))


def get_environment() -> str:
    configured = _clean_str(_push_config().get("environment"))
    if configured is None:
        return DEFAULT_ENVIRONMENT
    if configured not in _VALID_ENVIRONMENTS:
        raise ValueError(
            "push.environment must be 'development' or 'production' when set"
        )
    return configured


def _has_valid_key_path() -> bool:
    key_path = get_apns_key_path()
    if key_path is None or not key_path.is_absolute() or not key_path.is_file():
        return False
    try:
        key_path.read_text(encoding="utf-8")
    except OSError:
        return False
    return True


def is_configured() -> bool:
    if not (
        get_apns_key_path()
        and get_apns_key_id()
        and get_apns_team_id()
        and get_bundle_id()
    ):
        return False
    try:
        get_environment()
    except ValueError:
        return False
    return _has_valid_key_path()


__all__ = [
    "DEFAULT_ENVIRONMENT",
    "get_apns_key_id",
    "get_apns_key_path",
    "get_apns_team_id",
    "get_bundle_id",
    "get_environment",
    "is_configured",
]
