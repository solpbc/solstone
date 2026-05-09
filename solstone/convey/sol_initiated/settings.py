# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Settings loader for sol-initiated chat policy."""

from __future__ import annotations

import copy
import fcntl
import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from solstone.convey.sol_initiated.copy import CATEGORY_CAP_DEFAULTS
from solstone.convey.sol_initiated.dedup import parse_dedupe_window
from solstone.think.utils import get_config, get_journal

logger = logging.getLogger(__name__)
_MISSING = object()


@dataclass(frozen=True)
class MuteWindowSettings:
    enabled: bool
    start_hour_local: int
    end_hour_local: int


@dataclass(frozen=True)
class SolVoiceSettings:
    daily_cap: int
    category_caps: dict[str, int]
    rate_floor_minutes: int
    mute_window: MuteWindowSettings
    category_self_mute_hours: int
    default_dedupe_window: str
    category_self_mute_clear_markers: dict[str, int] = field(default_factory=dict)
    system_notifications_macos: bool = False
    system_notifications_linux: bool = False
    debug_show_throttled: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return the on-disk ``sol_voice`` shape."""
        return {
            "daily_cap": self.daily_cap,
            "category_caps": dict(self.category_caps),
            "rate_floor_minutes": self.rate_floor_minutes,
            "mute_window": {
                "enabled": self.mute_window.enabled,
                "start_hour_local": self.mute_window.start_hour_local,
                "end_hour_local": self.mute_window.end_hour_local,
            },
            "category_self_mute_hours": self.category_self_mute_hours,
            "category_self_mute_clear_markers": dict(
                self.category_self_mute_clear_markers
            ),
            "default_dedupe_window": self.default_dedupe_window,
            "system_notifications": {
                "macos": self.system_notifications_macos,
                "linux": self.system_notifications_linux,
            },
            "debug_show_throttled": self.debug_show_throttled,
        }


DEFAULT_SETTINGS = SolVoiceSettings(
    daily_cap=5,
    category_caps=dict(CATEGORY_CAP_DEFAULTS),
    rate_floor_minutes=20,
    mute_window=MuteWindowSettings(
        enabled=False,
        start_hour_local=22,
        end_hour_local=7,
    ),
    category_self_mute_hours=24,
    category_self_mute_clear_markers={},
    default_dedupe_window="24h",
    system_notifications_macos=False,
    system_notifications_linux=False,
    debug_show_throttled=False,
)


def load_settings() -> SolVoiceSettings:
    """Load sol-initiated chat settings, falling back field-by-field."""
    return _parse_settings(get_config().get("sol_voice"), strict=False)


def save_settings(updates: dict[str, Any]) -> SolVoiceSettings:
    """Deep-merge and persist sol-initiated chat settings."""
    if not isinstance(updates, dict):
        raise ValueError("sol_voice update must be an object")

    config = get_config()
    raw_root = config.get("sol_voice")
    if not isinstance(raw_root, dict):
        raw_root = {}

    merged = _deep_merge(raw_root, updates)
    settings = _parse_settings(merged, strict=True)
    config["sol_voice"] = settings.to_dict()
    _write_config_atomic(config)
    return settings


def _parse_settings(raw_root: object, *, strict: bool) -> SolVoiceSettings:
    if not isinstance(raw_root, dict):
        _reject("sol_voice", raw_root, strict)
        raw_root = {}

    raw_mute = raw_root.get("mute_window", _MISSING)
    if raw_mute is _MISSING:
        raw_mute = {}
    elif not isinstance(raw_mute, dict):
        _reject("mute_window", raw_mute, strict)
        raw_mute = {}

    raw_notifications = raw_root.get("system_notifications", _MISSING)
    if raw_notifications is _MISSING:
        raw_notifications = {}
    elif not isinstance(raw_notifications, dict):
        _reject("system_notifications", raw_notifications, strict)
        raw_notifications = {}

    if strict:
        _validate_known_keys(
            "sol_voice",
            raw_root,
            {
                "daily_cap",
                "category_caps",
                "rate_floor_minutes",
                "mute_window",
                "category_self_mute_hours",
                "category_self_mute_clear_markers",
                "default_dedupe_window",
                "system_notifications",
                "debug_show_throttled",
            },
        )
        _validate_known_keys(
            "mute_window",
            raw_mute,
            {"enabled", "start_hour_local", "end_hour_local"},
        )
        _validate_known_keys(
            "system_notifications", raw_notifications, {"macos", "linux"}
        )

    return SolVoiceSettings(
        daily_cap=_nonnegative_int(
            "daily_cap",
            raw_root.get("daily_cap", _MISSING),
            DEFAULT_SETTINGS.daily_cap,
            strict,
        ),
        category_caps=_category_caps(raw_root.get("category_caps", _MISSING), strict),
        rate_floor_minutes=_nonnegative_int(
            "rate_floor_minutes",
            raw_root.get("rate_floor_minutes", _MISSING),
            DEFAULT_SETTINGS.rate_floor_minutes,
            strict,
        ),
        mute_window=MuteWindowSettings(
            enabled=_bool(
                "mute_window.enabled",
                raw_mute.get("enabled", _MISSING),
                DEFAULT_SETTINGS.mute_window.enabled,
                strict,
            ),
            start_hour_local=_hour(
                "mute_window.start_hour_local",
                raw_mute.get("start_hour_local", _MISSING),
                DEFAULT_SETTINGS.mute_window.start_hour_local,
                strict,
            ),
            end_hour_local=_hour(
                "mute_window.end_hour_local",
                raw_mute.get("end_hour_local", _MISSING),
                DEFAULT_SETTINGS.mute_window.end_hour_local,
                strict,
            ),
        ),
        category_self_mute_hours=_nonnegative_int(
            "category_self_mute_hours",
            raw_root.get("category_self_mute_hours", _MISSING),
            DEFAULT_SETTINGS.category_self_mute_hours,
            strict,
        ),
        category_self_mute_clear_markers=_category_self_mute_clear_markers(
            raw_root.get("category_self_mute_clear_markers", _MISSING),
            strict,
        ),
        default_dedupe_window=_dedupe_window(
            raw_root.get("default_dedupe_window", _MISSING),
            DEFAULT_SETTINGS.default_dedupe_window,
            strict,
        ),
        system_notifications_macos=_bool(
            "system_notifications.macos",
            raw_notifications.get("macos", _MISSING),
            DEFAULT_SETTINGS.system_notifications_macos,
            strict,
        ),
        system_notifications_linux=_bool(
            "system_notifications.linux",
            raw_notifications.get("linux", _MISSING),
            DEFAULT_SETTINGS.system_notifications_linux,
            strict,
        ),
        debug_show_throttled=_bool(
            "debug_show_throttled",
            raw_root.get("debug_show_throttled", _MISSING),
            DEFAULT_SETTINGS.debug_show_throttled,
            strict,
        ),
    )


def _warn_invalid(key: str, raw_value: object) -> None:
    logger.warning(
        "sol_voice settings invalid key=%s value=%r; falling back to default",
        key,
        raw_value,
    )


def _reject(key: str, raw_value: object, strict: bool) -> None:
    if raw_value is _MISSING:
        return
    if strict:
        raise ValueError(f"{key} has invalid value: {raw_value!r}")
    _warn_invalid(key, raw_value)


def _validate_known_keys(
    key: str, raw_value: dict[str, Any], allowed: set[str]
) -> None:
    for candidate in raw_value:
        if candidate not in allowed:
            raise ValueError(f"{key}.{candidate} is not a recognized setting")


def _nonnegative_int(key: str, raw_value: object, default: int, strict: bool) -> int:
    if raw_value is _MISSING:
        return default
    if (
        isinstance(raw_value, int)
        and not isinstance(raw_value, bool)
        and raw_value >= 0
    ):
        return raw_value
    _reject(key, raw_value, strict)
    return default


def _bool(key: str, raw_value: object, default: bool, strict: bool) -> bool:
    if raw_value is _MISSING:
        return default
    if isinstance(raw_value, bool):
        return raw_value
    _reject(key, raw_value, strict)
    return default


def _hour(key: str, raw_value: object, default: int, strict: bool) -> int:
    if raw_value is _MISSING:
        return default
    if (
        isinstance(raw_value, int)
        and not isinstance(raw_value, bool)
        and 0 <= raw_value <= 23
    ):
        return raw_value
    _reject(key, raw_value, strict)
    return default


def _category_caps(raw_value: object, strict: bool = False) -> dict[str, int]:
    if raw_value is _MISSING:
        raw_value = {}
    if not isinstance(raw_value, dict):
        _reject("category_caps", raw_value, strict)
        raw_value = {}

    caps: dict[str, int] = {}
    for category, default in CATEGORY_CAP_DEFAULTS.items():
        caps[category] = _nonnegative_int(
            f"category_caps.{category}",
            raw_value.get(category, _MISSING),
            default,
            strict,
        )
    for category, value in raw_value.items():
        if category in caps:
            continue
        if not isinstance(category, str):
            _reject(f"category_caps.{category!r}", category, strict)
            continue
        caps[category] = _nonnegative_int(
            f"category_caps.{category}",
            value,
            0,
            strict,
        )
    return caps


def _category_self_mute_clear_markers(
    raw_value: object,
    strict: bool,
) -> dict[str, int]:
    if raw_value is _MISSING:
        return {}
    if not isinstance(raw_value, dict):
        _reject("category_self_mute_clear_markers", raw_value, strict)
        return {}

    markers: dict[str, int] = {}
    for category, marker in raw_value.items():
        if not isinstance(category, str):
            _reject(f"category_self_mute_clear_markers.{category!r}", category, strict)
            continue
        if isinstance(marker, int) and not isinstance(marker, bool) and marker >= 0:
            markers[category] = marker
            continue
        _reject(f"category_self_mute_clear_markers.{category}", marker, strict)
    return markers


def _dedupe_window(raw_value: object, default: str, strict: bool) -> str:
    if raw_value is _MISSING:
        return default
    if isinstance(raw_value, str):
        try:
            parse_dedupe_window(raw_value)
        except ValueError:
            pass
        else:
            return raw_value
    _reject("default_dedupe_window", raw_value, strict)
    return default


def _deep_merge(base: object, updates: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(base, dict):
        base = {}
    merged = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
            continue
        merged[key] = copy.deepcopy(value)
    return merged


def _write_config_atomic(config: dict[str, Any]) -> None:
    config_path = Path(get_journal()) / "config" / "journal.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        temp_path: Path | None = None
        try:
            fd, raw_temp_path = tempfile.mkstemp(
                dir=config_path.parent,
                prefix=".journal.",
                suffix=".tmp",
                text=True,
            )
            temp_path = Path(raw_temp_path)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(config, handle, indent=2, ensure_ascii=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, config_path)
            os.chmod(config_path, 0o600)
            _fsync_dir(config_path.parent)
        finally:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink()
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _fsync_dir(path: Path) -> None:
    dir_fd = os.open(path, os.O_DIRECTORY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)
