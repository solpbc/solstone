# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Settings loader for sol-initiated chat policy."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from solstone.convey.sol_initiated.copy import CATEGORY_CAP_DEFAULTS
from solstone.convey.sol_initiated.dedup import parse_dedupe_window
from solstone.think.utils import get_config

logger = logging.getLogger(__name__)


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
    category_self_mute_clear_marker_ts: int
    default_dedupe_window: str


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
    category_self_mute_clear_marker_ts=0,
    default_dedupe_window="24h",
)


def load_settings() -> SolVoiceSettings:
    """Load sol-initiated chat settings, falling back field-by-field."""
    raw_root = get_config().get("sol_voice")
    if not isinstance(raw_root, dict):
        _warn_invalid("sol_voice", raw_root)
        raw_root = {}

    raw_mute = raw_root.get("mute_window")
    if not isinstance(raw_mute, dict):
        _warn_invalid("mute_window", raw_mute)
        raw_mute = {}

    return SolVoiceSettings(
        daily_cap=_nonnegative_int(
            "daily_cap",
            raw_root.get("daily_cap"),
            DEFAULT_SETTINGS.daily_cap,
        ),
        category_caps=_category_caps(raw_root.get("category_caps")),
        rate_floor_minutes=_nonnegative_int(
            "rate_floor_minutes",
            raw_root.get("rate_floor_minutes"),
            DEFAULT_SETTINGS.rate_floor_minutes,
        ),
        mute_window=MuteWindowSettings(
            enabled=_bool(
                "mute_window.enabled",
                raw_mute.get("enabled"),
                DEFAULT_SETTINGS.mute_window.enabled,
            ),
            start_hour_local=_hour(
                "mute_window.start_hour_local",
                raw_mute.get("start_hour_local"),
                DEFAULT_SETTINGS.mute_window.start_hour_local,
            ),
            end_hour_local=_hour(
                "mute_window.end_hour_local",
                raw_mute.get("end_hour_local"),
                DEFAULT_SETTINGS.mute_window.end_hour_local,
            ),
        ),
        category_self_mute_hours=_nonnegative_int(
            "category_self_mute_hours",
            raw_root.get("category_self_mute_hours"),
            DEFAULT_SETTINGS.category_self_mute_hours,
        ),
        category_self_mute_clear_marker_ts=_nonnegative_int(
            "category_self_mute_clear_marker_ts",
            raw_root.get("category_self_mute_clear_marker_ts"),
            DEFAULT_SETTINGS.category_self_mute_clear_marker_ts,
        ),
        default_dedupe_window=_dedupe_window(
            raw_root.get("default_dedupe_window"),
            DEFAULT_SETTINGS.default_dedupe_window,
        ),
    )


def _warn_invalid(key: str, raw_value: object) -> None:
    logger.warning(
        "sol_voice settings invalid key=%s value=%r; falling back to default",
        key,
        raw_value,
    )


def _nonnegative_int(key: str, raw_value: object, default: int) -> int:
    if (
        isinstance(raw_value, int)
        and not isinstance(raw_value, bool)
        and raw_value >= 0
    ):
        return raw_value
    _warn_invalid(key, raw_value)
    return default


def _bool(key: str, raw_value: object, default: bool) -> bool:
    if isinstance(raw_value, bool):
        return raw_value
    _warn_invalid(key, raw_value)
    return default


def _hour(key: str, raw_value: object, default: int) -> int:
    if (
        isinstance(raw_value, int)
        and not isinstance(raw_value, bool)
        and 0 <= raw_value <= 23
    ):
        return raw_value
    _warn_invalid(key, raw_value)
    return default


def _category_caps(raw_value: object) -> dict[str, int]:
    if not isinstance(raw_value, dict):
        _warn_invalid("category_caps", raw_value)
        raw_value = {}

    caps: dict[str, int] = {}
    for category, default in CATEGORY_CAP_DEFAULTS.items():
        caps[category] = _nonnegative_int(
            f"category_caps.{category}",
            raw_value.get(category),
            default,
        )
    return caps


def _dedupe_window(raw_value: object, default: str) -> str:
    if isinstance(raw_value, str):
        try:
            parse_dedupe_window(raw_value)
        except ValueError:
            pass
        else:
            return raw_value
    _warn_invalid("default_dedupe_window", raw_value)
    return default
