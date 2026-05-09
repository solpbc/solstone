# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Policy checks for sol-initiated chat."""

from __future__ import annotations

from datetime import datetime

from solstone.convey.sol_initiated.copy import (
    KIND_OWNER_CHAT_DISMISSED,
    KIND_SOL_CHAT_REQUEST,
    THROTTLE_CATEGORY_CAP,
    THROTTLE_CATEGORY_SELF_MUTE,
    THROTTLE_DAILY_CAP,
    THROTTLE_MUTE_WINDOW,
    THROTTLE_RATE_FLOOR,
)
from solstone.convey.sol_initiated.dedup import _is_live_for_dedup
from solstone.convey.sol_initiated.settings import SolVoiceSettings


def check_mute_window(
    settings: SolVoiceSettings,
    now_local_dt: datetime,
) -> str | None:
    """Throttle when the current owner-local hour is muted."""
    window = settings.mute_window
    if not window.enabled:
        return None

    hour = now_local_dt.hour
    start = window.start_hour_local
    end = window.end_hour_local
    if start == end:
        return THROTTLE_MUTE_WINDOW
    if start < end:
        if start <= hour < end:
            return THROTTLE_MUTE_WINDOW
        return None
    if hour >= start or hour < end:
        return THROTTLE_MUTE_WINDOW
    return None


def check_rate_floor(
    settings: SolVoiceSettings,
    recent_events: list[dict],
    now_ms: int,
) -> str | None:
    """Throttle when another sol-initiated request is too recent."""
    floor_ms = settings.rate_floor_minutes * 60_000
    if floor_ms <= 0:
        return None

    for event in reversed(recent_events):
        if event.get("kind") != KIND_SOL_CHAT_REQUEST:
            continue
        request_ts = int(event.get("ts", 0) or 0)
        if now_ms - request_ts < floor_ms:
            return THROTTLE_RATE_FLOOR
        return None
    return None


def check_category_self_mute(
    settings: SolVoiceSettings,
    events_today: list[dict],
    category: str,
    now_ms: int,
) -> str | None:
    """Throttle a category after a recent owner dismissal."""
    mute_ms = settings.category_self_mute_hours * 3_600_000
    if mute_ms <= 0:
        return None

    request_categories: dict[str, str] = {}
    for event in events_today:
        kind = event.get("kind")
        if kind == KIND_SOL_CHAT_REQUEST:
            request_id = str(event.get("request_id") or "")
            if request_id:
                request_categories[request_id] = str(event.get("category") or "")
            continue
        if kind != KIND_OWNER_CHAT_DISMISSED:
            continue
        dismissed_ts = int(event.get("ts", 0) or 0)
        if dismissed_ts <= settings.category_self_mute_clear_marker_ts:
            continue
        if now_ms - dismissed_ts > mute_ms:
            continue
        request_category = request_categories.get(str(event.get("request_id") or ""))
        if request_category == category:
            return THROTTLE_CATEGORY_SELF_MUTE
    return None


def check_category_cap(
    settings: SolVoiceSettings,
    events_today: list[dict],
    category: str,
) -> str | None:
    """Throttle when the category has consumed its daily cap."""
    cap = settings.category_caps[category]
    count = sum(
        1
        for event in events_today
        if event.get("kind") == KIND_SOL_CHAT_REQUEST
        and event.get("category") == category
    )
    if count >= cap:
        return THROTTLE_CATEGORY_CAP
    return None


def check_daily_cap(
    settings: SolVoiceSettings,
    events_today: list[dict],
) -> str | None:
    """Throttle when the daily cap has been consumed."""
    count = sum(
        1 for event in events_today if event.get("kind") == KIND_SOL_CHAT_REQUEST
    )
    if count >= settings.daily_cap:
        return THROTTLE_DAILY_CAP
    return None


def check_dedup(
    events_today: list[dict],
    dedupe_key: str,
    window_ms: int,
    now_ms: int,
) -> bool:
    """Return whether a live duplicate exists."""
    return _is_live_for_dedup(events_today, dedupe_key, window_ms, now_ms)
