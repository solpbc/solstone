# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Start sol-initiated chat requests."""

from __future__ import annotations

import re
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone

import solstone.convey.chat_stream as chat_stream
from solstone.convey.sol_initiated.copy import (
    CATEGORIES,
    KIND_SOL_CHAT_REQUEST,
    KIND_SOL_CHAT_REQUEST_SUPERSEDED,
)
from solstone.convey.sol_initiated.dedup import (
    _is_unresolved_for_supersede,
    parse_dedupe_window,
)
from solstone.convey.sol_initiated.nudge_log import record_nudge_log
from solstone.convey.sol_initiated.policy import (
    check_category_cap,
    check_category_self_mute,
    check_daily_cap,
    check_dedup,
    check_mute_window,
    check_rate_floor,
)
from solstone.convey.sol_initiated.settings import load_settings
from solstone.think.utils import get_owner_timezone, now_ms

_TRIGGER_TALENT_RE = re.compile(r"^[a-z0-9][a-z0-9_.:-]*$")


@dataclass(frozen=True)
class StartChatResult:
    written: bool
    deduped: bool
    throttled: str | None
    request_id: str | None


def start_chat(
    *,
    summary: str,
    message: str | None,
    category: str,
    dedupe: str,
    dedupe_window: str | None,
    since_ts: int,
    trigger_talent: str,
) -> StartChatResult:
    """Start a sol-initiated chat request if policy allows it."""
    settings = load_settings()
    clean_summary = _validate_summary(summary)
    clean_message = _validate_message(message)
    clean_category = _validate_category(category)
    clean_dedupe = _validate_dedupe(dedupe)
    clean_trigger_talent = _validate_trigger_talent(trigger_talent)
    current_ms = now_ms()
    clean_since_ts = _validate_since_ts(since_ts, current_ms)
    window_spec = str(dedupe_window or settings.default_dedupe_window)
    window_ms = parse_dedupe_window(window_spec)

    stored_events: list[dict] = []
    result: StartChatResult
    with chat_stream._CHAT_LOCK:
        day = chat_stream._day_for_ts(current_ms)
        events_today = chat_stream.read_chat_events(day)
        utc_day_events = _read_utc_day_events(current_ms)
        throttle = (
            check_mute_window(settings, datetime.now(get_owner_timezone()))
            or check_rate_floor(settings, events_today, current_ms)
            or check_category_self_mute(
                settings,
                events_today,
                clean_category,
                current_ms,
            )
            or check_category_cap(settings, utc_day_events, clean_category)
            or check_daily_cap(settings, utc_day_events)
        )
        if throttle is not None:
            record_nudge_log(
                KIND_SOL_CHAT_REQUEST,
                clean_dedupe,
                clean_category,
                f"throttled:{throttle}",
            )
            return StartChatResult(False, False, throttle, None)

        if check_dedup(events_today, clean_dedupe, window_ms, current_ms):
            record_nudge_log(
                KIND_SOL_CHAT_REQUEST,
                clean_dedupe,
                clean_category,
                "deduped",
            )
            return StartChatResult(False, True, None, None)

        request_id = secrets.token_hex(16)
        event_pairs: list[tuple[str, dict]] = []
        superseded_request_id = _is_unresolved_for_supersede(events_today)
        if superseded_request_id is not None:
            event_pairs.append(
                (
                    KIND_SOL_CHAT_REQUEST_SUPERSEDED,
                    {
                        "ts": current_ms,
                        "request_id": superseded_request_id,
                        "replaced_by": request_id,
                    },
                )
            )
        event_pairs.append(
            (
                KIND_SOL_CHAT_REQUEST,
                {
                    "ts": current_ms,
                    "request_id": request_id,
                    "summary": clean_summary,
                    "message": clean_message,
                    "category": clean_category,
                    "dedupe": clean_dedupe,
                    "dedupe_window": window_spec,
                    "since_ts": clean_since_ts,
                    "trigger_talent": clean_trigger_talent,
                },
            )
        )
        stored_events = chat_stream.append_chat_events_locked(
            event_pairs,
            _lock_already_held=True,
        )
        record_nudge_log(
            KIND_SOL_CHAT_REQUEST,
            clean_dedupe,
            clean_category,
            "written",
        )
        result = StartChatResult(True, False, None, request_id)

    chat_stream._finalize_chat_event_appends(stored_events)
    return result


def _read_utc_day_events(current_ms: int) -> list[dict]:
    current_utc = datetime.fromtimestamp(current_ms / 1000, timezone.utc)
    start_dt = current_utc.replace(hour=0, minute=0, second=0, microsecond=0)
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = start_ms + 86_400_000
    candidate_days = {
        chat_stream._day_for_ts(start_ms),
        chat_stream._day_for_ts(end_ms - 1),
    }

    events: list[dict] = []
    for day in sorted(candidate_days):
        events.extend(chat_stream.read_chat_events(day))
    return [
        event for event in events if start_ms <= int(event.get("ts", 0) or 0) < end_ms
    ]


def _validate_summary(summary: str) -> str:
    value = str(summary or "").strip()
    if not value:
        raise ValueError("summary is required")
    if len(value) > 80:
        raise ValueError("summary must be 80 characters or fewer")
    return value


def _validate_message(message: str | None) -> str | None:
    if message is None:
        return None
    value = str(message).strip()
    if len(value) > 500:
        raise ValueError("message must be 500 characters or fewer")
    return value or None


def _validate_category(category: str) -> str:
    value = str(category or "").strip()
    if value not in CATEGORIES:
        raise ValueError(f"category must be one of: {', '.join(CATEGORIES)}")
    return value


def _validate_dedupe(dedupe: str) -> str:
    value = str(dedupe or "").strip()
    if not value:
        raise ValueError("dedupe is required")
    return value


def _validate_since_ts(since_ts: int, current_ms: int) -> int:
    if not isinstance(since_ts, int) or isinstance(since_ts, bool):
        raise ValueError("since_ts must be an int")
    if since_ts <= 0:
        raise ValueError("since_ts must be positive")
    if since_ts > current_ms:
        raise ValueError("since_ts must not be in the future")
    return since_ts


def _validate_trigger_talent(trigger_talent: str) -> str:
    value = str(trigger_talent or "").strip()
    if not value:
        raise ValueError("trigger_talent is required")
    if _TRIGGER_TALENT_RE.fullmatch(value) is None:
        raise ValueError("trigger_talent must be a slug name")
    return value
