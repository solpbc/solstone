# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

import json
from datetime import datetime, timezone
from pathlib import Path

import solstone.convey.chat_stream as chat_stream
import solstone.convey.sol_initiated.start as start_module
from solstone.convey.chat_stream import append_chat_event
from solstone.convey.sol_initiated.copy import (
    CATEGORIES,
    CATEGORY_CAP_DEFAULTS,
    KIND_OWNER_CHAT_DISMISSED,
    KIND_SOL_CHAT_REQUEST,
    THROTTLE_CATEGORY_CAP,
    THROTTLE_CATEGORY_SELF_MUTE,
    THROTTLE_DAILY_CAP,
    THROTTLE_MUTE_WINDOW,
    THROTTLE_RATE_FLOOR,
)
from solstone.convey.sol_initiated.policy import (
    check_category_cap,
    check_category_self_mute,
    check_daily_cap,
    check_mute_window,
    check_rate_floor,
)
from solstone.convey.sol_initiated.settings import (
    MuteWindowSettings,
    SolVoiceSettings,
    load_settings,
)


def _settings(**overrides) -> SolVoiceSettings:
    values = {
        "daily_cap": 5,
        "category_caps": dict(CATEGORY_CAP_DEFAULTS),
        "rate_floor_minutes": 20,
        "mute_window": MuteWindowSettings(False, 22, 7),
        "category_self_mute_hours": 24,
        "category_self_mute_clear_marker_ts": 0,
        "default_dedupe_window": "24h",
    }
    values.update(overrides)
    return SolVoiceSettings(**values)


def _request(category: str = CATEGORIES[0], *, ts: int = 1_000) -> dict:
    return {
        "kind": KIND_SOL_CHAT_REQUEST,
        "ts": ts,
        "request_id": f"r-{ts}",
        "category": category,
    }


def _write_config(journal: Path, payload: dict) -> None:
    config_dir = journal / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "journal.json").write_text(json.dumps(payload), encoding="utf-8")


def test_mute_window_crosses_midnight() -> None:
    settings = _settings(mute_window=MuteWindowSettings(True, 22, 7))

    assert check_mute_window(settings, datetime(2026, 5, 9, 23)) == THROTTLE_MUTE_WINDOW
    assert check_mute_window(settings, datetime(2026, 5, 9, 6)) == THROTTLE_MUTE_WINDOW
    assert check_mute_window(settings, datetime(2026, 5, 9, 12)) is None


def test_rate_floor_checks_most_recent_request() -> None:
    settings = _settings(rate_floor_minutes=10)

    assert (
        check_rate_floor(settings, [_request(ts=1_000)], 500_000) == THROTTLE_RATE_FLOOR
    )
    assert check_rate_floor(settings, [_request(ts=1_000)], 700_000) is None


def test_category_self_mute_uses_dismissal_category() -> None:
    category = CATEGORIES[1]
    settings = _settings(category_self_mute_hours=2)
    events = [
        _request(category, ts=1_000),
        {
            "kind": KIND_OWNER_CHAT_DISMISSED,
            "ts": 2_000,
            "request_id": "r-1000",
        },
    ]

    assert (
        check_category_self_mute(settings, events, category, 3_000)
        == THROTTLE_CATEGORY_SELF_MUTE
    )
    assert check_category_self_mute(settings, events, CATEGORIES[0], 3_000) is None


def test_category_and_daily_caps_count_requests() -> None:
    settings = _settings(
        daily_cap=2, category_caps={**CATEGORY_CAP_DEFAULTS, CATEGORIES[0]: 1}
    )
    events = [_request(CATEGORIES[0], ts=1_000), _request(CATEGORIES[1], ts=2_000)]

    assert check_category_cap(settings, events, CATEGORIES[0]) == THROTTLE_CATEGORY_CAP
    assert check_daily_cap(settings, events) == THROTTLE_DAILY_CAP


def test_start_chat_daily_cap_counts_current_utc_day_across_stream_days(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("SOLSTONE_JOURNAL", str(tmp_path))
    _write_config(
        tmp_path,
        {
            "sol_voice": {
                "daily_cap": 1,
                "category_caps": {key: 10 for key in CATEGORY_CAP_DEFAULTS},
                "rate_floor_minutes": 0,
                "mute_window": {
                    "enabled": False,
                    "start_hour_local": 22,
                    "end_hour_local": 7,
                },
                "category_self_mute_hours": 0,
                "category_self_mute_clear_marker_ts": 0,
                "default_dedupe_window": "24h",
            }
        },
    )
    previous_stream_day_ms = int(
        datetime(2026, 5, 9, 1, tzinfo=timezone.utc).timestamp() * 1000
    )
    current_ms = int(datetime(2026, 5, 9, 7, tzinfo=timezone.utc).timestamp() * 1000)

    def stream_day(ts_ms: int) -> str:
        if ts_ms < current_ms:
            return "20260508"
        return "20260509"

    monkeypatch.setattr(chat_stream, "_day_for_ts", stream_day)
    monkeypatch.setattr(start_module, "now_ms", lambda: current_ms)
    append_chat_event(
        KIND_SOL_CHAT_REQUEST,
        ts=previous_stream_day_ms,
        request_id="existing",
        summary="existing",
        message=None,
        category=CATEGORIES[0],
        dedupe="existing",
        dedupe_window="24h",
        since_ts=1,
        trigger_talent="reflection",
    )

    result = start_module.start_chat(
        summary="blocked",
        message=None,
        category=CATEGORIES[0],
        dedupe="new-key",
        dedupe_window=None,
        since_ts=1,
        trigger_talent="reflection",
    )

    assert result.throttled == THROTTLE_DAILY_CAP


def test_load_settings_defaults_and_warns(monkeypatch, tmp_path, caplog) -> None:
    monkeypatch.setenv("SOLSTONE_JOURNAL", str(tmp_path))
    _write_config(
        tmp_path,
        {
            "sol_voice": {
                "daily_cap": "bad",
                "category_caps": {CATEGORIES[0]: "bad"},
                "rate_floor_minutes": 0,
                "mute_window": {"enabled": True, "start_hour_local": 24},
                "default_dedupe_window": "bad",
            }
        },
    )

    settings = load_settings()

    assert settings.daily_cap == 5
    assert settings.category_caps[CATEGORIES[0]] == CATEGORY_CAP_DEFAULTS[CATEGORIES[0]]
    assert settings.rate_floor_minutes == 0
    assert settings.mute_window.enabled is True
    assert settings.mute_window.start_hour_local == 22
    assert settings.default_dedupe_window == "24h"
    assert "key=daily_cap value='bad'" in caplog.text
    assert "key=mute_window.start_hour_local value=24" in caplog.text
