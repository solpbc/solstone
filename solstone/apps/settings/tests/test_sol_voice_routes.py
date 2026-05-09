# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

import json

from solstone.apps.settings import routes as settings_routes
from solstone.convey import create_app
from solstone.convey.chat_stream import append_chat_event
from solstone.convey.sol_initiated import start as start_module
from solstone.convey.sol_initiated.copy import (
    CATEGORY_CAP_DEFAULTS,
    KIND_OWNER_CHAT_DISMISSED,
    KIND_SOL_CHAT_REQUEST,
)
from solstone.convey.sol_initiated.settings import load_settings


def _base_config() -> dict:
    return {
        "setup": {"completed_at": "2026-05-09T00:00:00Z"},
        "convey": {"trust_localhost": True},
    }


def _sol_voice_config() -> dict:
    return {
        "daily_cap": 10,
        "category_caps": {key: 10 for key in CATEGORY_CAP_DEFAULTS},
        "rate_floor_minutes": 0,
        "mute_window": {
            "enabled": False,
            "start_hour_local": 22,
            "end_hour_local": 7,
        },
        "category_self_mute_hours": 2,
        "category_self_mute_clear_markers": {},
        "default_dedupe_window": "24h",
    }


def _client(journal_path):
    app = create_app(str(journal_path))
    app.config["TESTING"] = True
    return app.test_client()


def test_sol_voice_api_get_returns_defaults_when_unconfigured(settings_env):
    journal_path, _config = settings_env(_base_config())
    client = _client(journal_path)

    response = client.get("/app/settings/api/sol_voice")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["daily_cap"] == 5
    assert payload["category_self_mute_clear_markers"] == {}
    assert payload["system_notifications"] == {"linux": False, "macos": False}
    assert payload["debug_show_throttled"] is False


def test_sol_voice_api_put_round_trips(settings_env):
    journal_path, _config = settings_env(_base_config())
    client = _client(journal_path)

    response = client.put("/app/settings/api/sol_voice", json={"daily_cap": 7})

    assert response.status_code == 200
    assert response.get_json()["daily_cap"] == 7
    assert client.get("/app/settings/api/sol_voice").get_json()["daily_cap"] == 7


def test_sol_voice_api_put_rejects_invalid_shape(settings_env):
    journal_path, _config = settings_env(_base_config())
    client = _client(journal_path)

    non_object = client.put("/app/settings/api/sol_voice", json=["bad"])
    wrong_type = client.put("/app/settings/api/sol_voice", json={"daily_cap": "bad"})

    assert non_object.status_code == 400
    assert wrong_type.status_code == 400


def test_sol_voice_api_put_per_category_clear_marker(settings_env):
    journal_path, _config = settings_env(_base_config())
    client = _client(journal_path)
    clear_ts = 1_700_000_000_000

    response = client.put(
        "/app/settings/api/sol_voice",
        json={"category_self_mute_clear_markers": {"notice": clear_ts}},
    )

    assert response.status_code == 200
    assert response.get_json()["category_self_mute_clear_markers"]["notice"] == clear_ts

    response = client.put(
        "/app/settings/api/sol_voice",
        json={"category_self_mute_clear_markers": {"notice": 0}},
    )

    assert response.status_code == 200
    assert response.get_json()["category_self_mute_clear_markers"]["notice"] == 0


def test_sol_voice_api_per_category_clear_marker_unmutes(settings_env, monkeypatch):
    fixed_now = 1_778_263_800_000
    request_ts = fixed_now - 30 * 60_000
    dismissed_ts = fixed_now - 10 * 60_000
    config = _base_config()
    config["sol_voice"] = _sol_voice_config()
    journal_path, _config = settings_env(config)
    monkeypatch.setattr(settings_routes, "now_ms", lambda: fixed_now)
    client = _client(journal_path)

    append_chat_event(
        KIND_SOL_CHAT_REQUEST,
        ts=request_ts,
        request_id="old-request",
        summary="old request",
        message="",
        category="notice",
        dedupe="old-key",
        dedupe_window="24h",
        since_ts=request_ts - 1,
        trigger_talent="reflection",
    )
    append_chat_event(
        KIND_OWNER_CHAT_DISMISSED,
        ts=dismissed_ts,
        request_id="old-request",
        surface="settings",
        reason="dismissed",
    )

    before = client.get("/app/settings/api/sol_voice").get_json()
    state = before["category_mute_state"]["notice"]
    assert state == {
        "muted": True,
        "expires_ts": dismissed_ts + 2 * 3_600_000,
    }
    assert state["expires_ts"] > fixed_now

    response = client.put(
        "/app/settings/api/sol_voice",
        json={"category_self_mute_clear_markers": {"notice": fixed_now}},
    )
    assert response.status_code == 200
    after = client.get("/app/settings/api/sol_voice").get_json()
    assert after["category_mute_state"]["notice"] == {
        "muted": False,
        "expires_ts": None,
    }

    monkeypatch.setattr(start_module, "now_ms", lambda: fixed_now + 1_000)
    result = start_module.start_chat(
        summary="new request",
        message="",
        category="notice",
        dedupe="new-key",
        dedupe_window="24h",
        since_ts=fixed_now - 1,
        trigger_talent="reflection",
    )

    assert result.written is True
    assert result.throttled is None


def test_sol_voice_throttled_endpoint_filters_correctly(settings_env):
    journal_path, _config = settings_env(_base_config())
    log_path = journal_path / "push" / "nudge_log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "ts": 1,
            "kind": KIND_SOL_CHAT_REQUEST,
            "category": "notice",
            "dedupe_key": "old",
            "outcome": "deduped",
        },
        {
            "ts": 2,
            "kind": KIND_SOL_CHAT_REQUEST,
            "category": "notice",
            "dedupe_key": "written",
            "outcome": "written",
        },
        {
            "ts": 3,
            "kind": "other",
            "category": "notice",
            "dedupe_key": "other",
            "outcome": "muted",
        },
        {
            "ts": 4,
            "kind": KIND_SOL_CHAT_REQUEST,
            "category": "commitment",
            "dedupe_key": "new",
            "outcome": "rate_floor",
        },
        {
            "ts": 5,
            "category": "notice",
            "dedupe_key": "missing-kind",
            "outcome": "muted",
        },
    ]
    log_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    client = _client(journal_path)

    response = client.get("/app/settings/api/sol_voice/throttled?limit=1")

    assert response.status_code == 200
    assert response.get_json() == [
        {
            "ts": 4,
            "category": "commitment",
            "dedupe_key": "new",
            "outcome": "rate_floor",
        }
    ]


def test_sol_voice_throttled_endpoint_handles_missing_file(settings_env):
    journal_path, _config = settings_env(_base_config())
    client = _client(journal_path)

    response = client.get("/app/settings/api/sol_voice/throttled")

    assert response.status_code == 200
    assert response.get_json() == []


def test_sol_voice_settings_persist_through_load_settings(settings_env):
    journal_path, _config = settings_env(_base_config())
    client = _client(journal_path)

    response = client.put(
        "/app/settings/api/sol_voice",
        json={
            "daily_cap": 8,
            "system_notifications": {"macos": True, "linux": True},
            "debug_show_throttled": True,
        },
    )

    assert response.status_code == 200
    settings = load_settings()
    assert settings.daily_cap == 8
    assert settings.system_notifications_macos is True
    assert settings.system_notifications_linux is True
    assert settings.debug_show_throttled is True
