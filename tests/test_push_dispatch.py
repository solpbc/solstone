# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import patch

import httpx
import jwt

from think.push import dispatch

TEST_KEY = """-----BEGIN PRIVATE KEY-----
MIGHAgEAMBMGByqGSM49AgEGCCqGSM49AwEHBG0wawIBAQQg+Zj7Bk6Dzp080/PU
jTZnJ6kP4KtlHErFO/WuVRTQvkShRANCAARW8djY5HF7K8noSZQRfjP38mIzaufi
/YPI38YuaWmiPIqRmwDOu5rICl4PPLem4k+qtb950rlYCGx3J+MQN9tO
-----END PRIVATE KEY-----
"""


def _write_key(tmp_path: Path) -> Path:
    key_path = tmp_path / "keys" / "apns.p8"
    key_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.write_text(TEST_KEY, encoding="utf-8")
    return key_path


def _configure_push(monkeypatch, tmp_path: Path) -> None:
    key_path = _write_key(tmp_path)
    monkeypatch.setattr(dispatch, "get_apns_key_path", lambda: key_path)
    monkeypatch.setattr(dispatch, "get_apns_key_id", lambda: "KEY123")
    monkeypatch.setattr(dispatch, "get_apns_team_id", lambda: "TEAM123")
    monkeypatch.setattr(dispatch, "get_bundle_id", lambda: "org.solpbc.solstone-swift")
    monkeypatch.setattr(dispatch, "get_environment", lambda: "development")
    dispatch._APNS_JWT_CACHE.clear()


def test_mint_apns_jwt_has_expected_header_and_claims(monkeypatch, tmp_path):
    _configure_push(monkeypatch, tmp_path)

    token = dispatch._mint_apns_jwt(now=1713528000)

    assert jwt.get_unverified_header(token) == {
        "alg": "ES256",
        "kid": "KEY123",
        "typ": "JWT",
    }
    assert jwt.decode(token, options={"verify_signature": False}) == {
        "iss": "TEAM123",
        "iat": 1713528000,
    }


def test_mint_apns_jwt_reuses_cached_token_within_55_minutes(monkeypatch, tmp_path):
    _configure_push(monkeypatch, tmp_path)

    first = dispatch._mint_apns_jwt(now=1000)
    second = dispatch._mint_apns_jwt(now=1000 + 55 * 60)

    assert first == second


def test_mint_apns_jwt_refreshes_after_55_minutes(monkeypatch, tmp_path):
    _configure_push(monkeypatch, tmp_path)

    first = dispatch._mint_apns_jwt(now=1000)
    second = dispatch._mint_apns_jwt(now=1000 + 55 * 60 + 1)

    assert first != second


def test_daily_briefing_payload_shape():
    payload = dispatch.build_daily_briefing_payload(
        day="20260419", generated="2026-04-19T06:45:00", needs_attention_count=3
    )

    assert payload["aps"]["category"] == dispatch.CATEGORY_DAILY_BRIEFING
    assert payload["aps"]["sound"] == "default"
    assert payload["aps"]["mutable-content"] == 1
    assert payload["aps"]["content-available"] == 1
    assert "interruption-level" not in payload["aps"]
    assert payload["data"] == {
        "action": "open_briefing",
        "day": "20260419",
        "generated": "2026-04-19T06:45:00",
        "needs_attention_count": 3,
    }


def test_pre_meeting_payload_shape():
    payload = dispatch.build_pre_meeting_payload(
        activity={
            "id": "anticipated_meeting_090000_0420",
            "start": "09:00",
            "title": "Launch sync",
            "location": "Room A",
            "prep_notes": "Bring launch notes",
            "participation": [
                {"name": "Juliet Capulet", "role": "attendee"},
                {"name": "Observer", "role": "organizer"},
            ],
        },
        facet="work",
        day="20260420",
    )

    assert payload["aps"]["category"] == dispatch.CATEGORY_PRE_MEETING_PREP
    assert payload["aps"]["interruption-level"] == "time-sensitive"
    assert payload["data"]["action"] == "open_pre_meeting"
    assert payload["data"]["participants"] == ["Juliet Capulet"]


def test_agent_alert_payload_shape():
    payload = dispatch.build_agent_alert_payload(
        title="Agent Alert", body="Needs review", context_id="ctx-1"
    )

    assert payload["aps"]["category"] == dispatch.CATEGORY_AGENT_ALERT
    assert payload["data"] == {"action": "open_alert", "context_id": "ctx-1"}
    assert "interruption-level" not in payload["aps"]


def test_agent_alert_payload_includes_route_when_present():
    payload = dispatch.build_agent_alert_payload(
        title="Agent Alert",
        body="Needs review",
        context_id="ctx-1",
        route="/app/reflections/20260308",
    )

    assert payload["data"] == {
        "action": "open_alert",
        "context_id": "ctx-1",
        "route": "/app/reflections/20260308",
    }


def test_commitment_payload_shape():
    payload = dispatch.build_commitment_payload(ledger_id="lg_123")

    assert payload["aps"]["category"] == dispatch.CATEGORY_COMMITMENT_NUDGE
    assert payload["data"] == {"action": "open_commitment", "ledger_id": "lg_123"}


def test_collapse_ids():
    assert dispatch.build_daily_briefing_collapse_id("20260419") == "briefing.20260419"
    assert (
        dispatch.build_pre_meeting_collapse_id("anticipated_meeting_090000_0420")
        == "meeting.anticipated_meeting_090000_0420"
    )
    assert dispatch.build_agent_alert_collapse_id("ctx-1") == "alert.ctx-1"
    assert dispatch.build_commitment_collapse_id("lg_123") == "commitment.lg_123"


def test_send_removes_bad_device_token(monkeypatch, tmp_path):
    _configure_push(monkeypatch, tmp_path)
    removed: list[str] = []
    monkeypatch.setattr(
        dispatch.devices, "remove_device", lambda token: removed.append(token) or True
    )

    async def fake_post(self, url, *, headers, json):
        return httpx.Response(400, json={"reason": "BadDeviceToken"})

    with patch.object(httpx.AsyncClient, "post", new=fake_post):
        ok, reason = dispatch.send(
            {"token": "a" * 64},
            dispatch.build_agent_alert_payload(
                title="Agent Alert", body="Needs review", context_id="ctx-1"
            ),
            collapse_id="alert.ctx-1",
        )

    assert ok is False
    assert reason == "BadDeviceToken"
    assert removed == ["a" * 64]


def test_send_removes_unregistered_device_on_410(monkeypatch, tmp_path):
    _configure_push(monkeypatch, tmp_path)
    removed: list[str] = []
    monkeypatch.setattr(
        dispatch.devices, "remove_device", lambda token: removed.append(token) or True
    )

    async def fake_post(self, url, *, headers, json):
        return httpx.Response(410, json={"reason": "Unregistered"})

    with patch.object(httpx.AsyncClient, "post", new=fake_post):
        ok, reason = dispatch.send(
            {"token": "b" * 64},
            dispatch.build_agent_alert_payload(
                title="Agent Alert", body="Needs review", context_id="ctx-1"
            ),
            collapse_id="alert.ctx-1",
        )

    assert ok is False
    assert reason == "Unregistered"
    assert removed == ["b" * 64]


def test_send_many_reuses_client_and_redacts_tokens(monkeypatch, tmp_path, caplog):
    _configure_push(monkeypatch, tmp_path)
    calls: list[dict[str, object]] = []
    caplog.set_level("WARNING", logger="solstone.push.dispatch")

    async def fake_post(self, url, *, headers, json):
        calls.append({"url": url, "headers": headers, "json": json})
        return httpx.Response(500, json={"reason": "InternalServerError"})

    with patch.object(httpx.AsyncClient, "post", new=fake_post):
        sent, failed = dispatch.send_many(
            [
                {"token": "c" * 64},
                {"token": "d" * 64},
            ],
            dispatch.build_daily_briefing_payload(
                day="20260419",
                generated="2026-04-19T06:45:00",
                needs_attention_count=1,
            ),
            collapse_id="briefing.20260419",
        )

    assert sent == 0
    assert failed == 2
    assert len(calls) == 2
    assert calls[0]["headers"]["apns-collapse-id"] == "briefing.20260419"
    assert calls[0]["headers"]["apns-priority"] == "10"
    assert calls[0]["headers"]["apns-push-type"] == "alert"
    assert calls[0]["headers"]["apns-topic"] == "org.solpbc.solstone-swift"
    assert "push rejected token=...cccc" in caplog.text
    assert all(record.levelname == "WARNING" for record in caplog.records)
    assert re.search(r"[0-9a-f]{64}", caplog.text) is None
