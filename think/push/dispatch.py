# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""APNs transport for push notifications."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

import httpx
import jwt

from think.push import devices
from think.push.config import (
    get_apns_key_id,
    get_apns_key_path,
    get_apns_team_id,
    get_bundle_id,
    get_environment,
)

logger = logging.getLogger("solstone.push.dispatch")

CATEGORY_DAILY_BRIEFING = "SOLSTONE_DAILY_BRIEFING"
CATEGORY_PRE_MEETING_PREP = "SOLSTONE_PRE_MEETING_PREP"
CATEGORY_AGENT_ALERT = "SOLSTONE_AGENT_ALERT"
CATEGORY_COMMITMENT_NUDGE = "SOLSTONE_COMMITMENT_NUDGE"
CATEGORIES = (
    CATEGORY_DAILY_BRIEFING,
    CATEGORY_PRE_MEETING_PREP,
    CATEGORY_AGENT_ALERT,
    CATEGORY_COMMITMENT_NUDGE,
)
_JWT_MAX_AGE_SECONDS = 55 * 60
_APNS_JWT_CACHE: dict[tuple[str, str], tuple[str, int]] = {}
_APNS_JWT_CACHE_LOCK = threading.Lock()


def _require_bundle_id() -> str:
    bundle_id = get_bundle_id()
    if bundle_id is None:
        raise RuntimeError("push.bundle_id is not configured")
    return bundle_id


def _require_key_id() -> str:
    key_id = get_apns_key_id()
    if key_id is None:
        raise RuntimeError("push.apns_key_id is not configured")
    return key_id


def _require_team_id() -> str:
    team_id = get_apns_team_id()
    if team_id is None:
        raise RuntimeError("push.apns_team_id is not configured")
    return team_id


def _require_key_path() -> Path:
    key_path = get_apns_key_path()
    if key_path is None:
        raise RuntimeError("push.apns_key_path is not configured")
    if not key_path.is_absolute():
        raise ValueError("push.apns_key_path must be an absolute path")
    if not key_path.exists():
        raise FileNotFoundError(f"APNs key file not found: {key_path}")
    if not key_path.is_file():
        raise RuntimeError(f"APNs key file is not a regular file: {key_path}")
    return key_path


def _mint_apns_jwt(*, now: int | None = None) -> str:
    issued_at = int(time.time()) if now is None else now
    key_id = _require_key_id()
    team_id = _require_team_id()
    cache_key = (key_id, team_id)
    with _APNS_JWT_CACHE_LOCK:
        cached = _APNS_JWT_CACHE.get(cache_key)
        if cached and issued_at - cached[1] <= _JWT_MAX_AGE_SECONDS:
            return cached[0]
        token = jwt.encode(
            {"iss": team_id, "iat": issued_at},
            _require_key_path().read_text(encoding="utf-8"),
            algorithm="ES256",
            headers={"alg": "ES256", "kid": key_id},
        )
        _APNS_JWT_CACHE[cache_key] = (token, issued_at)
        return token


def build_daily_briefing_collapse_id(day: str) -> str:
    return f"briefing.{day}"


def build_pre_meeting_collapse_id(activity_id: str) -> str:
    return f"meeting.{activity_id}"


def build_agent_alert_collapse_id(context_id: str) -> str:
    return f"alert.{context_id}"


def build_commitment_collapse_id(ledger_id: str) -> str:
    return f"commitment.{ledger_id}"


def build_daily_briefing_payload(
    *, day: str, generated: str | None, needs_attention_count: int
) -> dict[str, Any]:
    return {
        "aps": {
            "alert": {
                "title": "Daily Briefing",
                "body": "Your briefing is ready — tap to view",
            },
            "category": CATEGORY_DAILY_BRIEFING,
            "sound": "default",
            "mutable-content": 1,
            "content-available": 1,
        },
        "data": {
            "action": "open_briefing",
            "day": day,
            "generated": generated,
            "needs_attention_count": needs_attention_count,
        },
    }


def build_pre_meeting_payload(
    *, activity: dict[str, Any], facet: str, day: str
) -> dict[str, Any]:
    participants = [
        str(entry.get("name") or "").strip()
        for entry in activity.get("participation", [])
        if isinstance(entry, dict)
        and entry.get("role") == "attendee"
        and str(entry.get("name") or "").strip()
    ]
    return {
        "aps": {
            "alert": {
                "title": "Pre-Meeting Prep",
                "body": "Meeting in 15 minutes — tap to view",
            },
            "category": CATEGORY_PRE_MEETING_PREP,
            "sound": "default",
            "mutable-content": 1,
            "content-available": 1,
            "interruption-level": "time-sensitive",
        },
        "data": {
            "action": "open_pre_meeting",
            "activity_id": str(activity.get("id") or ""),
            "facet": facet,
            "day": day,
            "start": str(activity.get("start") or ""),
            "title": str(activity.get("title") or ""),
            "location": str(activity.get("location") or ""),
            "participants": participants,
            "prep_notes": str(activity.get("prep_notes") or ""),
        },
    }


def build_agent_alert_payload(
    *, title: str, body: str, context_id: str
) -> dict[str, Any]:
    return {
        "aps": {
            "alert": {"title": title, "body": body},
            "category": CATEGORY_AGENT_ALERT,
            "sound": "default",
            "mutable-content": 1,
            "content-available": 1,
        },
        "data": {"action": "open_alert", "context_id": context_id},
    }


def build_commitment_payload(*, ledger_id: str) -> dict[str, Any]:
    return {
        "aps": {
            "alert": {
                "title": "Commitment Nudge",
                "body": "A commitment needs attention — tap to view",
            },
            "category": CATEGORY_COMMITMENT_NUDGE,
            "sound": "default",
            "mutable-content": 1,
            "content-available": 1,
        },
        "data": {"action": "open_commitment", "ledger_id": ledger_id},
    }


def _apns_host() -> str:
    environment = get_environment()
    if environment == "production":
        return "https://api.push.apple.com"
    return "https://api.sandbox.push.apple.com"


def _headers(*, collapse_id: str, priority: int) -> dict[str, str]:
    return {
        "apns-topic": _require_bundle_id(),
        "apns-collapse-id": collapse_id,
        "apns-priority": str(priority),
        "apns-push-type": "alert",
        "authorization": f"bearer {_mint_apns_jwt()}",
    }


def _response_reason(response: httpx.Response) -> str | None:
    try:
        payload = response.json()
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    reason = payload.get("reason")
    return str(reason) if isinstance(reason, str) and reason else None


def _run_async(coro: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    result: dict[str, Any] = {}
    error: dict[str, BaseException] = {}

    def runner() -> None:
        try:
            result["value"] = asyncio.run(coro)
        except BaseException as exc:
            error["value"] = exc

    thread = threading.Thread(target=runner, name="push-dispatch", daemon=True)
    thread.start()
    thread.join()
    if "value" in error:
        raise error["value"]
    return result.get("value")


async def _send_with_client(
    client: httpx.AsyncClient,
    device: dict[str, Any],
    payload: dict[str, Any],
    *,
    collapse_id: str,
    priority: int,
) -> tuple[bool, str | None]:
    token = str(device.get("token") or "")
    masked_token = devices.mask_token(token)
    try:
        response = await client.post(
            f"{_apns_host()}/3/device/{token}",
            headers=_headers(collapse_id=collapse_id, priority=priority),
            json=payload,
        )
    except Exception as exc:
        logger.warning("push delivery failed token=%s error=%s", masked_token, exc)
        return False, str(exc)
    reason = _response_reason(response)
    if response.status_code == 200:
        return True, None
    if response.status_code == 410 or reason in {"BadDeviceToken", "Unregistered"}:
        devices.remove_device(token)
        logger.warning(
            "push pruning token=%s status=%s reason=%s",
            masked_token,
            response.status_code,
            reason or "",
        )
        return False, reason
    if 500 <= response.status_code:
        logger.warning(
            "push rejected token=%s status=%s reason=%s",
            masked_token,
            response.status_code,
            reason or "",
        )
        return False, reason
    logger.error(
        "push rejected token=%s status=%s reason=%s",
        masked_token,
        response.status_code,
        reason or "",
    )
    return False, reason


async def _send_async(
    device: dict[str, Any],
    payload: dict[str, Any],
    *,
    collapse_id: str,
    priority: int = 10,
) -> tuple[bool, str | None]:
    async with httpx.AsyncClient(http2=True, timeout=10.0) as client:
        return await _send_with_client(
            client, device, payload, collapse_id=collapse_id, priority=priority
        )


def send(
    device: dict[str, Any],
    payload: dict[str, Any],
    *,
    collapse_id: str,
    priority: int = 10,
) -> tuple[bool, str | None]:
    return _run_async(
        _send_async(device, payload, collapse_id=collapse_id, priority=priority)
    )


async def _send_many_async(
    push_devices: list[dict[str, Any]],
    payload: dict[str, Any],
    *,
    collapse_id: str,
    priority: int = 10,
) -> tuple[int, int]:
    sent = 0
    failed = 0
    async with httpx.AsyncClient(http2=True, timeout=10.0) as client:
        for device in push_devices:
            ok, _ = await _send_with_client(
                client, device, payload, collapse_id=collapse_id, priority=priority
            )
            if ok:
                sent += 1
            else:
                failed += 1
    return sent, failed


def send_many(
    push_devices: list[dict[str, Any]],
    payload: dict[str, Any],
    *,
    collapse_id: str,
    priority: int = 10,
) -> tuple[int, int]:
    return _run_async(
        _send_many_async(
            push_devices, payload, collapse_id=collapse_id, priority=priority
        )
    )


__all__ = [
    "CATEGORIES",
    "CATEGORY_AGENT_ALERT",
    "CATEGORY_COMMITMENT_NUDGE",
    "CATEGORY_DAILY_BRIEFING",
    "CATEGORY_PRE_MEETING_PREP",
    "build_agent_alert_collapse_id",
    "build_agent_alert_payload",
    "build_commitment_collapse_id",
    "build_commitment_payload",
    "build_daily_briefing_collapse_id",
    "build_daily_briefing_payload",
    "build_pre_meeting_collapse_id",
    "build_pre_meeting_payload",
    "send",
    "send_many",
]
