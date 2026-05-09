# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Deduplication and supersede readers for sol-initiated chat."""

from __future__ import annotations

import re

from solstone.convey.sol_initiated.copy import (
    KIND_OWNER_CHAT_DISMISSED,
    KIND_SOL_CHAT_REQUEST,
    KIND_SOL_CHAT_REQUEST_SUPERSEDED,
)

_WINDOW_RE = re.compile(r"^(\d+)([smhd])$")
_WINDOW_MULTIPLIERS_MS = {
    "s": 1_000,
    "m": 60_000,
    "h": 3_600_000,
    "d": 86_400_000,
}


def parse_dedupe_window(spec: str) -> int:
    """Parse a dedupe window like ``24h`` into milliseconds."""
    match = _WINDOW_RE.fullmatch(str(spec or "").strip())
    if match is None:
        raise ValueError("dedupe_window must match <number><s|m|h|d>")
    amount = int(match.group(1))
    if amount <= 0:
        raise ValueError("dedupe_window amount must be positive")
    return amount * _WINDOW_MULTIPLIERS_MS[match.group(2)]


def _is_live_for_dedup(
    events: list[dict],
    dedupe_key: str,
    window_ms: int,
    now_ms: int,
) -> bool:
    """Return whether a matching request is still live for deduplication."""
    requests: dict[str, tuple[str, int]] = {}
    released: set[str] = set()

    for event in events:
        kind = event.get("kind")
        if kind == KIND_SOL_CHAT_REQUEST:
            request_id = str(event.get("request_id") or "")
            if request_id:
                requests[request_id] = (
                    str(event.get("dedupe") or ""),
                    int(event.get("ts", 0) or 0),
                )
            continue
        if kind == KIND_OWNER_CHAT_DISMISSED:
            request_id = str(event.get("request_id") or "")
            if request_id in requests:
                released.add(request_id)

    for request_id, (request_dedupe, request_ts) in requests.items():
        if request_id in released:
            continue
        if request_dedupe != dedupe_key:
            continue
        if request_ts + window_ms > now_ms:
            return True
    return False


def _is_unresolved_for_supersede(events: list[dict]) -> str | None:
    """Return the most recent unresolved sol-initiated request id."""
    replaced: set[str] = set()

    for event in reversed(events):
        kind = event.get("kind")
        if kind == "sol_message":
            return None
        if kind == KIND_SOL_CHAT_REQUEST_SUPERSEDED:
            request_id = str(event.get("request_id") or "")
            if request_id:
                replaced.add(request_id)
            continue
        if kind == KIND_SOL_CHAT_REQUEST:
            request_id = str(event.get("request_id") or "")
            if request_id and request_id not in replaced:
                return request_id
    return None
