# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Deduplication and supersede readers for sol-initiated chat."""

from __future__ import annotations

import re

from solstone.convey.sol_initiated.copy import (
    KIND_OWNER_CHAT_OPEN,
    KIND_OWNER_MESSAGE,
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
    """Live if a still-pending request with matching dedupe is within window.

    Engagement releases pending requests: chat open by request_id, owner message
    by timestamp (strict <). Dismissal is not a release event.
    """
    pending: list[tuple[str, str, int]] = []

    for event in events:
        kind = event.get("kind")
        if kind == KIND_SOL_CHAT_REQUEST:
            request_id = str(event.get("request_id") or "")
            if not request_id:
                continue
            pending.append(
                (
                    request_id,
                    str(event.get("dedupe") or ""),
                    int(event.get("ts", 0) or 0),
                )
            )
            continue
        if kind == KIND_OWNER_CHAT_OPEN:
            request_id = str(event.get("request_id") or "")
            if not request_id:
                continue
            pending = [item for item in pending if item[0] != request_id]
            continue
        if kind == KIND_OWNER_MESSAGE:
            event_ts = int(event.get("ts", 0) or 0)
            pending = [item for item in pending if item[2] >= event_ts]

    return any(
        request_dedupe == dedupe_key and request_ts + window_ms > now_ms
        for _, request_dedupe, request_ts in pending
    )


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
