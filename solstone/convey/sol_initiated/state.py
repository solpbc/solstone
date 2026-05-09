# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Read helpers for sol-initiated chat stream state."""

from __future__ import annotations

from solstone.convey.sol_initiated.copy import (
    KIND_OWNER_CHAT_DISMISSED,
    KIND_OWNER_CHAT_OPEN,
    KIND_SOL_CHAT_REQUEST,
    KIND_SOL_CHAT_REQUEST_SUPERSEDED,
)


def latest_unresolved_sol_chat_request(events: list[dict]) -> dict | None:
    """Return the latest unresolved sol-initiated request from chronological events.

    Resolved means the request_id appears in a later owner-open, owner-dismiss,
    or supersede event.

    Returns dict with keys: request_id, summary, ts, event_index. None if no
    unresolved request exists.
    """
    resolved_request_ids: set[str] = set()
    requests: list[dict] = []

    for index, event in enumerate(events):
        kind = event.get("kind")
        if kind in {
            KIND_OWNER_CHAT_OPEN,
            KIND_OWNER_CHAT_DISMISSED,
            KIND_SOL_CHAT_REQUEST_SUPERSEDED,
        }:
            request_id = str(event.get("request_id") or "").strip()
            if request_id:
                resolved_request_ids.add(request_id)
            continue

        if kind != KIND_SOL_CHAT_REQUEST:
            continue

        request_id = str(event.get("request_id") or "").strip()
        if not request_id:
            continue
        requests.append(
            {
                "request_id": request_id,
                "summary": str(event.get("summary") or ""),
                "ts": event.get("ts"),
                "event_index": index,
            }
        )

    for request in reversed(requests):
        if request["request_id"] not in resolved_request_ids:
            return request
    return None
