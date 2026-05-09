# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Owner lifecycle event writers for sol-initiated chat."""

from __future__ import annotations

from solstone.convey.chat_stream import append_chat_event
from solstone.convey.sol_initiated.copy import (
    KIND_OWNER_CHAT_DISMISSED,
    KIND_OWNER_CHAT_OPEN,
)


def record_owner_chat_open(request_id: str, surface: str) -> dict:
    """Record that the owner opened a sol-initiated chat request."""
    return append_chat_event(
        KIND_OWNER_CHAT_OPEN,
        request_id=request_id,
        surface=surface,
    )


def record_owner_chat_dismissed(
    request_id: str,
    surface: str,
    reason: str | None = None,
) -> dict:
    """Record that the owner dismissed a sol-initiated chat request."""
    return append_chat_event(
        KIND_OWNER_CHAT_DISMISSED,
        request_id=request_id,
        surface=surface,
        reason=reason,
    )
