# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Observer-side filter for sol-initiated chat events."""

from __future__ import annotations

from solstone.convey.sol_initiated.copy import (
    KIND_OWNER_CHAT_DISMISSED,
    KIND_OWNER_CHAT_OPEN,
    KIND_SOL_CHAT_REQUEST,
    KIND_SOL_CHAT_REQUEST_SUPERSEDED,
)

_FIELDS_BY_KIND: dict[str, tuple[str, ...]] = {
    KIND_SOL_CHAT_REQUEST: (
        "request_id",
        "summary",
        "message",
        "category",
        "dedupe",
        "dedupe_window",
        "since_ts",
        "trigger_talent",
    ),
    KIND_SOL_CHAT_REQUEST_SUPERSEDED: ("request_id", "replaced_by"),
    KIND_OWNER_CHAT_OPEN: ("request_id", "surface"),
    KIND_OWNER_CHAT_DISMISSED: ("request_id", "surface", "reason"),
}


def filter_sol_chat_event(frame: dict) -> dict | None:
    """Normalize a callosum/chronicle frame for sol-initiated chat kinds.

    Accepts either a callosum frame (tract+event) or a chronicle stream event
    (kind). Returns a normalized dict with `kind`, `ts`, plus the kind-specific
    fields. Returns None for any frame that does not match the four kinds or
    that lacks a callosum chat tract when one is present.
    """
    if not isinstance(frame, dict):
        return None
    kind = frame.get("event") or frame.get("kind")
    if kind not in _FIELDS_BY_KIND:
        return None
    tract = frame.get("tract")
    if tract is not None and tract != "chat":
        return None
    out: dict = {"kind": kind, "ts": frame.get("ts")}
    for field in _FIELDS_BY_KIND[kind]:
        out[field] = frame.get(field)
    return out
