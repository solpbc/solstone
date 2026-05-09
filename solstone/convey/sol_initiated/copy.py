# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Locked literals for sol-initiated chat."""

KIND_SOL_CHAT_REQUEST = "sol_chat_request"
KIND_SOL_CHAT_REQUEST_SUPERSEDED = "sol_chat_request_superseded"
KIND_OWNER_CHAT_OPEN = "owner_chat_open"
KIND_OWNER_CHAT_DISMISSED = "owner_chat_dismissed"

TRIGGER_LABEL_SOL_INITIATED = "sol_initiated"
SYNTHETIC_TRIGGER_LABEL = "synthetic"

THROTTLE_MUTE_WINDOW = "mute-window"
THROTTLE_RATE_FLOOR = "rate-floor"
THROTTLE_CATEGORY_SELF_MUTE = "category-self-mute"
THROTTLE_CATEGORY_CAP = "category-cap"
THROTTLE_DAILY_CAP = "daily-cap"

CATEGORIES = ("briefing", "pattern", "commitment", "error", "arrival", "notice")
CATEGORY_CAP_DEFAULTS = {
    "briefing": 3,
    "pattern": 2,
    "commitment": 2,
    "error": 2,
    "arrival": 3,
    "notice": 2,
}

ENV_GUARD_LITERAL = "SOLSTONE_SOL_CHAT_REQUEST"
