# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Locked copy for observer localhost-only CLI warnings."""

from __future__ import annotations

OBSERVER_LOCALHOST_BANNER_LINE_1 = "⚠️  convey is in localhost-only mode."
OBSERVER_LOCALHOST_BANNER_LINE_2 = (
    "this observer key won't be reachable from remote devices."
)
OBSERVER_LOCALHOST_BANNER_LINE_3 = "enabling network access should only be done on trusted networks and requires a password."
OBSERVER_LOCALHOST_BANNER_LINE_4 = (
    "to allow network access: sol call settings convey network-access enable"
)
OBSERVER_LOCALHOST_REMINDER = "reminder: convey is in localhost-only mode. this observer can't reach convey from another device until you run: sol call settings convey network-access enable"


__all__ = [
    "OBSERVER_LOCALHOST_BANNER_LINE_1",
    "OBSERVER_LOCALHOST_BANNER_LINE_2",
    "OBSERVER_LOCALHOST_BANNER_LINE_3",
    "OBSERVER_LOCALHOST_BANNER_LINE_4",
    "OBSERVER_LOCALHOST_REMINDER",
]
