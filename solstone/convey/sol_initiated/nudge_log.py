# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Nudge-log writer for sol-initiated chat attempts."""

from __future__ import annotations

import time

from solstone.think.push.triggers import _append_nudge_log


def record_nudge_log(
    kind: str,
    dedupe_key: str,
    category: str,
    outcome: str,
) -> None:
    """Append one sol-initiated row.

    Older rows written by push send accounting do not include ``kind``. This
    writer leaves those rows unchanged.
    """
    _append_nudge_log(
        {
            "ts": int(time.time()),
            "kind": kind,
            "dedupe_key": dedupe_key,
            "category": category,
            "outcome": outcome,
        }
    )
