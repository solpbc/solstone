# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Pre-hook for decisionalizer talent — skips days with no decision outputs."""

from pathlib import Path

from think.utils import get_journal


def pre_process(context: dict) -> dict | None:
    """Skip days that have no decision activity outputs."""
    day = context["day"]
    if not any(Path(get_journal()).glob(f"facets/*/activities/{day}/*/decisions.md")):
        return {"skip_reason": "no decision outputs for day"}
    return {}
