# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Pre-hook: provide routine state template vars for exec."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def pre_process(context: dict) -> dict:
    """Build routine-state template vars for the exec talent prompt."""
    del context

    active_routines = ""
    routine_suggestion = ""

    try:
        from solstone.talent._routine_context import render_active_routines

        active_routines = render_active_routines()
    except Exception:
        logger.debug("exec_context: failed to render active routines", exc_info=True)

    try:
        from solstone.talent._routine_context import render_routine_suggestion

        routine_suggestion = render_routine_suggestion()
    except Exception:
        logger.debug("exec_context: failed to render routine suggestion", exc_info=True)

    return {
        "template_vars": {
            "active_routines": active_routines,
            "routine_suggestion": routine_suggestion,
        }
    }
