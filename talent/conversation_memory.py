# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Pre-hook: inject conversation memory into unified talent context.

Loaded via hook config: {"hook": {"pre": "conversation_memory"}}

Replaces CONVERSATION_MEMORY_INJECTION_POINT in the unified talent's
user instruction with recent conversation exchanges and today's summary.
This gives the agent awareness of past conversations without needing
to search — recent interactions are always in context.
"""

import logging

logger = logging.getLogger(__name__)


def pre_process(context: dict) -> dict | None:
    """Inject conversation memory into the unified talent's user instruction.

    Args:
        context: Full agent config dict.

    Returns:
        Dict with modified user_instruction, or None if no injection needed.
    """
    from think.conversation import INJECTION_MARKER, build_memory_context, inject_memory

    user_instruction = context.get("user_instruction", "")
    if INJECTION_MARKER not in user_instruction:
        return None

    facet = context.get("facet")

    try:
        memory_context = build_memory_context(facet=facet, recent_limit=10)
        new_instruction = inject_memory(user_instruction, memory_context)

        if new_instruction != user_instruction:
            return {"user_instruction": new_instruction}
    except Exception:
        logger.exception("Conversation memory injection failed")

    return None
