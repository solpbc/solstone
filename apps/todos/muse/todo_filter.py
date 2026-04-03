# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Pre-filter hook for the todo detector agent.

Scans activity transcripts for commitment-language signals. When no
signals are found, returns skip_reason so the agent skips without
LLM invocation.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

_SIGNAL_PATTERNS = (
    # Action commitment
    re.compile(
        r"I'll|I will|I need to|I should|we should|we need to|let's|let me",
        re.IGNORECASE,
    ),
    # Follow-up
    re.compile(r"follow up|follow-up|get back to|circle back", re.IGNORECASE),
    # Reminders
    re.compile(r"remind me|don't forget|make sure to|remember to", re.IGNORECASE),
    # Deadlines
    re.compile(
        r"by (?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|tomorrow|end of|next week)|deadline|due date|due by",
        re.IGNORECASE,
    ),
    # Explicit markers
    re.compile(r"\bTODO\b|\bFIXME\b|action items?|next steps?", re.IGNORECASE),
    # Task creation
    re.compile(r"add to\b.*\blist|put on\b.*\blist|add a task|create a task", re.IGNORECASE),
)


def pre_process(context: dict) -> dict | None:
    """Skip the todo detector when the transcript has no commitment signals.

    Args:
        context: Agent config dict with transcript, day, activity, etc.

    Returns:
        Dict with skip_reason when no signals found, or None to proceed.
    """
    transcript = context.get("transcript") or ""
    if not transcript.strip():
        logger.info("todo_filter: skipping, empty transcript")
        return {"skip_reason": "no commitment signals in transcript"}

    if any(pattern.search(transcript) for pattern in _SIGNAL_PATTERNS):
        logger.debug("todo_filter: commitment signal found, proceeding")
        return None

    logger.info(
        "todo_filter: skipping, no commitment signals (transcript_len=%d)",
        len(transcript),
    )
    return {"skip_reason": "no commitment signals in transcript"}
