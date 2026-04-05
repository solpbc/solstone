# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""First-day check-in hooks.

Pre-hook: guards on awareness state — skips immediately (zero API cost)
unless onboarding is complete and ~1 hour has elapsed since completion.
One-shot: once the check-in fires, it never fires again.

Post-hook: records that the check-in was sent and spawns a support
agent chat for the user.
"""

from __future__ import annotations

import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# Minimum hours after onboarding completion before check-in fires
MIN_HOURS_AFTER_COMPLETE = 1.0


def pre_process(context: dict) -> dict | None:
    """Guard: skip unless onboarding is complete and 1+ hour has elapsed.

    Returns dict with skip_reason to skip, or None to proceed.
    """
    from think.awareness import get_onboarding

    onboarding = get_onboarding()
    status = onboarding.get("status")

    # Only fire after onboarding completes
    if status != "complete":
        return {"skip_reason": "not_complete"}

    # Only fire once
    if onboarding.get("firstday_checkin_sent"):
        return {"skip_reason": "already_sent"}

    # Check elapsed time since onboarding started
    # (onboarding.started is when they began, but we want time since completion;
    # since complete_onboarding() doesn't record a timestamp, use started + a
    # generous window — 1 hour after they started is a reasonable proxy for
    # "settled in")
    started = onboarding.get("started", "")
    if not started:
        return {"skip_reason": "no_start_time"}

    hours = _elapsed_hours(started)
    if hours < MIN_HOURS_AFTER_COMPLETE:
        return {"skip_reason": "too_soon"}

    # All conditions met — proceed
    return None


def post_process(result: str, context: dict) -> str | None:
    """Record check-in and spawn support agent chat."""
    from think.awareness import append_log, update_state

    # Record that we sent it (prevents repeat)
    update_state("onboarding", {"firstday_checkin_sent": _now_iso()})
    append_log(
        "state",
        key="onboarding.firstday_checkin_sent",
        message="First-day check-in sent to user",
    )

    # Spawn support agent check-in and surface through conversation panel
    try:
        from think.callosum import callosum_send
        from think.cortex_client import cortex_request

        prompt = (
            "The user recently completed onboarding. This is your first-day "
            "check-in. Send a warm, brief message: ask how things are going, "
            "if anything is surprising or confusing, and remind them you're "
            "here to help or capture feedback anytime. Keep it short and "
            "conversational — one message, not a wall of text."
        )
        agent_id = cortex_request(prompt=prompt, name="support")
        if agent_id:
            callosum_send(
                "notification",
                "show",
                title="Check-in",
                message="How's everything going? I'm here if you need anything.",
                icon="👋",
                app="conversation",
            )
            logger.info("Spawned first-day check-in agent: %s", agent_id)
    except Exception:
        logger.exception("Failed to spawn first-day check-in agent")

    return result


def _elapsed_hours(started_iso: str) -> float:
    """Calculate hours elapsed since the started timestamp."""
    if not started_iso:
        return 0.0
    try:
        start = datetime.strptime(started_iso, "%Y%m%dT%H:%M:%S")
        elapsed = (datetime.now() - start).total_seconds()
        return elapsed / 3600
    except (ValueError, TypeError):
        return 0.0


def _now_iso() -> str:
    """Return current time as compact ISO string."""
    return datetime.now().strftime("%Y%m%dT%H:%M:%S")
