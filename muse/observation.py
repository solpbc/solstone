# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Observation hooks for Path A onboarding.

Pre-hook: guards on awareness state — skips immediately (zero API cost)
when the user is not in Path A observation mode.

Post-hook: writes LLM findings to the awareness log, sends callosum
notifications for interesting discoveries, and transitions to "ready"
when the observation threshold is met.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# Observation thresholds
MIN_SEGMENTS = 10
MIN_HOURS = 4.0

# Maximum nudge notifications during observation
MAX_NUDGES = 4


# ---------------------------------------------------------------------------
# Pre-hook
# ---------------------------------------------------------------------------


def pre_process(context: dict) -> dict | None:
    """Guard: skip if not in Path A observation mode.

    Args:
        context: PreHookContext with day, segment, output_path, transcript, meta

    Returns:
        Dict with skip_reason if not observing, or None to proceed.
    """
    from think.awareness import get_onboarding

    onboarding = get_onboarding()
    if onboarding.get("status") != "observing":
        return {"skip_reason": "not_observing"}

    # Observing — let the LLM analyze the segment transcript
    return None


# ---------------------------------------------------------------------------
# Post-hook
# ---------------------------------------------------------------------------


def post_process(result: str, context: dict) -> str | None:
    """Process LLM observation output — log, notify, check threshold.

    Args:
        result: LLM JSON output with observation findings
        context: Full config dict with day, segment, etc.

    Returns:
        The result string unchanged (output still written to segment dir).
    """
    from think.awareness import append_log, get_onboarding, update_state
    from think.callosum import callosum_send

    day = context.get("day", "")
    segment = context.get("segment", "")

    # Parse LLM output
    try:
        findings = json.loads(result)
    except (json.JSONDecodeError, TypeError):
        logger.warning("observation post-hook: failed to parse LLM output")
        return result

    if not isinstance(findings, dict):
        logger.warning("observation post-hook: LLM output is not a dict")
        return result

    # Write observation to awareness log
    append_log(
        "observation",
        key=f"segment.{day}.{segment}",
        message=findings.get("summary", ""),
        data=findings,
        day=day,
        segment=segment,
    )

    # Update observation count
    onboarding = get_onboarding()
    count = onboarding.get("observation_count", 0) + 1
    update_state("onboarding", {"observation_count": count})

    # Check if we should send a nudge notification
    nudges_sent = onboarding.get("nudges_sent", 0)
    if nudges_sent < MAX_NUDGES:
        nudge = _check_nudge(findings, count, nudges_sent, onboarding)
        if nudge:
            callosum_send(
                "notification",
                "show",
                title=nudge["title"],
                message=nudge["message"],
                icon=nudge.get("icon", "🔍"),
                app="observation",
            )
            update_state("onboarding", {"nudges_sent": nudges_sent + 1})
            append_log(
                "nudge",
                key="onboarding.nudge",
                message=nudge["message"],
                data={"title": nudge["title"], "nudge_number": nudges_sent + 1},
            )

    # Check observation threshold
    if _threshold_met(onboarding, count):
        _transition_to_ready(day)

    return result


def _check_nudge(
    findings: dict,
    observation_count: int,
    nudges_sent: int,
    onboarding: dict,
) -> dict | None:
    """Decide whether this segment's findings warrant a notification.

    Returns a nudge dict with title/message/icon, or None.
    Nudge triggers (in order of priority):
      0: First meeting detected
      1: First entity cluster (3+ named people)
      2: After 5 segments — progress update
      3: Nearing threshold — "almost ready"
    """
    # Nudge 0: First meeting
    if nudges_sent == 0 and findings.get("has_meeting"):
        speaker_count = findings.get("speaker_count", 0)
        topic = findings.get("meeting_topic") or "a conversation"
        return {
            "title": "Meeting detected",
            "message": f"Noticed {speaker_count} people discussing {topic}.",
            "icon": "🎙️",
        }

    # Nudge 1: First entity cluster
    if nudges_sent <= 1:
        people = findings.get("people", [])
        if len(people) >= 3:
            names = ", ".join(people[:3])
            return {
                "title": "Learning your network",
                "message": f"Spotted several people: {names}.",
                "icon": "👥",
            }

    # Nudge 2: Progress update at 5 segments
    if nudges_sent <= 2 and observation_count == 5:
        return {
            "title": "Still learning",
            "message": "Building a picture of your work patterns. Keep going!",
            "icon": "📊",
        }

    # Nudge 3: Almost ready
    if nudges_sent <= 3 and observation_count >= MIN_SEGMENTS - 1:
        started = onboarding.get("started", "")
        hours = _elapsed_hours(started)
        if hours >= MIN_HOURS * 0.75:
            return {
                "title": "Almost ready",
                "message": "Have enough data to make suggestions soon.",
                "icon": "✨",
            }

    return None


def _threshold_met(onboarding: dict, count: int) -> bool:
    """Check if observation period is complete.

    Requires both minimum segments AND minimum elapsed time.
    """
    if count < MIN_SEGMENTS:
        return False

    started = onboarding.get("started", "")
    hours = _elapsed_hours(started)
    return hours >= MIN_HOURS


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


def _transition_to_ready(day: str) -> None:
    """Transition onboarding to 'ready' state and send chat redirect."""
    from think.awareness import append_log, update_state

    update_state("onboarding", {"status": "ready"})
    append_log(
        "state",
        key="onboarding.ready",
        message="Observation threshold met — recommendations ready",
        day=day,
    )

    # Send chat redirect to open the recommendation review
    try:
        from apps.utils import get_app_storage_path
        from convey.utils import save_json
        from think.callosum import callosum_send
        from think.cortex_client import cortex_request
        from think.utils import now_ms

        prompt = (
            "The user chose Path A onboarding — passive observation. "
            "The observation period is complete. Read the accumulated "
            "observations and present your recommendations for facets "
            "and entities. Be warm and enthusiastic about what you learned."
        )
        agent_id = cortex_request(prompt=prompt, name="observation_review")
        if agent_id:
            chat_record = {
                "ts": now_ms(),
                "muse": "observation_review",
                "title": "Your journal suggestions are ready",
                "agent_ids": [agent_id],
            }
            chats_dir = get_app_storage_path("chat", "chats")
            save_json(chats_dir / f"{agent_id}.json", chat_record)
            callosum_send("navigate", "request", path=f"/app/chat#{agent_id}")
            logger.info("Sent observation review redirect: %s", agent_id)
    except Exception:
        logger.exception("Failed to send observation review redirect")
        # Non-fatal — user can still trigger review via triage
