# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Callosum event handlers for conversation exchange recording.

Records triage agent completions to the conversation memory service
so exchanges persist for future context injection.
"""

import logging

from apps.events import EventContext, on_event
from think.conversation import record_exchange
from think.cortex_client import read_agent_events

logger = logging.getLogger(__name__)

TRIAGE_AGENT_NAMES = {"unified", "triage", "onboarding"}


@on_event("cortex", "finish")
def record_triage_exchange(ctx: EventContext) -> None:
    """Record completed triage agent exchanges to conversation memory."""
    name = ctx.msg.get("name")
    if name not in TRIAGE_AGENT_NAMES:
        return

    agent_id = ctx.msg.get("agent_id")
    if not agent_id:
        return

    try:
        events = read_agent_events(agent_id)
        facet = ""
        app = ""
        path = ""
        user_message = ""
        for event in events:
            if event.get("event") == "request":
                facet = event.get("facet", "")
                app = event.get("app", "")
                path = event.get("path", "")
                user_message = event.get("user_message", "")
                break

        result = ctx.msg.get("result", "")
        record_exchange(
            facet=facet,
            app=app,
            path=path,
            user_message=user_message,
            agent_response=result,
            muse=name,
            agent_id=agent_id,
        )
    except Exception:
        logger.debug(
            "Failed to record conversation exchange for agent %s", agent_id, exc_info=True
        )
