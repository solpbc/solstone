# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Observer app event handlers for observer segment processing state."""

import logging

from apps.events import EventContext, on_event
from think.utils import now_ms

from .utils import append_history_record, find_observer_by_name, increment_stat

logger = logging.getLogger(__name__)


@on_event("observe", "observed")
def handle_observed(ctx: EventContext) -> None:
    """Track observe.observed events for observer-originated segments.

    When a segment from an observer completes processing, append
    an 'observed' record to that observer's sync history. This enables
    observers to verify end-to-end success via the segments API.
    """
    observer_name = ctx.msg.get("observer")
    if not observer_name:
        return  # Not an observer segment

    segment = ctx.msg.get("segment")
    day = ctx.msg.get("day")
    if not segment or not day:
        logger.warning(
            f"observe.observed missing segment/day for observer {observer_name}"
        )
        return

    # Find observer by name to get key prefix
    observer = find_observer_by_name(observer_name)
    if not observer:
        logger.debug(f"Observer not found for observed event: {observer_name}")
        return

    key_prefix = observer.get("key", "")[:8]
    if not key_prefix:
        return

    # Append observed record to history
    record = {
        "ts": now_ms(),
        "type": "observed",
        "segment": segment,
    }
    append_history_record(key_prefix, day, record)

    # Update stats
    increment_stat(key_prefix, "segments_observed")

    logger.debug(
        f"Recorded observed status for observer {observer_name}: {day}/{segment}"
    )
