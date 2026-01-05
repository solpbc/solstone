# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Remote app event handlers - track processing status of remote segments.

Listens for observe.observed events for remote-originated segments and
records their completion in the sync history. This enables remote observers
to verify end-to-end processing success via the segments endpoint.
"""

import logging
import time

from apps.events import EventContext, on_event

from .utils import append_history_record, find_remote_by_name, increment_stat

logger = logging.getLogger(__name__)


@on_event("observe", "observed")
def handle_observed(ctx: EventContext) -> None:
    """Track observe.observed events for remote-originated segments.

    When a segment from a remote observer completes processing, append
    an 'observed' record to that remote's sync history. This enables
    remote observers to verify end-to-end success via the segments API.
    """
    remote_name = ctx.msg.get("remote")
    if not remote_name:
        return  # Not a remote segment

    segment = ctx.msg.get("segment")
    day = ctx.msg.get("day")
    if not segment or not day:
        logger.warning(f"observe.observed missing segment/day for remote {remote_name}")
        return

    # Find remote by name to get key prefix
    remote = find_remote_by_name(remote_name)
    if not remote:
        logger.debug(f"Remote not found for observed event: {remote_name}")
        return

    key_prefix = remote.get("key", "")[:8]
    if not key_prefix:
        return

    # Append observed record to history
    record = {
        "ts": int(time.time() * 1000),
        "type": "observed",
        "segment": segment,
    }
    append_history_record(key_prefix, day, record)

    # Update stats
    increment_stat(key_prefix, "segments_observed")

    logger.debug(f"Recorded observed status for remote {remote_name}: {day}/{segment}")
