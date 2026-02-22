# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Dev app event handlers - debug logging for all Callosum events.

This module demonstrates server-side event handling for apps.
Enable verbose logging to see events in the Convey log.
"""

import logging

from apps.events import EventContext, on_event

logger = logging.getLogger(__name__)


@on_event("*", "*")
def log_all_events(ctx: EventContext) -> None:
    """Log all Callosum events for debugging.

    This handler matches all events via wildcards and logs them at DEBUG level.
    Useful for understanding event flow during development.
    """
    logger.debug(f"[dev] Event: {ctx.tract}/{ctx.event} - keys: {list(ctx.msg.keys())}")
