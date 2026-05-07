# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Callosum event handlers for proactive support detection.

Listens for repeated errors on the callosum bus and surfaces suggestions
to the user when ``support.proactive`` is enabled.  Never sends data
automatically — only alerts the user locally.
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from pathlib import Path

from solstone.apps.events import EventContext, on_event

logger = logging.getLogger(__name__)

# Track error counts per service in memory (resets on convey restart).
_error_counts: dict[str, list[float]] = defaultdict(list)
_WINDOW_SECONDS = 3600  # 1 hour
_THRESHOLD = 3  # notify after 3 errors in the window


def _is_proactive_enabled() -> bool:
    """Check if proactive detection is enabled in settings."""
    try:
        from solstone.think.utils import get_journal

        config_path = Path(get_journal()) / "config" / "config.json"
        if config_path.is_file():
            config = json.loads(config_path.read_text())
            support = config.get("support", {})
            return support.get("enabled", True) and support.get("proactive", True)
    except Exception:
        pass
    return True


@on_event("*", "error")
def detect_repeated_errors(ctx: EventContext) -> None:
    """Track error events and notify when a threshold is reached.

    When the same service emits 3+ errors within an hour and proactive
    mode is on, fires a notification suggesting the user investigate.
    """
    if not _is_proactive_enabled():
        return

    service = ctx.msg.get("service") or ctx.tract or "unknown"
    now = time.time()

    # Append and prune old entries
    timestamps = _error_counts[service]
    timestamps.append(now)
    cutoff = now - _WINDOW_SECONDS
    _error_counts[service] = [t for t in timestamps if t > cutoff]

    if len(_error_counts[service]) >= _THRESHOLD:
        count = len(_error_counts[service])
        logger.info(
            "Proactive support: %d errors from %s in the last hour",
            count,
            service,
        )

        # Fire a notification via callosum so the background service picks it up
        try:
            from solstone.think.callosum import callosum_send

            callosum_send(
                "support",
                "proactive_suggestion",
                service=service,
                count=count,
                message=(
                    f"I noticed {service} has had {count} errors in the last hour. "
                    "Want me to diagnose this?"
                ),
            )
        except Exception:
            logger.debug("Failed to send proactive notification", exc_info=True)

        # Reset counter to avoid spamming
        _error_counts[service] = []
