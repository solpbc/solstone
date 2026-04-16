# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Live capture-health derivation.

Read-time pull from `apps.observer.utils.list_observers()`. No cache, no
write path -- every call returns fresh state. On any exception below the
observer layer, returns ``{"status": "unknown", "observers": []}`` rather
than propagating; callers render a neutral UI instead of crashing.
"""

from __future__ import annotations

from think.utils import now_ms

_CONNECTED_MS = 30_000
_STALE_MS = 120_000


def get_capture_health() -> dict:
    """Return {"status": ..., "observers": [...]}.

    status ∈ {"active", "stale", "offline", "no_observers", "unknown"}.
    Overall rollup: active if any observer is active, else stale if any is
    stale, else offline. "no_observers" when the filtered list is empty.
    """
    from apps.observer.utils import list_observers

    try:
        observers = list_observers()
        # Filter to active (non-revoked, enabled) observers
        active = [
            o
            for o in observers
            if not o.get("revoked", False) and o.get("enabled", True)
        ]

        if not active:
            return {
                "status": "no_observers",
                "observers": [],
            }

        now = now_ms()
        observer_summaries = []
        statuses = []

        for o in active:
            last_seen = o.get("last_seen")
            if last_seen is None:
                obs_status = "offline"
            else:
                elapsed = now - last_seen
                if elapsed < _CONNECTED_MS:
                    obs_status = "active"
                elif elapsed < _STALE_MS:
                    obs_status = "stale"
                else:
                    obs_status = "offline"

            statuses.append(obs_status)
            observer_summaries.append(
                {
                    "name": o.get("name", "unknown"),
                    "last_seen": last_seen,
                    "status": obs_status,
                }
            )

        # Overall status is best healthy state across observers.
        if "active" in statuses:
            overall = "active"
        elif "stale" in statuses:
            overall = "stale"
        else:
            overall = "offline"

        return {
            "status": overall,
            "observers": observer_summaries,
        }
    except Exception:
        return {"status": "unknown", "observers": []}
