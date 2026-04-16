# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""System health API endpoint."""

from __future__ import annotations

import logging
import time
from typing import Any

from flask import Blueprint, jsonify

logger = logging.getLogger(__name__)

bp = Blueprint("system", __name__, url_prefix="/api/system")

# Freshness thresholds (ms) — matches apps/observer/workspace.html:744-749
_CONNECTED_MS = 30_000
_STALE_MS = 120_000

# Version check cache TTL: 24 hours
_VERSION_CACHE_TTL_S = 86400


def collect_version() -> str | None:
    """Return the installed solstone version string."""
    from apps.support.diagnostics import collect_version as _collect_version

    return _collect_version()


def list_observers() -> list[dict]:
    """Return observer metadata records."""
    from apps.observer.utils import list_observers as _list_observers

    return _list_observers()


def _check_latest_version() -> dict[str, Any] | None:
    """Fetch latest release from GitHub. Returns None on any failure."""
    import json as _json
    import urllib.request

    url = "https://api.github.com/repos/solpbc/solstone/releases/latest"
    req = urllib.request.Request(
        url,
        headers={"Accept": "application/vnd.github.v3+json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = _json.loads(resp.read())
        tag = data.get("tag_name", "")
        # Strip leading 'v' if present
        return {"latest": tag.lstrip("v")}
    except Exception:
        logger.debug("GitHub version check failed", exc_info=True)
        return None


def _get_version_info() -> dict[str, Any]:
    """Get version info with cached GitHub check."""
    from think.awareness import get_current, update_state

    current = collect_version() or "unknown"
    result: dict[str, Any] = {"current": current}

    # Check cache
    awareness = get_current()
    cached = awareness.get("version", {})
    checked_at = cached.get("checked_at", 0)
    now = time.time()

    if now - checked_at < _VERSION_CACHE_TTL_S and "latest" in cached:
        result["latest"] = cached["latest"]
    else:
        # Fetch fresh
        fresh = _check_latest_version()
        if fresh:
            result["latest"] = fresh["latest"]
            update_state(
                "version",
                {
                    "latest": fresh["latest"],
                    "checked_at": now,
                },
            )
        elif "latest" in cached:
            # Use stale cache on failure
            result["latest"] = cached["latest"]

    if "latest" in result:
        result["update_available"] = result["latest"] != current

    return result


def _get_capture_health() -> dict[str, Any]:
    """Derive capture health from observer last_seen timestamps."""
    from think.utils import now_ms

    observers = list_observers()
    # Filter to active (non-revoked, enabled) observers
    active = [
        o for o in observers if not o.get("revoked", False) and o.get("enabled", True)
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


@bp.route("/status")
def system_status():
    """Return system health: version, capture status, overall ok."""
    version = _get_version_info()
    capture = _get_capture_health()

    ok = capture["status"] in ("active", "no_observers")

    return jsonify(
        {
            "version": version,
            "capture": capture,
            "ok": ok,
        }
    )
