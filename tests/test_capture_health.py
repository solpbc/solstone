# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for live capture-health derivation."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from solstone.think.capture_health import get_capture_health


def test_no_last_seen_is_offline(monkeypatch):
    monkeypatch.setattr(
        "solstone.apps.observer.utils.list_observers",
        lambda: [{"name": "x", "enabled": True}],
    )

    result = get_capture_health()

    assert result["observers"][0]["status"] == "offline"


def test_disabled_observers_excluded(monkeypatch):
    monkeypatch.setattr(
        "solstone.apps.observer.utils.list_observers",
        lambda: [{"name": "x", "last_seen": 1000, "enabled": False}],
    )

    result = get_capture_health()

    assert result["status"] == "no_observers"


def test_list_observers_raises_returns_unknown(monkeypatch):
    def _raise() -> list[dict]:
        raise RuntimeError("boom")

    monkeypatch.setattr("solstone.apps.observer.utils.list_observers", _raise)

    result = get_capture_health()

    assert result == {"status": "unknown", "observers": []}
