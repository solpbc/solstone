# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

import json
from pathlib import Path

from think.push import devices


def _devices_path(tmp_path: Path) -> Path:
    return tmp_path / "config" / "push_devices.json"


def test_load_devices_returns_empty_for_missing_store(monkeypatch, tmp_path):
    monkeypatch.setenv("SOLSTONE_JOURNAL", str(tmp_path))

    assert devices.load_devices() == []


def test_register_load_remove_round_trip(monkeypatch, tmp_path):
    monkeypatch.setenv("SOLSTONE_JOURNAL", str(tmp_path))

    count = devices.register_device(
        token="a" * 64,
        bundle_id="org.solpbc.solstone-swift",
        environment="development",
        platform="ios",
    )

    assert count == 1
    stored = devices.load_devices()
    assert stored == [
        {
            "token": "a" * 64,
            "bundle_id": "org.solpbc.solstone-swift",
            "environment": "development",
            "platform": "ios",
            "registered_at": stored[0]["registered_at"],
        }
    ]

    removed = devices.remove_device("a" * 64)
    assert removed is True
    assert devices.load_devices() == []


def test_register_device_updates_existing_token(monkeypatch, tmp_path):
    monkeypatch.setenv("SOLSTONE_JOURNAL", str(tmp_path))
    times = iter([1000, 2000])
    monkeypatch.setattr(devices.time, "time", lambda: next(times))

    first = devices.register_device(
        token="b" * 64,
        bundle_id="org.solpbc.solstone-swift",
        environment="development",
        platform="ios",
    )
    second = devices.register_device(
        token="b" * 64,
        bundle_id="org.solpbc.solstone-swift",
        environment="production",
        platform="ios",
    )

    assert first == 1
    assert second == 1
    assert devices.load_devices() == [
        {
            "token": "b" * 64,
            "bundle_id": "org.solpbc.solstone-swift",
            "environment": "production",
            "platform": "ios",
            "registered_at": 2000,
        }
    ]


def test_remove_device_returns_false_for_unknown_token(monkeypatch, tmp_path):
    monkeypatch.setenv("SOLSTONE_JOURNAL", str(tmp_path))
    devices.register_device(
        token="c" * 64,
        bundle_id="org.solpbc.solstone-swift",
        environment="development",
        platform="ios",
    )

    assert devices.remove_device("d" * 64) is False
    assert len(devices.load_devices()) == 1


def test_load_devices_returns_empty_for_malformed_store(monkeypatch, tmp_path, caplog):
    monkeypatch.setenv("SOLSTONE_JOURNAL", str(tmp_path))
    path = _devices_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"devices": "bad"}', encoding="utf-8")

    loaded = devices.load_devices()

    assert loaded == []
    assert "push device store unreadable" in caplog.text


def test_status_view_masks_token(monkeypatch, tmp_path):
    monkeypatch.setenv("SOLSTONE_JOURNAL", str(tmp_path))
    path = _devices_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "devices": [
                    {
                        "token": "0123456789abcdef",
                        "bundle_id": "org.solpbc.solstone-swift",
                        "environment": "development",
                        "platform": "ios",
                        "registered_at": 1713528000,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    device = devices.load_devices()[0]
    view = devices.status_view(device)

    assert devices.mask_token("0123456789abcdef") == "...cdef"
    assert view == {
        "token_suffix": "...cdef",
        "bundle_id": "org.solpbc.solstone-swift",
        "environment": "development",
        "platform": "ios",
        "registered_at": "2024-04-19T12:00:00Z",
    }
