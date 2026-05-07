# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

import json

from solstone.think.pairing import devices


def _devices_path(journal_copy):
    return journal_copy / "config" / "paired_devices.json"


def test_load_devices_returns_empty_for_missing_store(journal_copy):
    assert devices.load_devices() == []


def test_register_load_and_remove_round_trip(journal_copy):
    device = devices.register_device(
        name="Phone",
        platform="ios",
        public_key="ssh-ed25519 AAAAphone",
        session_key_hash="sha256:abc",
        bundle_id="org.solpbc.solstone-swift",
        app_version="0.1.0",
        paired_at="2026-04-20T15:31:02Z",
    )

    loaded = devices.load_devices()

    assert loaded == [
        {
            "id": device["id"],
            "name": "Phone",
            "platform": "ios",
            "public_key": "ssh-ed25519 AAAAphone",
            "session_key_hash": "sha256:abc",
            "bundle_id": "org.solpbc.solstone-swift",
            "app_version": "0.1.0",
            "paired_at": "2026-04-20T15:31:02Z",
            "last_seen_at": None,
        }
    ]
    assert devices.find_device_by_id(device["id"]) == loaded[0]
    assert devices.find_device_by_session_key_hash("sha256:abc") == loaded[0]
    assert devices.remove_device(device["id"]) is True
    assert devices.load_devices() == []


def test_register_device_upserts_by_public_key(journal_copy):
    first = devices.register_device(
        name="Phone",
        platform="ios",
        public_key="ssh-ed25519 AAAAsame",
        session_key_hash="sha256:first",
        bundle_id="org.solpbc.solstone-swift",
        app_version="0.1.0",
        paired_at="2026-04-20T15:31:02Z",
    )
    second = devices.register_device(
        name="Phone 2",
        platform="ios",
        public_key="ssh-ed25519 AAAAsame",
        session_key_hash="sha256:second",
        bundle_id="org.solpbc.solstone-swift",
        app_version="0.2.0",
        paired_at="2026-04-20T16:00:00Z",
    )

    assert first["id"] == second["id"]
    assert devices.load_devices() == [
        {
            "id": first["id"],
            "name": "Phone 2",
            "platform": "ios",
            "public_key": "ssh-ed25519 AAAAsame",
            "session_key_hash": "sha256:second",
            "bundle_id": "org.solpbc.solstone-swift",
            "app_version": "0.2.0",
            "paired_at": "2026-04-20T16:00:00Z",
            "last_seen_at": None,
        }
    ]


def test_touch_last_seen_updates_existing_device(journal_copy):
    device = devices.register_device(
        name="Phone",
        platform="ios",
        public_key="ssh-ed25519 AAAAseen",
        session_key_hash="sha256:seen",
        bundle_id="org.solpbc.solstone-swift",
        app_version="0.1.0",
        paired_at="2026-04-20T15:31:02Z",
    )

    touched = devices.touch_last_seen(device["id"], last_seen_at="2026-04-20T16:01:00Z")

    assert touched is True
    assert devices.find_device_by_id(device["id"]) is not None
    assert (
        devices.find_device_by_id(device["id"])["last_seen_at"]
        == "2026-04-20T16:01:00Z"
    )


def test_load_devices_recovers_from_malformed_store(journal_copy, caplog):
    path = _devices_path(journal_copy)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"devices": "bad"}', encoding="utf-8")

    assert devices.load_devices() == []
    assert "paired device store unreadable" in caplog.text

    healed = devices.register_device(
        name="Phone",
        platform="ios",
        public_key="ssh-ed25519 AAAAheal",
        session_key_hash="sha256:heal",
        bundle_id="org.solpbc.solstone-swift",
        app_version="0.1.0",
        paired_at="2026-04-20T15:31:02Z",
    )
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload == {
        "devices": [
            {
                "id": healed["id"],
                "name": "Phone",
                "platform": "ios",
                "public_key": "ssh-ed25519 AAAAheal",
                "session_key_hash": "sha256:heal",
                "bundle_id": "org.solpbc.solstone-swift",
                "app_version": "0.1.0",
                "paired_at": "2026-04-20T15:31:02Z",
                "last_seen_at": None,
            }
        ]
    }


def test_status_view_redacts_secret_fields(journal_copy):
    device = devices.register_device(
        name="Phone",
        platform="ios",
        public_key="ssh-ed25519 AAAAredact",
        session_key_hash="sha256:redact",
        bundle_id="org.solpbc.solstone-swift",
        app_version="0.1.0",
        paired_at="2026-04-20T15:31:02Z",
    )

    assert devices.status_view(device) == {
        "id": device["id"],
        "name": "Phone",
        "platform": "ios",
        "paired_at": "2026-04-20T15:31:02Z",
        "last_seen_at": None,
    }
