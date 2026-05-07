# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

from solstone.convey import create_app
from solstone.think.pairing.devices import load_devices


def _owner_login(client) -> None:
    with client.session_transaction() as session:
        session["logged_in"] = True
        session.permanent = True


def _public_key() -> str:
    return (
        ed25519.Ed25519PrivateKey.generate()
        .public_key()
        .public_bytes(
            encoding=serialization.Encoding.OpenSSH,
            format=serialization.PublicFormat.OpenSSH,
        )
        .decode("utf-8")
    )


def test_pairing_round_trip(journal_copy):
    app = create_app(str(journal_copy))
    app.config["TESTING"] = True

    owner_client = app.test_client()
    _owner_login(owner_client)
    bearer_client = app.test_client()
    public_key = _public_key()

    create_response = owner_client.post("/api/pairing/create", json={})
    create_body = create_response.get_json()
    token = create_body["token"]

    confirm_payload = {
        "token": token,
        "public_key": public_key,
        "device_name": "Round Trip Phone",
        "platform": "ios",
        "bundle_id": "org.solpbc.solstone-swift",
        "app_version": "0.1.0",
    }
    confirm_response = owner_client.post("/api/pairing/confirm", json=confirm_payload)
    confirm_body = confirm_response.get_json()
    session_key = confirm_body["session_key"]
    device_id = confirm_body["device_id"]

    assert create_response.status_code == 200
    assert confirm_response.status_code == 200
    assert session_key.startswith("dsk_")
    assert device_id.startswith("dev_")

    second_confirm = owner_client.post("/api/pairing/confirm", json=confirm_payload)
    assert second_confirm.status_code == 410
    assert second_confirm.get_json()["reason"] == "token_consumed"

    wrong_bearer = bearer_client.get(
        "/api/pairing/devices",
        headers={
            "Authorization": "Bearer dsk_wrong",
            "X-Forwarded-For": "1.2.3.4",
        },
    )
    assert wrong_bearer.status_code == 401
    assert wrong_bearer.get_json()["reason"] == "auth_required"

    bearer_list = bearer_client.get(
        "/api/pairing/devices",
        headers={
            "Authorization": f"Bearer {session_key}",
            "X-Forwarded-For": "1.2.3.4",
        },
    )
    assert bearer_list.status_code == 200
    assert bearer_list.get_json()["devices"] == [
        {
            "id": device_id,
            "name": "Round Trip Phone",
            "platform": "ios",
            "paired_at": load_devices()[0]["paired_at"],
            "last_seen_at": None,
        }
    ]

    heartbeat = bearer_client.post(
        "/api/pairing/heartbeat",
        headers={
            "Authorization": f"Bearer {session_key}",
            "X-Forwarded-For": "1.2.3.4",
        },
        json={},
    )
    assert heartbeat.status_code == 200
    assert heartbeat.get_json() == {"ok": True}

    owner_list = owner_client.get("/api/pairing/devices")
    devices = owner_list.get_json()["devices"]
    assert owner_list.status_code == 200
    assert devices[0]["id"] == device_id
    assert devices[0]["last_seen_at"] is not None

    unpair = owner_client.delete(f"/api/pairing/devices/{device_id}")
    assert unpair.status_code == 200
    assert unpair.get_json() == {"unpaired": True}
    assert load_devices() == []

    stale_bearer = bearer_client.get(
        "/api/pairing/devices",
        headers={
            "Authorization": f"Bearer {session_key}",
            "X-Forwarded-For": "1.2.3.4",
        },
    )
    assert stale_bearer.status_code == 401
    assert stale_bearer.get_json()["reason"] == "auth_required"
