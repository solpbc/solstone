# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

import json

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

from convey import create_app
from think.pairing.devices import find_device_by_id, load_devices
from think.pairing.tokens import create_token as mint_pairing_token


def _write_config(journal_copy, payload: dict) -> None:
    (journal_copy / "config" / "journal.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def _read_config(journal_copy) -> dict:
    return json.loads((journal_copy / "config" / "journal.json").read_text("utf-8"))


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


@pytest.fixture
def pairing_app(journal_copy):
    app = create_app(str(journal_copy))
    app.config["TESTING"] = True
    return app


@pytest.fixture
def pairing_client(pairing_app):
    client = pairing_app.test_client()
    _owner_login(client)
    return client


def test_create_token_happy_path(pairing_client):
    response = pairing_client.post("/api/pairing/create", json={})

    body = response.get_json()
    assert response.status_code == 200
    assert body["token"].startswith("ptk_")
    assert body["expires_at"] > 0
    assert body["pairing_url"].startswith("solstone://pair?token=")
    assert body["qr_data"] == body["pairing_url"]


def test_create_token_rejects_non_object(pairing_client):
    response = pairing_client.post("/api/pairing/create", json=["bad"])

    assert response.status_code == 400
    assert response.get_json() == {
        "error": "request body must be a JSON object",
        "reason": "invalid_request",
    }


def test_confirm_pairing_happy_path(pairing_client, journal_copy):
    config = _read_config(journal_copy)
    config["identity"] = {"name": "Sol", "preferred": "Preferred Sol"}
    _write_config(journal_copy, config)

    create_response = pairing_client.post("/api/pairing/create", json={})
    token = create_response.get_json()["token"]
    public_key = _public_key()

    response = pairing_client.post(
        "/api/pairing/confirm",
        json={
            "token": token,
            "public_key": public_key,
            "device_name": "Phone",
            "platform": "ios",
            "bundle_id": "org.solpbc.solstone-swift",
            "app_version": "0.1.0",
        },
    )

    body = response.get_json()
    assert response.status_code == 200
    assert body["session_key"].startswith("dsk_")
    assert body["device_id"].startswith("dev_")
    assert body["journal_root"] == str(journal_copy.resolve())
    assert body["owner_identity"] == "Preferred Sol"
    assert body["server_version"] == "unknown"
    stored = find_device_by_id(body["device_id"])
    assert stored is not None
    assert stored["session_key_hash"] != body["session_key"]


def test_confirm_pairing_rejects_bad_public_key(pairing_client):
    token = pairing_client.post("/api/pairing/create", json={}).get_json()["token"]

    response = pairing_client.post(
        "/api/pairing/confirm",
        json={
            "token": token,
            "public_key": "ssh-ed25519 bad",
            "device_name": "Phone",
            "platform": "ios",
            "bundle_id": "org.solpbc.solstone-swift",
            "app_version": "0.1.0",
        },
    )

    assert response.status_code == 400
    assert response.get_json() == {
        "error": "public_key must be a valid ssh-ed25519 key",
        "reason": "invalid_public_key",
    }


def test_confirm_pairing_distinguishes_expired_and_consumed_tokens(
    pairing_client, monkeypatch
):
    expired = mint_pairing_token(ttl_seconds=60, now=1000)
    monkeypatch.setattr("convey.pairing.time.time", lambda: 1060)

    expired_response = pairing_client.post(
        "/api/pairing/confirm",
        json={
            "token": expired.token,
            "public_key": _public_key(),
            "device_name": "Phone",
            "platform": "ios",
            "bundle_id": "org.solpbc.solstone-swift",
            "app_version": "0.1.0",
        },
    )

    assert expired_response.status_code == 410
    assert expired_response.get_json()["reason"] == "token_expired"

    token = pairing_client.post("/api/pairing/create", json={}).get_json()["token"]
    body = {
        "token": token,
        "public_key": _public_key(),
        "device_name": "Phone",
        "platform": "ios",
        "bundle_id": "org.solpbc.solstone-swift",
        "app_version": "0.1.0",
    }
    assert pairing_client.post("/api/pairing/confirm", json=body).status_code == 200

    consumed_response = pairing_client.post("/api/pairing/confirm", json=body)

    assert consumed_response.status_code == 410
    assert consumed_response.get_json()["reason"] == "token_consumed"


def test_heartbeat_requires_valid_bearer(pairing_client):
    response = pairing_client.post("/api/pairing/heartbeat")

    assert response.status_code == 401
    assert response.get_json() == {
        "error": "paired device required",
        "reason": "auth_required",
    }


def test_list_devices_allows_bearer_or_owner(pairing_client):
    confirm = pairing_client.post(
        "/api/pairing/confirm",
        json={
            "token": pairing_client.post("/api/pairing/create", json={}).get_json()[
                "token"
            ],
            "public_key": _public_key(),
            "device_name": "Phone",
            "platform": "ios",
            "bundle_id": "org.solpbc.solstone-swift",
            "app_version": "0.1.0",
        },
    ).get_json()

    owner_response = pairing_client.get("/api/pairing/devices")
    bearer_response = pairing_client.get(
        "/api/pairing/devices",
        headers={"Authorization": f"Bearer {confirm['session_key']}"},
    )
    anon_client = pairing_client.application.test_client()
    anon_response = anon_client.get(
        "/api/pairing/devices", headers={"X-Forwarded-For": "1.2.3.4"}
    )

    assert owner_response.status_code == 200
    assert bearer_response.status_code == 200
    assert owner_response.get_json() == bearer_response.get_json()
    assert owner_response.get_json()["devices"] == [
        {
            "id": confirm["device_id"],
            "name": "Phone",
            "platform": "ios",
            "paired_at": load_devices()[0]["paired_at"],
            "last_seen_at": None,
        }
    ]
    assert anon_response.status_code == 401
    assert anon_response.get_json()["reason"] == "auth_required"


def test_unpair_device_returns_404_for_unknown_device(pairing_client):
    response = pairing_client.delete("/api/pairing/devices/dev_missing")

    assert response.status_code == 404
    assert response.get_json() == {
        "error": "paired device not found",
        "reason": "device_not_found",
    }
