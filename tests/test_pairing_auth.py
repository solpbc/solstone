# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

import base64
import json

from flask import Flask, g, jsonify, session
from werkzeug.security import generate_password_hash

from convey.auth import (
    extract_bearer_token,
    is_owner_authed,
    require_paired_device,
    resolve_paired_device,
)
from think.pairing.devices import register_device
from think.pairing.keys import hash_session_key


def _write_config(journal_copy, payload: dict) -> None:
    (journal_copy / "config" / "journal.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def _read_config(journal_copy) -> dict:
    return json.loads((journal_copy / "config" / "journal.json").read_text("utf-8"))


def _create_app() -> Flask:
    app = Flask(__name__)
    app.secret_key = "test-secret"
    app.config["TESTING"] = True

    @app.get("/bearer")
    def bearer_view():
        return jsonify(
            {
                "token": extract_bearer_token(),
                "device_id": (resolve_paired_device() or {}).get("id"),
            }
        )

    @app.get("/owner")
    def owner_view():
        return jsonify({"owner_authed": is_owner_authed()})

    @app.get("/paired")
    @require_paired_device
    def paired_view():
        return jsonify({"device_id": g.paired_device["id"]})

    @app.get("/session-login")
    def session_login():
        session["logged_in"] = True
        session.permanent = True
        return jsonify({"ok": True})

    return app


def test_extract_bearer_token_handles_missing_and_malformed_headers(journal_copy):
    app = _create_app()
    client = app.test_client()

    assert client.get("/bearer").get_json() == {"token": None, "device_id": None}
    assert client.get("/bearer", headers={"Authorization": "Bearer"}).get_json() == {
        "token": None,
        "device_id": None,
    }
    assert client.get("/bearer", headers={"Authorization": "Basic abc"}).get_json() == {
        "token": None,
        "device_id": None,
    }


def test_resolve_paired_device_requires_matching_hash(journal_copy):
    app = _create_app()
    client = app.test_client()
    register_device(
        name="Phone",
        platform="ios",
        public_key="ssh-ed25519 AAAAauth",
        session_key_hash=hash_session_key("dsk_real"),
        bundle_id="org.solpbc.solstone-swift",
        app_version="0.1.0",
        paired_at="2026-04-20T15:31:02Z",
    )

    missing = client.get("/bearer", headers={"Authorization": "Bearer dsk_wrong"})
    found = client.get("/bearer", headers={"Authorization": "Bearer dsk_real"})

    assert missing.get_json() == {"token": "dsk_wrong", "device_id": None}
    assert found.get_json()["token"] == "dsk_real"
    assert found.get_json()["device_id"].startswith("dev_")


def test_is_owner_authed_via_basic_auth(journal_copy):
    payload = _read_config(journal_copy)
    payload["convey"]["password_hash"] = generate_password_hash("test123")
    payload["setup"] = {"completed_at": 1700000000000}
    _write_config(journal_copy, payload)

    app = _create_app()
    client = app.test_client()
    creds = base64.b64encode(b":test123").decode("ascii")

    response = client.get(
        "/owner",
        headers={
            "Authorization": f"Basic {creds}",
            "X-Forwarded-For": "1.2.3.4",
        },
    )

    assert response.get_json() == {"owner_authed": True}


def test_is_owner_authed_via_session_cookie(journal_copy):
    app = _create_app()
    client = app.test_client()

    client.get("/session-login")
    response = client.get("/owner")

    assert response.get_json() == {"owner_authed": True}


def test_is_owner_authed_via_trust_localhost(journal_copy):
    payload = _read_config(journal_copy)
    payload["convey"]["trust_localhost"] = True
    payload["setup"] = {"completed_at": 1700000000000}
    payload["convey"].pop("password_hash", None)
    _write_config(journal_copy, payload)

    app = _create_app()
    client = app.test_client()
    response = client.get("/owner")

    assert response.get_json() == {"owner_authed": True}


def test_require_paired_device_returns_401_json_without_bearer(journal_copy):
    app = _create_app()
    client = app.test_client()

    response = client.get("/paired")

    assert response.status_code == 401
    assert response.get_json() == {
        "error": "paired device required",
        "reason": "auth_required",
    }


def test_require_paired_device_sets_g_paired_device(journal_copy):
    device = register_device(
        name="Phone",
        platform="ios",
        public_key="ssh-ed25519 AAAApaired",
        session_key_hash=hash_session_key("dsk_paired"),
        bundle_id="org.solpbc.solstone-swift",
        app_version="0.1.0",
        paired_at="2026-04-20T15:31:02Z",
    )
    app = _create_app()
    client = app.test_client()

    response = client.get("/paired", headers={"Authorization": "Bearer dsk_paired"})

    assert response.status_code == 200
    assert response.get_json() == {"device_id": device["id"]}
