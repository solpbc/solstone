# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

import json

import pytest
from flask import Flask, g, jsonify
from werkzeug.security import generate_password_hash

from solstone.convey import create_app
from solstone.convey.auth import require_paired_device
from solstone.think.pairing.devices import register_device
from solstone.think.pairing.keys import hash_session_key


def _read_config(journal_copy) -> dict:
    return json.loads((journal_copy / "config" / "journal.json").read_text("utf-8"))


def _write_config(journal_copy, payload: dict) -> None:
    (journal_copy / "config" / "journal.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def _set_auth_matrix_cell(
    journal_copy,
    *,
    allow_network_access: bool,
    trust_localhost: bool,
) -> None:
    payload = _read_config(journal_copy)
    payload["convey"]["allow_network_access"] = allow_network_access
    payload["convey"]["trust_localhost"] = trust_localhost
    payload["convey"]["password_hash"] = generate_password_hash("test123")
    payload["setup"] = {"completed_at": 1700000000000}
    _write_config(journal_copy, payload)


def _paired_device_app() -> Flask:
    app = Flask(__name__)
    app.secret_key = "test-secret"
    app.config["TESTING"] = True

    @app.get("/paired")
    @require_paired_device
    def paired_view():
        return jsonify({"device_id": g.paired_device["id"]})

    return app


@pytest.mark.parametrize("allow_network_access", [False, True])
@pytest.mark.parametrize("trust_localhost", [True, False])
@pytest.mark.parametrize("remote_addr", ["127.0.0.1", "10.0.0.9"])
def test_require_login_matrix(
    journal_copy,
    allow_network_access: bool,
    trust_localhost: bool,
    remote_addr: str,
):
    _set_auth_matrix_cell(
        journal_copy,
        allow_network_access=allow_network_access,
        trust_localhost=trust_localhost,
    )
    client = create_app(str(journal_copy)).test_client()

    response = client.get("/", environ_overrides={"REMOTE_ADDR": remote_addr})

    assert response.status_code == 302
    if remote_addr == "127.0.0.1" and trust_localhost:
        assert "/login" not in response.headers["Location"]
        assert "/init" not in response.headers["Location"]
    else:
        assert "/login" in response.headers["Location"]


@pytest.mark.parametrize("allow_network_access", [False, True])
@pytest.mark.parametrize("trust_localhost", [True, False])
@pytest.mark.parametrize("remote_addr", ["127.0.0.1", "10.0.0.9"])
def test_bearer_auth_succeeds_in_every_matrix_cell(
    journal_copy,
    allow_network_access: bool,
    trust_localhost: bool,
    remote_addr: str,
):
    _set_auth_matrix_cell(
        journal_copy,
        allow_network_access=allow_network_access,
        trust_localhost=trust_localhost,
    )
    device = register_device(
        name="Phone",
        platform="ios",
        public_key="ssh-ed25519 AAAAbearer",
        session_key_hash=hash_session_key("dsk_real"),
        bundle_id="org.solpbc.solstone-swift",
        app_version="0.1.0",
        paired_at="2026-04-20T15:31:02Z",
    )
    client = _paired_device_app().test_client()

    response = client.get(
        "/paired",
        headers={"Authorization": "Bearer dsk_real"},
        environ_overrides={"REMOTE_ADDR": remote_addr},
    )

    assert response.status_code == 200
    assert response.get_json() == {"device_id": device["id"]}
