# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import requests
from flask import Flask, g, jsonify

from tests.link.client import Client, EnrolledDevice
from tests.link.live_helpers import (
    CONVEY_PASSWORD,
    running_convey_server,
)

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_pl_identity_stamp_via_direct_dial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tmp_journal = tmp_path / "journal"
    tmp_journal.mkdir()
    monkeypatch.setenv("SOLSTONE_JOURNAL", str(tmp_journal))

    with running_convey_server(
        tmp_journal,
        configure_app=_install_identity_probe,
    ) as base_url:
        identity = Client.pair(base_url, device_label="pytest-identity-device")
        enrolled = EnrolledDevice(device_token="", identity=identity)

        session = await Client.dial_direct("127.0.0.1", enrolled)
        async with session:
            status, _headers, body = await session.request("GET", "/test/identity")

    assert status == 200
    payload = json.loads(body)
    assert payload["mode"] == "pl-via-spl"
    assert payload["fingerprint"] == identity.fingerprint
    assert payload["device_label"] == "pytest-identity-device"
    assert payload["paired_at"] is not None
    assert payload["session_id"] is None


def test_dl_identity_synthesis(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tmp_journal = tmp_path / "journal"
    tmp_journal.mkdir()
    monkeypatch.setenv("SOLSTONE_JOURNAL", str(tmp_journal))

    with running_convey_server(
        tmp_journal,
        configure_app=_install_identity_probe,
    ) as base_url:
        session = requests.Session()
        login = session.post(
            f"{base_url}/login",
            data={"password": CONVEY_PASSWORD},
            allow_redirects=False,
            timeout=10,
        )
        assert login.status_code == 302
        response = session.get(f"{base_url}/test/identity", timeout=10)

    response.raise_for_status()
    payload = response.json()
    assert payload == {
        "mode": "dl",
        "fingerprint": None,
        "device_label": None,
        "paired_at": None,
        "session_id": None,
    }


def _install_identity_probe(app: Flask) -> None:
    @app.get("/test/identity")
    def test_identity() -> Any:
        identity = g.identity
        return jsonify(
            {
                "mode": identity.mode,
                "fingerprint": identity.fingerprint,
                "device_label": identity.device_label,
                "paired_at": identity.paired_at,
                "session_id": identity.session_id,
            }
        )
