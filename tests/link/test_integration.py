# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path

import pytest

from tests.link.client import Client, StreamResetError
from tests.link.live_helpers import (
    CONVEY_PASSWORD,
    RELAY_URL,
    list_devices,
    running_convey_server,
    running_link_service,
    skip_unless_live_relay,
    unpair_device,
)

pytestmark = pytest.mark.integration
skip_unless_live_relay()


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_pair_enroll_dial_roundtrip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tmp_journal = tmp_path / "journal"
    tmp_journal.mkdir()
    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_journal))

    with (
        running_convey_server(tmp_journal) as base_url,
        running_link_service(tmp_journal),
    ):
        identity = Client.pair(base_url, device_label="pytest-device")
        assert identity.home_instance_id
        assert identity.client_cert_pem.startswith("-----BEGIN CERTIFICATE-----")
        assert len(identity.home_attestation.split(".")) == 3

        enrolled = Client.enroll_device(RELAY_URL, identity)
        assert enrolled.device_token

        before = next(
            device
            for device in list_devices(base_url)
            if device["fingerprint"] == identity.fingerprint
        )
        assert before["last_seen_at"] is None

        session = await Client.dial(RELAY_URL, enrolled)
        async with session:
            auth = base64.b64encode(f":{CONVEY_PASSWORD}".encode("utf-8")).decode(
                "ascii"
            )
            status, headers, body = await session.request(
                "GET",
                "/",
                headers={"authorization": f"Basic {auth}"},
            )
            assert status == 302
            assert headers["location"] == "/app/home/"

            status, headers, body = await session.request(
                "GET",
                "/app/link/api/status",
                headers={"authorization": f"Basic {auth}"},
            )
        assert status == 200
        assert headers["content-type"] == "application/json"
        status_payload = json.loads(body)
        assert status_payload["instance_id"] == identity.home_instance_id

        after = next(
            device
            for device in list_devices(base_url)
            if device["fingerprint"] == identity.fingerprint
        )
        assert after["last_seen_at"] is not None

        unpaired = unpair_device(base_url, identity.fingerprint)
        assert unpaired["unpaired"] == identity.fingerprint

        await asyncio.sleep(1)
        failed = await Client.dial(RELAY_URL, enrolled)
        async with failed:
            with pytest.raises(StreamResetError):
                auth = base64.b64encode(f":{CONVEY_PASSWORD}".encode("utf-8")).decode(
                    "ascii"
                )
                await failed.request(
                    "GET",
                    "/app/link/api/status",
                    headers={"authorization": f"Basic {auth}"},
                )
