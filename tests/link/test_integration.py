# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path

import pytest
from wsproto import WSConnection
from wsproto.connection import ConnectionType
from wsproto.events import AcceptConnection, CloseConnection, Request, TextMessage

import convey.bridge as convey_bridge
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
    monkeypatch.setenv("SOLSTONE_JOURNAL", str(tmp_journal))

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


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_websocket_upgrade_and_bidirectional_flow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tmp_journal = tmp_path / "journal"
    tmp_journal.mkdir()
    monkeypatch.setenv("SOLSTONE_JOURNAL", str(tmp_journal))

    with (
        running_convey_server(tmp_journal) as base_url,
        running_link_service(tmp_journal),
    ):
        identity = Client.pair(base_url, device_label="pytest-device")
        enrolled = Client.enroll_device(RELAY_URL, identity)

        session = await Client.dial(RELAY_URL, enrolled)
        async with session:
            raw_stream = await session._mux.open_stream()
            ws = WSConnection(ConnectionType.CLIENT)
            await raw_stream.write(
                ws.send(Request(host="127.0.0.1", target="/ws/events"))
            )

            accepted = False
            while not accepted:
                chunk = await asyncio.wait_for(_read_next_chunk(raw_stream), timeout=5)
                ws.receive_data(chunk)
                for event in ws.events():
                    if isinstance(event, AcceptConnection):
                        accepted = True
                        break

            await _wait_for_ws_client()
            convey_bridge._broadcast_to_websockets({"tract": "test", "event": "ping"})

            payload = None
            while payload is None:
                chunk = await asyncio.wait_for(_read_next_chunk(raw_stream), timeout=5)
                ws.receive_data(chunk)
                for event in ws.events():
                    if isinstance(event, TextMessage):
                        payload = event.data
                        break

            assert payload == json.dumps({"tract": "test", "event": "ping"})

            await raw_stream.write(ws.send(CloseConnection(code=1000)))
            await raw_stream.close()


async def _read_next_chunk(raw_stream: object) -> bytes:
    return await anext(raw_stream.read())


async def _wait_for_ws_client() -> None:
    deadline = asyncio.get_running_loop().time() + 5
    while asyncio.get_running_loop().time() < deadline:
        if convey_bridge._WEBSOCKET_CLIENTS:
            return
        await asyncio.sleep(0.05)
    raise AssertionError("websocket client never registered")
