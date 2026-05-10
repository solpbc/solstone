# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

import asyncio
import base64
import platform
import subprocess
from pathlib import Path

import pytest

from tests.link.client import Client, TlsError
from tests.link.live_helpers import (
    CONVEY_PASSWORD,
    RELAY_URL,
    LinkProcessCapture,
    list_devices,
    running_convey_server,
    running_link_service,
    skip_unless_live_relay,
    unpair_device,
)

pytestmark = pytest.mark.integration
skip_unless_live_relay()

_TCP_READY_LINE = "tcp listener bound on 0.0.0.0:7657"


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_pair_enroll_direct_dial_roundtrip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tmp_journal = tmp_path / "journal"
    tmp_journal.mkdir()
    monkeypatch.setenv("SOLSTONE_JOURNAL", str(tmp_journal))

    with (
        running_convey_server(tmp_journal) as base_url,
        running_link_service(tmp_journal) as link_capture,
    ):
        link_capture.wait_for_line(_TCP_READY_LINE, timeout=15)
        identity = Client.pair(base_url, device_label="pytest-direct-device")
        enrolled = Client.enroll_device(RELAY_URL, identity)

        before = next(
            device
            for device in list_devices(base_url)
            if device["fingerprint"] == identity.fingerprint
        )
        assert before["last_seen_at"] is None

        session = await Client.dial_direct("127.0.0.1", enrolled)
        async with session:
            auth = base64.b64encode(f":{CONVEY_PASSWORD}".encode("utf-8")).decode(
                "ascii"
            )
            status, headers, _body = await session.request(
                "GET",
                "/",
                headers={"authorization": f"Basic {auth}"},
            )
            assert status == 302
            assert headers["location"] == "/app/home/"

        after = next(
            device
            for device in list_devices(base_url)
            if device["fingerprint"] == identity.fingerprint
        )
        assert after["last_seen_at"] is not None

        _assert_single_tcp_listener(link_capture)

        unpaired = unpair_device(base_url, identity.fingerprint)
        assert unpaired["unpaired"] == identity.fingerprint

        await asyncio.sleep(1)
        with pytest.raises((TlsError, ConnectionError, OSError, asyncio.TimeoutError)):
            await Client.dial_direct("127.0.0.1", enrolled)


def _assert_single_tcp_listener(capture: LinkProcessCapture) -> None:
    if platform.system() != "Linux":
        pytest.skip("ss listener assertion is Linux-only")
    try:
        result = subprocess.run(
            ["ss", "-tlnp"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except FileNotFoundError:
        pytest.skip("ss is not available")
    assert result.returncode == 0, result.stderr
    pid_token = f"pid={capture.proc.pid},"
    lines = [
        line
        for line in result.stdout.splitlines()
        if pid_token in line and ":7657 " in line
    ]
    assert len(lines) == 1, result.stdout
