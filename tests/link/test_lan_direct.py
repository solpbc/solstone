# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

import asyncio
import base64
import os
import platform
import subprocess
import urllib.parse
from pathlib import Path

import pytest

from tests.link.client import Client, EnrolledDevice, TlsError
from tests.link.live_helpers import (
    CONVEY_PASSWORD,
    list_devices,
    running_convey_server,
    unpair_device,
)

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_pair_enroll_direct_dial_roundtrip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tmp_journal = tmp_path / "journal"
    tmp_journal.mkdir()
    monkeypatch.setenv("SOLSTONE_JOURNAL", str(tmp_journal))

    with running_convey_server(tmp_journal) as base_url:
        identity = Client.pair(base_url, device_label="pytest-direct-device")
        enrolled = EnrolledDevice(device_token="", identity=identity)

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

        _assert_convey_owns_both_ports(os.getpid(), _port_from_base_url(base_url))

        unpaired = unpair_device(base_url, identity.fingerprint)
        assert unpaired["unpaired"] == identity.fingerprint

        await asyncio.sleep(1)
        with pytest.raises((TlsError, ConnectionError, OSError, asyncio.TimeoutError)):
            rejected = await Client.dial_direct("127.0.0.1", enrolled)
            async with rejected:
                await rejected.request("GET", "/")


def _port_from_base_url(base_url: str) -> int:
    parsed = urllib.parse.urlparse(base_url)
    assert parsed.port is not None
    return parsed.port


def _assert_convey_owns_both_ports(
    convey_pid: int,
    dl_port: int,
    pl_port: int = 7657,
    link_pid: int | None = None,
) -> None:
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
    convey_token = f"pid={convey_pid},"
    lines = result.stdout.splitlines()
    dl_lines = [
        line for line in lines if convey_token in line and f":{dl_port} " in line
    ]
    pl_lines = [
        line for line in lines if convey_token in line and f":{pl_port} " in line
    ]
    assert len(dl_lines) == 1, result.stdout
    assert len(pl_lines) == 1, result.stdout
    if link_pid is not None:
        link_token = f"pid={link_pid},"
        link_lines = [line for line in lines if link_token in line]
        assert not link_lines, result.stdout
