# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import AsyncIterator, Callable
from pathlib import Path

import pytest

from solstone.convey import bridge as convey_bridge
from tests.link.client import Client, EnrolledDevice, _http_request_bytes
from tests.link.live_helpers import running_convey_server

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_pl_sse_disconnect_unregisters_subscriber(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tmp_journal = tmp_path / "journal"
    tmp_journal.mkdir()
    monkeypatch.setenv("SOLSTONE_JOURNAL", str(tmp_journal))

    with running_convey_server(tmp_journal) as base_url:
        identity = Client.pair(base_url, device_label="pytest-sse-device")
        enrolled = EnrolledDevice(device_token="", identity=identity)
        baseline_subscriptions = convey_bridge.subscription_count("convey-ui")

        session = await Client.dial_direct("127.0.0.1", enrolled)
        async with session:
            stream = await session._mux.open_stream(  # noqa: SLF001
                _http_request_bytes(
                    "GET",
                    "/sse/events",
                    headers={"Accept": "text/event-stream"},
                    body=b"",
                )
            )
            await stream.close()
            reader = stream.read()
            received = bytearray()

            await _read_until(reader, received, b": heartbeat\n\n")
            convey_bridge._broadcast_callosum_event(  # noqa: SLF001
                {"tract": "pytest", "event": "pl_sse_probe", "ts": 1}
            )
            await _read_until(reader, received, b"pl_sse_probe")

            await stream.reset()
            _wait_until(
                lambda: (
                    convey_bridge.subscription_count("convey-ui")
                    == baseline_subscriptions
                ),
                timeout=2.0,
            )

    _wait_until(lambda: not _secure_listener_wsgi_threads(), timeout=2.0)


async def _read_until(
    reader: AsyncIterator[bytes],
    received: bytearray,
    needle: bytes,
) -> None:
    async with asyncio.timeout(5.0):
        while needle not in received:
            received.extend(await anext(reader))


def _wait_until(check: Callable[[], bool], *, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if check():
            return
        time.sleep(0.05)
    raise AssertionError("timed out waiting for condition")


def _secure_listener_wsgi_threads() -> list[threading.Thread]:
    return [
        thread
        for thread in threading.enumerate()
        if thread.name.startswith("secure-listener-wsgi")
    ]
