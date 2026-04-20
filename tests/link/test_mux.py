# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

import asyncio

import pytest

from think.link.framing import (
    FLAG_CLOSE,
    FLAG_DATA,
    FLAG_RESET,
    Frame,
    FrameDecoder,
    build_close,
    build_data,
    build_open,
)
from think.link.mux import Multiplexer


def _decode_frames(chunks: list[bytes]) -> list[Frame]:
    decoder = FrameDecoder()
    for chunk in chunks:
        decoder.feed(chunk)
    return decoder.drain()


@pytest.mark.asyncio
async def test_open_with_initial_payload_hits_handler() -> None:
    handler_seen: dict[int, bytes] = {}

    async def handler(
        reader: asyncio.StreamReader, writer: object
    ) -> None:  # pragma: no cover - typed by mux
        data = await reader.readuntil(b"\n")
        handler_seen[1] = data
        await writer.write(b"ack\n")  # type: ignore[attr-defined]
        await writer.close()  # type: ignore[attr-defined]

    sent: list[bytes] = []

    async def send(data: bytes) -> None:
        sent.append(data)

    mux = Multiplexer(send, handler, is_listener=True)
    await mux.feed(build_open(1, b"hello\n").encode() + build_close(1).encode())

    for _ in range(20):
        await asyncio.sleep(0.005)
        if handler_seen.get(1):
            break

    assert handler_seen.get(1) == b"hello\n"

    frames = _decode_frames(sent)
    flags = [frame.flags for frame in frames]
    assert any(flag & FLAG_DATA for flag in flags)
    assert any(flag & FLAG_CLOSE for flag in flags)
    assert (
        b"".join(frame.payload for frame in frames if frame.flags & FLAG_DATA)
        == b"ack\n"
    )
    await mux.close()


@pytest.mark.asyncio
async def test_wrong_parity_stream_id_gets_reset() -> None:
    sent: list[bytes] = []

    async def send(data: bytes) -> None:
        sent.append(data)

    async def handler(*_: object) -> None:
        pytest.fail("handler should not be reached for wrong-parity stream ids")

    mux = Multiplexer(send, handler, is_listener=True)
    await mux.feed(build_open(2).encode())

    frames = _decode_frames(sent)
    assert any(frame.stream_id == 2 and frame.flags & FLAG_RESET for frame in frames)
    await mux.close()


@pytest.mark.asyncio
async def test_unknown_stream_data_gets_reset() -> None:
    sent: list[bytes] = []

    async def send(data: bytes) -> None:
        sent.append(data)

    async def handler(*_: object) -> None:
        return

    mux = Multiplexer(send, handler, is_listener=True)
    await mux.feed(build_data(99, b"x").encode())

    frames = _decode_frames(sent)
    assert any(frame.stream_id == 99 and frame.flags & FLAG_RESET for frame in frames)
    await mux.close()


@pytest.mark.asyncio
async def test_concurrent_streams_do_not_interfere() -> None:
    responses: dict[int, bytes] = {}

    async def handler(
        reader: asyncio.StreamReader, writer: object
    ) -> None:  # pragma: no cover - typed by mux
        payload = await reader.readuntil(b"\n")
        await writer.write(payload)  # type: ignore[attr-defined]
        await writer.close()  # type: ignore[attr-defined]

    sent: list[bytes] = []

    async def send(data: bytes) -> None:
        sent.append(data)

    mux = Multiplexer(send, handler, is_listener=True)
    bulk = bytearray()
    for stream_id in (1, 3, 5, 7, 9):
        bulk.extend(build_open(stream_id, f"stream-{stream_id}\n".encode()).encode())
        bulk.extend(build_close(stream_id).encode())

    await mux.feed(bytes(bulk))

    for _ in range(50):
        await asyncio.sleep(0.005)
        frames = _decode_frames(sent)
        for frame in frames:
            if frame.flags & FLAG_DATA:
                responses.setdefault(frame.stream_id, b"")
                responses[frame.stream_id] += frame.payload
        if all(stream_id in responses for stream_id in (1, 3, 5, 7, 9)):
            break

    for stream_id in (1, 3, 5, 7, 9):
        assert responses.get(stream_id) == f"stream-{stream_id}\n".encode()
    await mux.close()


@pytest.mark.asyncio
async def test_validates_open_reopen_is_protocol_error() -> None:
    sent: list[bytes] = []

    async def send(data: bytes) -> None:
        sent.append(data)

    gate = asyncio.Event()

    async def handler(
        reader: asyncio.StreamReader, writer: object
    ) -> None:  # pragma: no cover - typed by mux
        await gate.wait()
        await writer.close()  # type: ignore[attr-defined]

    mux = Multiplexer(send, handler, is_listener=True)
    await mux.feed(build_open(1).encode())
    await asyncio.sleep(0.01)
    await mux.feed(build_open(1).encode())

    frames = _decode_frames(sent)
    assert any(frame.stream_id == 1 and frame.flags & FLAG_RESET for frame in frames)

    gate.set()
    await mux.close()
