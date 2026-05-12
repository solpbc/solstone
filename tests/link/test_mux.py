# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

import asyncio

import pytest

from solstone.convey.secure_listener.framing import (
    FLAG_CLOSE,
    FLAG_DATA,
    FLAG_PING,
    FLAG_PONG,
    FLAG_RESET,
    Frame,
    FrameDecoder,
    build_close,
    build_data,
    build_open,
    build_ping,
    build_pong,
)
from solstone.convey.secure_listener.mux import Multiplexer


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


# ---- streamID==0 PING/PONG keepalive responder ------------------------------


@pytest.mark.asyncio
async def test_ping_emits_matching_pong() -> None:
    sent: list[bytes] = []

    async def send(data: bytes) -> None:
        sent.append(data)

    async def handler(*_: object) -> None:
        pytest.fail("handler should not be invoked for control frames")

    nonce = bytes(range(1, 9))
    mux = Multiplexer(send, handler, is_listener=True)
    await mux.feed(build_ping(nonce).encode())

    frames = _decode_frames(sent)
    pongs = [f for f in frames if f.stream_id == 0 and f.flags & FLAG_PONG]
    assert len(pongs) == 1
    assert pongs[0].payload == nonce
    await mux.close()


@pytest.mark.asyncio
async def test_repeated_pings_each_get_matching_pong() -> None:
    sent: list[bytes] = []

    async def send(data: bytes) -> None:
        sent.append(data)

    async def handler(*_: object) -> None:
        pytest.fail("handler should not be invoked for control frames")

    nonces = [bytes([i]) * 8 for i in range(1, 6)]
    mux = Multiplexer(send, handler, is_listener=True)
    for nonce in nonces:
        await mux.feed(build_ping(nonce).encode())

    pongs = [f for f in _decode_frames(sent) if f.flags & FLAG_PONG]
    assert [p.payload for p in pongs] == nonces
    await mux.close()


@pytest.mark.asyncio
async def test_unsolicited_pong_is_silently_dropped() -> None:
    sent: list[bytes] = []

    async def send(data: bytes) -> None:
        sent.append(data)

    async def handler(*_: object) -> None:
        pytest.fail("handler should not be invoked for control frames")

    mux = Multiplexer(send, handler, is_listener=True)
    await mux.feed(build_pong(b"\x00" * 8).encode())

    # No emit on stray PONG — neither RESET nor PONG nor any other frame.
    assert sent == []
    await mux.close()


@pytest.mark.asyncio
async def test_ping_on_nonzero_stream_is_protocol_error() -> None:
    sent: list[bytes] = []

    async def send(data: bytes) -> None:
        sent.append(data)

    async def handler(*_: object) -> None:
        return

    mux = Multiplexer(send, handler, is_listener=True)
    # PING on stream 5 (illegal — control frames are streamID==0 only).
    illegal = Frame(stream_id=5, flags=FLAG_PING, payload=b"\x00" * 8).encode()
    await mux.feed(illegal)

    # Behavior parity with other top-level protocol errors: a RESET stamps the
    # tunnel as broken; we don't have streams to reset here, so the side effect
    # is internal teardown. The wire effect is no PONG emission.
    frames = _decode_frames(sent)
    assert not any(f.flags & FLAG_PONG for f in frames)
    await mux.close()


@pytest.mark.asyncio
async def test_pings_interleave_with_open_streams() -> None:
    # Keepalive cadence is 500ms, which is faster than most app traffic; the
    # responder must not block on or be blocked by concurrent data streams.
    handler_seen: dict[int, bytes] = {}

    async def handler(
        reader: asyncio.StreamReader, writer: object
    ) -> None:  # pragma: no cover - typed by mux
        payload = await reader.readuntil(b"\n")
        handler_seen[1] = payload
        await writer.write(b"ok\n")  # type: ignore[attr-defined]
        await writer.close()  # type: ignore[attr-defined]

    sent: list[bytes] = []

    async def send(data: bytes) -> None:
        sent.append(data)

    mux = Multiplexer(send, handler, is_listener=True)
    nonce = b"\xab" * 8
    # PING arrives mid-stream between OPEN and CLOSE.
    await mux.feed(build_open(1, b"hello\n").encode())
    await mux.feed(build_ping(nonce).encode())
    await mux.feed(build_close(1).encode())

    for _ in range(20):
        await asyncio.sleep(0.005)
        if handler_seen.get(1):
            break

    assert handler_seen.get(1) == b"hello\n"
    frames = _decode_frames(sent)
    pongs = [f for f in frames if f.flags & FLAG_PONG]
    assert len(pongs) == 1 and pongs[0].payload == nonce
    data_payload = b"".join(f.payload for f in frames if f.flags & FLAG_DATA)
    assert data_payload == b"ok\n"
    await mux.close()
