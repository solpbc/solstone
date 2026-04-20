# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Multiplex driver: framing-layer state + per-stream asyncio I/O.

Bytes in from the TLS-plaintext side are fed into this module; we produce
frames to send back, and each logical stream surfaces as an
`asyncio.StreamReader`/`StreamWriter` pair that the HTTP app can drive.

Flow-control uses the 1 MiB initial window per the spl framing spec — this
side grants credit as bytes drain into the app; the peer uses its granted
credit to send more data. For MVP the default "grant on every drained
chunk" policy is fine.

Concurrent stream cap: 256 per direction. OPENs beyond cap RESET with
STREAM_LIMIT_EXCEEDED.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .framing import (
    FLAG_CLOSE,
    FLAG_DATA,
    FLAG_OPEN,
    FLAG_RESET,
    FLAG_WINDOW,
    INITIAL_WINDOW,
    MAX_CONCURRENT_STREAMS,
    RECOMMENDED_CHUNK,
    RESET_FLOW_CONTROL_ERROR,
    RESET_INTERNAL_ERROR,
    RESET_PROTOCOL_ERROR,
    RESET_STREAM_LIMIT_EXCEEDED,
    Frame,
    FrameDecoder,
    ProtocolError,
    build_close,
    build_data,
    build_open,
    build_reset,
    build_window,
    parse_reset_reason,
    parse_window_credit,
)

if TYPE_CHECKING:
    StreamHandler = Callable[[asyncio.StreamReader, "StreamWriter"], Awaitable[None]]
else:
    StreamHandler = object


@dataclass
class _StreamState:
    stream_id: int
    reader: asyncio.StreamReader
    reader_closed: bool = False
    writer_closed: bool = False
    send_credit: int = INITIAL_WINDOW
    recv_credit: int = INITIAL_WINDOW
    unacked_recv: int = 0
    credit_event: asyncio.Event = field(default_factory=asyncio.Event)
    task: asyncio.Task[None] | None = None


class StreamWriter:
    """Per-stream writer. Calls into the mux to emit DATA/CLOSE/RESET frames."""

    def __init__(self, mux: Multiplexer, state: _StreamState) -> None:
        self._mux = mux
        self._state = state

    async def write(self, data: bytes) -> None:
        if self._state.writer_closed:
            raise ConnectionError(f"stream {self._state.stream_id} writer is closed")
        view = memoryview(data)
        while view:
            chunk_len = min(len(view), RECOMMENDED_CHUNK, self._state.send_credit)
            if chunk_len <= 0:
                self._state.credit_event.clear()
                await self._state.credit_event.wait()
                continue
            chunk = bytes(view[:chunk_len])
            view = view[chunk_len:]
            self._state.send_credit -= chunk_len
            await self._mux._emit(build_data(self._state.stream_id, chunk))

    async def close(self) -> None:
        if self._state.writer_closed:
            return
        self._state.writer_closed = True
        await self._mux._emit(build_close(self._state.stream_id))

    async def reset(self, reason: int = RESET_INTERNAL_ERROR) -> None:
        if self._state.writer_closed and self._state.reader_closed:
            return
        self._state.writer_closed = True
        self._state.reader_closed = True
        await self._mux._emit(build_reset(self._state.stream_id, reason))
        self._state.reader.feed_eof()
        self._mux._forget(self._state.stream_id)


class Multiplexer:
    """Frame-level state. Caller pumps incoming bytes with `feed`."""

    def __init__(
        self,
        send_frame: Callable[[bytes], Awaitable[None]],
        handler: StreamHandler,
        *,
        is_listener: bool = True,
    ) -> None:
        """If `is_listener=True`, this side expects odd stream_ids from the peer."""
        self._decoder = FrameDecoder()
        self._send_frame = send_frame
        self._handler = handler
        self._is_listener = is_listener
        self._streams: dict[int, _StreamState] = {}
        self._closed = False

    async def feed(self, plaintext: bytes) -> None:
        if not plaintext:
            return
        self._decoder.feed(plaintext)
        while True:
            try:
                frame = self._decoder.next()
            except ProtocolError:
                await self._reset_all(RESET_PROTOCOL_ERROR)
                return
            if frame is None:
                return
            await self._dispatch(frame)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for state in list(self._streams.values()):
            state.reader.feed_eof()
            state.writer_closed = True
            if state.task and not state.task.done():
                state.task.cancel()
        self._streams.clear()

    async def _dispatch(self, frame: Frame) -> None:
        if frame.flags & FLAG_OPEN:
            if not self._valid_peer_stream_id(frame.stream_id):
                await self._emit(build_reset(frame.stream_id, RESET_PROTOCOL_ERROR))
                return
            if frame.stream_id in self._streams:
                await self._emit(build_reset(frame.stream_id, RESET_PROTOCOL_ERROR))
                return
            if len(self._streams) >= MAX_CONCURRENT_STREAMS:
                await self._emit(build_reset(frame.stream_id, RESET_STREAM_LIMIT_EXCEEDED))
                return
            state = self._open_stream(frame.stream_id)
            if frame.payload:
                state.reader.feed_data(frame.payload)
                state.recv_credit -= len(frame.payload)
            if frame.flags & FLAG_CLOSE:
                state.reader.feed_eof()
                state.reader_closed = True
            return

        maybe_state = self._streams.get(frame.stream_id)
        if maybe_state is None:
            await self._emit(build_reset(frame.stream_id, RESET_PROTOCOL_ERROR))
            return
        state = maybe_state

        if frame.flags & FLAG_DATA:
            if len(frame.payload) > state.recv_credit:
                await self._emit(build_reset(frame.stream_id, RESET_FLOW_CONTROL_ERROR))
                self._terminate(state)
                return
            state.reader.feed_data(frame.payload)
            state.recv_credit -= len(frame.payload)
            state.unacked_recv += len(frame.payload)
            if state.unacked_recv >= INITIAL_WINDOW // 2:
                grant = state.unacked_recv
                state.recv_credit += grant
                state.unacked_recv = 0
                await self._emit(build_window(frame.stream_id, grant))
        if frame.flags & FLAG_CLOSE:
            state.reader.feed_eof()
            state.reader_closed = True
            if state.writer_closed:
                self._forget(frame.stream_id)
        if frame.flags & FLAG_WINDOW:
            try:
                credit = parse_window_credit(frame)
            except ProtocolError:
                await self._emit(build_reset(frame.stream_id, RESET_PROTOCOL_ERROR))
                self._terminate(state)
                return
            state.send_credit += credit
            state.credit_event.set()
        if frame.flags & FLAG_RESET:
            try:
                _ = parse_reset_reason(frame)
            except ProtocolError:
                pass
            state.reader.feed_eof()
            self._terminate(state)

    def _open_stream(self, stream_id: int) -> _StreamState:
        reader = asyncio.StreamReader()
        state = _StreamState(stream_id=stream_id, reader=reader)
        state.credit_event.set()
        self._streams[stream_id] = state
        writer = StreamWriter(self, state)

        async def runner() -> None:
            try:
                await self._handler(reader, writer)
            except Exception:
                await writer.reset(RESET_INTERNAL_ERROR)
            finally:
                if not state.writer_closed:
                    try:
                        await writer.close()
                    except Exception:
                        pass
                self._forget(stream_id)

        state.task = asyncio.create_task(runner(), name=f"link-stream-{stream_id}")
        return state

    def _terminate(self, state: _StreamState) -> None:
        state.writer_closed = True
        state.reader_closed = True
        if state.task and not state.task.done():
            state.task.cancel()
        self._forget(state.stream_id)

    def _forget(self, stream_id: int) -> None:
        self._streams.pop(stream_id, None)

    def _valid_peer_stream_id(self, stream_id: int) -> bool:
        if stream_id == 0:
            return False
        return (stream_id % 2 == 1) if self._is_listener else (stream_id % 2 == 0)

    async def _emit(self, frame: Frame) -> None:
        if self._closed:
            return
        try:
            encoded = frame.encode()
        except ProtocolError:
            return
        await self._send_frame(encoded)

    async def _reset_all(self, reason: int) -> None:
        for state in list(self._streams.values()):
            await self._emit(build_reset(state.stream_id, reason))
            self._terminate(state)

    async def open_stream(
        self,
        initial_payload: bytes = b"",
    ) -> tuple[asyncio.StreamReader, StreamWriter]:
        next_id = self._next_local_stream_id()
        reader = asyncio.StreamReader()
        state = _StreamState(stream_id=next_id, reader=reader)
        state.credit_event.set()
        self._streams[next_id] = state
        writer = StreamWriter(self, state)
        frame = build_open(next_id, initial_payload)
        if initial_payload:
            state.send_credit -= len(initial_payload)
        await self._emit(frame)
        return reader, writer

    def _next_local_stream_id(self) -> int:
        start = 2 if self._is_listener else 1
        cur = start
        while cur in self._streams:
            cur += 2
            if cur > 0xFFFFFFFF:
                raise RuntimeError("stream_id space exhausted")
        return cur
