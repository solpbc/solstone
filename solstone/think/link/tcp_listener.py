# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""LAN-direct TCP listener for link tunnels."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import uuid
from collections.abc import Callable
from typing import Any

from OpenSSL import SSL

from .auth import AuthorizedClients
from .mux import Multiplexer, StreamWriter
from .tcp_pipe import ConveyUnreachable, PipeMetadata, pump_stream
from .tls_adapter import TlsError, drive_tls, new_server

CallosumEmit = Callable[[str, dict[str, Any]], None]

log = logging.getLogger("link.tcp_listener")


class TcpListener:
    def __init__(
        self,
        *,
        host: str = "0.0.0.0",
        port: int = 7657,
        tls_ctx: SSL.Context,
        authorized: AuthorizedClients,
        callosum_emit: CallosumEmit | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._host = host
        self._port = port
        self._tls_ctx = tls_ctx
        self._authorized = authorized
        self._emit = callosum_emit or (lambda _event, _fields: None)
        self._log = logger or log
        self._server: asyncio.AbstractServer | None = None
        self._tasks: set[asyncio.Task[None]] = set()

    async def start(self) -> None:
        if self._server is not None:
            return
        self._server = await asyncio.start_server(
            self._accept,
            host=self._host,
            port=self._port,
        )
        self._log.info("tcp listener bound on %s:%d", self._host, self._port)

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        if not self._tasks:
            return
        done, pending = await asyncio.wait(self._tasks, timeout=2.0)
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        for task in done:
            with contextlib.suppress(Exception):
                task.result()
        self._tasks.clear()

    def _accept(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        task = asyncio.create_task(
            self._on_connect(reader, writer),
            name="link-tcp-connection",
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _on_connect(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        connection_id = uuid.uuid4().hex
        self._log.info({"event": "accept", "conn": connection_id})
        reason = "eof"
        try:
            await self._pump_connection(reader, writer, connection_id)
        except TlsError as exc:
            reason = _tls_close_reason(exc)
            self._log.warning(
                {"event": "close", "conn": connection_id, "reason": reason}
            )
        except (BrokenPipeError, ConnectionResetError):
            reason = "reset"
            self._log.info({"event": "close", "conn": connection_id, "reason": reason})
        except asyncio.CancelledError:
            reason = "cancelled"
            raise
        except Exception:
            reason = "error"
            self._log.exception(
                {"event": "close", "conn": connection_id, "reason": reason}
            )
        else:
            self._log.info({"event": "close", "conn": connection_id, "reason": reason})
        finally:
            writer.close()
            with contextlib.suppress(OSError, RuntimeError):
                await writer.wait_closed()

    async def _pump_connection(
        self,
        tcp_reader: asyncio.StreamReader,
        tcp_writer: asyncio.StreamWriter,
        connection_id: str,
    ) -> None:
        tls = new_server(self._tls_ctx)
        send_queue: asyncio.Queue[bytes] = asyncio.Queue()
        fingerprint_touched = False

        async def write_ciphertext(data: bytes) -> None:
            if not data:
                return
            tcp_writer.write(data)
            await tcp_writer.drain()

        async def send_frame(frame: bytes) -> None:
            send_queue.put_nowait(frame)

        async def handle_stream(
            reader: asyncio.StreamReader,
            writer: StreamWriter,
        ) -> None:
            try:
                meta: PipeMetadata = await pump_stream(
                    reader,
                    writer,
                    tunnel_id=connection_id,
                    stream_id=writer.stream_id,
                )
            except ConveyUnreachable:
                raise
            self._log.debug(
                "link stream closed tunnel=%s stream_id=%d bytes_in=%d bytes_out=%d reason=%s",
                meta.tunnel_id,
                meta.stream_id,
                meta.bytes_in,
                meta.bytes_out,
                meta.closed_reason,
            )

        mux = Multiplexer(send_frame, handle_stream, is_listener=True)

        async def tcp_reader_loop() -> None:
            nonlocal fingerprint_touched
            while True:
                inbound = await tcp_reader.read(65536)
                if not inbound:
                    return
                outbound, plaintext = drive_tls(tls, inbound=inbound)
                await write_ciphertext(outbound)
                if plaintext:
                    await mux.feed(plaintext)
                if (
                    not fingerprint_touched
                    and tls.handshake_done
                    and tls.peer_fingerprint
                ):
                    fingerprint_touched = True
                    self._authorized.touch_last_seen(tls.peer_fingerprint)
                    self._emit(
                        "last_seen",
                        {
                            "fingerprint": tls.peer_fingerprint,
                            "tunnel_id": connection_id,
                        },
                    )
                await _drain_send_queue(tls, write_ciphertext, send_queue)

        async def app_writer_loop() -> None:
            while True:
                data = await send_queue.get()
                outbound = _encrypt(tls, data)
                await write_ciphertext(outbound)

        reader_task = asyncio.create_task(
            tcp_reader_loop(),
            name=f"tcp-reader-{connection_id}",
        )
        writer_task = asyncio.create_task(
            app_writer_loop(),
            name=f"tcp-writer-{connection_id}",
        )
        try:
            await reader_task
        finally:
            writer_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await writer_task
            await mux.close()


def _encrypt(tls: Any, plaintext: bytes) -> bytes:
    outbound, _ = drive_tls(tls, inbound=b"", plaintext_out=plaintext)
    return outbound


async def _drain_send_queue(
    tls: Any,
    write_ciphertext: Callable[[bytes], asyncio.Future[Any] | Any],
    queue: asyncio.Queue[bytes],
) -> None:
    drained: list[bytes] = []
    while not queue.empty():
        try:
            drained.append(queue.get_nowait())
        except asyncio.QueueEmpty:
            break
    for chunk in drained:
        outbound = _encrypt(tls, chunk)
        result = write_ciphertext(outbound)
        if asyncio.iscoroutine(result):
            await result


def _tls_close_reason(exc: TlsError) -> str:
    text = str(exc).lower()
    if "certificate" in text or "verify" in text:
        return "verify_failed"
    return "tls_alert"


__all__ = ["TcpListener"]
