# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Asyncio accept loop for secure PL listener connections."""

from __future__ import annotations

import asyncio
import contextlib
import ipaddress
import logging
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from OpenSSL import SSL

from solstone.think.link.auth import AuthorizedClients

from .identity import ConveyIdentity
from .mux import Multiplexer, StreamWriter
from .tls import TlsError, drive_tls, new_server
from .wsgi import dispatch_stream

CallosumEmit = Callable[[str, dict[str, Any]], None]

log = logging.getLogger("convey.secure_listener.accept")


class SecureListener:
    def __init__(
        self,
        *,
        app: Any,
        tls_ctx: SSL.Context,
        authorized: AuthorizedClients,
        executor: ThreadPoolExecutor,
        callosum_emit: CallosumEmit | None = None,
        host: str = "0.0.0.0",
        port: int = 7657,
        logger: logging.Logger | None = None,
    ) -> None:
        self._app = app
        self._tls_ctx = tls_ctx
        self._authorized = authorized
        self._executor = executor
        self._emit = callosum_emit or (lambda _event, _fields: None)
        self._host = host
        self._port = port
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
        self._log.info("secure_listener bound on %s:%d", self._host, self._port)

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
            name="secure-listener-connection",
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _on_connect(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        connection_id = uuid.uuid4().hex
        mode = _mode_from_peername(writer.get_extra_info("peername"))
        self._log.info(
            "secure connection accepted conn=%s mode=%s",
            connection_id,
            mode,
        )
        reason = "eof"
        try:
            await self._pump_connection(reader, writer, connection_id, mode)
        except TlsError as exc:
            reason = _tls_close_reason(exc)
            self._log.warning(
                "secure connection closed conn=%s reason=%s",
                connection_id,
                reason,
            )
        except (BrokenPipeError, ConnectionResetError):
            reason = "reset"
            self._log.info(
                "secure connection closed conn=%s reason=%s",
                connection_id,
                reason,
            )
        except asyncio.CancelledError:
            reason = "cancelled"
            raise
        except Exception:
            reason = "error"
            self._log.exception(
                "secure connection closed conn=%s reason=%s",
                connection_id,
                reason,
            )
        else:
            self._log.info(
                "secure connection closed conn=%s reason=%s",
                connection_id,
                reason,
            )
        finally:
            writer.close()
            with contextlib.suppress(OSError, RuntimeError):
                await writer.wait_closed()

    async def _pump_connection(
        self,
        tcp_reader: asyncio.StreamReader,
        tcp_writer: asyncio.StreamWriter,
        connection_id: str,
        mode: str,
    ) -> None:
        tls = new_server(self._tls_ctx)
        send_queue: asyncio.Queue[bytes] = asyncio.Queue()
        identity: ConveyIdentity | None = None

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
            if identity is None:
                await writer.reset()
                return
            self._log.debug(
                "secure stream opened conn=%s stream_id=%d",
                connection_id,
                writer.stream_id,
            )
            try:
                await dispatch_stream(
                    self._app,
                    identity,
                    reader,
                    writer,
                    asyncio.get_running_loop(),
                    self._executor,
                )
            finally:
                self._log.debug(
                    "secure stream closed conn=%s stream_id=%d",
                    connection_id,
                    writer.stream_id,
                )

        mux = Multiplexer(send_frame, handle_stream, is_listener=True)

        async def tcp_reader_loop() -> None:
            nonlocal identity
            while True:
                inbound = await tcp_reader.read(65536)
                if not inbound:
                    return
                outbound, plaintext = drive_tls(tls, inbound=inbound)
                await write_ciphertext(outbound)
                if identity is None and tls.handshake_done and tls.peer_fingerprint:
                    identity = self._identity_for_peer(mode, tls.peer_fingerprint)
                    self._authorized.touch_last_seen(tls.peer_fingerprint)
                    self._emit(
                        "last_seen",
                        {
                            "fingerprint": tls.peer_fingerprint,
                            "tunnel_id": connection_id,
                        },
                    )
                    self._log.info(
                        "secure TLS handshake completed conn=%s fingerprint_short=%s",
                        connection_id,
                        tls.peer_fingerprint.replace("sha256:", "")[:16],
                    )
                if plaintext:
                    await mux.feed(plaintext)
                await _drain_send_queue(tls, write_ciphertext, send_queue)

        async def app_writer_loop() -> None:
            while True:
                data = await send_queue.get()
                outbound = _encrypt(tls, data)
                await write_ciphertext(outbound)

        reader_task = asyncio.create_task(
            tcp_reader_loop(),
            name=f"secure-tcp-reader-{connection_id}",
        )
        writer_task = asyncio.create_task(
            app_writer_loop(),
            name=f"secure-tcp-writer-{connection_id}",
        )
        try:
            await reader_task
        finally:
            writer_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await writer_task
            await mux.close()

    def _identity_for_peer(self, mode: str, fingerprint: str) -> ConveyIdentity:
        entry = self._authorized.get(fingerprint)
        return ConveyIdentity(
            mode=mode,  # type: ignore[arg-type]
            fingerprint=fingerprint,
            device_label=entry.device_label if entry else None,
            paired_at=entry.paired_at if entry else None,
            session_id=None,
        )


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


def _mode_from_peername(peername: object) -> str:
    host = ""
    if isinstance(peername, tuple) and peername:
        host = str(peername[0])
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        return "pl-direct"
    mapped = getattr(addr, "ipv4_mapped", None)
    if addr.is_loopback or (mapped is not None and mapped.is_loopback):
        return "pl-via-spl"
    return "pl-direct"


def _tls_close_reason(exc: TlsError) -> str:
    text = str(exc).lower()
    if "certificate" in text or "verify" in text:
        return "verify_failed"
    return "tls_alert"


__all__ = ["SecureListener"]
