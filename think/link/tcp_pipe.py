# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Byte pump for tunnel→convey.

Opens a loopback TCP connection to the running convey server and pumps
bytes bidirectionally with half-close handling. Does not parse HTTP or
inspect payloads — the pipe carries whatever the tunnel stream carries.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from think.utils import read_service_port

from .mux import StreamWriter

log = logging.getLogger("link.pipe")

_BUF = 65536


@dataclass
class PipeMetadata:
    tunnel_id: str
    stream_id: int
    bytes_in: int
    bytes_out: int
    closed_reason: str


class ConveyUnreachable(RuntimeError): ...


async def _pump_up(src: asyncio.StreamReader, dst: asyncio.StreamWriter) -> int:
    total = 0
    try:
        while True:
            data = await src.read(_BUF)
            if not data:
                try:
                    dst.write_eof()
                except (OSError, RuntimeError):
                    pass
                return total
            dst.write(data)
            await dst.drain()
            total += len(data)
    except (BrokenPipeError, ConnectionResetError):
        return total


async def _pump_down(src: asyncio.StreamReader, dst: StreamWriter) -> int:
    total = 0
    try:
        while True:
            data = await src.read(_BUF)
            if not data:
                await dst.close()
                return total
            await dst.write(data)
            total += len(data)
    except (BrokenPipeError, ConnectionResetError):
        return total


async def pump_stream(
    reader: asyncio.StreamReader,
    writer: StreamWriter,
    *,
    tunnel_id: str,
    stream_id: int,
) -> PipeMetadata:
    port = read_service_port("convey")
    if port is None:
        log.debug(
            "convey unreachable for stream tunnel=%s stream_id=%d: no convey port",
            tunnel_id,
            stream_id,
        )
        raise ConveyUnreachable("convey port not published")

    try:
        tcp_reader, tcp_writer = await asyncio.open_connection("127.0.0.1", port)
    except (ConnectionRefusedError, OSError) as err:
        log.debug(
            "convey unreachable for stream tunnel=%s stream_id=%d: %s",
            tunnel_id,
            stream_id,
            err,
        )
        raise ConveyUnreachable(str(err)) from err

    bytes_in = 0
    bytes_out = 0
    closed_reason = "both_eof"
    try:
        if hasattr(asyncio, "TaskGroup"):
            async with asyncio.TaskGroup() as tg:
                up = tg.create_task(_pump_up(reader, tcp_writer))
                down = tg.create_task(_pump_down(tcp_reader, writer))
            bytes_in, bytes_out = up.result(), down.result()
        else:
            tasks = [
                asyncio.create_task(_pump_up(reader, tcp_writer)),
                asyncio.create_task(_pump_down(tcp_reader, writer)),
            ]
            try:
                bytes_in, bytes_out = await asyncio.gather(*tasks)
            except BaseException:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise
        if reader.at_eof() and tcp_reader.at_eof():
            closed_reason = "both_eof"
        elif reader.at_eof():
            closed_reason = "client_eof"
        elif tcp_reader.at_eof():
            closed_reason = "server_eof"
    except asyncio.CancelledError:
        closed_reason = "cancelled"
        raise
    except Exception:
        closed_reason = "error"
        raise
    finally:
        try:
            tcp_writer.close()
            await tcp_writer.wait_closed()
        except (OSError, RuntimeError):
            pass

    return PipeMetadata(tunnel_id, stream_id, bytes_in, bytes_out, closed_reason)
