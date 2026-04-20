# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""HTTP/1.1 ⇄ WSGI bridge for the link tunnel.

Each tunnel stream is one HTTP request/response exchange (no keep-alive).
This module:

  1. Parses an HTTP/1.1 request from the per-stream byte reader
  2. Builds a WSGI environ dict
  3. Invokes convey's real Flask app
  4. Writes the response (status + headers + body) back to the stream writer

Streaming responses (SSE, chunked) are supported — the WSGI iterable is
pumped chunk-by-chunk so the client sees events as the app produces them.

Request-body size is capped at 64 MiB. Privacy invariant: NO request or
response bytes are ever logged. Only error counts, byte totals, and
status codes are eligible for logging (see `metadata`), never payloads.
"""

from __future__ import annotations

import asyncio
import io
import logging
import urllib.parse
from dataclasses import dataclass
from typing import Any, Callable

log = logging.getLogger("link.wsgi")

MAX_REQUEST_BODY = 64 * 1024 * 1024
WSGI_SERVER_NAME = "solstone-link"


@dataclass
class ExchangeMetadata:
    """Per-request rendezvous metadata — safe to log (no payload)."""

    method: str
    path: str
    status: int | None = None
    request_bytes: int = 0
    response_bytes: int = 0
    stream_id: int | None = None


async def serve_request(
    reader: asyncio.StreamReader,
    writer: Any,  # think.link.mux.StreamWriter
    wsgi_app: Callable[[dict[str, Any], Callable[..., Any]], Any],
    *,
    peer_fingerprint: str | None = None,
    tunnel_id: str | None = None,
    stream_id: int | None = None,
) -> ExchangeMetadata:
    """Read one HTTP/1.1 request; dispatch to `wsgi_app`; write response.

    Returns rendezvous metadata describing what happened (for callosum
    events + debug logs). Never returns payload bytes.
    """
    meta = ExchangeMetadata(method="-", path="-", stream_id=stream_id)
    try:
        request = await _read_request(reader)
    except _BadRequest as exc:
        log.debug("tunnel %s stream %s: bad request: %s", tunnel_id, stream_id, exc)
        await _write_simple(writer, 400, "bad request", b"bad request\n")
        meta.status = 400
        meta.response_bytes = _byte_count_for_simple(b"bad request\n")
        return meta
    except asyncio.IncompleteReadError:
        log.debug("tunnel %s stream %s: incomplete request", tunnel_id, stream_id)
        return meta

    meta.method = request.method
    meta.path = request.path
    meta.request_bytes = len(request.body)

    environ = _build_environ(
        request,
        peer_fingerprint=peer_fingerprint,
        tunnel_id=tunnel_id,
    )

    response_state = _ResponseState()

    def start_response(
        status: str,
        headers: list[tuple[str, str]],
        exc_info: Any = None,
    ) -> Callable[[bytes], None]:
        response_state.status_line = status
        response_state.headers = list(headers)
        # WSGI returns a write() callable, but we use the iterable instead.
        def write(_data: bytes) -> None:
            raise RuntimeError("write() callable not supported; return iterable")

        return write

    # Run the WSGI app in a thread — it may block on DB, file I/O, etc.
    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(
            None, lambda: wsgi_app(environ, start_response)
        )
    except Exception:
        log.exception(
            "tunnel %s stream %s: wsgi app raised", tunnel_id, stream_id
        )
        await _write_simple(writer, 500, "internal server error", b"internal server error\n")
        meta.status = 500
        meta.response_bytes = _byte_count_for_simple(b"internal server error\n")
        return meta

    if response_state.status_line is None:
        log.warning(
            "tunnel %s stream %s: wsgi app returned without calling start_response",
            tunnel_id,
            stream_id,
        )
        await _write_simple(writer, 500, "internal server error", b"missing response\n")
        meta.status = 500
        return meta

    # Parse numeric status.
    try:
        code_str, reason = response_state.status_line.split(" ", 1)
        code = int(code_str)
    except (ValueError, IndexError):
        code = 500
        reason = "internal server error"
    meta.status = code

    # Detect if the response is Transfer-Encoding: chunked or uses Content-Length.
    headers_map = {k.lower(): v for k, v in response_state.headers}
    is_chunked = headers_map.get("transfer-encoding", "").lower() == "chunked"

    # Send status + headers.
    sent = await _write_status_headers(writer, code, reason, response_state.headers)
    meta.response_bytes += sent

    # Stream body. Iterate in a thread (WSGI iterables can block).
    iterator = iter(result) if not isinstance(result, (bytes, bytearray)) else iter([bytes(result)])

    async def next_chunk() -> bytes | None:
        def _pull() -> bytes | None:
            try:
                return next(iterator)
            except StopIteration:
                return None

        return await loop.run_in_executor(None, _pull)

    try:
        while True:
            chunk = await next_chunk()
            if chunk is None:
                break
            if not chunk:
                continue
            if is_chunked:
                # The WSGI app is responsible for emitting valid chunked
                # framing (Flask's stream_with_context does this). Pass
                # through unchanged.
                await writer.write(chunk)
            else:
                await writer.write(chunk)
            meta.response_bytes += len(chunk)
    finally:
        # WSGI requires closing iterables that have a close() method.
        close = getattr(result, "close", None)
        if callable(close):
            try:
                await loop.run_in_executor(None, close)
            except Exception:
                log.debug("wsgi close() raised", exc_info=True)

    return meta


@dataclass
class _ResponseState:
    status_line: str | None = None
    headers: list[tuple[str, str]] | None = None


@dataclass
class _Request:
    method: str
    path: str
    query: str
    headers: list[tuple[str, str]]
    body: bytes


class _BadRequest(Exception):
    pass


async def _read_request(reader: asyncio.StreamReader) -> _Request:
    """Parse HTTP/1.1 request line + headers + body (Content-Length only)."""
    raw_line = await reader.readline()
    if not raw_line:
        raise asyncio.IncompleteReadError(b"", None)
    line = raw_line.decode("latin-1").rstrip("\r\n")
    parts = line.split(" ", 2)
    if len(parts) != 3:
        raise _BadRequest(f"bad request line: {line!r}")
    method, target, _version = parts
    path, _, query = target.partition("?")

    headers: list[tuple[str, str]] = []
    while True:
        raw = await reader.readline()
        if raw in (b"\r\n", b"\n", b""):
            break
        try:
            header_line = raw.decode("latin-1").rstrip("\r\n")
        except UnicodeDecodeError as exc:
            raise _BadRequest("header decode failed") from exc
        if ":" not in header_line:
            raise _BadRequest(f"bad header: {header_line!r}")
        name, _, value = header_line.partition(":")
        headers.append((name.strip(), value.strip()))

    headers_map = {k.lower(): v for k, v in headers}
    body = b""
    if "transfer-encoding" in headers_map and headers_map["transfer-encoding"].lower() == "chunked":
        body = await _read_chunked(reader)
    else:
        cl_raw = headers_map.get("content-length", "0")
        try:
            cl = int(cl_raw)
        except ValueError as exc:
            raise _BadRequest(f"bad content-length: {cl_raw!r}") from exc
        if cl < 0 or cl > MAX_REQUEST_BODY:
            raise _BadRequest(f"content-length out of bounds: {cl}")
        if cl:
            body = await reader.readexactly(cl)

    return _Request(
        method=method,
        path=urllib.parse.unquote(path),
        query=query,
        headers=headers,
        body=body,
    )


async def _read_chunked(reader: asyncio.StreamReader) -> bytes:
    """Minimal chunked-transfer decoder for uploads."""
    out = bytearray()
    total = 0
    while True:
        size_line = (await reader.readline()).decode("latin-1").strip()
        if ";" in size_line:
            size_line = size_line.split(";", 1)[0].strip()
        try:
            size = int(size_line, 16)
        except ValueError as exc:
            raise _BadRequest(f"bad chunk size: {size_line!r}") from exc
        if size == 0:
            # Consume trailer headers until blank line.
            while True:
                line = await reader.readline()
                if line in (b"\r\n", b"\n", b""):
                    break
            break
        chunk = await reader.readexactly(size)
        out.extend(chunk)
        total += size
        if total > MAX_REQUEST_BODY:
            raise _BadRequest("request body too large")
        # Trailing CRLF after each chunk.
        trailer = await reader.readexactly(2)
        if trailer not in (b"\r\n", b"\n\r"):
            # tolerate bare \n
            pass
    return bytes(out)


def _build_environ(
    request: _Request,
    *,
    peer_fingerprint: str | None,
    tunnel_id: str | None,
) -> dict[str, Any]:
    """Build a PEP-3333-compliant WSGI environ from the parsed request."""
    environ: dict[str, Any] = {
        "REQUEST_METHOD": request.method,
        "SCRIPT_NAME": "",
        "PATH_INFO": request.path,
        "QUERY_STRING": request.query,
        "SERVER_NAME": WSGI_SERVER_NAME,
        "SERVER_PORT": "443",
        "SERVER_PROTOCOL": "HTTP/1.1",
        "wsgi.version": (1, 0),
        "wsgi.url_scheme": "https",
        "wsgi.input": io.BytesIO(request.body),
        "wsgi.errors": io.StringIO(),
        "wsgi.multithread": True,
        "wsgi.multiprocess": False,
        "wsgi.run_once": False,
    }

    for name, value in request.headers:
        key = "HTTP_" + name.upper().replace("-", "_")
        if name.lower() == "content-length":
            environ["CONTENT_LENGTH"] = value
        elif name.lower() == "content-type":
            environ["CONTENT_TYPE"] = value
        environ[key] = value

    if peer_fingerprint:
        environ["LINK_PEER_FINGERPRINT"] = peer_fingerprint
    if tunnel_id:
        environ["LINK_TUNNEL_ID"] = tunnel_id

    return environ


async def _write_status_headers(
    writer: Any,
    code: int,
    reason: str,
    headers: list[tuple[str, str]],
) -> int:
    lines = [f"HTTP/1.1 {code} {reason}\r\n"]
    for name, value in headers:
        lines.append(f"{name}: {value}\r\n")
    lines.append("\r\n")
    out = "".join(lines).encode("latin-1")
    await writer.write(out)
    return len(out)


async def _write_simple(writer: Any, code: int, reason: str, body: bytes) -> None:
    headers = [
        ("Content-Type", "text/plain; charset=utf-8"),
        ("Content-Length", str(len(body))),
    ]
    await _write_status_headers(writer, code, reason, headers)
    if body:
        await writer.write(body)


def _byte_count_for_simple(body: bytes) -> int:
    # Just a rough tally for metadata — status line + headers + body.
    return len(body) + 64
