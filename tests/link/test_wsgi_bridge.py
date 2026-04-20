# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import os
import time
from collections.abc import Iterator

import pytest
from flask import Flask, Response, jsonify, request, stream_with_context

from think.link.wsgi_bridge import ExchangeMetadata, serve_request


class _WriterStub:
    def __init__(self) -> None:
        self.writes: list[bytes] = []

    async def write(self, data: bytes) -> None:
        self.writes.append(data)

    async def drain(self) -> None:
        return None

    def close(self) -> None:
        return None

    async def wait_closed(self) -> None:
        return None

    def joined(self) -> bytes:
        return b"".join(self.writes)


def _build_app(*, propagate_exceptions: bool = False) -> Flask:
    app = Flask(__name__)
    app.config["PROPAGATE_EXCEPTIONS"] = propagate_exceptions

    @app.get("/hello")
    def hello() -> Response:
        return Response(b"hello", mimetype="text/plain")

    @app.get("/stream")
    def stream() -> Response:
        @stream_with_context
        def generate() -> Iterator[bytes]:
            for chunk in (b"part-1\n", b"part-2\n", b"part-3\n"):
                time.sleep(0.01)
                yield chunk

        return Response(generate(), mimetype="text/plain")

    @app.post("/upload")
    def upload() -> Response:
        body = request.get_data()
        return jsonify(
            {"sha256": hashlib.sha256(body).hexdigest(), "length": len(body)}
        )

    @app.get("/boom")
    def boom() -> Response:
        raise RuntimeError("bridge test failure")

    return app


def _make_reader(request_bytes: bytes) -> asyncio.StreamReader:
    reader = asyncio.StreamReader()
    reader.feed_data(request_bytes)
    reader.feed_eof()
    return reader


async def _serve(
    request_bytes: bytes,
    *,
    app: Flask | None = None,
    stream_id: int = 1,
) -> tuple[ExchangeMetadata, _WriterStub]:
    writer = _WriterStub()
    metadata = await serve_request(
        _make_reader(request_bytes),
        writer,
        (app or _build_app()).wsgi_app,
        stream_id=stream_id,
    )
    return metadata, writer


def _split_response(raw: bytes) -> tuple[bytes, bytes]:
    head, sep, body = raw.partition(b"\r\n\r\n")
    assert sep == b"\r\n\r\n"
    return head, body


@pytest.mark.asyncio
async def test_get_returns_200_and_body() -> None:
    meta, writer = await _serve(
        b"GET /hello HTTP/1.1\r\nHost: link.test\r\nContent-Length: 0\r\n\r\n",
    )

    head, body = _split_response(writer.joined())

    assert meta == ExchangeMetadata(
        method="GET",
        path="/hello",
        status=200,
        request_bytes=0,
        response_bytes=len(writer.joined()),
        stream_id=1,
    )
    assert set(ExchangeMetadata.__dataclass_fields__) == {
        "method",
        "path",
        "status",
        "request_bytes",
        "response_bytes",
        "stream_id",
    }
    assert head.startswith(b"HTTP/1.1 200 OK\r\n")
    assert body == b"hello"


@pytest.mark.asyncio
async def test_post_upload_roundtrips_sha256() -> None:
    payload = os.urandom(1024 * 1024)
    digest = hashlib.sha256(payload).hexdigest()
    request_bytes = (
        b"POST /upload HTTP/1.1\r\n"
        b"Host: link.test\r\n"
        b"Content-Type: application/octet-stream\r\n"
        b"Content-Length: " + str(len(payload)).encode("ascii") + b"\r\n\r\n" + payload
    )

    meta, writer = await _serve(request_bytes)
    head, body = _split_response(writer.joined())
    parsed = json.loads(body)

    assert meta.method == "POST"
    assert meta.path == "/upload"
    assert meta.status == 200
    assert meta.request_bytes == 1024 * 1024
    assert meta.response_bytes == len(writer.joined())
    assert head.startswith(b"HTTP/1.1 200 OK\r\n")
    assert parsed == {"sha256": digest, "length": 1024 * 1024}


@pytest.mark.asyncio
async def test_streaming_response_arrives_in_chunks() -> None:
    meta, writer = await _serve(
        b"GET /stream HTTP/1.1\r\nHost: link.test\r\nContent-Length: 0\r\n\r\n",
    )

    assert meta.status == 200
    assert meta.method == "GET"
    assert meta.path == "/stream"
    assert writer.writes[0].startswith(b"HTTP/1.1 200 OK\r\n")
    assert writer.writes[1:] == [b"part-1\n", b"part-2\n", b"part-3\n"]
    assert len(writer.writes) == 4
    assert meta.response_bytes == len(writer.joined())


@pytest.mark.asyncio
async def test_malformed_request_line_returns_400() -> None:
    meta, writer = await _serve(b"NOTAREALREQUEST\r\n\r\n")

    head, body = _split_response(writer.joined())

    assert meta.method == "-"
    assert meta.path == "-"
    assert meta.status == 400
    assert head.startswith(b"HTTP/1.1 400 bad request\r\n")
    assert body == b"bad request\n"


@pytest.mark.asyncio
async def test_wsgi_exception_returns_500() -> None:
    meta, writer = await _serve(
        b"GET /boom HTTP/1.1\r\nHost: link.test\r\nContent-Length: 0\r\n\r\n",
        app=_build_app(propagate_exceptions=True),
    )

    head, body = _split_response(writer.joined())

    assert meta.method == "GET"
    assert meta.path == "/boom"
    assert meta.status == 500
    assert head.startswith(b"HTTP/1.1 500 internal server error\r\n")
    assert body == b"internal server error\n"


@pytest.mark.asyncio
async def test_metadata_has_no_payload_fields() -> None:
    meta, _ = await _serve(
        b"GET /hello HTTP/1.1\r\nHost: link.test\r\nContent-Length: 0\r\n\r\n",
    )

    assert [field.name for field in dataclasses.fields(meta)] == [
        "method",
        "path",
        "status",
        "request_bytes",
        "response_bytes",
        "stream_id",
    ]
