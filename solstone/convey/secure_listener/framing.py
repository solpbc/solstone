# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Multiplex framing per the spl protocol (see github.com/solpbc/spl proto/framing.md).

Wire format (8-byte header + payload):

    +------+------+---+-------+
    | sid4 | flg1 | len3      |  header
    +------+------+-----------+
    | payload (len bytes)     |
    +-------------------------+

This module implements encoding, decoding, the flag bitfield, the reset
reason codes, and a small `Stream` helper that tracks per-stream credit +
close state per the spec. Protocol violations (reserved bits, illegal flag
combos) raise `ProtocolError`; the caller is expected to translate that
into a RESET frame with reason PROTOCOL_ERROR.

The relay does not parse frames — this code runs on both tunnel endpoints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

# Flag bits — each frame must carry exactly one of OPEN / DATA / CLOSE /
# RESET / WINDOW / PING / PONG, except OPEN|DATA (open with initial bytes)
# and DATA|CLOSE (last data + half-close). PING and PONG ride on stream_id
# 0 only (control channel) and carry an 8-byte nonce.
FLAG_OPEN: Final[int] = 0x01
FLAG_DATA: Final[int] = 0x02
FLAG_CLOSE: Final[int] = 0x04
FLAG_RESET: Final[int] = 0x08
FLAG_WINDOW: Final[int] = 0x10
FLAG_PING: Final[int] = 0x20
FLAG_PONG: Final[int] = 0x40
FLAG_RESERVED_MASK: Final[int] = 0x80  # bit 7 must be zero on send

# Reset reason codes — 1-byte payload of RESET frames.
RESET_PROTOCOL_ERROR: Final[int] = 0x01
RESET_FLOW_CONTROL_ERROR: Final[int] = 0x02
RESET_STREAM_LIMIT_EXCEEDED: Final[int] = 0x03
RESET_INTERNAL_ERROR: Final[int] = 0x04
RESET_CANCEL: Final[int] = 0x05
RESET_UNSPECIFIED: Final[int] = 0xFF

# Spec constants.
HEADER_LEN: Final[int] = 8
MAX_PAYLOAD: Final[int] = (1 << 24) - 1  # 16 MiB - 1
INITIAL_WINDOW: Final[int] = 1 << 20  # 1 MiB
MAX_CONCURRENT_STREAMS: Final[int] = 256
RECOMMENDED_CHUNK: Final[int] = 64 * 1024
CONTROL_NONCE_LEN: Final[int] = 8


class ProtocolError(ValueError):
    """Raised when a frame violates the spl framing spec — caller should RESET."""


@dataclass(frozen=True)
class Frame:
    stream_id: int
    flags: int
    payload: bytes

    def encode(self) -> bytes:
        if not 0 <= self.stream_id <= 0xFFFFFFFF:
            raise ProtocolError(f"stream_id out of range: {self.stream_id}")
        if not 0 <= self.flags <= 0xFF:
            raise ProtocolError(f"flags out of range: {self.flags}")
        if self.flags & FLAG_RESERVED_MASK:
            raise ProtocolError(f"reserved flag bits set: {self.flags:#x}")
        length = len(self.payload)
        if length > MAX_PAYLOAD:
            raise ProtocolError(f"payload exceeds 16 MiB - 1: {length}")
        header = bytearray(HEADER_LEN)
        header[0:4] = self.stream_id.to_bytes(4, "big")
        header[4] = self.flags
        header[5:8] = length.to_bytes(3, "big")
        return bytes(header) + self.payload


@dataclass
class FrameDecoder:
    """Stream decoder. Feed bytes, pull frames until no complete frame remains.

    WebSocket message boundaries are ignored — we re-frame ourselves so that
    coalescing or fragmentation at the transport layer doesn't affect
    framing-layer semantics.
    """

    _buf: bytearray = field(default_factory=bytearray)

    def feed(self, data: bytes | bytearray | memoryview) -> None:
        self._buf.extend(data)

    def next(self) -> Frame | None:
        if len(self._buf) < HEADER_LEN:
            return None
        stream_id = int.from_bytes(self._buf[0:4], "big")
        flags = self._buf[4]
        length = int.from_bytes(self._buf[5:8], "big")
        if flags & FLAG_RESERVED_MASK:
            raise ProtocolError(f"reserved flag bits set: {flags:#x}")
        end = HEADER_LEN + length
        if len(self._buf) < end:
            return None
        payload = bytes(self._buf[HEADER_LEN:end])
        del self._buf[:end]
        return Frame(stream_id=stream_id, flags=flags, payload=payload)

    def drain(self) -> list[Frame]:
        out: list[Frame] = []
        while True:
            frame = self.next()
            if frame is None:
                return out
            out.append(frame)


def build_open(stream_id: int, payload: bytes = b"") -> Frame:
    flags = FLAG_OPEN
    if payload:
        flags |= FLAG_DATA
    return Frame(stream_id=stream_id, flags=flags, payload=payload)


def build_data(stream_id: int, payload: bytes, *, close: bool = False) -> Frame:
    if not payload and not close:
        return Frame(stream_id=stream_id, flags=FLAG_DATA, payload=b"")
    flags = FLAG_DATA
    if close:
        flags |= FLAG_CLOSE
    return Frame(stream_id=stream_id, flags=flags, payload=payload)


def build_close(stream_id: int, payload: bytes = b"") -> Frame:
    return Frame(stream_id=stream_id, flags=FLAG_CLOSE, payload=payload)


def build_reset(stream_id: int, reason: int = RESET_UNSPECIFIED) -> Frame:
    if not 0 <= reason <= 0xFF:
        raise ProtocolError(f"reset reason out of range: {reason}")
    return Frame(stream_id=stream_id, flags=FLAG_RESET, payload=bytes([reason]))


def build_window(stream_id: int, credit: int) -> Frame:
    if not 0 <= credit <= 0xFFFFFFFF:
        raise ProtocolError(f"window credit out of range: {credit}")
    return Frame(
        stream_id=stream_id, flags=FLAG_WINDOW, payload=credit.to_bytes(4, "big")
    )


def build_ping(nonce: bytes) -> Frame:
    if len(nonce) != CONTROL_NONCE_LEN:
        raise ProtocolError(
            f"PING nonce must be {CONTROL_NONCE_LEN} bytes, got {len(nonce)}"
        )
    return Frame(stream_id=0, flags=FLAG_PING, payload=bytes(nonce))


def build_pong(nonce: bytes) -> Frame:
    if len(nonce) != CONTROL_NONCE_LEN:
        raise ProtocolError(
            f"PONG nonce must be {CONTROL_NONCE_LEN} bytes, got {len(nonce)}"
        )
    return Frame(stream_id=0, flags=FLAG_PONG, payload=bytes(nonce))


def validate_flags(flags: int) -> None:
    if flags & FLAG_RESERVED_MASK:
        raise ProtocolError(f"reserved flag bits set: {flags:#x}")
    exclusive = flags & (
        FLAG_OPEN | FLAG_DATA | FLAG_CLOSE | FLAG_RESET | FLAG_WINDOW | FLAG_PING | FLAG_PONG
    )
    if not exclusive:
        raise ProtocolError(
            "frame has no OPEN/DATA/CLOSE/RESET/WINDOW/PING/PONG bit set"
        )
    allowed = {
        FLAG_OPEN,
        FLAG_DATA,
        FLAG_CLOSE,
        FLAG_RESET,
        FLAG_WINDOW,
        FLAG_PING,
        FLAG_PONG,
        FLAG_OPEN | FLAG_DATA,
        FLAG_DATA | FLAG_CLOSE,
    }
    if exclusive not in allowed:
        raise ProtocolError(f"illegal flag combination: {flags:#x}")


def parse_window_credit(frame: Frame) -> int:
    if frame.flags & (FLAG_OPEN | FLAG_DATA | FLAG_CLOSE | FLAG_RESET) or not (
        frame.flags & FLAG_WINDOW
    ):
        raise ProtocolError("not a WINDOW frame")
    if len(frame.payload) != 4:
        raise ProtocolError(f"WINDOW payload must be 4 bytes, got {len(frame.payload)}")
    return int.from_bytes(frame.payload, "big")


def parse_reset_reason(frame: Frame) -> int:
    if not (frame.flags & FLAG_RESET):
        raise ProtocolError("not a RESET frame")
    if len(frame.payload) != 1:
        raise ProtocolError(f"RESET payload must be 1 byte, got {len(frame.payload)}")
    return frame.payload[0]


def parse_control_nonce(frame: Frame) -> bytes:
    if not (frame.flags & (FLAG_PING | FLAG_PONG)):
        raise ProtocolError("not a PING/PONG frame")
    if len(frame.payload) != CONTROL_NONCE_LEN:
        raise ProtocolError(
            f"control nonce must be {CONTROL_NONCE_LEN} bytes, got {len(frame.payload)}"
        )
    return frame.payload
