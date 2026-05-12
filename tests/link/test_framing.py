# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Framing encode/decode round-trip + flag validation (forked from spl/home)."""

from __future__ import annotations

import pytest

from solstone.convey.secure_listener.framing import (
    CONTROL_NONCE_LEN,
    FLAG_CLOSE,
    FLAG_DATA,
    FLAG_OPEN,
    FLAG_PING,
    FLAG_PONG,
    FLAG_RESERVED_MASK,
    FLAG_RESET,
    FLAG_WINDOW,
    HEADER_LEN,
    RESET_PROTOCOL_ERROR,
    Frame,
    FrameDecoder,
    ProtocolError,
    build_close,
    build_data,
    build_open,
    build_ping,
    build_pong,
    build_reset,
    build_window,
    parse_control_nonce,
    parse_reset_reason,
    parse_window_credit,
    validate_flags,
)


def test_header_is_8_bytes() -> None:
    assert HEADER_LEN == 8


def test_encode_decode_roundtrip() -> None:
    original = Frame(stream_id=7, flags=FLAG_DATA, payload=b"hello world")
    encoded = original.encode()
    decoder = FrameDecoder()
    decoder.feed(encoded)
    got = decoder.next()
    assert got == original


def test_decoder_handles_fragmented_feeds() -> None:
    frame = Frame(stream_id=5, flags=FLAG_DATA, payload=b"fragmented")
    encoded = frame.encode()
    decoder = FrameDecoder()
    for byte in encoded:
        decoder.feed(bytes([byte]))
    assert decoder.next() == frame


def test_decoder_returns_none_when_incomplete() -> None:
    decoder = FrameDecoder()
    decoder.feed(b"\x00\x00\x00\x01")  # partial header
    assert decoder.next() is None


def test_multiple_frames_in_one_buffer() -> None:
    frames = [build_data(1, b"a"), build_data(3, b"bb"), build_data(5, b"ccc")]
    decoder = FrameDecoder()
    for f in frames:
        decoder.feed(f.encode())
    assert decoder.drain() == frames


def test_reserved_bits_rejected_on_encode() -> None:
    with pytest.raises(ProtocolError):
        Frame(stream_id=1, flags=FLAG_RESERVED_MASK, payload=b"").encode()


def test_reserved_bits_rejected_on_decode() -> None:
    bad = Frame(stream_id=1, flags=FLAG_DATA, payload=b"")
    encoded = bytearray(bad.encode())
    encoded[4] |= 0x80
    decoder = FrameDecoder()
    decoder.feed(bytes(encoded))
    with pytest.raises(ProtocolError):
        decoder.next()


def test_payload_length_bound() -> None:
    build_data(1, b"").encode()
    ok = Frame(stream_id=1, flags=FLAG_DATA, payload=b"x" * ((1 << 24) - 1))
    ok.encode()


def test_open_with_initial_bytes_carries_both_flags() -> None:
    f = build_open(1, b"seed")
    assert f.flags & FLAG_OPEN and f.flags & FLAG_DATA
    assert f.payload == b"seed"


def test_open_without_bytes_is_pure_open() -> None:
    f = build_open(1)
    assert f.flags & FLAG_OPEN
    assert not (f.flags & FLAG_DATA)


def test_data_with_close_carries_both() -> None:
    f = build_data(1, b"last", close=True)
    assert f.flags & FLAG_DATA and f.flags & FLAG_CLOSE


def test_close_carries_only_close_flag() -> None:
    f = build_close(1)
    assert f.flags & FLAG_CLOSE
    assert not (f.flags & FLAG_OPEN)


def test_window_credit_parse_roundtrip() -> None:
    f = build_window(1, 65536)
    assert parse_window_credit(f) == 65536


def test_reset_reason_parse_roundtrip() -> None:
    f = build_reset(1, RESET_PROTOCOL_ERROR)
    assert parse_reset_reason(f) == RESET_PROTOCOL_ERROR


def test_validate_flags_allows_only_legal_combos() -> None:
    for flag in (FLAG_OPEN, FLAG_DATA, FLAG_CLOSE, FLAG_RESET, FLAG_WINDOW):
        validate_flags(flag)
    validate_flags(FLAG_OPEN | FLAG_DATA)
    validate_flags(FLAG_DATA | FLAG_CLOSE)
    with pytest.raises(ProtocolError):
        validate_flags(FLAG_OPEN | FLAG_CLOSE)
    with pytest.raises(ProtocolError):
        validate_flags(FLAG_DATA | FLAG_WINDOW)


def test_window_frame_requires_4_byte_payload() -> None:
    f = Frame(stream_id=1, flags=FLAG_WINDOW, payload=b"abc")
    with pytest.raises(ProtocolError):
        parse_window_credit(f)


def test_reset_frame_requires_1_byte_payload() -> None:
    f = Frame(stream_id=1, flags=FLAG_RESET, payload=b"")
    with pytest.raises(ProtocolError):
        parse_reset_reason(f)


# ---- streamID==0 PING/PONG control frames -----------------------------------


def test_ping_round_trip_wire_format() -> None:
    nonce = bytes(range(1, 9))
    f = build_ping(nonce)
    encoded = f.encode()
    # 4-byte big-endian stream_id (0) + 1 flag byte + 3-byte big-endian length (8) + 8-byte nonce.
    assert encoded == bytes([0, 0, 0, 0, FLAG_PING, 0, 0, 8]) + nonce
    decoder = FrameDecoder()
    decoder.feed(encoded)
    got = decoder.next()
    assert got == f
    assert got.stream_id == 0
    assert got.flags == FLAG_PING


def test_pong_round_trip_wire_format() -> None:
    nonce = bytes(range(1, 9))
    f = build_pong(nonce)
    assert f.encode() == bytes([0, 0, 0, 0, FLAG_PONG, 0, 0, 8]) + nonce


def test_ping_requires_8_byte_nonce() -> None:
    with pytest.raises(ProtocolError):
        build_ping(b"short")
    with pytest.raises(ProtocolError):
        build_ping(b"")
    with pytest.raises(ProtocolError):
        build_ping(b"x" * 9)


def test_pong_requires_8_byte_nonce() -> None:
    with pytest.raises(ProtocolError):
        build_pong(b"short")


def test_parse_control_nonce_round_trip() -> None:
    nonce = b"\xde\xad\xbe\xef\xfe\xed\xfa\xce"
    assert parse_control_nonce(build_ping(nonce)) == nonce
    assert parse_control_nonce(build_pong(nonce)) == nonce


def test_parse_control_nonce_rejects_non_control_frame() -> None:
    f = build_data(1, b"x" * 8)
    with pytest.raises(ProtocolError):
        parse_control_nonce(f)


def test_parse_control_nonce_rejects_wrong_payload_length() -> None:
    f = Frame(stream_id=0, flags=FLAG_PING, payload=b"short")
    with pytest.raises(ProtocolError):
        parse_control_nonce(f)


def test_validate_flags_allows_solo_ping_and_pong() -> None:
    validate_flags(FLAG_PING)
    validate_flags(FLAG_PONG)


def test_validate_flags_rejects_ping_combined_with_other_flags() -> None:
    for other in (FLAG_OPEN, FLAG_DATA, FLAG_CLOSE, FLAG_RESET, FLAG_WINDOW, FLAG_PONG):
        with pytest.raises(ProtocolError):
            validate_flags(FLAG_PING | other)


def test_validate_flags_rejects_pong_combined_with_other_flags() -> None:
    for other in (FLAG_OPEN, FLAG_DATA, FLAG_CLOSE, FLAG_RESET, FLAG_WINDOW):
        with pytest.raises(ProtocolError):
            validate_flags(FLAG_PONG | other)


def test_control_nonce_len_is_eight() -> None:
    assert CONTROL_NONCE_LEN == 8


def test_reserved_mask_collapsed_to_bit_seven() -> None:
    # bits 5 and 6 are PING/PONG; only bit 7 (0x80) remains reserved.
    assert FLAG_RESERVED_MASK == 0x80
