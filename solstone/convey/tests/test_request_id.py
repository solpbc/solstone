# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for Convey request ID generation."""

from __future__ import annotations

import itertools
import re

import solstone.convey.utils as convey_utils

REQUEST_ID_RE = re.compile(r"[0123456789ABCDEFGHJKMNPQRSTVWXYZ]{12}")


def test_generate_request_id_shape() -> None:
    assert REQUEST_ID_RE.fullmatch(convey_utils.generate_request_id())


def test_generate_request_id_calls_are_unique(monkeypatch) -> None:
    timestamp_ms = itertools.count(1_700_000_000_000)
    monkeypatch.setattr(
        convey_utils.time,
        "time_ns",
        lambda: next(timestamp_ms) * 1_000_000,
    )
    monkeypatch.setattr(convey_utils.secrets, "token_bytes", lambda size: b"\x00\x01")

    ids = [convey_utils.generate_request_id() for _ in range(50)]

    assert len(set(ids)) == len(ids)
