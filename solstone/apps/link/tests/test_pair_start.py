# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Regression tests for the link pair-start response contract."""

from __future__ import annotations

import re

PAIR_START_KEYS = [
    "nonce",
    "pair_link",
    "manual_code",
    "expires_in",
    "device_label",
    "lan_url",
    "ca_fingerprint",
]


def test_pair_start_shape_and_locked_order(link_env) -> None:
    env = link_env()

    response = env.client.post(
        "/app/link/pair-start",
        json={"device_label": "Test Phone"},
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert list(payload.keys()) == PAIR_START_KEYS
    assert re.fullmatch(
        r"https://link\.solpbc\.org/p#h=[^&]+&t=[a-f0-9]+&f=[a-f0-9]+&l=[^&]*&v=1",
        payload["pair_link"],
    )
    assert re.fullmatch(r"^[A-Z2-9]{4}-[A-Z2-9]{4}$", payload["manual_code"])
    assert "://" not in payload["lan_url"]
    assert "pair_url" not in payload
    assert "qr_payload" not in payload


def test_pair_start_mints_distinct_nonce_and_manual_code(link_env) -> None:
    env = link_env()

    first = env.client.post(
        "/app/link/pair-start",
        json={"device_label": "First Phone"},
    ).get_json()
    second = env.client.post(
        "/app/link/pair-start",
        json={"device_label": "Second Phone"},
    ).get_json()

    assert first["nonce"] != second["nonce"]
    assert first["manual_code"] != second["manual_code"]
