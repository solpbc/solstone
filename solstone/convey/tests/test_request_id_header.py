# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for Convey request ID response headers."""

from __future__ import annotations

import re

REQUEST_ID_RE = re.compile(r"[0123456789ABCDEFGHJKMNPQRSTVWXYZ]{12}")


def _assert_request_id_header(response) -> None:
    request_id = response.headers.get("X-Solstone-Request-Id")
    assert request_id is not None
    assert REQUEST_ID_RE.fullmatch(request_id)


def test_request_id_header_on_success(convey_env) -> None:
    env = convey_env()

    response = env.client.get("/login")

    _assert_request_id_header(response)


def test_request_id_header_on_not_found(convey_env) -> None:
    env = convey_env()

    response = env.client.get("/does-not-exist")

    assert response.status_code == 404
    _assert_request_id_header(response)
