# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

import base64
from pathlib import Path

import pytest

from tests.link.client import Client
from tests.link.live_helpers import (
    CONVEY_PASSWORD,
    RELAY_URL,
    running_convey_server,
    running_link_service,
    runtime_texts,
    skip_unless_live_relay,
)

pytestmark = pytest.mark.integration
skip_unless_live_relay()

FORBIDDEN_TOKENS = [
    "authorization",
    "bearer ",
    "cookie",
    "ca private",
    "BEGIN PRIVATE",
    "client_cert",
    "home_attestation",
    "x-api-key",
    "payload",
]


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_privacy_scan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tmp_journal = tmp_path / "journal"
    tmp_journal.mkdir()
    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_journal))

    capture = None
    with (
        running_convey_server(tmp_journal) as base_url,
        running_link_service(tmp_journal) as link_capture,
    ):
        capture = link_capture
        identity = Client.pair(base_url, device_label="pytest-device")
        enrolled = Client.enroll_device(RELAY_URL, identity)

        session = await Client.dial(RELAY_URL, enrolled)
        async with session:
            auth = base64.b64encode(f":{CONVEY_PASSWORD}".encode("utf-8")).decode(
                "ascii"
            )
            status, headers, _body = await session.request(
                "GET",
                "/app/link/api/status",
                headers={"authorization": f"Basic {auth}"},
            )
        assert status == 200
        assert headers["content-type"] == "application/json"

    assert capture is not None
    texts = runtime_texts(tmp_journal, capture)
    dynamic_tokens = [
        identity.home_attestation,
        enrolled.device_token,
        identity.client_cert_pem[:100],
        identity.home_attestation.split(".")[1],
    ]

    for token in FORBIDDEN_TOKENS:
        _assert_token_absent(texts, token, ignore_case=True)
    for token in dynamic_tokens:
        _assert_token_absent(texts, token, ignore_case=False)


def _assert_token_absent(
    texts: dict[str, str],
    token: str,
    *,
    ignore_case: bool,
) -> None:
    if not token:
        return
    needle = token.lower() if ignore_case else token
    for source, text in texts.items():
        haystack = text.lower() if ignore_case else text
        match_at = haystack.find(needle)
        if match_at < 0:
            continue
        line_start = text.rfind("\n", 0, match_at) + 1
        line_end = text.find("\n", match_at)
        if line_end < 0:
            line_end = len(text)
        context = text[line_start:line_end]
        raise AssertionError(
            f"forbidden token {token!r} found in {source}: {context!r}"
        )
